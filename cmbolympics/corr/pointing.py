# Copyright (C) 2025 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Pointing measurements on HEALPix maps."""

import os
import tempfile
from contextlib import contextmanager

import healpy as hp
import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange
from tqdm.auto import tqdm


def pointing_enclosed_profile(map_in, ell_deg, b_deg, radii_arcmin, mask=None):
    """
    Mean enclosed profile around some pointing (ell_deg, b_deg) in
    degrees, `radii_arcmin` can be a scalar or array of radii in arcminutes.
    """
    assert isinstance(ell_deg, (float, int)), "`ell_deg` must be a scalar."
    assert isinstance(b_deg, (float, int)), "`b_deg` must be a scalar"

    radii_arcmin = np.atleast_1d(radii_arcmin)
    nside = hp.get_nside(map_in)
    if mask is None:
        mask = np.isfinite(map_in) & (map_in != hp.UNSEEN)
    else:
        mask = (mask > 0) & np.isfinite(map_in) & (map_in != hp.UNSEEN)

    th0 = np.radians(90.0 - b_deg)
    ph0 = np.radians(ell_deg)
    v0 = hp.ang2vec(th0, ph0)

    out = np.full(len(radii_arcmin), np.nan)
    for i, r_arcmin in enumerate(radii_arcmin):
        r_rad = np.radians(r_arcmin / 60.0)
        idx = hp.query_disc(nside, v0, r_rad, inclusive=False, fact=4)
        good = mask[idx]
        if np.any(good):
            out[i] = np.mean(map_in[idx][good])

    # If input was scalar, return scalar mean for convenience
    if out.size == 1:
        return out[0]

    return out


def pointing_enclosed_profile_per_source(map_in, ell_deg, b_deg, radii_arcmin,
                                         mask=None):
    """
    Mean enclosed profile around multiple pointings (ell_deg, b_deg)
    in degrees, with sizes of `radii_arcmin`.
    """
    # Parse and check inputs
    ell_deg = np.atleast_1d(ell_deg)
    b_deg = np.atleast_1d(b_deg)
    radii_arcmin = np.atleast_1d(radii_arcmin)

    ell_deg = np.asarray(ell_deg)
    b_deg = np.asarray(b_deg)
    radii_arcmin = np.asarray(radii_arcmin)
    assert ell_deg.shape == b_deg.shape == radii_arcmin.shape, "Input arrays must have the same shape."  # noqa

    n_pointings = ell_deg.size

    out = np.full(n_pointings, np.nan)
    for i in trange(n_pointings, desc="Measuring profiles"):
        out[i] = pointing_enclosed_profile(
           map_in, ell_deg[i], b_deg[i], radii_arcmin[i], mask=mask)

    return out  # (n_pointings,)


@contextmanager
def tqdm_joblib(tqdm_object):
    class _TQDMCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TQDMCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def randpoint_enclosed_profiles(map_in, radii_arcmin, n_points, abs_b_min=None,
                                mask=None, seed=None, n_jobs=1,
                                prefer="processes", batch_size="auto"):
    """
    Mean radial profiles at `n_points` random HEALPix positions with radii
    `radii_arcmin`. If `abs_b_min` (deg) is given, require |b| >= abs_b_min.
    Parallelised with joblib+tqdm. Memory-maps `map_in` and cleans up.

    `prefer` can be "threads" or "processes".
    """
    gen = np.random.default_rng(seed)

    if abs_b_min is not None:
        smin = np.sin(np.deg2rad(abs_b_min))
        b = np.arcsin(gen.uniform(smin, 1.0, size=n_points))
        b[gen.random(n_points) < 0.5] *= -1.0
    else:
        b = np.arcsin(gen.uniform(-1.0, 1.0, size=n_points))

    b_deg = np.rad2deg(b)
    ell_deg = np.rad2deg(gen.uniform(0.0, 2.0 * np.pi, size=n_points))
    radii_arcmin = np.asarray(radii_arcmin)

    with tempfile.NamedTemporaryFile(suffix=".mmap", delete=False) as tmp:
        mmap_path = tmp.name
    try:
        joblib.dump(map_in, mmap_path, compress=0)
        map_in_mm = joblib.load(mmap_path, mmap_mode="r")

        if n_jobs == 1:
            results = []
            for i in tqdm(range(n_points), desc="Measuring profiles"):
                prof = pointing_enclosed_profile(
                    map_in_mm, ell_deg[i], b_deg[i], radii_arcmin, mask=mask
                )
                results.append(prof)
        else:
            with tqdm_joblib(tqdm(total=n_points, desc="Measuring profiles")):
                results = Parallel(
                    n_jobs=n_jobs, prefer=prefer, batch_size=batch_size
                )(
                    delayed(pointing_enclosed_profile)(
                        map_in_mm, ell_deg[i], b_deg[i], radii_arcmin,
                        mask=mask) for i in range(n_points)
                )
    finally:
        if os.path.exists(mmap_path):
            os.remove(mmap_path)

    profiles, num_skipped = [], 0
    for prof in results:
        if np.any(np.isnan(prof)):
            num_skipped += 1
        else:
            profiles.append(prof)

    print(f"Skipped {num_skipped} / {n_points} profiles due to NaNs.")
    return np.asarray(profiles, dtype=float)


def get_pointing_pvalue(signal_source, theta200, theta_rand, signal_rand):
    """
    Get empirical p-values for source signals based on random pointing signals.

    Defined as the fraction of random signals greater than or equal to the
    source signal at the closest matching aperture size.
    """
    assert theta_rand.ndim == 1 and signal_rand.ndim == 2
    assert signal_rand.shape[1] == len(theta_rand)

    signal_rand = np.sort(signal_rand, axis=0)
    nr = len(signal_rand)

    pval = np.full(len(theta200), np.nan)

    for i in range(len(theta200)):
        k = np.argmin(np.abs(theta_rand - theta200[i]))
        pval[i] = 1 - np.searchsorted(signal_rand[:, k], signal_source[i]) / nr

    return pval
