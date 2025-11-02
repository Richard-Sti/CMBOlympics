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


def pointing_enclosed_profile_per_source(map_in, ell_deg, b_deg, radii_arcmin,
                                         mask=None, n_jobs=-1,
                                         prefer="processes",
                                         batch_size="auto"):
    """
    Mean enclosed profile around multiple pointings (ell_deg, b_deg)
    with sizes of `radii_arcmin` (one per pointing).
    """
    ell_deg = np.asarray(np.atleast_1d(ell_deg))
    b_deg = np.asarray(np.atleast_1d(b_deg))
    radii_arcmin = np.asarray(np.atleast_1d(radii_arcmin))
    assert ell_deg.shape == b_deg.shape == radii_arcmin.shape

    n = ell_deg.size
    out = np.full(n, np.nan, dtype=float)

    with tempfile.NamedTemporaryFile(suffix=".mmap", delete=False) as tmp:
        mmap_path = tmp.name
    try:
        joblib.dump(map_in, mmap_path, compress=0)
        map_mm = joblib.load(mmap_path, mmap_mode="r")

        if n_jobs == 1:
            for i in tqdm(range(n), desc="Measuring profiles"):
                out[i] = pointing_enclosed_profile(
                    map_mm, ell_deg[i], b_deg[i], radii_arcmin[i], mask=mask
                )
        else:
            from contextlib import nullcontext
            ctx = tqdm_joblib(tqdm(total=n, desc="Measuring profiles")) \
                if 'tqdm_joblib' in globals() else nullcontext()

            with ctx:
                res = Parallel(n_jobs=n_jobs, prefer=prefer,
                               batch_size=batch_size)(
                    delayed(pointing_enclosed_profile)(
                        map_mm, ell_deg[i], b_deg[i], radii_arcmin[i],
                        mask=mask)
                    for i in range(n)
                )
            out[:] = np.asarray(res, dtype=float)
    finally:
        if os.path.exists(mmap_path):
            os.remove(mmap_path)

    return out  # (n,)


def random_sky_positions(n_points, abs_b_min=None, seed=None):
    """
    Generate random sky positions in Galactic coordinates.

    Parameters
    ----------
    n_points : int
        Number of random positions to generate.
    abs_b_min : float or None, optional
        If given, require |b| >= abs_b_min [deg].
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    ell_deg : ndarray
        Galactic longitudes [deg].
    b_deg : ndarray
        Galactic latitudes [deg].
    gen : np.random.Generator
        The random number generator used.
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

    return ell_deg, b_deg, gen


def randpoint_enclosed_profiles(map_in, radii_arcmin, n_points, abs_b_min=None,
                                mask=None, seed=None, n_jobs=1,
                                prefer="processes", batch_size="auto"):
    """
    Mean radial profiles at `n_points` random HEALPix positions with radii
    `radii_arcmin`. If `abs_b_min` (deg) is given, require |b| >= abs_b_min.
    Parallelised with joblib+tqdm. Memory-maps `map_in` and cleans up.

    `prefer` can be "threads" or "processes".
    """
    ell_deg, b_deg, __ = random_sky_positions(n_points, abs_b_min, seed)
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


###############################################################################
#              Class for 2D cutouts from (scalar) HEALPix maps                #
###############################################################################


class Pointing2DCutout:
    """
    Class for extracting and normalizing 2D cutouts from HEALPix maps.

    Parameters
    ----------
    m : array_like
        HEALPix map (Galactic), RING by default unless nest=True.
    npix : int, optional
        Pixels per side for cutouts (odd keeps center on pixel). Default: 151.
    nest : bool, optional
        True if map is NESTED ordering, else RING. Default: False.
    mask : array_like or None, optional
        HEALPix mask in the same NSIDE/order as map. Default: None.
    nbins : int, optional
        Number of bins for normalized grid. Default: 40.
    grid_halfsize : float or None, optional
        Half-size of normalized grid in theta200 units. If None, auto-sized.
        Default: None.
    """

    def __init__(self, m, npix=301, nest=False, mask=None, nbins=128,
                 grid_halfsize=None):
        self.m = m
        self.npix = npix
        self.nest = nest
        self.mask = mask
        self.nbins = nbins
        self.grid_halfsize = grid_halfsize

    def get_cutout_2d(self, ell_deg, b_deg, size_arcmin):
        """
        Flat-sky cutout (nearest-neighbour) from a HEALPix map in Galactic
        coordinates

        Parameters
        ----------
        ell_deg, b_deg : float
            Center in Galactic longitude/latitude [deg].
        size_arcmin : float
            Side length of the square cutout [arcmin].

        Returns
        -------
        cutout : ndarray, shape (npix, npix)
            Cutout array; invalid pixels are NaN.
        extent : tuple
            (xmin, xmax, ymin, ymax) in arcmin for imshow.
        """
        npix = self.npix
        if npix % 2 == 0:
            npix += 1

        arcmin2rad = np.pi / (180.0 * 60.0)
        size_rad = size_arcmin * arcmin2rad
        step = size_rad / npix

        x = -size_rad/2 + (np.arange(npix) + 0.5) * step
        y = -size_rad/2 + (np.arange(npix) + 0.5) * step
        xg, yg = np.meshgrid(x, y, indexing="xy")

        ell0 = np.deg2rad(float(ell_deg))
        b0 = np.deg2rad(float(b_deg))

        c = np.hypot(xg, yg)
        A = np.arctan2(xg, yg)

        sb0, cb0 = np.sin(b0), np.cos(b0)
        sc, cc = np.sin(c),  np.cos(c)

        b = np.arcsin(sb0*cc + cb0*sc*np.cos(A))
        d_ell = np.arctan2(sc*np.sin(A), cb0*cc - sb0*sc*np.cos(A))
        ell = (ell0 + d_ell) % (2.0 * np.pi)

        theta = np.pi/2 - b
        phi = ell

        nside = hp.get_nside(self.m)
        pix = hp.ang2pix(nside, theta, phi, nest=self.nest)

        # Base cutout from map
        cutout = np.asarray(self.m[pix], dtype=float)

        # Invalid if map is UNSEEN or non-finite
        invalid = ~np.isfinite(cutout)
        if np.any(self.m == hp.UNSEEN):
            invalid |= (self.m[pix] == hp.UNSEEN)

        # Apply mask if provided
        if self.mask is not None:
            if hp.get_nside(self.mask) != nside:
                raise ValueError("mask NSIDE differs from map NSIDE.")
            mval = np.asarray(self.mask[pix], dtype=float, copy=False)
            invalid |= (~np.isfinite(mval)) | (mval <= 0)
            if np.any(self.mask == hp.UNSEEN):
                invalid |= (self.mask[pix] == hp.UNSEEN)

        # Write NaNs where invalid
        if invalid.any():
            cutout = cutout.copy()
            cutout[invalid] = np.nan

        extent = (-size_arcmin/2, size_arcmin/2, -size_arcmin/2, size_arcmin/2)
        return cutout, extent

    def normalize_by_theta200(self, cutout, size_arcmin, theta200_arcmin):
        """
        Normalize and rebin a tangent-plane cutout onto
        `(u, v) = (x, y) / theta200`

        Parameters
        ----------
        cutout : ndarray
            The cutout array from get_cutout_2d.
        size_arcmin : float
            Original cutout size [arcmin].
        theta200_arcmin : float
            Normalization scale theta200 [arcmin].

        Returns
        -------
        y_grid : ndarray, shape (nbins, nbins)
            Normalized and binned cutout.
        counts : ndarray, shape (nbins, nbins)
            Number of pixels per bin.
        extent : tuple
            (umin, umax, vmin, vmax) in theta200 units for imshow.
        """
        arcmin2rad = np.pi / (180.0 * 60.0)
        npix = cutout.shape[0]

        # automatic grid size if not provided
        grid_halfsize = self.grid_halfsize
        if grid_halfsize is None:
            grid_halfsize = 0.5 * size_arcmin / theta200_arcmin

        # pixel grid in radians
        size_rad = float(size_arcmin) * arcmin2rad
        step = size_rad / npix
        x = -size_rad / 2 + (np.arange(npix) + 0.5) * step
        y = -size_rad / 2 + (np.arange(npix) + 0.5) * step
        xg, yg = np.meshgrid(x, y, indexing="xy")

        # normalised coordinates
        theta200_rad = float(theta200_arcmin) * arcmin2rad
        u = xg / theta200_rad
        v = yg / theta200_rad

        # bin edges in (u, v)
        half = float(grid_halfsize)
        edges_u = np.linspace(-half, +half, self.nbins + 1)
        edges_v = np.linspace(-half, +half, self.nbins + 1)
        extent = [edges_u[0], edges_u[-1], edges_v[0], edges_v[-1]]

        # handle missing data
        valid = np.isfinite(cutout)
        if not np.any(valid):
            y_grid = np.full((self.nbins, self.nbins), np.nan)
            w_grid = np.zeros((self.nbins, self.nbins))
            return y_grid, w_grid, extent

        # bin the data
        u_sel, v_sel, z_sel = u[valid], v[valid], cutout[valid]
        z_sum, _, _ = np.histogram2d(u_sel, v_sel, bins=[edges_u, edges_v],
                                     weights=z_sel)
        counts, _, _ = np.histogram2d(u_sel, v_sel, bins=[edges_u, edges_v])

        with np.errstate(invalid="ignore", divide="ignore"):
            y_grid = np.where(counts > 0, z_sum / counts, np.nan)

        return y_grid, counts, extent

    def stack_cutouts(self, ell_deg, b_deg, size_arcmin, theta200_arcmin):
        """
        Get cutouts at multiple positions, normalize by theta200, and return
        the mean stack.

        Parameters
        ----------
        ell_deg : array_like
            Array of Galactic longitudes [deg].
        b_deg : array_like
            Array of Galactic latitudes [deg].
        size_arcmin : array_like or float
            Array of cutout sizes [arcmin], or a single size for all.
        theta200_arcmin : array_like or float
            Array of theta200 values [arcmin] for normalization, or a single
            value for all.

        Returns
        -------
        individual_normalized : ndarray, shape (n, nbins, nbins)
            Individual normalized cutout arrays.
        mean_stack : ndarray, shape (nbins, nbins)
            Mean of all normalized cutouts (ignoring NaNs).
        extent : tuple
            (umin, umax, vmin, vmax) in theta200 units for imshow.
        """
        ell_deg = np.atleast_1d(ell_deg)
        b_deg = np.atleast_1d(b_deg)
        size_arcmin = np.atleast_1d(size_arcmin)
        theta200_arcmin = np.atleast_1d(theta200_arcmin)

        if size_arcmin.size == 1:
            size_arcmin = np.full(len(ell_deg), size_arcmin[0])
        if theta200_arcmin.size == 1:
            theta200_arcmin = np.full(len(ell_deg), theta200_arcmin[0])

        cond = (len(ell_deg) == len(b_deg) == len(size_arcmin) ==
                len(theta200_arcmin))
        assert cond, ("ell_deg, b_deg, size_arcmin, and theta200_arcmin "
                      "must have the same length")

        n = len(ell_deg)
        individual_normalized = np.full((n, self.nbins, self.nbins), np.nan)

        for i in trange(n, desc="Stacking cutouts"):
            cutout, _ = self.get_cutout_2d(ell_deg[i], b_deg[i],
                                           size_arcmin[i])
            normalized, _, extent = self.normalize_by_theta200(
                cutout, size_arcmin[i], theta200_arcmin[i]
            )
            individual_normalized[i] = normalized

        # Compute mean stack, ignoring NaNs
        mean_stack = np.nanmean(individual_normalized, axis=0)

        return individual_normalized, mean_stack, extent

    def stack_random_cutouts(self, theta200_arcmin, n_stack, size_factor=5.0,
                             abs_b_min=10, seed=None):
        """
        Get cutouts at random sky positions with resampled angular sizes,
        normalize by theta200, and return the mean stack.

        Parameters
        ----------
        theta200_arcmin : array_like
            Array of theta200 values [arcmin] to sample from.
        n_stack : int
            Number of random cutouts to generate and stack.
        size_factor : float, optional
            Multiplicative factor for cutout size relative to theta200.
            Default: 5.0 (i.e., size_arcmin = 5 * theta200_arcmin).
        abs_b_min : float or None, optional
            If given, require |b| >= abs_b_min [deg] for random positions.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        individual_normalized : ndarray, shape (n_stack, nbins, nbins)
            Individual normalized cutout arrays.
        mean_stack : ndarray, shape (nbins, nbins)
            Mean of all normalized cutouts (ignoring NaNs).
        extent : tuple
            (umin, umax, vmin, vmax) in theta200 units for imshow.
        """
        theta200_arcmin = np.atleast_1d(theta200_arcmin)

        # Generate random sky positions
        ell_deg, b_deg, gen = random_sky_positions(n_stack, abs_b_min, seed)

        # Resample theta200 with replacement
        theta200_resampled = gen.choice(theta200_arcmin, size=n_stack,
                                        replace=True)

        # Compute cutout sizes
        size_arcmin = size_factor * theta200_resampled

        # Use stack_cutouts with random positions
        return self.stack_cutouts(ell_deg, b_deg, size_arcmin,
                                  theta200_resampled)
