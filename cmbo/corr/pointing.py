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
from scipy.interpolate import interp1d
from tqdm import trange
from tqdm.auto import tqdm


class PointingEnclosedProfile:
    """
    Class for computing enclosed radial profiles from HEALPix maps.

    Parameters
    ----------
    m : array_like
        HEALPix map (Galactic).
    mask : array_like or None, optional
        HEALPix mask in the same NSIDE/order as map. Default: None.
    n_jobs : int, optional
        Number of parallel jobs. Default: 1.
    prefer : str, optional
        Joblib parallelization preference ("processes" or "threads").
        Default: "processes".
    batch_size : str or int, optional
        Joblib batch size. Default: "auto".
    fwhm_arcmin : float, optional
        Width of the annular aperture for background subtraction [arcmin].
        Default: 9.66 (Planck 100 GHz FWHM).
    """

    def __init__(self, m, mask=None, n_jobs=1, prefer="processes",
                 batch_size="auto", fwhm_arcmin=9.66):
        self.m = m
        self.nside = hp.get_nside(m)

        # Set up mask
        if mask is None:
            self.mask = np.isfinite(m) & (m != hp.UNSEEN)
        else:
            self.mask = (mask > 0) & np.isfinite(m) & (m != hp.UNSEEN)

        self.n_jobs = n_jobs
        self.prefer = prefer
        self.batch_size = batch_size
        self.fwhm_arcmin = fwhm_arcmin

    def get_profile(self, ell_deg, b_deg, radii_arcmin):
        """
        Mean enclosed profile at specified aperture radii.

        Parameters
        ----------
        ell_deg : float
            Galactic longitude [deg].
        b_deg : float
            Galactic latitude [deg].
        radii_arcmin : float or array_like
            Aperture radius/radii [arcmin] for signal measurement.

        Returns
        -------
        signal : float or ndarray
            Mean value(s) within aperture(s).
        """
        assert isinstance(ell_deg, (float, int)), \
            "`ell_deg` must be a scalar."
        assert isinstance(b_deg, (float, int)), \
            "`b_deg` must be a scalar."

        radii_arcmin = np.atleast_1d(radii_arcmin)

        th0 = np.radians(90.0 - b_deg)
        ph0 = np.radians(ell_deg)
        v0 = hp.ang2vec(th0, ph0)

        signal = np.full(len(radii_arcmin), np.nan)

        for i, r_arcmin in enumerate(radii_arcmin):
            r_rad = np.radians(r_arcmin / 60.0)

            # Signal aperture
            idx_signal = hp.query_disc(
                self.nside, v0, r_rad, inclusive=False, fact=4)
            if idx_signal.size > 0:
                values_signal = self.m[idx_signal]
                good_signal = self.mask[idx_signal]
                if np.any(good_signal):
                    signal[i] = np.mean(values_signal[good_signal])

        # If input was scalar, return scalar
        if signal.size == 1:
            return float(signal[0])

        return signal

    def get_background(self, ell_deg, b_deg, radii_inner_arcmin,
                       radii_outer_arcmin=None):
        """
        Mean background in annular regions.

        Parameters
        ----------
        ell_deg : float
            Galactic longitude [deg].
        b_deg : float
            Galactic latitude [deg].
        radii_inner_arcmin : float or array_like
            Inner radius/radii [arcmin] for background annulus.
        radii_outer_arcmin : float or array_like, optional
            Outer radius/radii [arcmin] for background annulus. If None,
            defaults to 2 * radii_inner_arcmin.

        Returns
        -------
        background : float or ndarray
            Mean value(s) in annular region(s).
        """
        assert isinstance(ell_deg, (float, int)), \
            "`ell_deg` must be a scalar."
        assert isinstance(b_deg, (float, int)), \
            "`b_deg` must be a scalar."

        radii_inner_arcmin = np.atleast_1d(radii_inner_arcmin)
        if radii_outer_arcmin is None:
            radii_outer_arcmin = 2.0 * radii_inner_arcmin
        else:
            radii_outer_arcmin = np.atleast_1d(radii_outer_arcmin)

        if radii_inner_arcmin.shape != radii_outer_arcmin.shape:
            raise ValueError(
                "radii_inner_arcmin and radii_outer_arcmin must have "
                "the same shape."
            )

        th0 = np.radians(90.0 - b_deg)
        ph0 = np.radians(ell_deg)
        v0 = hp.ang2vec(th0, ph0)

        background = np.full(len(radii_inner_arcmin), np.nan)

        for i, (r_in, r_out) in enumerate(zip(radii_inner_arcmin,
                                              radii_outer_arcmin)):
            r_in_rad = np.radians(r_in / 60.0)
            r_out_rad = np.radians(r_out / 60.0)

            # Get pixels in outer disc
            idx_outer = hp.query_disc(
                self.nside, v0, r_out_rad, inclusive=False, fact=4)
            # Get pixels in inner disc
            idx_inner = hp.query_disc(
                self.nside, v0, r_in_rad, inclusive=False, fact=4)

            # Annulus is outer minus inner
            idx_annulus = np.setdiff1d(idx_outer, idx_inner)

            if idx_annulus.size > 0:
                values_bg = self.m[idx_annulus]
                good_bg = self.mask[idx_annulus]
                if np.any(good_bg):
                    background[i] = np.mean(values_bg[good_bg])

        # If input was scalar, return scalar
        if background.size == 1:
            return float(background[0])

        return background

    def get_profiles_per_source(self, ell_deg, b_deg, radii_arcmin,
                                verbose=True):
        """
        Mean enclosed profiles for multiple sources.

        Parameters
        ----------
        ell_deg : array_like
            Galactic longitudes [deg].
        b_deg : array_like
            Galactic latitudes [deg].
        radii_arcmin : array_like
            Aperture radii [arcmin], one per pointing.
        verbose : bool, optional
            If True, show progress bar. Default: True.

        Returns
        -------
        signal : ndarray
            Mean values within each aperture.
        """
        ell_deg = np.asarray(np.atleast_1d(ell_deg))
        b_deg = np.asarray(np.atleast_1d(b_deg))
        radii_arcmin = np.asarray(np.atleast_1d(radii_arcmin))

        assert ell_deg.shape == b_deg.shape == radii_arcmin.shape

        n = ell_deg.size
        signal = np.full(n, np.nan, dtype=float)

        with tempfile.NamedTemporaryFile(suffix=".mmap",
                                         delete=False) as tmp:
            mmap_path = tmp.name
        try:
            joblib.dump(self.m, mmap_path, compress=0)
            map_mm = joblib.load(mmap_path, mmap_mode="r")

            temp_obj = PointingEnclosedProfile(
                map_mm, mask=self.mask, n_jobs=1,
                prefer=self.prefer, batch_size=self.batch_size,
                fwhm_arcmin=self.fwhm_arcmin
            )

            if self.n_jobs == 1:
                iterator = tqdm(range(n), desc="Measuring profiles",
                                disable=not verbose)
                for i in iterator:
                    signal[i] = temp_obj.get_profile(
                        ell_deg[i], b_deg[i], radii_arcmin[i]
                    )
            else:
                from contextlib import nullcontext
                ctx = tqdm_joblib(
                    tqdm(total=n, desc="Measuring profiles",
                         disable=not verbose)) \
                    if 'tqdm_joblib' in globals() and verbose \
                    else nullcontext()

                with ctx:
                    signal[:] = Parallel(n_jobs=self.n_jobs,
                                         prefer=self.prefer,
                                         batch_size=self.batch_size,
                                         timeout=300)(
                        delayed(temp_obj.get_profile)(
                            ell_deg[i], b_deg[i], radii_arcmin[i])
                        for i in range(n)
                    )
        finally:
            if os.path.exists(mmap_path):
                os.remove(mmap_path)

        return signal

    def stack_normalized_profiles(self, ell_deg, b_deg, theta_ref_arcmin,
                                  radii_norm, n_boot=10000, seed=None,
                                  return_individual=False,
                                  return_random_profiles=False,
                                  random_profile_pool=None,
                                  random_pool_radii=None,
                                  random_pool_samples=1000):
        """
        Stack radial profiles normalised by each source's size.

        Parameters
        ----------
        ell_deg, b_deg : array_like
            Galactic longitudes/latitudes [deg] of the sources.
        theta_ref_arcmin : array_like
            Reference angular scales per source (e.g. theta500) in
            arcminutes.
        radii_norm : array_like
            Dimensionless radii where the profile is evaluated. The
            physical aperture for each source is
            ``radii_norm * theta_ref_arcmin``.
        n_boot : int, optional
            Number of bootstrap resamples for the stacked profile.
            Default: 10000.
        seed : int or None, optional
            Seed for the bootstrap generator.
        return_individual : bool, optional
            If True, also return the per-source profile array used in the
            stack. Default: False.
        return_random_profiles : bool, optional
            If True, also return the individual random profile stacks.
            Default: False.
        random_profile_pool : array_like or None, optional
            Precomputed random signal profiles of shape (n_random, n_cols).
            When provided, these profiles are resampled instead of calling
            :meth:`get_random_profiles`. If ``random_pool_radii`` is None,
            ``n_cols`` must match ``len(radii_norm)``; otherwise profiles are
            interpolated onto the requested radii.
        random_pool_radii : array_like or None, optional
            Physical radii [arcmin] associated with the columns of
            ``random_profile_pool``. Required when the pool was measured on
            a different radius grid than ``radii_norm``.
        random_pool_samples : int or None, optional
            Number of random stacks to draw from the pool. Defaults to all
            available rows when None.

        Returns
        -------
        stacked_profile : ndarray
            Bootstrap mean profile across sources at each normalised radius.
            NaNs are ignored.
        stacked_error : ndarray
            Standard deviation of bootstrap means.
        random_profile : ndarray, optional
            Mean stacked profile from random sky positions (included when
            ``random_profile_pool`` is provided).
        random_error : ndarray, optional
            Standard deviation across random stacks (same availability as
            ``random_profile``).
        individual_profiles : ndarray, optional
            Array of shape (n_sources, n_radii) with per-source profiles,
            returned when ``return_individual`` is True.
        random_profiles : ndarray, optional
            Array of shape (n_random_stacks, n_sources, n_radii) with
            unstacked random profiles, returned when
            ``return_random_profiles`` is True.
        """
        ell_deg = np.asarray(np.atleast_1d(ell_deg), dtype=float)
        b_deg = np.asarray(np.atleast_1d(b_deg), dtype=float)
        theta_ref_arcmin = np.asarray(
            np.atleast_1d(theta_ref_arcmin), dtype=float)
        radii_norm = np.asarray(np.atleast_1d(radii_norm), dtype=float)

        assert ell_deg.shape == b_deg.shape == theta_ref_arcmin.shape, (
            "ell_deg, b_deg, and theta_ref_arcmin must share the same shape.")

        if radii_norm.ndim != 1:
            raise ValueError(
                "radii_norm must be a 1D array of dimensionless radii.")

        pool = None
        pool_radii = None
        if random_profile_pool is not None:
            pool = np.asarray(random_profile_pool, dtype=float)
            if pool.ndim != 2:
                raise ValueError("random_profile_pool must be a 2D array.")

            if random_pool_radii is not None:
                pool_radii = np.asarray(random_pool_radii, dtype=float)
                if pool_radii.ndim != 1:
                    raise ValueError("random_pool_radii must be 1D.")
                if pool_radii.size != pool.shape[1]:
                    raise ValueError(
                        "random_pool_radii length must match pool "
                        "column count.")
                if not np.all(np.diff(pool_radii) >= 0):
                    order = np.argsort(pool_radii)
                    pool_radii = pool_radii[order]
                    pool = pool[:, order]
            elif pool.shape[1] != radii_norm.size:
                raise ValueError(
                    "random_profile_pool width must match len(radii_norm) "
                    "when random_pool_radii is not provided."
                )
        else:
            raise ValueError(
                "random_profile_pool is required to compute random stacks."
            )

        if random_pool_samples is None:
            random_samples_eff = pool.shape[0]
        else:
            if random_pool_samples <= 0:
                raise ValueError("random_pool_samples must be positive.")
            random_samples_eff = min(int(random_pool_samples), pool.shape[0])
        n_sources = ell_deg.size
        n_radii = radii_norm.size

        radius_targets = theta_ref_arcmin[:, None] * radii_norm[None, :]
        ell_tiled = np.repeat(ell_deg, n_radii)
        b_tiled = np.repeat(b_deg, n_radii)
        radii_flat = radius_targets.reshape(-1)

        signal_flat = self.get_profiles_per_source(
            ell_tiled,
            b_tiled,
            radii_flat,
        )
        profiles = signal_flat.reshape(n_sources, n_radii)

        stacked, stacked_error = bootstrap_profile_std(
            profiles, n_boot=n_boot, seed=seed)

        random_mean = random_error = None
        rng = np.random.default_rng(seed)
        rand_means = np.full((random_samples_eff, n_radii), np.nan,
                             dtype=float)
        rand_profiles = np.full((random_samples_eff, n_sources, n_radii),
                                np.nan, dtype=float) \
            if return_random_profiles else None

        for s in trange(random_samples_eff,
                        desc="Stacking random profiles",
                        disable=random_samples_eff < 10):
            idx = rng.choice(pool.shape[0], size=n_sources,
                             replace=True)
            selected_sig = pool[idx]

            if pool_radii is None:
                # Direct use without interpolation
                profiles_to_stack = selected_sig

                if return_random_profiles:
                    rand_profiles[s] = profiles_to_stack
                rand_means[s] = np.nanmean(profiles_to_stack, axis=0)
            else:
                # Interpolation needed
                theta_draw = rng.choice(theta_ref_arcmin,
                                        size=n_sources,
                                        replace=True)
                radii_random = theta_draw[:, None] * radii_norm[None, :]
                interp_vals = np.empty((n_sources, n_radii), dtype=float)
                for p in range(n_sources):
                    prof = selected_sig[p]
                    interp_vals[p] = np.interp(
                        radii_random[p],
                        pool_radii,
                        prof,
                        left=prof[0],
                        right=prof[-1],
                    )
                profiles_to_stack = interp_vals

                if return_random_profiles:
                    rand_profiles[s] = profiles_to_stack
                rand_means[s] = np.nanmean(profiles_to_stack, axis=0)

        valid_rows = ~np.isnan(rand_means).all(axis=1)
        if np.any(valid_rows):
            rand_subset = rand_means[valid_rows]
            random_mean = np.nanmean(rand_subset, axis=0)
            random_error = np.nanstd(rand_subset, axis=0)
        else:
            random_mean = np.full(n_radii, np.nan, dtype=float)
            random_error = np.full(n_radii, np.nan, dtype=float)

        outputs = [stacked, stacked_error]
        outputs.extend([random_mean, random_error])
        if return_individual:
            outputs.append(profiles)
        if return_random_profiles:
            outputs.append(rand_profiles)

        return tuple(outputs)

    def get_random_profiles(self, radii_arcmin, n_points, abs_b_min=None,
                            seed=None):
        """
        Mean radial profiles at random HEALPix positions.

        Parameters
        ----------
        radii_arcmin : float or array_like
            Aperture radius/radii [arcmin] to measure at each random
            position. Can be 1D (shared for all points) or 2D of shape
            (n_points, n_radii) to provide per-point apertures.
        n_points : int
            Number of random positions to generate.
        abs_b_min : float or None, optional
            If given, require |b| >= abs_b_min [deg].
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        signal : ndarray
            Signal profiles at random positions (excluding NaN profiles).
        """
        ell_deg, b_deg, __ = random_sky_positions(n_points, abs_b_min, seed)
        radii_arr = np.asarray(radii_arcmin, dtype=float)

        if radii_arr.ndim == 1:
            radii_per_point = np.tile(radii_arr, (n_points, 1))
        elif radii_arr.ndim == 2:
            if radii_arr.shape[0] != n_points:
                raise ValueError(
                    "For 2D radii_arcmin, shape[0] must equal n_points.")
            radii_per_point = radii_arr
        else:
            raise ValueError("radii_arcmin must be 1D or 2D array-like.")

        with tempfile.NamedTemporaryFile(suffix=".mmap",
                                         delete=False) as tmp:
            mmap_path = tmp.name
        try:
            joblib.dump(self.m, mmap_path, compress=0)
            map_in_mm = joblib.load(mmap_path, mmap_mode="r")

            temp_obj = PointingEnclosedProfile(
                map_in_mm, mask=self.mask, n_jobs=1,
                prefer=self.prefer, batch_size=self.batch_size,
                fwhm_arcmin=self.fwhm_arcmin
            )

            if self.n_jobs == 1:
                results = []
                for i in tqdm(range(n_points),
                              desc="Measuring profiles"):
                    sig = temp_obj.get_profile(
                        ell_deg[i], b_deg[i], radii_per_point[i]
                    )
                    results.append(sig)
            else:
                with tqdm_joblib(tqdm(total=n_points,
                                      desc="Measuring profiles")):
                    results = Parallel(
                        n_jobs=self.n_jobs, prefer=self.prefer,
                        batch_size=self.batch_size, timeout=300
                    )(
                        delayed(temp_obj.get_profile)(
                            ell_deg[i], b_deg[i], radii_per_point[i])
                        for i in range(n_points)
                    )
        finally:
            if os.path.exists(mmap_path):
                os.remove(mmap_path)

        signal_list, num_skipped = [], 0
        for sig in results:
            if np.any(np.isnan(sig)):
                num_skipped += 1
            else:
                signal_list.append(sig)

        print(f"Skipped {num_skipped} / {n_points} profiles due to "
              f"NaNs.")
        return np.asarray(signal_list, dtype=float)

    def signal_to_pvalue(self, theta_arcmin, signal, theta_rand, map_rand):
        """
        Convert signal values to empirical p-values using random
        pointings.

        For each source, interpolates the random profile at the signal
        radius and computes an empirical p-value.

        Parameters
        ----------
        theta_arcmin : array_like
            Angular sizes (radii) for signal measurement [arcmin].
        signal : array_like
            Measured signal per source. Must match theta_arcmin in
            shape.
        theta_rand : array_like
            Radii grid for the random signal profiles [arcmin].
        map_rand : array_like
            Random signal profile measurements. Shape (n_random, n_theta)
            where n_theta = len(theta_rand).

        Returns
        -------
        pval : ndarray
            Empirical p-values for each signal measurement.
        """
        theta_arcmin = np.asarray(theta_arcmin, dtype=float)
        signal = np.asarray(signal, dtype=float)
        theta_rand = np.asarray(theta_rand, dtype=float)
        map_rand = np.asarray(map_rand, dtype=float)

        if theta_arcmin.shape != signal.shape:
            raise ValueError(
                "theta_arcmin and signal must have the same shape."
            )

        if map_rand.ndim != 2:
            raise ValueError("map_rand must be 2D.")

        n_theta = map_rand.shape[1]
        if theta_rand.shape[0] != n_theta:
            raise ValueError(
                "theta_rand length must equal map_rand second dimension."
            )

        def _pvalue_from_pool(val, pool):
            rank = np.searchsorted(pool, val, side="left")
            return 1.0 - rank / pool.size

        # Build 2D interpolator
        # Interpolate along theta axis (axis=1) for all random profiles
        interp_signal = interp1d(
            theta_rand, map_rand, kind='linear', axis=1,
            bounds_error=False, fill_value='extrapolate'
        )

        n_data = signal.size
        pval = np.empty(n_data, dtype=float)

        for i, (r_sig, sig) in enumerate(zip(theta_arcmin, signal)):
            # Interpolate all random profiles at once
            rand_signal = interp_signal(r_sig)
            pool = np.sort(rand_signal)
            pval[i] = _pvalue_from_pool(sig, pool)

        return pval

    def compute_profiles_observed_clusters(self, obs_clusters, radii_arcmin,
                                           radii_background_arcmin=None,
                                           verbose=True):
        """
        Compute 1D radial profiles for observed clusters.

        For each cluster with map_fit information, computes signal and
        background profiles at the specified radii and stores the results
        in the cluster's map_fit dictionary.

        Parameters
        ----------
        obs_clusters
            ObservedClusterCatalogue instance with map_fit populated.
        radii_arcmin
            Array of aperture radii [arcmin] at which to measure profiles.
        radii_background_arcmin
            Array of inner radii [arcmin] for background annulus. If None,
            defaults to radii_arcmin. Can be scalar or array matching
            radii_arcmin. Default: None.
        verbose
            If True, show progress bar. Default: True.
        """
        radii_arcmin = np.asarray(radii_arcmin, dtype=float)

        if radii_background_arcmin is None:
            radii_background_arcmin = radii_arcmin
        else:
            radii_background_arcmin = np.asarray(
                radii_background_arcmin, dtype=float)
            if radii_background_arcmin.size == 1:
                radii_background_arcmin = np.full(
                    radii_arcmin.shape, radii_background_arcmin.item())

        # Filter clusters with map_fit
        clusters_to_process = [
            (i, cluster) for i, cluster in enumerate(obs_clusters)
            if cluster.map_fit is not None
        ]

        if not clusters_to_process:
            raise ValueError(
                "No clusters with map_fit found. Run "
                "find_centers_observed_clusters first."
            )

        iterator = tqdm(clusters_to_process, desc="Computing profiles",
                        disable=not verbose)

        for i, cluster in iterator:
            ell = cluster.map_fit['ell']
            b = cluster.map_fit['b']

            # Compute signal profile at all radii
            signal = self.get_profile(ell, b, radii_arcmin)

            # Compute background profile
            radii_background_arcmin = np.asarray(
                radii_background_arcmin, dtype=float
            )
            radii_outer_arcmin = radii_background_arcmin + self.fwhm_arcmin
            background = self.get_background(
                ell, b, radii_background_arcmin, radii_outer_arcmin
            )

            # Store in map_fit
            obs_clusters.clusters[i].map_fit['radii_arcmin'] = radii_arcmin
            obs_clusters.clusters[i].map_fit['signal_profile'] = signal
            obs_clusters.clusters[i].map_fit[
                'radii_background_arcmin'
            ] = radii_background_arcmin
            obs_clusters.clusters[i].map_fit[
                'background_profile'
            ] = background


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


###############################################################################
#                          Bootstrap utilities                                #
###############################################################################


def bootstrap_profile_std(profiles, n_boot=1000, seed=None):
    """
    Bootstrap the mean stacked profile with standard deviation uncertainties.

    Parameters
    ----------
    profiles : array_like, shape (n_sources, n_radii)
        Per-source profiles. NaNs are ignored when computing statistics.
    n_boot : int, optional
        Number of bootstrap resamples. Default: 1000.
    seed : int or None, optional
        Seed for the random number generator. Default: None.
    Returns
    -------
    mean_profile : ndarray, shape (n_radii,)
        Mean profile across all sources (NaNs ignored).
    std_profile : ndarray, shape (n_radii,)
        Standard deviation of the bootstrap mean profiles.
    """
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2:
        raise ValueError("profiles must be a 2D array.")
    if n_boot <= 0:
        raise ValueError("n_boot must be a positive integer.")

    n_sources, n_radii = profiles.shape
    rng = np.random.default_rng(seed)

    boot_means = np.empty((n_boot, n_radii), dtype=float)
    iterator = trange(
        n_boot,
        desc="Bootstrapping profiles",
        disable=n_boot < 50,
    )
    for i in iterator:
        draw = rng.integers(0, n_sources, size=n_sources)
        sample = profiles[draw]
        boot_means[i] = np.nanmean(sample, axis=0)

    mean_profile = np.nanmean(profiles, axis=0)
    std_profile = np.nanstd(boot_means, axis=0)

    return mean_profile, std_profile


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
    fwhm_arcmin : float
        FWHM of the map smoothing [arcmin]. Used as the default
        final search radius in find_center().
    npix : int, optional
        Pixels per side for cutouts (odd keeps center on pixel).
        Default: 301.
    nest : bool, optional
        True if map is NESTED ordering, else RING. Default: False.
    mask : array_like or None, optional
        HEALPix mask in the same NSIDE/order as map. Default: None.
    nbins : int, optional
        Number of bins for normalized grid. Default: 128.
    grid_halfsize : float or None, optional
        Half-size of normalized grid in theta200 units. If None,
        auto-sized. Default: None.
    """

    def __init__(self, m, fwhm_arcmin, npix=301, nest=False, mask=None,
                 nbins=128, grid_halfsize=None):
        self.m = m
        self.fwhm_arcmin = fwhm_arcmin
        self.npix = npix
        self.nest = nest
        self.mask = mask
        self.nbins = nbins
        self.grid_halfsize = grid_halfsize

    def get_cutout_2d(self, ell_deg, b_deg, size_arcmin):
        """
        Flat-sky cutout (nearest-neighbor) from a HEALPix map in Galactic
        coordinates.

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

    def find_center(self, ell_init, b_init, size_arcmin=240.0,
                    initial_radius_arcmin=None, final_radius_arcmin=None,
                    shrink_factor=0.9, max_iterations=1000, verbose=True):
        """
        Find signal center using two-stage shrinking aperture method.

        Stage 1: Starting from an initial guess, computes the center
        of mass within a circular aperture, recenters on that position,
        shrinks the aperture, and repeats until the aperture radius
        falls below a threshold.

        Stage 2: Finds the pixel with maximum signal within the final
        aperture and recenters on that peak position.

        Parameters
        ----------
        ell_init : float
            Initial guess for Galactic longitude [deg].
        b_init : float
            Initial guess for Galactic latitude [deg].
        size_arcmin : float, optional
            Size of cutout for searching [arcmin]. Default: 240.0.
        initial_radius_arcmin : float or None, optional
            Initial search aperture radius [arcmin]. If None, defaults
            to size_arcmin / sqrt(2) to cover entire cutout including
            corners. Default: None.
        final_radius_arcmin : float or None, optional
            Final aperture radius [arcmin]. Iteration stops when the
            search radius falls below this threshold. If None, uses
            the FWHM of the map. Default: None.
        shrink_factor : float, optional
            Factor by which to shrink the aperture each iteration.
            Must be between 0 and 1. Default: 0.9.
        max_iterations : int, optional
            Maximum number of iterations. Default: 1000.
        verbose : bool, optional
            If True, print iteration progress. Default: True.

        Returns
        -------
        ell_refined : float
            Refined Galactic longitude [deg].
        b_refined : float
            Refined Galactic latitude [deg].
        result : dict
            Dictionary with keys:
            - 'converged': bool, whether iteration converged
            - 'n_iterations': int, number of iterations performed
            - 'final_radius_arcmin': float, final search radius
        cutout_centered : ndarray
            Centered cutout at final position.
        extent : tuple
            Extent for imshow: (xmin, xmax, ymin, ymax) in arcmin.
        """
        if not isinstance(ell_init,
                          (int, float, np.integer, np.floating)):
            raise TypeError("ell_init must be a scalar, not an array")
        if not isinstance(b_init,
                          (int, float, np.integer, np.floating)):
            raise TypeError("b_init must be a scalar, not an array")
        if not 0 < shrink_factor < 1:
            raise ValueError("shrink_factor must be between 0 and 1")

        # Default: search entire cutout including corners
        if initial_radius_arcmin is None:
            initial_radius_arcmin = size_arcmin / np.sqrt(2)

        # Default: converge to map FWHM
        if final_radius_arcmin is None:
            final_radius_arcmin = self.fwhm_arcmin

        ell_deg = float(ell_init)
        b_deg = float(b_init)
        search_radius = float(initial_radius_arcmin)

        converged = False

        for iteration in range(max_iterations):
            cutout, extent = self.get_cutout_2d(
                ell_deg, b_deg, size_arcmin)

            npix = cutout.shape[0]
            pixel_size = size_arcmin / npix

            # Create radial distance map from center
            axis = (np.arange(npix) - (npix - 1) / 2.0) * pixel_size
            xg, yg = np.meshgrid(axis, axis, indexing='xy')
            r_map = np.hypot(xg, yg)

            # Mask pixels outside search radius
            aperture_mask = r_map <= search_radius
            valid = np.isfinite(cutout) & aperture_mask

            if not np.any(valid):
                if verbose:
                    msg = f"Iteration {iteration}: no valid pixels"
                    print(msg)
                break

            # Compute center of mass within aperture
            values = cutout[valid]
            x_coords = xg[valid]
            y_coords = yg[valid]

            # Use positive values only for weighting
            weights = np.maximum(values, 0.0)
            total_weight = np.sum(weights)

            if total_weight == 0:
                if verbose:
                    msg = f"Iteration {iteration}: zero total weight"
                    print(msg)
                break

            x_offset = np.sum(x_coords * weights) / total_weight
            y_offset = np.sum(y_coords * weights) / total_weight
            r_offset = np.hypot(x_offset, y_offset)

            if verbose:
                msg = (f"Iteration {iteration}: "
                       f"radius = {search_radius:.2f} arcmin, "
                       f"offset = ({x_offset:.3f}, {y_offset:.3f}), "
                       f"r = {r_offset:.3f} arcmin")
                print(msg)

            # Update center using small-angle approximation
            ell_deg += x_offset / 60.0 / np.cos(np.radians(b_deg))
            b_deg += y_offset / 60.0

            # Shrink search radius
            search_radius *= shrink_factor

            if search_radius < final_radius_arcmin:
                converged = True
                if verbose:
                    msg = (f"Converged after {iteration + 1} "
                           f"iteration(s)")
                    print(msg)
                break

        # Final refinement: find max signal within final aperture
        if converged:
            cutout, extent = self.get_cutout_2d(
                ell_deg, b_deg, size_arcmin)

            npix = cutout.shape[0]
            pixel_size = size_arcmin / npix
            axis = (np.arange(npix) - (npix - 1) / 2.0) * pixel_size
            xg, yg = np.meshgrid(axis, axis, indexing='xy')
            r_map = np.hypot(xg, yg)

            # Mask pixels within final radius
            final_mask = r_map <= final_radius_arcmin
            valid_final = np.isfinite(cutout) & final_mask

            if np.any(valid_final):
                # Find maximum signal position
                cutout_masked = np.where(valid_final, cutout, -np.inf)
                max_idx = np.nanargmax(cutout_masked)
                iy_max, ix_max = np.unravel_index(max_idx, cutout.shape)

                x_peak = axis[ix_max]
                y_peak = axis[iy_max]
                if verbose:
                    print(f"Peak refinement: offset = "
                          f"({x_peak:.3f}, {y_peak:.3f}) arcmin, "
                          f"signal = {cutout[iy_max, ix_max]:.3e}")

                # Update to peak position
                ell_deg += x_peak / 60.0 / np.cos(np.radians(b_deg))
                b_deg += y_peak / 60.0

        result = {
            'converged': converged,
            'n_iterations': iteration + 1,
            'final_radius_arcmin': search_radius,
        }

        cutout_centered, extent = self.get_cutout_2d(
            ell_deg, b_deg, size_arcmin)

        return ell_deg, b_deg, result, cutout_centered, extent

    def find_centers_observed_clusters(self, obs_clusters, size_arcmin=240.0,
                                       initial_radius_arcmin=None,
                                       final_radius_arcmin=None,
                                       shrink_factor=0.9,
                                       max_iterations=1000,
                                       verbose=True):
        """
        Find tSZ peak centers for all observed clusters.

        For each cluster in the catalogue, finds the tSZ signal peak
        and stores the result in the cluster's map_fit dictionary.

        Parameters
        ----------
        obs_clusters
            ObservedClusterCatalogue instance.
        size_arcmin
            Size of cutout for searching [arcmin]. Default: 240.0.
        initial_radius_arcmin
            Initial search aperture radius [arcmin]. If None, defaults
            to size_arcmin / sqrt(2). Default: None.
        final_radius_arcmin
            Final aperture radius [arcmin]. If None, uses the FWHM of
            the map. Default: None.
        shrink_factor
            Factor by which to shrink the aperture each iteration.
            Must be between 0 and 1. Default: 0.9.
        max_iterations
            Maximum number of iterations. Default: 1000.
        verbose
            If True, print progress for each cluster. Default: True.
        """
        gal_coords = obs_clusters.galactic_coordinates

        for i, (ell_obs, b_obs) in enumerate(gal_coords):
            cluster_name = obs_clusters.names[i]

            if verbose:
                print(f"Processing {cluster_name} ({i+1}/"
                      f"{len(obs_clusters)})...")

            # Find the tSZ center
            ell_center, b_center, info, cutout, extent = self.find_center(
                ell_obs, b_obs,
                size_arcmin=size_arcmin,
                initial_radius_arcmin=initial_radius_arcmin,
                final_radius_arcmin=final_radius_arcmin,
                shrink_factor=shrink_factor,
                max_iterations=max_iterations,
                verbose=False
            )

            # Store in cluster object
            obs_clusters.clusters[i].map_fit = {
                'ell': float(ell_center),
                'b': float(b_center),
                'converged': info['converged'],
                'n_iterations': info['n_iterations'],
                'final_radius_arcmin': info['final_radius_arcmin'],
                'fwhm_arcmin': float(self.fwhm_arcmin),
                'cutout': cutout,
                'extent': extent,
            }

            if verbose:
                offset = np.sqrt(
                    (ell_center - ell_obs)**2 + (b_center - b_obs)**2
                ) * 60.0
                print(f"  Input: ({ell_obs:.3f}, {b_obs:.3f})")
                print(f"  Peak:  ({ell_center:.3f}, {b_center:.3f})")
                print(f"  Offset: {offset:.2f} arcmin")
                if not info['converged']:
                    print("WARNING: Did not converge!")
