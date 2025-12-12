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
"""
Sky mask integration for probabilistic matching.

Computes the fraction of a von Mises-Fisher angular uncertainty distribution
that falls within an unmasked region of the sky, defined by a galactic latitude
cut |b| >= b_lim. Uses HEALPix for efficient spherical integration.
"""
import healpy as hp
import jax.numpy as jnp
import numpy as np
from jax import vmap
from tqdm import tqdm

from ..utils.logging import fprint


def angular_sep(b, ell, b0, ell0):
    """Angular separation on the sphere between (ell, b) and (ell0, b0)."""
    cos_theta = jnp.sin(b) * jnp.sin(b0) + jnp.cos(b) * jnp.cos(b0) * jnp.cos(
        ell - ell0)
    return jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))


def masked_integral_for_sigma(b0, ell0, b_lim, sigma, nside=512, n_sigma=5.0,
                              verbose=False):
    """
    Compute the masked sky integral of a von Mises-Fisher likelihood.

    Evaluates ∫_{|b|>=b_lim} L(θ | σ) dΩ, where L is the von Mises-Fisher PDF
    centered at (ell0, b0) with concentration κ = 1/σ². Returns 1 when the
    source is entirely in the unmasked region, <1 when masked.

    Parameters
    ----------
    b0 : scalar
        Galactic latitude of source center, degrees.
    ell0 : scalar
        Galactic longitude of source center, degrees.
    b_lim : scalar
        Latitude mask threshold, degrees. Region |b| < b_lim is masked.
    sigma : scalar or array
        Angular uncertainty, radians. Can be scalar or 1D array.
    nside : int
        HEALPix resolution parameter.
    n_sigma : scalar
        Query radius in units of sigma. Pixels beyond n_sigma * max(sigma)
        from source are not queried.
    verbose : bool
        Print integration diagnostics if True.

    Returns
    -------
    result : scalar or array
        Integrated probability in unmasked region. Scalar if sigma is scalar,
        array if sigma is array. Equals 1 when source is far from mask.

    Notes
    -----
    Uses von Mises-Fisher distribution L(θ) = (κ/(4π sinh(κ))) exp(κ cos(θ))
    with κ = 1/σ². For array sigma, queries HEALPix once at max(sigma) then
    filters by distance for efficiency.
    """
    b0_rad = jnp.deg2rad(b0)
    ell0_rad = jnp.deg2rad(ell0)
    b_lim_rad = jnp.deg2rad(b_lim)

    theta0 = jnp.pi / 2 - b0_rad
    phi0 = ell0_rad

    sigma_array = jnp.atleast_1d(sigma)
    radius = n_sigma * jnp.max(sigma_array)

    vec = hp.ang2vec(float(theta0), float(phi0))
    pixels = hp.query_disc(
        nside, vec, float(radius), nest=False, inclusive=True)

    theta_pix, phi_pix = hp.pix2ang(nside, pixels, nest=False)
    b_pix = jnp.pi / 2 - theta_pix

    lat_mask = jnp.abs(b_pix) >= b_lim_rad
    theta_sep = angular_sep(b_pix, phi_pix, b0_rad, ell0_rad)
    pixel_area = hp.nside2pixarea(nside)

    if verbose:
        pixel_size_arcmin = jnp.rad2deg(jnp.sqrt(pixel_area)) * 60
        n_pix_total = len(pixels)
        sigma_min = jnp.min(sigma_array)
        n_pix_min = jnp.sum(theta_sep <= n_sigma * sigma_min)

        fprint(f"HEALPix integration: nside={nside} "
               f"({pixel_size_arcmin:.2f}'/pix)")
        fprint(f"Pixels queried: {n_pix_total} (σ_max) | "
               f"{n_pix_min} (σ_min)")

    def integrate_single_sigma(s):
        mask = lat_mask & (theta_sep <= n_sigma * s)
        kappa = 1.0 / s**2
        # Numerically stable form: work in log space to avoid sinh(κ) overflow
        # L(θ) = (κ/(4π sinh(κ))) exp(κ cos(θ))
        # log(sinh(κ)) = κ + log1p(-exp(-2κ)) - log(2) for κ > 0
        log_norm = (
            jnp.log(kappa / (4 * jnp.pi))
            - (kappa + jnp.log1p(-jnp.exp(-2 * kappa)) - jnp.log(2.0)))
        log_likelihood = log_norm + kappa * jnp.cos(theta_sep)
        return jnp.sum(jnp.exp(log_likelihood) * mask) * pixel_area

    result = vmap(integrate_single_sigma)(sigma_array)
    return result[0] if jnp.ndim(sigma) == 0 else result


class MaskedSkyInterpolator:
    """
    JAX-based interpolator for masked-sky integrals as a function of σ.

    For halos at (ell, b) and a fixed latitude cut b_lim, precomputes:
        f(σ) = ∫_{|b|>=b_lim} L(θ | σ) dΩ
    for each halo on a grid of σ values, then interpolates.

    Parameters
    ----------
    ell : array
        Galactic longitudes of halos, degrees.
    b : array
        Galactic latitudes of halos, degrees.
    b_lim : scalar
        Latitude mask threshold, degrees.
    sigma_min : scalar
        Minimum sigma for grid, degrees.
    sigma_max : scalar
        Maximum sigma for grid, degrees.
    n_sigma_grid : int
        Number of grid points in sigma.
    nside : int
        HEALPix resolution parameter.
    n_sigma : scalar
        Query radius in units of sigma.
    verbose : bool
        Print progress and diagnostics if True.
    """

    def __init__(self, ell, b, b_lim, sigma_min, sigma_max, n_sigma_grid=101,
                 nside=512, n_sigma=5.0, verbose=True):
        self.ell = jnp.asarray(ell)
        self.b = jnp.asarray(b)
        self.b_lim = float(b_lim)
        self.n_halos = len(self.ell)

        sigma_min_rad = jnp.deg2rad(sigma_min)
        sigma_max_rad = jnp.deg2rad(sigma_max)
        self._sig_grid = jnp.linspace(
            sigma_min_rad, sigma_max_rad, n_sigma_grid)
        val_grid = np.zeros((self.n_halos, n_sigma_grid))

        if verbose:
            fprint(f"Precomputing grid for {self.n_halos} halos, "
                   f"{n_sigma_grid} σ values...")

        for i in tqdm(range(self.n_halos), disable=not verbose):
            val_grid[i] = masked_integral_for_sigma(
                self.b[i], self.ell[i], self.b_lim, self._sig_grid,
                nside=nside, n_sigma=n_sigma, verbose=False
            )

        self._val_grid = jnp.asarray(val_grid)

    def __call__(self, sigma):
        """
        Interpolate f(σ) at given σ for all halos.

        Parameters
        ----------
        sigma : scalar or JAX array
            Angular uncertainty, degrees.

        Returns
        -------
        result : array, shape (n_halos,)
            Interpolated values for each halo.
        """
        sigma_rad = jnp.deg2rad(sigma)
        return vmap(lambda vals: jnp.interp(
            sigma_rad, self._sig_grid, vals))(self._val_grid)
