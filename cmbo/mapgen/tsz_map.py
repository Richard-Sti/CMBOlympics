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
"""JAX helpers for painting thermal SZ maps from halo catalogues."""

from dataclasses import dataclass

from jax import config

config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp


import jax.scipy.integrate as jsci

@dataclass
class GNFWParameters:
    """
    Battaglia16-like GNFW parameterisation.

    The parameters capture the mass- and redshift-dependence of the pressure
    normalisation and shape. Units assume M200c is provided in Msun and angles
    in radians; the overall normalisation is dimensionless and can be rescaled
    as needed.
    """

    Omega_c: float = 0.2589
    Omega_b: float = 0.0486
    h: float = 0.6774
    p0_base: float = 18.1
    p0_mass_slope: float = 0.154
    p0_z_slope: float = -0.758
    xc_base: float = 0.497
    xc_mass_slope: float = -0.00865
    xc_z_slope: float = 0.731
    beta_base: float = 4.35
    beta_mass_slope: float = 0.0393
    beta_z_slope: float = 0.415
    alpha: float = 1.0
    gamma: float = -0.3
    mpivot: float = 1.0e14
    y_norm: float = 1.0
    p_e_factor: float = 0.5178


class GNFWProfile:
    """Evaluate a projected GNFW pressure profile."""

    def __init__(self, params):
        self.params = params
        self._dtype = jnp.float64
        self._setup_constants()
        self._setup_interpolator()

    def _setup_constants(self):
        self._G = self._dtype(6.67430e-11)
        self._c = self._dtype(299792458.0)
        self._sigma_T = self._dtype(6.6524587321e-29)
        self._m_e = self._dtype(9.10938356e-31)
        self._M_sun = self._dtype(1.98847e30)
        self._Mpc_to_m = self._dtype(3.085677581491367e22)
        self._Omega_m = self._dtype(self.params.Omega_b + self.params.Omega_c)
        self._f_b = self._dtype(self.params.Omega_b / self._Omega_m)
        self._H0 = self._dtype(self.params.h * 100.0 * 1000.0 / self._Mpc_to_m)
        self._y_prefac = self._sigma_T / (self._m_e * self._c**2)

    def _setup_interpolator(self, n_l=100, n_x=100):
        # Set up a grid for line-of-sight integral
        l_grid = jnp.geomspace(1e-4, 1e2, n_l)
        x_grid = jnp.geomspace(1e-4, 1e2, n_x)
        
        # Perform the integral for g(x) = integral( gnfw(sqrt(l^2+x^2)) dl )
        lx_grid = jnp.sqrt(l_grid[:, None]**2 + x_grid[None, :]**2)
        
        # We need representative values for beta and xc
        # Since they vary slowly, we take them at the pivot mass and z=0
        _, xc, beta = self._scalings(jnp.array(self.params.mpivot), jnp.array(0.0))
        
        integrand = self._gnfw_3d(lx_grid, self.params.gamma, self.params.alpha, beta, xc)
        
        # Simple trapezoidal integration
        self.g_itg = jnp.trapz(integrand, l_grid, axis=0)
        self.x_itg = x_grid
        
    @staticmethod
    def _gnfw_3d(x, gamma, alpha, beta, xc):
        x_over = x / xc
        return (x_over**gamma) * (1 + x_over**alpha)**(-beta)

    def _scalings(self, mass, redshift):
        m = mass / self.params.mpivot
        z1 = 1.0 + redshift
        p0 = (
            self.params.p0_base
            * m ** self.params.p0_mass_slope
            * z1 ** self.params.p0_z_slope
        )
        xc = (
            self.params.xc_base
            * m ** self.params.xc_mass_slope
            * z1 ** self.params.xc_z_slope
        )
        beta = (
            self.params.beta_base
            * m ** self.params.beta_mass_slope
            * z1 ** self.params.beta_z_slope
        )
        return p0, xc, beta

    def _rho_crit(self, z):
        ez2 = self._Omega_m * (1.0 + z) ** 3 + (1.0 - self._Omega_m)
        H_z = self._H0 * jnp.sqrt(ez2)
        return 3.0 * H_z**2 / (8.0 * jnp.pi * self._G)

    def _angular_diameter_distance(self, z):
        # This is a simplified calculation for flat LCDM, good enough for this purpose.
        # For high-precision work, a proper cosmology library should be used.
        def integrand(z_prime):
            ez = jnp.sqrt(self._Omega_m * (1 + z_prime)**3 + (1 - self._Omega_m))
            return 1.0 / ez

        # Basic integration using trapezoidal rule for demonstration
        z_samples = jnp.linspace(0, z, 100)
        integral = jnp.trapz(integrand(z_samples), z_samples)
        
        comoving_distance = (self._c / self._H0) * integral
        return comoving_distance / (1 + z)

    def __call__(self, theta, mass, redshift, theta_scale):
        # Get mass- and redshift-dependent parameters
        p0, xc, beta = self._scalings(mass, redshift)
        
        # Calculate P_200 in SI units (Pascal)
        rho_c = self._rho_crit(redshift)
        r_200 = (3 * mass * self._M_sun / (4 * jnp.pi * 200 * rho_c))**(1/3)
        p_200 = 200 * self._G * mass * self._M_sun * rho_c * self._f_b / (2 * r_200)

        # Get the projected radius in units of R200
        x = theta / theta_scale[None, :]
        
        # Interpolate the integrated profile
        # Note: self.x_itg has shape (n_x,). beta and xc are used to get a single g_itg
        # A more advanced implementation would make g_itg depend on beta and xc.
        g_proj = jnp.interp(x, self.x_itg, self.g_itg)
        
        # Full projected profile in terms of pressure
        # The 3D profile P_3D = P_200 * P0 * gnfw_3d(r/R200)
        # The projected integral has been pre-calculated, so we multiply by R200
        # to give it units of pressure * length
        pressure_length = p_200[None,:] * p0[None,:] * self.params.p_e_factor * g_proj * r_200[None,:]

        return self.params.y_norm * 2 * self._y_prefac * pressure_length


class TSZMap:
    """
    Paint thermal SZ maps on a Healpix grid using JAX.

    Parameters
    ----------
    nside : int
        Healpix nside.
    profile : GNFWProfile
        Profile evaluator. Controls the shape/normalisation.
    theta_max_multiplier : float, optional
        Multiple of the supplied halo angular size used as a default cut-off
        radius when `theta_max` is not supplied.
    batch_size : int, optional
        Number of halos processed per batch to balance memory and throughput.
    nest : bool, optional
        Healpix ordering.
    """

    def __init__(self, nside, profile,
                 theta_max_multiplier=4.0, batch_size=256, nest=False):
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)
        self.profile = profile
        self.theta_max_multiplier = float(theta_max_multiplier)
        self.batch_size = int(batch_size)
        vecs = hp.pix2vec(self.nside, jnp.arange(self.npix), nest=nest)
        self._pix_vecs = jnp.stack(vecs, axis=1).astype(jnp.float64)

    @staticmethod
    def _radec_to_vec(ra, dec):
        ra_rad = jnp.deg2rad(ra)
        dec_rad = jnp.deg2rad(dec)
        cos_dec = jnp.cos(dec_rad)
        return jnp.stack(
            [cos_dec * jnp.cos(ra_rad), cos_dec * jnp.sin(ra_rad), jnp.sin(dec_rad)],
            axis=1,
        )

    def _paint_batch(self, halo_vecs, masses, redshifts,
                     theta_scale, theta_max):
        cos_theta = self._pix_vecs @ halo_vecs.T
        cos_theta = jnp.clip(cos_theta, -1.0 + 1e-9, 1.0 - 1e-9)
        theta = jnp.arccos(cos_theta)

        y = self.profile(theta, masses, redshifts, theta_scale)
        mask = theta <= theta_max[None, :]
        return jnp.where(mask, y, 0.0).sum(axis=1)

    def paint(self, ra, dec, mass, theta_scale, redshift, theta_max=None):
        """
        Paint the thermal SZ signal for a halo catalogue.

        Parameters
        ----------
        ra, dec : array_like
            Right ascension and declination (degrees).
        mass : array_like
            Halo masses M200c in solar masses.
        theta_scale : array_like
            Angular size per halo (e.g., theta_200c in radians).
        redshift : array_like
            Halo redshift.
        theta_max : array_like or float, optional
            Truncation radius per halo in radians. If None, uses
            `theta_max_multiplier * theta_scale`.

        Returns
        -------
        map_1d : jnp.ndarray
            Thermal SZ map in Healpix ring or nest ordering.
        """
        ra = jnp.asarray(ra, dtype=self.profile._dtype)
        dec = jnp.asarray(dec, dtype=self.profile._dtype)
        masses = jnp.asarray(mass, dtype=self.profile._dtype)
        theta_scale = jnp.asarray(theta_scale, dtype=self.profile._dtype)
        redshifts = jnp.asarray(redshift, dtype=self.profile._dtype)

        if not (ra.shape == dec.shape == masses.shape == theta_scale.shape):
            raise ValueError("ra, dec, mass and theta_scale must share shape.")

        if theta_max is None:
            theta_max_arr = self.theta_max_multiplier * theta_scale
        else:
            theta_max_arr = jnp.asarray(theta_max, dtype=self.profile._dtype)

        if theta_max_arr.shape != masses.shape:
            raise ValueError("theta_max must match halo array shape.")

        halo_vecs = self._radec_to_vec(ra, dec)
        nhalo = halo_vecs.shape[0]

        flat_map = jnp.zeros(self.npix, dtype=self.profile._dtype)
        print(f"Painting {nhalo} halos onto Healpix nside={self.nside} "
              f"(npix={self.npix}), batch_size={self.batch_size}")

        for start in range(0, nhalo, self.batch_size):
            end = min(start + self.batch_size, nhalo)
            sl = slice(start, end)
            batch = self._paint_batch(
                halo_vecs[sl], masses[sl], redshifts[sl],
                theta_scale[sl], theta_max_arr[sl],
            )
            flat_map = flat_map + batch
            print(f"  processed halos {start}:{end} "
                  f"({end - start} in batch)")

        return flat_map


__all__ = ["GNFWParameters", "GNFWProfile", "TSZMap"]