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

import healpy as hp
import numpy as np
from tqdm import trange
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simpson

from ..utils.coords import cartesian_to_r_theta_phi
from ..utils import fprint


def particle_ngp_projection(nside, pos, observer, Rmin=None, Rmax=None,
                            chunk=1000_000, verbose=True):
    """
    Nearest-grid pixel projection of particles onto a HEALPix map.

    `pos` and `observer` are in Cartesian coordinates, and in the same units.
    The latter specifies the position of the observer in the box.

    Note that no periodic wrapping is applied!
    """
    npix = hp.nside2npix(nside)
    out = np.zeros(npix, dtype=np.float64)
    npart = len(pos)

    iter_kwargs = {"desc": "Projecting particles",
                   "disable": not verbose or npart < chunk}
    for i in trange(0, npart, chunk, **iter_kwargs):
        sl = slice(i, min(i + chunk, npart))

        r, theta, phi = cartesian_to_r_theta_phi(
            pos[sl, 0], pos[sl, 1], pos[sl, 2], center=observer)

        if Rmax is not None:
            mask = r <= Rmax
            r, theta, phi = r[mask], theta[mask], phi[mask]

        if Rmin is not None:
            mask = r >= Rmin
            theta, phi = theta[mask], phi[mask]

        out += np.bincount(hp.ang2pix(nside, theta, phi), minlength=npix)

    return out


def grid_ngp_projection(nside, rho, boxsize, observer, Rmax,
                        dr, Rmin=0, chunksize=10_000, r_power=2, verbose=True):

    nx, ny, nz = rho.shape
    x = (np.arange(nx) + 0.5) * boxsize / nx
    y = (np.arange(ny) + 0.5) * boxsize / ny
    z = (np.arange(nz) + 0.5) * boxsize / nz

    # Interpolator (periodic handled by manual wrapping)
    fprint("building the 3D grid interpolator...", verbose=verbose)
    interp = RegularGridInterpolator((x, y, z), rho, bounds_error=True)

    # Radial samples
    r = np.arange(Rmin, Rmax + dr, dr)
    nr = r.size
    fprint(f"going to evaluate {nr} radial samples from {Rmin} to {Rmax}...",
           verbose=verbose)

    npix = hp.nside2npix(nside)
    map_out = np.zeros(npix, dtype=np.float64)

    # Pixel directions
    pix_rhat = np.array(hp.pix2vec(nside, np.arange(npix))).T  # (npix,3)

    norm = simpson(r**r_power, x=r)

    # Chunk over pixels to control memory
    iter_kwargs = {"desc": "Projecting grid",
                   "disable": not verbose or npix < chunksize}
    for i0 in trange(0, npix, chunksize, **iter_kwargs):
        i1 = min(i0 + chunksize, npix)
        nhat = pix_rhat[i0:i1]  # (chunk_size, 3)

        # Ray points: (nr, chunk_size, 3)
        pts = observer[None, None, :] + r[:, None, None] * nhat[None, :, :]
        pts = pts.reshape(-1, 3)

        # Interpolate rho along rays, reshape to (nr, C) and integrate
        vals = interp(pts).reshape(nr, i1 - i0)
        vals = simpson(r[:, None]**r_power * vals, x=r, axis=0) / norm

        map_out[i0:i1] = vals

    return map_out


def gaussian_smooth_map(input_map, fwhm_arcmin, nside_out=None):
    """Apply a Gaussian smoothing with a strictly positive kernel."""
    fwhm = np.deg2rad(fwhm_arcmin / 60.0)
    output_map = hp.sphtfunc.smoothing(input_map, fwhm=fwhm)

    if nside_out is not None and nside_out != hp.get_nside(input_map):
        output_map = hp.ud_grade(output_map, nside_out=nside_out,
                                 order_in="RING", order_out="RING", power=0)

    return output_map
