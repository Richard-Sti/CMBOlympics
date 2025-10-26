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

from ..utils.coords import cartesian_to_r_theta_phi


def particle_ngp_projection(nside, pos, observer, Rmax=None, chunk=1000_000,
                            verbose=True):
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
            theta, phi = theta[mask], phi[mask]

        out += np.bincount(hp.ang2pix(nside, theta, phi), minlength=npix)

    return out


def gaussian_smooth_map(input_map, fwhm_arcmin, nside_out=None):
    """Apply a Gaussian smoothing with a strictly positive kernel."""
    fwhm = np.deg2rad(fwhm_arcmin / 60.0)
    output_map = hp.sphtfunc.smoothing(input_map, fwhm=fwhm)

    if nside_out is not None and nside_out != hp.get_nside(input_map):
        output_map = hp.ud_grade(output_map, nside_out=nside_out,
                                 order_in="RING", order_out="RING", power=0)

    return output_map
