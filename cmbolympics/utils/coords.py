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
"""Coordinate transformations."""

import astropy.units as u
from tqdm import trange
import numpy as np
from astropy.coordinates import SkyCoord


def cartesian_to_r_theta_phi(x, y, z, center=[0.0, 0.0, 0.0]):
    """Convert Cartesian coordinates to spherical (r, theta, phi)."""
    x0, y0, z0 = center

    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    theta = np.arccos(np.clip((z - z0) / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y - y0, x - x0), 2.0 * np.pi)

    return r, theta, phi


def cartesian_icrs_to_galactic(pos, center, chunk=None):
    """
    Rotate ICRS Cartesian coordinates to Galactic Cartesian coordinates
    about a fixed pivot `center` (observer). The pivot remains numerically
    unchanged:
        x_gal = R @ (x_icrs - center) + center
    """
    c = np.asarray(center, dtype=float)
    if c.shape != (3,):
        raise ValueError("center must have shape (3,)")

    # --- Build correct rotation matrix R such that v_gal = R @ v_icrs ---
    icrs = SkyCoord(
        x=[1.0, 0.0, 0.0] * u.one,
        y=[0.0, 1.0, 0.0] * u.one,
        z=[0.0, 0.0, 1.0] * u.one,
        representation_type="cartesian",
        frame="icrs",
        )
    gal = icrs.galactic.cartesian

    # Rows are component names; columns are images of ICRS basis vectors:
    # column 0 = ex' = (x0,y0,z0)^T, etc.
    R = np.vstack([gal.x.value, gal.y.value, gal.z.value])  # shape (3,3)

    def _apply(block):
        return (R @ (block - c).T).T + c

    if chunk is None or pos.shape[0] <= (chunk or 0):
        out = _apply(pos)
    else:
        out = np.empty_like(pos)
        n = pos.shape[0]
        iter_kwargs = {"desc": "Rotating coordinates",
                       "disable": not n >= chunk}
        for i in trange(0, n, chunk, **iter_kwargs):
            sl = slice(i, min(i + chunk, n))
            out[sl] = _apply(pos[sl])

    return out
