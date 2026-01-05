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
"""Cross-correlations of scalar HEALPix maps."""

import numpy as np
import healpy as hp


def cross_cl_healpix(map1, map2, mask=None, lmax=None, deconvolve_pixwin=True,
                     shot1=0.0, shot2=0.0):
    """
    Cross angular power spectrum C_ell between two HEALPix maps up to some
    maximum multipole, by default `3 * nside - 1`.
    """
    nside1 = hp.get_nside(map1)
    nside2 = hp.get_nside(map2)

    if nside1 != nside2:
        raise ValueError("`map1` and `map2` must share the same nside.")
    nside = nside1

    if lmax is None:
        lmax = 3 * nside - 1

    ell = np.arange(lmax + 1)

    good = np.isfinite(map1) & np.isfinite(map2)
    good &= (map1 != hp.UNSEEN) & (map2 != hp.UNSEEN)
    if mask is not None:
        mask = np.asarray(mask)
        if hp.get_nside(mask) != nside:
            raise ValueError("mask nside must match maps.")
        good &= (mask > 0)

    f_sky = float(np.mean(good))
    # print(f"f_sky = {f_sky:.3f}")

    mm1 = hp.ma(map1)
    mm2 = hp.ma(map2)
    mm1.mask = ~good
    mm2.mask = ~good

    cl12 = hp.anafast(mm1, mm2, lmax=lmax)

    if deconvolve_pixwin:
        pw = hp.pixwin(nside, lmax=lmax)
        cl12 = cl12 / (pw**2)

    # Compute autos internally ONLY to form the variance; not returned.
    cl11 = hp.anafast(mm1, mm1, lmax=lmax, iter=0)
    cl22 = hp.anafast(mm2, mm2, lmax=lmax, iter=0)
    if deconvolve_pixwin:
        cl11 = cl11 / (pw**2)
        cl22 = cl22 / (pw**2)
    # Add optional white noise (e.g. 1/nbar for counts)
    cl11n = cl11 + float(shot1)
    cl22n = cl22 + float(shot2)
    var = (cl12**2 + cl11n * cl22n) / np.maximum((2*ell + 1) * f_sky, 1e-10)

    err_cl12 = np.sqrt(np.clip(var, 0.0, np.inf))
    return ell, cl12, err_cl12
