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
"""Various CMB readers."""

from astropy.io import fits
import numpy as np


def read_Planck_comptonSZ(fname, which="FULL"):
    d = fits.open(fname)[1].data
    return d[which].astype(np.float32)


def read_Planck_cluster_catalog(fname, extname="PSZ2_UNION"):
    """
    Load the Planck PSZ2 union cluster catalogue.

    Parameters
    ----------
    fname
        Path to the Planck PSZ2 union FITS catalogue.
    extname
        Name of the binary table extension containing the catalogue.

    Returns
    -------
    dict
        Dictionary with key astrophysical quantities for each cluster.
    """
    table = fits.getdata(fname, extname=extname)

    def _as_float(col, dtype=np.float64):
        return np.asarray(table[col], dtype=dtype)

    def _as_int(col, dtype=np.int32):
        return np.asarray(table[col], dtype=dtype)

    def _as_str(col):
        return np.char.strip(table[col].astype(str))

    catalog = {
        "index": _as_int("INDEX"),
        "name": _as_str("NAME"),
        "glon_deg": _as_float("GLON"),
        "glat_deg": _as_float("GLAT"),
        "ra_deg": _as_float("RA"),
        "dec_deg": _as_float("DEC"),
        "pos_err_arcmin": _as_float("POS_ERR"),
        "snr": _as_float("SNR", dtype=np.float32),
        "pipeline": _as_int("PIPELINE"),
        "pipe_det": _as_int("PIPE_DET"),
        "redshift_id": _as_str("REDSHIFT_ID"),
        "redshift": _as_float("REDSHIFT", dtype=np.float32),
        "y5r500": _as_float("Y5R500", dtype=np.float32),
        "y5r500_err": _as_float("Y5R500_ERR", dtype=np.float32),
        "msz": _as_float("MSZ", dtype=np.float32),
        "msz_err_up": _as_float("MSZ_ERR_UP", dtype=np.float32),
        "msz_err_low": _as_float("MSZ_ERR_LOW", dtype=np.float32),
        "validation": _as_int("VALIDATION"),
        "psz": _as_int("PSZ"),
        "pccs2": _as_int("PCCS2"),
        "ir_flag": np.asarray(table["IR_FLAG"], dtype=bool),
        "q_neural": _as_float("Q_NEURAL", dtype=np.float32),
        "mcxc": _as_str("MCXC"),
        "redmapper": _as_str("REDMAPPER"),
        "act": _as_str("ACT"),
        "spt": _as_str("SPT"),
        "wise_flag": _as_int("WISE_FLAG"),
        "comment": _as_str("COMMENT"),
    }
    return catalog
