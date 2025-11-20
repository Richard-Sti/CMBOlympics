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
from astropy.cosmology import FlatLambdaCDM
import numpy as np

from ..utils import E_z
from ..utils.coords import heliocentric_to_cmb

ARCMIN2_TO_SR = (np.pi / (180.0 * 60.0))**2


def read_Planck_comptonSZ(fname, which="FULL"):
    """
    Load a Planck Compton-y map column, warning when the key is missing.
    """
    data = fits.getdata(fname, ext=1)
    try:
        column = data[which]
    except KeyError as exc:
        names = data.dtype.names or ()
        available = ", ".join(names) if names else "none"
        raise KeyError(
            f"Column '{which}' not found in '{fname}'. "
            f"Available keys: {available}"
        ) from exc
    return np.asarray(column, dtype=np.float32)


def read_Planck_cluster_catalog(fname, extname="PSZ2_UNION", Om=0.306,
                                verbose=True):
    """
    Load the Planck PSZ2 union cluster catalogue.

    Filters out clusters with invalid redshifts (z <= 0 or NaN).

    Returns M500 (from MSZ field, scaled by 1e14) and eM500 (symmetric error
    computed from MSZ_ERR_UP and MSZ_ERR_LOW). Coordinates are stored in
    'RA' and 'DEC' fields (in degrees).

    Converts Y5R500 to physical units:
    - 'Y500': physical integrated Compton-y parameter
    - 'eY500': error on Y500
    - 'Y500_scaled': Y500 scaled by E(z)^(-2/3)
    - 'eY500_scaled': error on Y500_scaled

    Parameters
    ----------
    fname
        Path to the Planck PSZ2 union FITS catalogue.
    extname
        Name of the binary table extension containing the catalogue.
    Om : float, optional
        Matter density parameter for flat LCDM cosmology (default: 0.306).
    verbose : bool, optional
        If True, print diagnostic information about removed entries.

    Returns
    -------
    dict
        Dictionary with key astrophysical quantities for each cluster.
        Only clusters with valid redshifts (z > 0) are included.
        Coordinates in 'RA'/'DEC', mass in 'M500', error in 'eM500',
        physical Y500 parameters.
    """
    table = fits.getdata(fname, extname=extname)

    # Filter out clusters with invalid redshifts
    redshift_raw = np.asarray(table["REDSHIFT"], dtype=float)
    valid_z = np.isfinite(redshift_raw) & (redshift_raw > 0)
    if verbose:
        removed = int(np.count_nonzero(~valid_z))
        if removed:
            print(f"Removing {removed} Planck clusters with invalid "
                  "redshifts (z <= 0 or NaN).")
    table = np.array(table[valid_z], copy=True)

    # Convert heliocentric -> CMB-frame redshifts
    z_helio = np.asarray(table["REDSHIFT"], dtype=float)
    ra = np.asarray(table["RA"], dtype=float)
    dec = np.asarray(table["DEC"], dtype=float)
    z_cmb = heliocentric_to_cmb(z_helio, ra, dec)
    np.copyto(table["REDSHIFT"], np.asarray(z_cmb, dtype=float))
    if verbose:
        print("Converted Planck heliocentric redshifts to CMB frame.")

    def _as_float(col, dtype=np.float64):
        return np.asarray(table[col], dtype=dtype)

    def _as_int(col, dtype=np.int32):
        return np.asarray(table[col], dtype=dtype)

    def _as_str(col):
        return np.char.strip(table[col].astype(str))

    msz_err_up = _as_float("MSZ_ERR_UP", dtype=np.float32) * 1e14
    msz_err_low = _as_float("MSZ_ERR_LOW", dtype=np.float32) * 1e14

    catalog = {
        "index": _as_int("INDEX"),
        "name": _as_str("NAME"),
        "glon_deg": _as_float("GLON"),
        "glat_deg": _as_float("GLAT"),
        "RA": _as_float("RA"),
        "DEC": _as_float("DEC"),
        "pos_err_arcmin": _as_float("POS_ERR"),
        "snr": _as_float("SNR", dtype=np.float32),
        "pipeline": _as_int("PIPELINE"),
        "pipe_det": _as_int("PIPE_DET"),
        "redshift_id": _as_str("REDSHIFT_ID"),
        "redshift": _as_float("REDSHIFT", dtype=np.float32),
        "redshift_helio": np.asarray(z_helio, dtype=np.float32),
        "y5r500": _as_float("Y5R500", dtype=np.float32),
        "y5r500_err": _as_float("Y5R500_ERR", dtype=np.float32),
        "M500": _as_float("MSZ", dtype=np.float32) * 1e14,
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

    catalog["eM500"] = 0.5 * (msz_err_up + msz_err_low)

    # Convert Y5R500 to physical units (all redshifts are valid at this point)
    y5r500_arcmin2 = catalog["y5r500"]
    y5r500_err_arcmin2 = catalog["y5r500_err"]
    z = catalog["redshift"]

    # Convert Y5R500 to Y500 using spherical profile assumption
    y_arcmin2 = y5r500_arcmin2 / 1.81 * 1e-3
    y_err_arcmin2 = y5r500_err_arcmin2 / 1.81 * 1e-3

    # Use FlatLambdaCDM cosmology with h=1
    cosmo = FlatLambdaCDM(H0=100.0, Om0=Om)
    da_mpc = cosmo.angular_diameter_distance(z).value

    # Convert to physical units
    conversion = ARCMIN2_TO_SR * (da_mpc**2)
    y_phys = y_arcmin2 * conversion
    y_phys_err = y_err_arcmin2 * conversion

    # Apply E(z) scaling
    Ez_factor = E_z(z, Om)**(-2.0 / 3.0)
    y_scaled = y_phys * Ez_factor
    y_scaled_err = y_phys_err * Ez_factor

    catalog["Y500"] = y_phys
    catalog["eY500"] = y_phys_err
    catalog["Y500_scaled"] = y_scaled
    catalog["eY500_scaled"] = y_scaled_err

    return catalog
