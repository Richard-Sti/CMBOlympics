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
"""Readers for X-ray cluster catalogues."""

from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

from ..constants import SPEED_OF_LIGHT_KMS

__all__ = (
    "load_mcxc_catalogue",
    "load_erass_catalogue",
    "match_planck_to_mcxc",
    "match_planck_to_erass",
    "match_mcxc_to_erass",
    "build_matched_catalogues",
)


def _scale_fields(data, fields, factor):
    """Scale numeric structured-array fields in-place."""
    names = data.dtype.names or ()
    for field in fields:
        if field in names and np.issubdtype(data[field].dtype, np.number):
            np.multiply(data[field], factor, out=data[field], casting="unsafe")


def load_mcxc_catalogue(fname="data/MCXCII_2024.fits", verbose=True):
    """
    Load the MCXC-II catalogue into a NumPy structured array.

    Parameters
    ----------
    fname : str or path-like
        FITS file containing the MCXC-II binary table.
    """
    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue '{path}' not found.")

    with fits.open(path, memmap=False) as hdul:
        if len(hdul) <= 1:
            raise ValueError(
                f"FITS file '{path}' does not contain extension 1 with the "
                "MCXC-II table."
            )

        table = hdul[1]
        if table.data is None:
            raise ValueError(f"HDU 1 in '{path}' is empty.")

        data = np.array(table.data, dtype=table.data.dtype, copy=True)

    if {"ERRPM500", "ERRMM500"}.issubset(data.dtype.names):
        mask = (data["ERRPM500"] > 0.0) & (data["ERRMM500"] > 0.0)
        if verbose:
            removed = int(np.count_nonzero(~mask))
            if removed:
                print(
                    f"Removing {removed} MCXC entries with non-positive "
                    "mass uncertainties."
                )
        data = data[mask]

    _scale_fields(data, ("M500", "ERRMM500", "ERRPM500"), 1.0e14)

    return data


def load_erass_catalogue(fname="data/erass1cl_primary_v3.2.fits",
                         verbose=True):
    """
    Load the eRASS1 cluster catalogue into a NumPy structured array.

    Parameters
    ----------
    fname : str or path-like
        FITS file containing the eRASS1 catalogue.
    verbose : bool, optional
        If True, print how many invalid rows were removed.
    """
    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"Catalogue '{path}' not found.")

    with fits.open(path, memmap=False) as hdul:
        if len(hdul) <= 1 or hdul[1].data is None:
            raise ValueError(
                f"FITS file '{path}' does not contain extension 1 with data."
            )
        data = np.array(hdul[1].data, dtype=hdul[1].data.dtype, copy=True)

    if "M500" in data.dtype.names:
        mask = data["M500"] != -1
        if verbose:
            removed = int(np.count_nonzero(~mask))
            if removed:
                print(f"Removing {removed} eRASS entries with M500 = -1.")
        data = data[mask]

    erass_fields = ("M500", "M500_L", "M500_H")
    _scale_fields(data, erass_fields, 1.0e13)

    return data


def _match_catalogues(
    primary_catalogue,
    secondary_catalogue,
    fields_primary,
    fields_secondary,
    max_sep_arcmin,
    max_cz_diff_kms,
):
    primary = _as_structured(primary_catalogue)
    secondary = _as_structured(secondary_catalogue)

    primary_ra = np.asarray(primary[fields_primary["ra"]], dtype=np.float64)
    primary_dec = np.asarray(primary[fields_primary["dec"]], dtype=np.float64)
    primary_z = np.asarray(primary[fields_primary["z"]], dtype=np.float64)

    secondary_ra = np.asarray(secondary[fields_secondary["ra"]],
                              dtype=np.float64)
    secondary_dec = np.asarray(secondary[fields_secondary["dec"]],
                               dtype=np.float64)
    secondary_z = np.asarray(secondary[fields_secondary["z"]],
                             dtype=np.float64)

    primary_coord = SkyCoord(primary_ra * u.deg, primary_dec * u.deg)
    secondary_coord = SkyCoord(secondary_ra * u.deg, secondary_dec * u.deg)

    idx_secondary, idx_primary, sep2d, _ = primary_coord.search_around_sky(
        secondary_coord, max_sep_arcmin * u.arcmin
    )

    n_primary = primary.shape[0]
    matches = [None] * n_primary
    best_sep = np.full(n_primary, np.nan, dtype=np.float64)
    delta_cz = np.full(n_primary, np.nan, dtype=np.float64)

    if idx_primary.size:
        primary_cz = primary_z * SPEED_OF_LIGHT_KMS
        secondary_cz = secondary_z * SPEED_OF_LIGHT_KMS
        pcz = primary_cz[idx_primary]
        scz = secondary_cz[idx_secondary]
        valid = (
            np.isfinite(pcz)
            & np.isfinite(scz)
            & (pcz > 0.0)
            & (scz > 0.0)
            & (np.abs(pcz - scz) <= max_cz_diff_kms)
        )

        delta = pcz - scz
        for p_idx, s_idx, sep, dcz in zip(
            idx_primary[valid], idx_secondary[valid], sep2d[valid],
            delta[valid]
        ):
            sep_arcmin = float(sep.to_value(u.arcmin))
            if not np.isfinite(best_sep[p_idx]) or sep_arcmin < best_sep[p_idx]:  # noqa
                best_sep[p_idx] = sep_arcmin
                delta_cz[p_idx] = float(dcz)
                matches[p_idx] = int(s_idx)

    return matches, best_sep, delta_cz


def match_planck_to_mcxc(
    planck_catalog,
    mcxc_catalogue,
    max_sep_arcmin=60.0,
    max_cz_diff_kms=1000.0,
):
    """Match PSZ2 clusters to MCXC-II entries."""
    return _match_catalogues(
        planck_catalog,
        mcxc_catalogue,
        {"ra": "ra_deg", "dec": "dec_deg", "z": "redshift"},
        {"ra": "RAJ2000", "dec": "DEJ2000", "z": "Z"},
        max_sep_arcmin,
        max_cz_diff_kms,
    )


def match_planck_to_erass(
    planck_catalog,
    erass_catalogue,
    max_sep_arcmin=60.0,
    max_cz_diff_kms=1000.0,
):
    """Match PSZ2 clusters to the eRASS catalogue."""
    return _match_catalogues(
        planck_catalog,
        erass_catalogue,
        {"ra": "ra_deg", "dec": "dec_deg", "z": "redshift"},
        {"ra": "RA", "dec": "DEC", "z": "BEST_Z"},
        max_sep_arcmin,
        max_cz_diff_kms,
    )


def match_mcxc_to_erass(
    mcxc_catalogue,
    erass_catalogue,
    max_sep_arcmin=60.0,
    max_cz_diff_kms=1000.0,
):
    """Match MCXC-II clusters to the eRASS catalogue."""
    return _match_catalogues(
        mcxc_catalogue,
        erass_catalogue,
        {"ra": "RAJ2000", "dec": "DEJ2000", "z": "Z"},
        {"ra": "RA", "dec": "DEC", "z": "BEST_Z"},
        max_sep_arcmin,
        max_cz_diff_kms,
    )


def _dict_to_structured(data):
    keys = list(data.keys())
    arrays = [np.asarray(data[key]) for key in keys]
    if not keys or not arrays:
        return np.zeros(0, dtype=[])
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("All dictionary fields must share the same length.")
    return np.rec.array(
        np.core.records.fromarrays(arrays, names=",".join(keys)))


def _as_structured(catalogue):
    if isinstance(catalogue, dict):
        return _dict_to_structured(catalogue)
    arr = np.asarray(catalogue)
    if arr.dtype.names is None:
        raise ValueError("Catalogue must provide named columns.")
    return arr


def _subset_catalogue(catalogue, mask):
    arr = _as_structured(catalogue)
    return arr[mask]


def build_matched_catalogues(primary_catalogue, secondary_catalogue, matches):
    """
    Return two catalogues restricted to mutually matched entries.

    Parameters
    ----------
    primary_catalogue
        Catalogue describing the match order (e.g. Planck PSZ2 dict).
    secondary_catalogue
        Catalogue indexed by the provided matches (e.g. MCXC structured array).
    matches : Sequence
        Resulting match list from :func:`match_planck_to_mcxc`.

    Returns
    -------
    primary_matched
        Subset of ``primary_catalogue`` retaining only matched entries.
    secondary_matched
        Entries from ``secondary_catalogue`` aligned one-to-one with the
        primary subset.
    """
    matches = np.asarray(matches, dtype=object)
    matched_mask = np.array([m is not None for m in matches], dtype=bool)
    if not np.any(matched_mask):
        empty_primary = _subset_catalogue(primary_catalogue, slice(0, 0))
        empty_secondary = _subset_catalogue(secondary_catalogue, slice(0, 0))
        return empty_primary, empty_secondary

    primary_matched = _subset_catalogue(primary_catalogue, matched_mask)
    matched_indices = np.array(matches[matched_mask], dtype=np.int64)
    secondary_matched = _subset_catalogue(secondary_catalogue, matched_indices)
    return primary_matched, secondary_matched
