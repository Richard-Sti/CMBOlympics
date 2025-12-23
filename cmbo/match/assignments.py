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

"""Helpers assigning observed catalogues to halo associations."""

from __future__ import annotations

import numpy as np

from ..constants import SPEED_OF_LIGHT_KMS
from ..utils.arrays import mask_structured_array
from ..utils.associations import HaloAssociationList
from ..utils.logging import fprint
from ..utils.coords import cz_to_comoving_distance, radec_to_cartesian
from .cluster_matching import classical_matching


def match_catalogue_to_associations(
    catalogue,
    associations,
    ra_key,
    dec_key,
    redshift_key,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    min_member_fraction=0.5,
    median_halo_tsz_pval_max=None,
    use_median_halo_tsz_pval=False,
    cosmo_params=None,
    verbose=True,
):
    """
    Match a generic catalogue to halo associations.

    Parameters
    ----------
    catalogue : mapping
        Dictionary-like object with RA/Dec/redshift entries.
    associations : sequence
        Associations returned by :func:`cmbo.match.load_associations`.
    ra_key, dec_key, redshift_key : str
        Keys selecting RA, Dec (degrees) and redshift columns.
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching
        (default 30.0).
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching
        (default 500.0).
    min_member_fraction : float, optional
        Minimum fraction of member haloes that must satisfy angular/redshift
        cuts in classical matching.
    median_halo_tsz_pval_max : float, optional
        When set, classical matching only considers associations whose
        median halo_pval falls below this threshold.
    use_median_halo_tsz_pval : bool, optional
        If True, classical matching selects matches by minimising median
        halo_pval instead of 3D distance after angular/redshift filtering.
    cosmo_params : dict, optional
        Cosmological parameters for distance calculations.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue : structured ndarray
        Catalogue containing only successfully matched entries.
    matched_associations : HaloAssociationList
        List of matched associations (same length as matched_catalogue).
    pvals : ndarray
        Pfeifer p-values for each match (for Pfeifer methods), or angular
        separations in arcminutes (for classical method).
    distances : ndarray
        Centroid distances for each match in Mpc/h.
    n_matched : int
        Number of successfully matched objects.
    n_total : int
        Total number of objects in the catalogue.
    matches_mask : ndarray of bool
        Boolean mask indicating which entries in the input catalogue were
        matched (same length as input catalogue).
    """
    if not associations:
        raise ValueError("At least one association is required.")

    ra = np.asarray(catalogue[ra_key], dtype=float)
    dec = np.asarray(catalogue[dec_key], dtype=float)
    redshift = np.asarray(catalogue[redshift_key], dtype=float)

    matches_local = classical_matching(
        ra, dec, redshift,
        associations,
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        median_halo_tsz_pval_max=median_halo_tsz_pval_max,
        use_median_halo_tsz_pval=use_median_halo_tsz_pval,
        min_member_fraction=min_member_fraction,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )

    assoc_lookup = {id(assoc): idx for idx, assoc in enumerate(associations)}
    assoc_indices = np.empty(len(ra), dtype=int)
    assoc_indices.fill(-1)
    pvals = np.full(len(ra), np.nan, dtype=float)
    distances = np.full(len(ra), np.nan, dtype=float)
    for i, match in enumerate(matches_local):
        if match is None:
            continue
        assoc_obj, pval, distance = match
        assoc_idx = assoc_lookup.get(id(assoc_obj))
        if assoc_idx is None:
            raise ValueError(
                "Matched association not found in the associations list."
            )
        assoc_indices[i] = assoc_idx
        pvals[i] = np.nan
        distances[i] = distance

    matched_mask = assoc_indices != -1
    n_matched = np.sum(matched_mask)
    n_total = len(ra)

    if verbose:
        fprint(f"Matched {n_matched}/{n_total} objects "
               f"({100*n_matched/n_total:.1f}%)")

    filtered_catalogue = mask_structured_array(catalogue, matched_mask)
    matched_assoc_indices = assoc_indices[matched_mask]
    matched_associations = HaloAssociationList(
        [associations[i] for i in matched_assoc_indices]
    )

    return (filtered_catalogue, matched_associations,
            pvals[matched_mask], distances[matched_mask],
            n_matched, n_total, matched_mask)


def match_planck_catalog_to_associations(
    data_tsz,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    median_halo_tsz_pval_max=None,
    use_median_halo_tsz_pval=False,
    min_member_fraction=0.5,
    cosmo_params=None,
    verbose=True,
):
    """
    Match a Planck tSZ catalogue to halo associations (z/M500 cuts applied).

    Parameters
    ----------
    data_tsz : mapping
        Output of :func:`cmbo.io.read_Planck_cluster_catalog`.
    associations : sequence
        Associations returned by :func:`cmbo.match.load_associations`.
    z_max : float, optional
        Maximum redshift passed to the matcher (default 0.05).
    m500_min : float, optional
        Minimum Planck M500 mass (Msun/h) considered (default 1e14).
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
    median_halo_tsz_pval_max : float, optional
        When set, classical matching only considers associations with median
        halo_pval below this value.
    use_median_halo_tsz_pval : bool, optional
        If True, classical matching selects matches by minimising median
        halo_pval instead of 3D distance.
    min_member_fraction : float, optional
        Minimum fraction of member haloes that must satisfy angular/redshift
        cuts in classical matching.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances, n_matched,
    n_total :
        See :func:`match_catalogue_to_associations`.
    """

    redshift = np.asarray(data_tsz["redshift"], dtype=float)
    m500 = np.asarray(data_tsz["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_tsz, selection)

    result = match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="redshift",
        median_halo_tsz_pval_max=median_halo_tsz_pval_max,
        use_median_halo_tsz_pval=use_median_halo_tsz_pval,
        min_member_fraction=min_member_fraction,
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )
    return result + (filtered_data,)


def match_mcxc_catalog_to_associations(
    data_mcxc,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    cosmo_params=None,
    verbose=True,
    min_member_fraction=0.5,
    **kwargs,
):
    """
    Match an MCXC-II X-ray catalogue to halo associations (z/M500 cuts
    applied).

    Parameters
    ----------
    data_mcxc : mapping
        Output of :func:`cmbo.io.load_mcxc_catalogue`.
    associations : sequence
        Associations returned by :func:`cmbo.match.load_associations`.
    z_max : float, optional
        Maximum redshift passed to the matcher (default 0.05).
    m500_min : float, optional
        Minimum MCXC M500 mass (Msun/h) considered (default 1e14).
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.
    min_member_fraction : float, optional
        Minimum fraction of member haloes that must satisfy angular/redshift
        cuts in classical matching.
    **kwargs
        Additional arguments are ignored (e.g. classical_median_pval_max,
        classical_use_median_pval) to keep the function signature forgiving.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances, n_matched,
    n_total :
        See :func:`match_catalogue_to_associations`.
    """
    kwargs = kwargs.copy()

    redshift = np.asarray(data_mcxc["Z"], dtype=float)
    m500 = np.asarray(data_mcxc["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_mcxc, selection)

    kwargs.pop("median_halo_tsz_pval_max", None)
    kwargs.pop("use_median_halo_tsz_pval", None)

    result = match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="Z",
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        min_member_fraction=min_member_fraction,
        cosmo_params=cosmo_params,
        verbose=verbose,
        **kwargs,
    )
    return result + (filtered_data,)


def match_erass_catalog_to_associations(
    data_erass,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    cosmo_params=None,
    verbose=True,
    min_member_fraction=0.5,
    **kwargs,
):
    """
    Match an eRASS X-ray catalogue to halo associations (z/M500 cuts applied).

    Parameters
    ----------
    data_erass : mapping
        Output of :func:`cmbo.io.load_erass_catalogue`.
    associations : sequence
        Associations returned by :func:`cmbo.match.load_associations`.
    z_max : float, optional
        Maximum redshift passed to the matcher (default 0.05).
    m500_min : float, optional
        Minimum eRASS M500 mass (Msun/h) considered (default 1e14).
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.
    min_member_fraction : float, optional
        Minimum fraction of member haloes that must satisfy angular/redshift
        cuts in classical matching.
    **kwargs
        Additional arguments are ignored (e.g. classical_median_pval_max,
        classical_use_median_pval) to keep the function signature forgiving.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances, n_matched,
    n_total :
        See :func:`match_catalogue_to_associations`.
    """
    kwargs = kwargs.copy()

    redshift = np.asarray(data_erass["BEST_Z"], dtype=float)
    m500 = np.asarray(data_erass["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_erass, selection)

    kwargs.pop("median_halo_tsz_pval_max", None)
    kwargs.pop("use_median_halo_tsz_pval", None)

    result = match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="BEST_Z",
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        min_member_fraction=min_member_fraction,
        cosmo_params=cosmo_params,
        verbose=verbose,
        **kwargs,
    )
    return result + (filtered_data,)
