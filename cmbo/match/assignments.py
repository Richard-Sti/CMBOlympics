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
from .cluster_matching import (compute_matching_matrix_cartesian,
                               greedy_global_matching,
                               hungarian_global_matching,
                               classical_matching)


def match_catalogue_to_associations(
    catalogue,
    associations,
    ra_key,
    dec_key,
    redshift_key,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
    matching_method='greedy',
    median_halo_tsz_pval_max=None,
    use_median_halo_tsz_pval=False,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
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
    match_threshold : float, optional
        Maximum Pfeifer p-value accepted by the matching algorithm.
        Only used with 'greedy' or 'hungarian' matching methods.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Only used with greedy matching.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
        Only used with 'greedy' or 'hungarian' matching methods.
    matching_method : str, optional
        Algorithm for global matching: 'greedy' (default), 'hungarian', or
        'classical'.
    median_halo_tsz_pval_max : float, optional
        When set, classical matching only considers associations whose
        median halo_pval falls below this threshold.
    use_median_halo_tsz_pval : bool, optional
        If True, classical matching selects matches by minimising median
        halo_pval instead of 3D distance after angular/redshift filtering.
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching
        (default 30.0). Only used with 'classical' matching method.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching
        (default 500.0). Only used with 'classical' matching method.
    cosmo_params : dict, optional
        Cosmological parameters for Pfeifer matching.
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
    """
    if not associations:
        raise ValueError("At least one association is required.")

    if matching_method not in ('greedy', 'hungarian', 'classical'):
        raise ValueError(
            f"matching_method must be 'greedy', 'hungarian', or 'classical', "
            f"got '{matching_method}'"
        )

    print("median_halo_tsz_pval_max:", median_halo_tsz_pval_max)
    print("use_median_halo_tsz_pval:", use_median_halo_tsz_pval)

    ra = np.asarray(catalogue[ra_key], dtype=float)
    dec = np.asarray(catalogue[dec_key], dtype=float)
    redshift = np.asarray(catalogue[redshift_key], dtype=float)

    if matching_method == 'classical':
        # Classical matching uses angular separation + redshift criteria
        matches_local = classical_matching(
            ra, dec, redshift,
            associations,
            max_angular_sep=max_angular_sep,
            max_delta_cz=max_delta_cz,
            median_halo_tsz_pval_max=median_halo_tsz_pval_max,
            use_median_halo_tsz_pval=use_median_halo_tsz_pval,
            cosmo_params=cosmo_params,
            verbose=verbose,
        )
    else:
        # Pfeifer-based matching (greedy or hungarian)
        unit_vec = radec_to_cartesian(ra, dec)
        dist = cz_to_comoving_distance(redshift * SPEED_OF_LIGHT_KMS,
                                       **(cosmo_params or {}))
        x_obs = (unit_vec.T * dist).T

        pval_matrix, dist_matrix = compute_matching_matrix_cartesian(
            x_obs,
            associations,
            cosmo_params=cosmo_params,
            use_median_mass=use_median_mass,
            verbose=verbose,
        )

        if matching_method == 'greedy':
            matches_local = greedy_global_matching(
                pval_matrix,
                dist_matrix,
                associations,
                threshold=match_threshold,
                mass_preference_threshold=mass_preference_threshold,
                verbose=verbose,
            )
        else:  # hungarian
            matches_local = hungarian_global_matching(
                pval_matrix,
                dist_matrix,
                associations,
                threshold=match_threshold,
                mass_preference_threshold=mass_preference_threshold,
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
        pvals[i] = pval
        distances[i] = distance

    matched_mask = ~np.isnan(pvals)
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
            n_matched, n_total)


def match_planck_catalog_to_associations(
    data_tsz,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
    matching_method='greedy',
    median_halo_tsz_pval_max=None,
    use_median_halo_tsz_pval=False,
    max_angular_sep=30.0,
    max_delta_cz=500.0,
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
    match_threshold : float, optional
        Maximum Pfeifer p-value accepted by the matching algorithm.
        Only used with 'greedy' or 'hungarian' methods.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    matching_method : str, optional
        Algorithm for global matching: 'greedy' (default), 'hungarian', or
        'classical'.
    median_halo_tsz_pval_max : float, optional
        When set, classical matching only considers associations with median
        halo_pval below this value.
    use_median_halo_tsz_pval : bool, optional
        If True, classical matching selects matches by minimising median
        halo_pval instead of 3D distance.
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
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

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="redshift",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        use_median_mass=use_median_mass,
        matching_method=matching_method,
        median_halo_tsz_pval_max=median_halo_tsz_pval_max,
        use_median_halo_tsz_pval=use_median_halo_tsz_pval,
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )


def match_mcxc_catalog_to_associations(
    data_mcxc,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
    matching_method='greedy',
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    cosmo_params=None,
    verbose=True,
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
    match_threshold : float, optional
        Maximum Pfeifer p-value accepted by the matching algorithm.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    matching_method : str, optional
        Algorithm for global matching: 'greedy' (default), 'hungarian', or
        'classical'.
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.
    **kwargs
        Additional arguments are ignored (e.g. classical_median_pval_max,
        classical_use_median_pval) to keep the function signature forgiving.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances, n_matched,
    n_total :
        See :func:`match_catalogue_to_associations`.
    """
    redshift = np.asarray(data_mcxc["Z"], dtype=float)
    m500 = np.asarray(data_mcxc["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_mcxc, selection)

    # Drop legacy classical args that are unsupported here.
    kwargs.pop("median_halo_tsz_pval_max", None)
    kwargs.pop("use_median_halo_tsz_pval", None)

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="Z",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        use_median_mass=use_median_mass,
        matching_method=matching_method,
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        cosmo_params=cosmo_params,
        verbose=verbose,
        **kwargs,
    )


def match_erass_catalog_to_associations(
    data_erass,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
    matching_method='greedy',
    max_angular_sep=30.0,
    max_delta_cz=500.0,
    cosmo_params=None,
    verbose=True,
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
    match_threshold : float, optional
        Maximum Pfeifer p-value accepted by the matching algorithm.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    matching_method : str, optional
        Algorithm for global matching: 'greedy' (default), 'hungarian', or
        'classical'.
    max_angular_sep : float, optional
        Maximum angular separation in arcminutes for classical matching.
    max_delta_cz : float, optional
        Maximum velocity difference in km/s for classical matching.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.
    **kwargs
        Additional arguments are ignored (e.g. classical_median_pval_max,
        classical_use_median_pval) to keep the function signature forgiving.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances, n_matched,
    n_total :
        See :func:`match_catalogue_to_associations`.
    """
    redshift = np.asarray(data_erass["BEST_Z"], dtype=float)
    m500 = np.asarray(data_erass["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_erass, selection)

    # Drop legacy classical args that are unsupported here.
    kwargs.pop("median_halo_tsz_pval_max", None)
    kwargs.pop("use_median_halo_tsz_pval", None)

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="BEST_Z",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        use_median_mass=use_median_mass,
        matching_method=matching_method,
        max_angular_sep=max_angular_sep,
        max_delta_cz=max_delta_cz,
        cosmo_params=cosmo_params,
        verbose=verbose,
        **kwargs,
    )


# def match_planck_and_xray_joint(
#     data_tsz,
#     data_xray,
#     associations,
#     xray_kind='mcxc',
#     planck_kwargs=None,
#     xray_kwargs=None,
#     verbose=True,
# ):
#     """
#     Match Planck tSZ catalogue, then re-match the same associations with an
#     X-ray catalogue (MCXC or eROSITA), keeping only associations matched in
#     both. Returns the jointly matched catalogues.

#     Parameters
#     ----------
#     data_tsz : mapping
#         Planck catalogue passed to :func:`match_planck_catalog_to_associations`.  # noqa
#     data_xray : mapping
#         MCXC or eROSITA catalogue.
#     associations : sequence
#         Associations returned by :func:`cmbo.match.load_associations`.
#     xray_kind : {'mcxc', 'erass'}, optional
#         Which X-ray matcher to use (default 'mcxc').
#     planck_kwargs : dict, optional
#         Extra keyword arguments forwarded to
#         :func:`match_planck_catalog_to_associations`.
#     xray_kwargs : dict, optional
#         Extra keyword arguments forwarded to the X-ray matcher.
#     verbose : bool, optional
#         If True, print diagnostic information.

#     Returns
#     -------
#     planck_cat_joint, planck_assocs_joint, planck_pvals_joint,
#     planck_dists_joint, xray_cat_joint, xray_assocs_joint,
#     xray_pvals_joint, xray_dists_joint, n_joint :
#         Jointly matched Planck and X-ray catalogues and counts.
#     """
#     planck_kwargs = planck_kwargs or {}
#     xray_kwargs = xray_kwargs or {}

#     planck_match = match_planck_catalog_to_associations(
#         data_tsz,
#         associations,
#         verbose=verbose,
#         **planck_kwargs,
#     )
#     (planck_cat, planck_assocs, planck_pvals,
#      planck_dists, n_planck, n_planck_total) = planck_match

#     planck_assoc_ids = {id(a) for a in planck_assocs}
#     assoc_subset = HaloAssociationList(
#         [a for a in associations if id(a) in planck_assoc_ids]
#     )
#     if not assoc_subset:
#         if verbose:
#             fprint("Planck matching produced no associations; stopping.")
#         return (mask_structured_array(planck_cat, []),
#                 HaloAssociationList(),
#                 np.array([]),
#                 np.array([]),
#                 mask_structured_array(data_xray, []),
#                 HaloAssociationList(),
#                 np.array([]),
#                 np.array([]),
#                 0)

#     if xray_kind == 'mcxc':
#         xray_match = match_mcxc_catalog_to_associations(
#             data_xray,
#             assoc_subset,
#             verbose=verbose,
#             **xray_kwargs,
#         )
#     elif xray_kind == 'erass':
#         xray_match = match_erass_catalog_to_associations(
#             data_xray,
#             assoc_subset,
#             verbose=verbose,
#             **xray_kwargs,
#         )
#     else:
#         raise ValueError("xray_kind must be 'mcxc' or 'erass'.")

#     (xray_cat, xray_assocs, xray_pvals,
#      xray_dists, n_xray, n_xray_total) = xray_match

#     xray_assoc_ids = {id(a) for a in xray_assocs}
#     joint_ids = planck_assoc_ids & xray_assoc_ids

#     planck_joint_mask = np.array([id(a) in joint_ids for a in planck_assocs],
#                                  dtype=bool)
#     xray_joint_mask = np.array([id(a) in joint_ids for a in xray_assocs],
#                                dtype=bool)

#     planck_cat_joint = mask_structured_array(planck_cat, planck_joint_mask)
#     xray_cat_joint = mask_structured_array(xray_cat, xray_joint_mask)
#     planck_pvals_joint = np.asarray(planck_pvals)[planck_joint_mask]
#     planck_dists_joint = np.asarray(planck_dists)[planck_joint_mask]
#     xray_pvals_joint = np.asarray(xray_pvals)[xray_joint_mask]
#     xray_dists_joint = np.asarray(xray_dists)[xray_joint_mask]

#     planck_assocs_joint = HaloAssociationList(
#         [a for a in planck_assocs if id(a) in joint_ids]
#     )
#     xray_assocs_joint = HaloAssociationList(
#         [a for a in xray_assocs if id(a) in joint_ids]
#     )

#     n_joint = len(planck_assocs_joint)

#     if verbose:
#         fprint(
#             "Joint matching: "
#             f"{n_planck}/{n_planck_total} Planck matched, "
#             f"{n_xray}/{n_xray_total} {xray_kind} matched, "
#             f"{n_joint} in intersection."
#         )

#     return (planck_cat_joint, planck_assocs_joint,
#             planck_pvals_joint, planck_dists_joint,
#             xray_cat_joint, xray_assocs_joint,
#             xray_pvals_joint, xray_dists_joint,
#             n_joint)
