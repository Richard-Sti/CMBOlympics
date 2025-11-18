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
                               greedy_global_matching)


def match_catalogue_to_associations(
    catalogue,
    associations,
    ra_key,
    dec_key,
    redshift_key,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
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
        Maximum Pfeifer p-value accepted by :func:`greedy_global_matching`.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`greedy_global_matching`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
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
        Pfeifer p-values for each match.
    distances : ndarray
        Centroid distances for each match.
    """
    if not associations:
        raise ValueError("At least one association is required.")

    ra = np.asarray(catalogue[ra_key], dtype=float)
    dec = np.asarray(catalogue[dec_key], dtype=float)
    redshift = np.asarray(catalogue[redshift_key], dtype=float)

    unit_vec = radec_to_cartesian(ra, dec)
    dist = cz_to_comoving_distance(redshift * SPEED_OF_LIGHT_KMS,
                                   **(cosmo_params or {}))
    x_obs = (unit_vec.T * dist).T

    pval_matrix, dist_matrix = compute_matching_matrix_cartesian(
        x_obs,
        associations,
        box_size=None,
        mdef=None,
        cosmo_params=cosmo_params,
        use_median_mass=use_median_mass,
        verbose=verbose,
    )
    matches_local = greedy_global_matching(
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

    if verbose:
        fprint(f"Matched {n_matched}/{len(ra)} objects "
               f"({100*n_matched/len(ra):.1f}%)")

    filtered_catalogue = mask_structured_array(catalogue, matched_mask)
    matched_assoc_indices = assoc_indices[matched_mask]
    matched_associations = HaloAssociationList(
        [associations[i] for i in matched_assoc_indices]
    )

    return (filtered_catalogue, matched_associations,
            pvals[matched_mask], distances[matched_mask])


def match_planck_catalog_to_associations(
    data_tsz,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
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
        Maximum Pfeifer p-value accepted by :func:`greedy_global_matching`.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances :
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
    cosmo_params=None,
    verbose=True,
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
        Maximum Pfeifer p-value accepted by :func:`greedy_global_matching`.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances :
        See :func:`match_catalogue_to_associations`.
    """
    redshift = np.asarray(data_mcxc["Z"], dtype=float)
    m500 = np.asarray(data_mcxc["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_mcxc, selection)

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="Z",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        use_median_mass=use_median_mass,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )


def match_erass_catalog_to_associations(
    data_erass,
    associations,
    z_max=0.05,
    m500_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    use_median_mass=False,
    cosmo_params=None,
    verbose=True,
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
        Maximum Pfeifer p-value accepted by :func:`greedy_global_matching`.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue, matched_associations, pvals, distances :
        See :func:`match_catalogue_to_associations`.
    """
    redshift = np.asarray(data_erass["BEST_Z"], dtype=float)
    m500 = np.asarray(data_erass["M500"], dtype=float)
    selection = (redshift < z_max) & (m500 > m500_min)

    filtered_data = mask_structured_array(data_erass, selection)

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="RA",
        dec_key="DEC",
        redshift_key="BEST_Z",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        use_median_mass=use_median_mass,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )
