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
    cosmo_params : dict, optional
        Cosmological parameters for Pfeifer matching.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue : structured ndarray
        Catalogue containing only successfully matched entries.
    association_indices : ndarray
        Index of the matched association for each entry.
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

    return (filtered_catalogue, assoc_indices[matched_mask],
            pvals[matched_mask], distances[matched_mask])


def match_planck_catalog_to_associations(
    data_tsz,
    associations,
    z_max=0.05,
    msz_min=1.0e14,
    match_threshold=0.05,
    mass_preference_threshold=None,
    cosmo_params=None,
    verbose=True,
):
    """
    Match a Planck tSZ catalogue to halo associations (z/Msz cuts applied).

    Parameters
    ----------
    data_tsz : mapping
        Output of :func:`cmbo.io.read_Planck_cluster_catalog`.
    associations : sequence
        Associations returned by :func:`cmbo.match.load_associations`.
    z_max : float, optional
        Maximum redshift passed to the matcher (default 0.05).
    msz_min : float, optional
        Minimum Planck SZ mass proxy (Msun/h) considered (default 1e14).
    match_threshold : float, optional
        Maximum Pfeifer p-value accepted by :func:`greedy_global_matching`.
    mass_preference_threshold : float, optional
        When set, prefer associations with higher mean log mass among pairs
        with p-value below this threshold. Forwarded to
        :func:`match_catalogue_to_associations`.
    cosmo_params : dict, optional
        Cosmological parameters forwarded to the matcher.
    verbose : bool, optional
        If True, print diagnostic information.

    Returns
    -------
    matched_catalogue, association_indices, pvals, distances :
        See :func:`match_catalogue_to_associations`.
    """

    redshift = np.asarray(data_tsz["redshift"], dtype=float)
    msz = np.asarray(data_tsz["msz"], dtype=float)
    selection = (redshift < z_max) & (msz > msz_min)

    filtered_data = mask_structured_array(data_tsz, selection)

    return match_catalogue_to_associations(
        filtered_data,
        associations,
        ra_key="ra_deg",
        dec_key="dec_deg",
        redshift_key="redshift",
        match_threshold=match_threshold,
        mass_preference_threshold=mass_preference_threshold,
        cosmo_params=cosmo_params,
        verbose=verbose,
    )
