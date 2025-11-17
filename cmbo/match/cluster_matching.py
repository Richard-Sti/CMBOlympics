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
"""Match halo associations to observed clusters using Pfeifer p-values."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .pfeifer import MatchingProbability


def compute_matching_matrix_cartesian(x_obs, associations, box_size=None,
                                      mdef=None, cosmo_params=None,
                                      verbose=True):
    """
    Compute match matrices for provided Cartesian positions.

    Parameters
    ----------
    x_obs : array-like, shape (n_obs, 3)
        Cartesian positions of observed clusters in Mpc/h.
    associations
        Iterable of associations with `positions` and `masses` attributes.
    box_size : float, optional
        Simulation box size in Mpc/h. When omitted, the function attempts to
        read ``optional_data['box_size']`` from the first association.
    mdef : str, optional
        Mass definition (e.g. ``"500c"``). When omitted, the function tries
        to read ``optional_data['mass_definition']`` from the first
        association.
    cosmo_params
        Cosmological parameters passed to MatchingProbability.
    verbose
        If True, print clusters with no good match (min p-value > 0.05).

    Returns
    -------
    pval_matrix : ndarray of shape (n_obs, n_associations)
        Average p-value for each position-association pair.
    dist_matrix : ndarray of shape (n_obs, n_associations)
        Distance between association centroid and observed cluster position
        in Mpc/h.
    """
    if cosmo_params is None:
        cosmo_params = {
            'flat': True,
            'H0': 67.66,
            'Om0': 0.3111,
            'Ob0': 0.0489,
            'sigma8': 0.8101,
            'ns': 0.9665
        }
    x_obs = np.asarray(x_obs, dtype=float)
    if x_obs.ndim != 2 or x_obs.shape[1] != 3:
        raise ValueError("x_obs must have shape (n_obs, 3).")

    if box_size is None or mdef is None:
        if not associations:
            raise ValueError(
                "box_size and mdef must be provided when there are "
                "no associations."
            )
        first = associations[0]
        opt = getattr(first, "optional_data", {}) or {}
        if box_size is None:
            if "box_size" not in opt:
                raise ValueError(
                    "box_size not provided and missing from association "
                    "optional_data."
                )
            box_size = float(opt["box_size"])
        if mdef is None:
            mdef = opt.get("mass_definition")
            if mdef is None:
                raise ValueError(
                    "mdef not provided and missing from association "
                    "optional_data."
                )

    n_obs = x_obs.shape[0]
    associations = list(associations)
    n_assoc = len(associations)

    pval_matrix = np.empty((n_obs, n_assoc))
    dist_matrix = np.empty((n_obs, n_assoc))

    for j, assoc in tqdm(enumerate(associations), total=n_assoc):
        halo_pos = assoc.positions - box_size / 2
        halo_log_mass = np.log10(assoc.masses)
        centroid = np.mean(halo_pos, axis=0)

        matcher = MatchingProbability(
            halo_pos,
            halo_log_mass,
            mdef=mdef,
            cosmo_params=cosmo_params
        )

        for i in range(n_obs):
            pval, _, _ = matcher.cdf_per_halo(x_obs[i])
            pval_matrix[i, j] = np.mean(pval)
            dist_matrix[i, j] = np.linalg.norm(x_obs[i] - centroid)

    if verbose:
        for i in range(n_obs):
            min_pval = np.min(pval_matrix[i])
            if min_pval > 0.05:
                print(f"Position {i}: min p-value = {min_pval:.3e}")

    return pval_matrix, dist_matrix


def compute_matching_matrix_obs(obs_clusters, associations, box_size=None,
                                mdef=None, cosmo_params=None,
                                verbose=True):
    """
    Wrapper around :func:`compute_matching_matrix_cartesian` that extracts
    Cartesian positions from an ``ObservedCluster`` catalogue.

    Parameters
    ----------
    obs_clusters : ObservedClusterCatalogue
        Catalogue exposing ``icrs_cartesian()`` and ``names`` accessors.
    associations : sequence
        Halo associations as required by
        :func:`compute_matching_matrix_cartesian`.
    box_size : float, optional
        Simulation box size in Mpc/h. When ``None`` the function attempts to
        read ``optional_data['box_size']`` from the associations.
    mdef : str, optional
        Mass definition forwarded to the Pfeifer matcher (default: ``"500c"``).
    cosmo_params : dict, optional
        Cosmological parameters for :class:`MatchingProbability`. When omitted
        the default LCDM parameters are used.
    verbose : bool, optional
        If True, print observed cluster names whose minimum p-value exceeds
        0.05 indicating a poor match.

    Returns
    -------
    pval_matrix : ndarray
        Matrix of average p-values with shape ``(n_obs, n_assoc)``.
    dist_matrix : ndarray
        Matrix of centroid distances (Mpc/h) with the same shape.
    """
    x_obs = obs_clusters.icrs_cartesian()
    pval_matrix, dist_matrix = compute_matching_matrix_cartesian(
        x_obs,
        associations,
        box_size,
        mdef=mdef,
        cosmo_params=cosmo_params,
        verbose=False,
    )

    not_matched = []
    for i, name in enumerate(obs_clusters.names):
        min_pval = np.min(pval_matrix[i])
        if min_pval > 0.05:
            not_matched.append(name)
            if verbose:
                print(f"{name}: min p-value = {min_pval:.3e}")

    return pval_matrix, dist_matrix


def greedy_global_matching(pval_matrix, dist_matrix, associations,
                           obs_clusters=None, threshold=0.05,
                           mass_preference_threshold=None, verbose=True):
    """
    Assign association-cluster matches using global greedy algorithm.

    Iteratively selects the pair with the lowest p-value, assigns the match,
    and removes both from further consideration. Continues until all pairs
    are matched or the threshold is exceeded.

    When `mass_preference_threshold` is set, the algorithm prioritizes massive
    associations among good matches: if multiple pairs have p-values below this
    threshold, it selects the pair with the most massive association (by mean
    log mass). If no pairs satisfy this criterion, it falls back to selecting
    the lowest p-value pair.

    Parameters
    ----------
    pval_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Average p-value for each cluster-association pair.
    dist_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Distance between association centroid and observed cluster position.
    associations
        Iterable of associations.
    obs_clusters, optional
        Observed clusters with a `names` attribute.
    threshold : float, optional
        Maximum p-value to accept as a match. If None, matches all pairs.
    mass_preference_threshold : float, optional
        When set, among pairs with p-value below this threshold, prefer the
        association with the highest mean log mass. If None, always pick the
        lowest p-value pair (default behavior).
    verbose : bool, optional
        If True, print matching progress and eliminated clusters.

    Returns
    -------
    matches : list
        List of length n_obs_clusters. For each cluster:
        - (association_idx, association, pval, distance) if matched
        - None if not matched
    """
    pval = pval_matrix.copy()
    n_obs = pval_matrix.shape[0]
    associations = list(associations)
    matches = [None] * n_obs
    orphaned = set()

    if mass_preference_threshold is not None:
        assoc_mean_log_mass = np.array([np.mean(np.log10(assoc.masses))
                                        for assoc in associations])

    while True:
        if mass_preference_threshold is not None:
            good_matches = pval < mass_preference_threshold
            if np.any(good_matches):
                good_indices = np.where(good_matches)
                j_candidates = good_indices[1]
                best_mass_idx = np.argmax(assoc_mean_log_mass[j_candidates])
                i = int(good_indices[0][best_mass_idx])
                j = int(j_candidates[best_mass_idx])
            else:
                i, j = np.unravel_index(np.argmin(pval), pval.shape)
                i, j = int(i), int(j)
        else:
            i, j = np.unravel_index(np.argmin(pval), pval.shape)
            i, j = int(i), int(j)

        min_pval = float(pval[i, j])

        if threshold is not None and min_pval > threshold:
            break
        if not np.isfinite(min_pval):
            break

        matches[i] = (associations[j], min_pval, float(dist_matrix[i, j]))

        # Mark this cluster and association as used
        pval[i, :] = np.inf
        pval[:, j] = np.inf

        if verbose and obs_clusters is not None:
            # Check if any remaining clusters now have no good options
            for k in range(pval.shape[0]):
                if k != i and k not in orphaned:
                    finite_vals = pval[k, :][np.isfinite(pval[k, :])]
                    has_remaining = len(finite_vals) > 0
                    all_bad = np.all(finite_vals >= threshold)
                    if has_remaining and all_bad:
                        best_remaining = np.min(finite_vals)
                        name = obs_clusters.names[k]
                        print(f"Cluster {k} ({name}) now orphaned "
                              f"(best remaining p={best_remaining:.3e})")
                        orphaned.add(k)

    return matches
