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
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .pfeifer import MatchingProbability


def compute_matching_matrix_cartesian(x_obs, associations, box_size=None,
                                      mdef=None, cosmo_params=None,
                                      use_median_mass=False, verbose=True):
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
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
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

    if use_median_mass:
        median_log_mass = np.median(
            [np.mean(np.log10(a.masses)) for a in associations])

    for j, assoc in tqdm(enumerate(associations), total=n_assoc):
        halo_pos = assoc.positions - box_size / 2
        if use_median_mass:
            halo_log_mass = np.full(len(assoc.masses), median_log_mass)
        else:
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
                                use_median_mass=False, verbose=True):
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
    use_median_mass
        If True, all associations use the median of mean log masses instead
        of their own masses, giving each association equal weight.
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
        use_median_mass=use_median_mass,
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


def hungarian_global_matching(pval_matrix, dist_matrix, associations,
                              obs_clusters=None, threshold=0.05,
                              mass_preference_threshold=None, verbose=True):
    """
    Assign association-cluster matches using Hungarian algorithm.

    Uses the Hungarian (Munkres) algorithm to find the optimal assignment that
    minimizes the total cost (sum of p-values). This guarantees a globally
    optimal solution, unlike the greedy approach.

    The Hungarian algorithm requires a square cost matrix. When the number of
    clusters and associations differ, the matrix is padded with a large cost
    value (2.0, which is > max possible p-value of 1.0). Padded assignments
    are automatically rejected by threshold filtering. Any inf/nan values in
    the p-value matrix are also replaced with this large cost to ensure the
    optimization problem remains feasible.

    Note: mass_preference_threshold is ignored in this implementation as the
    Hungarian algorithm finds the global optimum based on p-values alone.

    Parameters
    ----------
    pval_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Average p-value for each cluster-association pair. P-values range
        from 0 (perfect match) to 1 (worst match).
    dist_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Distance between association centroid and observed cluster position.
    associations
        Iterable of associations.
    obs_clusters, optional
        Observed clusters with a `names` attribute.
    threshold : float, optional
        Maximum p-value to accept as a match. Matches above threshold are
        rejected after optimal assignment is found. Default 0.05.
    mass_preference_threshold : float, optional
        Not used in Hungarian algorithm (included for API compatibility).
    verbose : bool, optional
        If True, print matching progress and rejected matches.

    Returns
    -------
    matches : list
        List of length n_obs_clusters. For each cluster:
        - (association, pval, distance) if matched
        - None if not matched
    """
    if mass_preference_threshold is not None and verbose:
        print("Warning: mass_preference_threshold is ignored in Hungarian "
              "matching")

    n_obs = pval_matrix.shape[0]
    associations = list(associations)
    matches = [None] * n_obs

    # Hungarian algorithm requires square matrix, so we need to handle
    # rectangular matrices by padding with high but finite cost
    n_assoc = pval_matrix.shape[1]
    max_dim = max(n_obs, n_assoc)

    # Use a value larger than the maximum possible p-value (1.0) for padding
    # This ensures padded assignments will be rejected by threshold filtering
    large_cost = 2.0
    cost_matrix = np.full((max_dim, max_dim), large_cost)
    cost_matrix[:n_obs, :n_assoc] = pval_matrix

    # Replace any inf/nan values with large cost to ensure feasibility
    cost_matrix[~np.isfinite(cost_matrix)] = large_cost

    # Find optimal assignment minimizing total p-value
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Process assignments
    for i, j in zip(row_ind, col_ind):
        # Skip padding assignments
        if i >= n_obs or j >= n_assoc:
            continue

        pval = float(pval_matrix[i, j])

        # Apply threshold
        if threshold is not None and pval > threshold:
            if verbose and obs_clusters is not None:
                name = obs_clusters.names[i]
                print(f"Cluster {i} ({name}) rejected: p={pval:.3e} > "
                      f"threshold={threshold:.3e}")
            continue

        if not np.isfinite(pval):
            continue

        matches[i] = (associations[j], pval, float(dist_matrix[i, j]))

    return matches
