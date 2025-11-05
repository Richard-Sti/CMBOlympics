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


def compute_matching_matrix(obs_clusters, associations, box_size, mdef="200c",
                            cosmo_params=None, verbose=True):
    """
    Compute a 2D array of average p-values between observed clusters and
    halo associations.

    Parameters
    ----------
    obs_clusters
        Observed clusters with an `icrs_cartesian()` method.
    associations
        Iterable of associations with `positions` and `masses` attributes.
    box_size
        Simulation box size in Mpc/h.
    mdef
        Mass definition for the halo mass function.
    cosmo_params
        Cosmological parameters passed to MatchingProbability.
    verbose
        If True, print clusters with no good match (min p-value > 0.05).

    Returns
    -------
    pval_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Average p-value for each cluster-association pair.
    dist_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Distance between association centroid and observed cluster position
        in Mpc/h.
    not_matched : list
        Names of clusters with no good match (min p-value > 0.05).
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

    x_obs = obs_clusters.icrs_cartesian()
    n_obs = len(x_obs)
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

    not_matched = []
    for i, name in enumerate(obs_clusters.names):
        min_pval = np.min(pval_matrix[i])
        if min_pval > 0.05:
            not_matched.append(name)
            if verbose:
                print(f"{name}: min p-value = {min_pval:.3e}")

    return pval_matrix, dist_matrix


def greedy_global_matching(pval_matrix, dist_matrix, obs_clusters,
                           associations, threshold=0.05, verbose=True):
    """
    Assign association-cluster matches using global greedy algorithm.

    Iteratively selects the pair with the lowest p-value, assigns the match,
    and removes both from further consideration. Continues until all pairs
    are matched or the threshold is exceeded.

    Parameters
    ----------
    pval_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Average p-value for each cluster-association pair.
    dist_matrix : ndarray of shape (n_obs_clusters, n_associations)
        Distance between association centroid and observed cluster position.
    obs_clusters
        Observed clusters with a `names` attribute.
    associations
        Iterable of associations.
    threshold : float, optional
        Maximum p-value to accept as a match. If None, matches all pairs.
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

    while True:
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

        if verbose:
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
