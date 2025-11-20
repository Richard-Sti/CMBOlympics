"""Probabilistic matching between observed clusters and simulated halos."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.nn import logsumexp
from scipy.cluster.hierarchy import fclusterdata


def compute_gaussian_probabilities(x_obs, x_halo, sigma):
    """
    Compute Gaussian membership probabilities for all obs-halo pairs.

    Parameters
    ----------
    x_obs : (n_obs, 3) array
    x_halo : (n_halos, 3) array
    sigma : float
        Gaussian scatter (same units as positions).

    Returns
    -------
    prob : (n_obs, n_halos) array
        Unnormalized probabilities P(h|o) ∝ exp(-d²/2σ²).
    """
    diff = x_obs[:, None, :] - x_halo[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    return jnp.exp(-dist_sq / (2 * sigma**2))


def compute_group_probabilities(x_obs, x_halo, sigma, groups):
    """
    Compute Gaussian probabilities for each FoF group.

    Parameters
    ----------
    x_obs : (n_obs, 3) array
    x_halo : (n_halos, 3) array
    sigma : float
    groups : list of tuples
        Output from fof_tessellation: [(obs_indices, halo_indices), ...].

    Returns
    -------
    list of (n_obs_group, n_halo_group) arrays
        Gaussian probabilities for each group. Groups with no halos return
        an empty array of shape (n_obs_group, 0).
    """
    probs = []
    for obs_idx, halo_idx in groups:
        x_obs_group = x_obs[obs_idx]
        if len(halo_idx) == 0:
            probs.append(jnp.empty((len(obs_idx), 0)))
        else:
            x_halo_group = x_halo[halo_idx]
            probs.append(compute_gaussian_probabilities(x_obs_group,
                                                        x_halo_group, sigma))
    return probs


def fof_tessellation(x_obs, x_halo, linking_length):
    """
    Group clusters and halos using Friends-of-Friends.

    Parameters
    ----------
    x_obs : (n_obs, 3) array
        Observed cluster positions (Cartesian).
    x_halo : (n_halos, 3) array
        Halo positions (Cartesian).
    linking_length : float

    Returns
    -------
    list of tuples
        Each tuple is (obs_indices, halo_indices) for a group.
        Only groups with at least one observed cluster are returned.
    """
    n_obs = len(x_obs)
    n_halos = len(x_halo)

    if n_obs == 0:
        return []

    # Combine positions
    all_pos = np.vstack([x_obs, x_halo])
    is_obs = np.concatenate([np.ones(n_obs, dtype=bool),
                             np.zeros(n_halos, dtype=bool)])

    # FoF via single-linkage hierarchical clustering
    if len(all_pos) == 1:
        labels = np.array([0])
    else:
        labels = fclusterdata(all_pos, t=linking_length,
                              criterion='distance', method='single')

    # Group by label
    groups = []
    for label in np.unique(labels):
        mask = labels == label
        obs_idx = np.where(mask & is_obs)[0]
        halo_idx = np.where(mask & ~is_obs)[0] - n_obs

        if len(obs_idx) > 0:
            groups.append((obs_idx, halo_idx))

    return groups


def generate_valid_assignments(n_obs, n_halos):
    """
    Enumerate all partial injective mappings from observed clusters to halos.

    Each cluster can be assigned to a unique halo index or None (unmatched).
    No two clusters can be assigned to the same halo.

    Parameters
    ----------
    n_obs : int
    n_halos : int

    Returns
    -------
    list of tuples
        Each tuple has length n_obs, with elements being halo indices
        (0 to n_halos-1) or None.

    Example
    -------
    >>> generate_valid_assignments(2, 2)
    [
        (None, None),  # both unmatched
        (None, 0),     # obs 0 unmatched, obs 1 -> halo 0
        (None, 1),     # obs 0 unmatched, obs 1 -> halo 1
        (0, None),     # obs 0 -> halo 0, obs 1 unmatched
        (0, 1),        # obs 0 -> halo 0, obs 1 -> halo 1
        (1, None),     # obs 0 -> halo 1, obs 1 unmatched
        (1, 0),        # obs 0 -> halo 1, obs 1 -> halo 0
    ]
    """
    if n_obs == 0:
        return [()]

    if n_halos == 0:
        return [(None,) * n_obs]

    # Options for each cluster: None or any halo index
    options = [None] + list(range(n_halos))

    assignments = []
    for combo in product(options, repeat=n_obs):
        # Check no duplicate halo assignments
        used_halos = [h for h in combo if h is not None]
        if len(used_halos) == len(set(used_halos)):
            assignments.append(combo)

    return assignments


def compute_assignment_log_prob(prob_pos, log_L_obs, log_M_halo,
                                 rho_null, a, b, sigma_L, assignment):
    """
    Compute log probability of a single assignment.

    Parameters
    ----------
    prob_pos : (n_obs, n_halos) array
        Unnormalized Gaussian probabilities.
    log_L_obs : (n_obs,) array
    log_M_halo : (n_halos,) array
    rho_null : float
        Weight for unmatched option.
    a, b : float
        Scaling relation parameters.
    sigma_L : float
    assignment : tuple
        Assignment tuple from generate_valid_assignments.

    Returns
    -------
    float
        Log probability of this assignment.
    """
    log_prob = 0.0

    for o, h in enumerate(assignment):
        # Normalization: rho_null + sum of Gaussians for this cluster
        Z = rho_null + jnp.sum(prob_pos[o])

        if h is None:
            # Unmatched
            log_prob = log_prob + jnp.log(rho_null) - jnp.log(Z)
        else:
            # Matched
            log_prob = log_prob + jnp.log(prob_pos[o, h]) - jnp.log(Z)
            # Scaling relation likelihood
            residual = log_L_obs[o] - a - b * log_M_halo[h]
            log_prob = log_prob - 0.5 * (residual / sigma_L)**2

    return log_prob


class ScalingRelationModel:
    """
    NumPyro model for scaling relation inference with marginalized matching.

    Parameters
    ----------
    groups : list of tuples
        Output from fof_tessellation.
    group_probs : list of arrays
        Output from compute_group_probabilities.
    log_L_obs : (n_obs,) array
        Log luminosities for all observed clusters.
    log_M_halo : (n_halos,) array
        Log masses for all halos.
    """

    def __init__(self, groups, group_probs, log_L_obs, log_M_halo):
        self.groups = groups
        self.group_probs = group_probs
        self.log_L_obs = jnp.asarray(log_L_obs)
        self.log_M_halo = jnp.asarray(log_M_halo)
        self.n_obs = len(log_L_obs)

    def _marginalize_group_log_likelihood(self, prob_pos, log_L_obs, log_M_halo,
                                          rho_null, a, b, sigma_L):
        """Compute marginalized log-likelihood for one group."""
        n_obs, n_halos = prob_pos.shape

        # Generate all valid assignments
        assignments = generate_valid_assignments(n_obs, n_halos)

        # Compute log probability for each assignment
        log_probs = []
        for assignment in assignments:
            lp = compute_assignment_log_prob(prob_pos, log_L_obs, log_M_halo,
                                             rho_null, a, b, sigma_L, assignment)
            log_probs.append(lp)

        return logsumexp(jnp.array(log_probs))

    def __call__(self):
        """NumPyro model."""
        # Priors
        a = numpyro.sample("a", dist.Normal(0, 10))
        b = numpyro.sample("b", dist.Normal(1, 1))
        sigma_L = numpyro.sample("sigma_L", dist.HalfNormal(1))
        rho_null = numpyro.sample("rho_null", dist.HalfNormal(1))

        # Compute marginalized log-likelihood for each group
        log_lik = 0.0
        for (obs_idx, halo_idx), prob_pos in zip(self.groups, self.group_probs):
            if len(halo_idx) == 0:
                # No halos in group, all clusters unmatched (prob = 1)
                pass
            else:
                group_log_L = self.log_L_obs[obs_idx]
                group_log_M = self.log_M_halo[halo_idx]

                log_lik = log_lik + self._marginalize_group_log_likelihood(
                    prob_pos, group_log_L, group_log_M,
                    rho_null, a, b, sigma_L
                )

        numpyro.factor("log_lik", log_lik)
