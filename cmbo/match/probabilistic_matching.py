"""Probabilistic association between observed clusters and simulated halos."""

from itertools import product

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random, vmap
from jax.nn import logsumexp
from numpyro.infer import MCMC, NUTS
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
        Only groups with at least one cluster and one halo are returned.
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

        if len(obs_idx) > 0 and len(halo_idx) > 0:
            groups.append((obs_idx, halo_idx))

    return groups


def generate_valid_assignments(n_obs, n_halos, allow_unassociated=True):
    """
    Enumerate all partial injective mappings from observed clusters to halos.

    Each cluster can be associated with a unique halo index or None (unassociated).
    No two clusters can be associated with the same halo.

    Parameters
    ----------
    n_obs : int
    n_halos : int
    allow_unassociated : bool
        If False, each cluster must be associated with a halo.

    Returns
    -------
    list of tuples
        Each tuple has length n_obs, with elements being halo indices
        (0 to n_halos-1) or None (if allow_unassociated).

    Example
    -------
    >>> generate_valid_assignments(2, 2)
    [
        (None, None),  # both unassociated
        (None, 0),     # obs 0 unassociated, obs 1 -> halo 0
        (None, 1),     # obs 0 unassociated, obs 1 -> halo 1
        (0, None),     # obs 0 -> halo 0, obs 1 unassociated
        (0, 1),        # obs 0 -> halo 0, obs 1 -> halo 1
        (1, None),     # obs 0 -> halo 1, obs 1 unassociated
        (1, 0),        # obs 0 -> halo 1, obs 1 -> halo 0
    ]
    """
    if n_obs == 0:
        return [()]

    if n_halos == 0:
        if allow_unassociated:
            return [(None,) * n_obs]
        else:
            return []  # No valid assignments

    if not allow_unassociated and n_obs > n_halos:
        return []  # Not enough halos for all clusters

    # Options for each cluster
    if allow_unassociated:
        options = [None] + list(range(n_halos))
    else:
        options = list(range(n_halos))

    assignments = []
    for combo in product(options, repeat=n_obs):
        # Check no duplicate halo assignments
        used_halos = [h for h in combo if h is not None]
        if len(used_halos) == len(set(used_halos)):
            assignments.append(combo)

    return assignments


def assignments_to_array(assignments):
    """
    Convert list of assignment tuples to JAX array.

    None values are converted to -1.

    Parameters
    ----------
    assignments : list of tuples

    Returns
    -------
    (n_assignments, n_obs) array
    """
    return jnp.array([[h if h is not None else -1 for h in a]
                      for a in assignments])


def compute_all_assignment_log_probs(prob_pos, p_null, assignments_arr):
    """
    Compute log probabilities for all assignments (vectorized).

    Uses log-odds scaling: P(unassociated) when Σ Gaussians = 1 equals p_null.
    Higher Σ Gaussians → lower P(unassociated).

    Parameters
    ----------
    prob_pos : (n_obs, n_halos) array
        Unnormalized Gaussian probabilities.
    p_null : float
        Unassociated probability when Σ Gaussians = 1.
    assignments_arr : (n_assignments, n_obs) array
        Assignments with -1 for unassociated.

    Returns
    -------
    (n_assignments,) array
        Log probability of each assignment.
    """
    n_obs, n_halos = prob_pos.shape

    # Sum of Gaussians for each cluster: (n_obs,)
    sum_gauss = jnp.sum(prob_pos, axis=1)
    sum_gauss = jnp.maximum(sum_gauss, 1e-10)

    # Log-odds scaling
    # logit(p_null) - log(sum_gauss) = log(p_null/(1-p_null)) - log(sum_gauss)
    p_null_clipped = jnp.clip(p_null, 1e-6, 1 - 1e-6)
    logit_p_null = jnp.log(p_null_clipped / (1 - p_null_clipped))
    logit_p_unassoc = logit_p_null - jnp.log(sum_gauss)

    # P(unassociated) = sigmoid(logit_p_unassoc)
    p_unassoc = 1 / (1 + jnp.exp(-logit_p_unassoc))
    p_unassoc = jnp.clip(p_unassoc, 1e-10, 1 - 1e-10)

    # P(associated) = 1 - P(unassociated)
    p_assoc = 1 - p_unassoc

    # Normalize Gaussians over halos: (n_obs, n_halos)
    prob_halo_normalized = prob_pos / sum_gauss[:, None]

    # P(halo h) = P(associated) * normalized Gaussian
    prob_halo = prob_halo_normalized * p_assoc[:, None]

    # Extend with unassociated column: (n_obs, n_halos+1)
    prob_extended = jnp.concatenate(
        [prob_halo, p_unassoc[:, None]], axis=1
    )

    # Clip to avoid log(0)
    prob_extended = jnp.clip(prob_extended, 1e-10, 1.0)

    # Convert -1 to index n_halos (the unassociated column)
    assignments_idx = jnp.where(
        assignments_arr == -1, n_halos, assignments_arr)

    # Get log probabilities for each assignment: (n_assignments, n_obs)
    obs_indices = jnp.arange(n_obs)
    log_probs_per_cluster = jnp.log(
        prob_extended[obs_indices, assignments_idx]
    )

    # Sum over clusters: (n_assignments,)
    return jnp.sum(log_probs_per_cluster, axis=1)


def compute_assignment_log_prob(prob_pos, log_L_obs, log_M_halo,
                                rho_null, a, b, sigma_L, assignment):
    """
    Compute log probability of a single assignment with scaling relation.

    Parameters
    ----------
    prob_pos : (n_obs, n_halos) array
        Unnormalized Gaussian probabilities.
    log_L_obs : (n_obs,) array
    log_M_halo : (n_halos,) array
    rho_null : float
        Weight for unassociated option.
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
            # Unassociated
            log_prob = log_prob + jnp.log(rho_null) - jnp.log(Z)
        else:
            # Associated
            log_prob = log_prob + jnp.log(prob_pos[o, h]) - jnp.log(Z)
            # Scaling relation likelihood
            residual = log_L_obs[o] - a - b * log_M_halo[h]
            log_prob = log_prob - 0.5 * (residual / sigma_L)**2

    return log_prob


class AssociationModel:
    """
    NumPyro model for inferring associations based on positions only.

    Parameters
    ----------
    groups : list of tuples
        Output from fof_tessellation.
    group_probs : list of arrays
        Output from compute_group_probabilities.
    """

    def __init__(self, groups, group_probs):
        self.groups = groups
        self.group_probs = group_probs

        # Precompute assignments for each group
        self.group_assignments = []
        for obs_idx, halo_idx in groups:
            n_obs, n_halos = len(obs_idx), len(halo_idx)
            assignments = generate_valid_assignments(n_obs, n_halos)
            self.group_assignments.append(assignments_to_array(assignments))

    def __call__(self):
        """NumPyro model."""
        # Prior on unassociated probability when Σ Gaussians = 1
        # Beta(2, 2) is peaked at 0.5, avoids extremes
        # p_null = numpyro.sample("p_null", dist.Beta(2, 2))
        p_null = numpyro.sample("p_null", dist.Uniform(0, 1))

        # Compute marginalized log-likelihood for each group
        log_lik = 0.0
        for prob_pos, assignments_arr in zip(self.group_probs,
                                             self.group_assignments):
            log_probs = compute_all_assignment_log_probs(
                prob_pos, p_null, assignments_arr
            )
            log_lik = log_lik + logsumexp(log_probs)

        numpyro.factor("log_lik", log_lik)

    def run(self, seed=0, num_warmup=500, num_samples=1000, num_chains=1):
        """
        Run MCMC sampling.

        Parameters
        ----------
        seed : int
        num_warmup : int
        num_samples : int
        num_chains : int

        Returns
        -------
        mcmc : numpyro.infer.MCMC
        """
        rng_key = random.PRNGKey(seed)
        mcmc = MCMC(NUTS(self), num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains)
        mcmc.run(rng_key)
        return mcmc

    def get_association_probabilities(self, p_null):
        """
        Compute association probabilities for each cluster given p_null.

        Parameters
        ----------
        p_null : float or array
            Probability parameter (scalar or posterior samples).

        Returns
        -------
        list of dicts
            For each group, dict with:
            - 'obs_idx': original cluster indices
            - 'halo_idx': original halo indices
            - 'probs': (n_obs, n_halos+1) array of probabilities
                       Last column is probability of being unassociated.
        """
        p_null = jnp.atleast_1d(p_null)
        results = []

        for (obs_idx, halo_idx), prob_pos, assignments_arr in zip(
                self.groups, self.group_probs, self.group_assignments):

            n_obs, n_halos = prob_pos.shape
            n_assignments = len(assignments_arr)

            # Convert -1 to n_halos for one-hot encoding
            assignments_idx = jnp.where(
                assignments_arr == -1, n_halos, assignments_arr
            )

            # One-hot encoding: (n_assignments, n_obs, n_halos+1)
            one_hot = jnp.zeros((n_assignments, n_obs, n_halos + 1))
            obs_indices = jnp.arange(n_obs)
            for a_idx in range(n_assignments):
                one_hot = one_hot.at[a_idx, obs_indices,
                                     assignments_idx[a_idx]].set(1.0)

            def compute_for_sample(p):
                log_probs = compute_all_assignment_log_probs(
                    prob_pos, p, assignments_arr
                )
                assignment_probs = jnp.exp(log_probs - logsumexp(log_probs))
                # (n_obs, n_halos+1)
                return jnp.einsum('a,aoh->oh', assignment_probs, one_hot)

            # Map over samples
            all_probs = vmap(compute_for_sample)(p_null)

            # Average over samples
            mean_probs = jnp.mean(all_probs, axis=0)

            results.append({
                'obs_idx': obs_idx,
                'halo_idx': halo_idx,
                'probs': np.array(mean_probs),
            })

        return results

    def get_best_matches(self, p_null):
        """
        Get most likely halo match for each cluster.

        Parameters
        ----------
        p_null : float or array
            Probability parameter (scalar or posterior samples).

        Returns
        -------
        dict with:
            - 'obs_idx': (n_total_obs,) cluster indices
            - 'best_halo_idx': (n_total_obs,) best halo index or -1 if unassociated
            - 'best_prob': (n_total_obs,) probability of best match
        """
        results = self.get_association_probabilities(p_null)

        all_obs_idx = []
        all_best_halo = []
        all_best_prob = []

        for res in results:
            obs_idx = res['obs_idx']
            halo_idx = res['halo_idx']
            probs = res['probs']

            for i, o in enumerate(obs_idx):
                best_col = np.argmax(probs[i])
                best_prob = probs[i, best_col]

                if best_col == len(halo_idx):
                    # Unassociated
                    best_halo = -1
                else:
                    best_halo = halo_idx[best_col]

                all_obs_idx.append(o)
                all_best_halo.append(best_halo)
                all_best_prob.append(best_prob)

        return {
            'obs_idx': np.array(all_obs_idx),
            'best_halo_idx': np.array(all_best_halo),
            'best_prob': np.array(all_best_prob),
        }


def compute_all_assignment_log_probs_forced(prob_pos, assignments_arr):
    """
    Compute log probabilities for assignments where all clusters must match.

    Uses unnormalized Gaussians so distance affects likelihood.

    Parameters
    ----------
    prob_pos : (n_obs, n_halos) array
    assignments_arr : (n_assignments, n_obs) array

    Returns
    -------
    (n_assignments,) array
    """
    n_obs, n_halos = prob_pos.shape

    # Clip to avoid log(0)
    prob_pos_safe = jnp.clip(prob_pos, 1e-10, 1.0)

    # Get log probabilities for each assignment (unnormalized)
    obs_indices = jnp.arange(n_obs)
    log_probs_per_cluster = jnp.log(
        prob_pos_safe[obs_indices, assignments_arr]
    )

    return jnp.sum(log_probs_per_cluster, axis=1)


def compute_all_assignment_log_probs_with_scaling(
        prob_pos, assignments_arr, log_L_obs, log_L_err, log_M_halo,
        a, b, sigma_L):
    """
    Compute log probabilities for all assignments with scaling relation.

    All clusters must be associated (no unassociated option).

    Parameters
    ----------
    prob_pos : (n_obs, n_halos) array
    assignments_arr : (n_assignments, n_obs) array
    log_L_obs : (n_obs,) array
    log_L_err : (n_obs,) array
        Measurement errors on log_L.
    log_M_halo : (n_halos,) array
    a, b : float
        Scaling relation: log_L = a + b * log_M
    sigma_L : float
        Intrinsic scatter.

    Returns
    -------
    (n_assignments,) array
    """
    # Get positional log probabilities (normalized over halos)
    log_probs_pos = compute_all_assignment_log_probs_forced(
        prob_pos, assignments_arr
    )

    # Add scaling relation likelihood for all pairs
    # Predicted log_L for each halo
    log_L_pred = a + b * log_M_halo

    # Residuals: (n_assignments, n_obs)
    residuals = log_L_obs[None, :] - log_L_pred[assignments_arr]

    # Total variance: intrinsic scatter + measurement error
    total_var = sigma_L**2 + log_L_err[None, :]**2

    # Log likelihood contribution: -0.5 * residual^2 / total_var
    log_lik_scaling = -0.5 * residuals**2 / total_var

    # Sum over clusters
    log_probs_scaling = jnp.sum(log_lik_scaling, axis=1)

    return log_probs_pos + log_probs_scaling


class ScalingRelationModel:
    """
    NumPyro model for scaling relation inference with marginalized association.

    Parameters
    ----------
    groups : list of tuples
        Output from fof_tessellation.
    group_probs : list of arrays
        Output from compute_group_probabilities.
    log_L_obs : (n_obs,) array
        Log luminosities for all observed clusters.
    log_L_err : (n_obs,) array
        Measurement errors on log luminosities.
    log_M_halo : (n_halos,) array
        Log masses for all halos.
    """

    def __init__(self, groups, group_probs, log_L_obs, log_L_err, log_M_halo):
        self.groups = groups
        self.group_probs = group_probs
        self.log_L_obs = jnp.asarray(log_L_obs)
        self.log_L_err = jnp.asarray(log_L_err)
        self.log_M_halo = jnp.asarray(log_M_halo)
        self.n_obs = len(log_L_obs)

        # Precompute assignments for each group (forced association)
        self.group_assignments = []
        for obs_idx, halo_idx in groups:
            n_obs, n_halos = len(obs_idx), len(halo_idx)
            assignments = generate_valid_assignments(
                n_obs, n_halos, allow_unassociated=False
            )
            if len(assignments) == 0:
                raise ValueError(
                    f"Group has {n_obs} clusters but only {n_halos} halos. "
                    "Cannot force all clusters to be associated."
                )
            self.group_assignments.append(assignments_to_array(assignments))

    def __call__(self):
        """NumPyro model."""
        # Priors
        a = numpyro.sample("a", dist.Normal(0, 10))
        b = numpyro.sample("b", dist.Normal(1, 3))
        sigma_L = numpyro.sample("sigma_L", dist.HalfNormal(1))

        # Compute marginalized log-likelihood for each group
        log_lik = 0.0
        for (obs_idx, halo_idx), prob_pos, assignments_arr in zip(
                self.groups, self.group_probs, self.group_assignments):

            group_log_L = self.log_L_obs[obs_idx]
            group_log_L_err = self.log_L_err[obs_idx]
            group_log_M = self.log_M_halo[halo_idx]

            log_probs = compute_all_assignment_log_probs_with_scaling(
                prob_pos, assignments_arr,
                group_log_L, group_log_L_err, group_log_M, a, b, sigma_L
            )
            log_lik = log_lik + logsumexp(log_probs)

        numpyro.factor("log_lik", log_lik)

    def run(self, seed=0, num_warmup=500, num_samples=1000, num_chains=1):
        """Run MCMC sampling."""
        rng_key = random.PRNGKey(seed)
        mcmc = MCMC(NUTS(self), num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains)
        mcmc.run(rng_key)
        return mcmc
