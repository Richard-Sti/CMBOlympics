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
"""Probabilistic association between observed clusters and simulated halos."""

from dataclasses import dataclass
from itertools import permutations
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfc
from jax.scipy.stats import norm as jax_norm
from numpyro import factor, sample
from numpyro.distributions import Uniform, TruncatedNormal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from ..constants import SPEED_OF_LIGHT_KMS
from ..utils.coords import cz_to_comoving_distance, radec_to_cartesian


def von_mises_fisher_logpdf(theta, sigma):
    """
    Log PDF of von Mises-Fisher distribution for angular separation.

    Parameters
    ----------
    theta : scalar or array
        Angular separation, radians.
    sigma : scalar
        Angular uncertainty, radians.

    Returns
    -------
    log_prob : scalar or array
        Log probability density.

    Notes
    -----
    Uses numerically stable form to avoid sinh(κ) overflow for large κ.
    L(θ) = (κ/(4π sinh(κ))) exp(κ cos(θ)) with κ = 1/σ².
    """
    kappa = 1.0 / sigma**2
    # Numerically stable: log(sinh(κ)) = κ + log1p(-exp(-2κ)) - log(2)
    log_norm = (jnp.log(kappa / (4 * jnp.pi))
                - (kappa + jnp.log1p(-jnp.exp(-2 * kappa)) - jnp.log(2.0)))
    return log_norm + kappa * jnp.cos(theta)


def truncated_normal_logpdf(x, mu, sigma, delta):
    """
    Log PDF of truncated normal distribution.

    Truncates the distribution to [mu - delta, mu + delta].
    Returns a floor value for values outside bounds.

    Parameters
    ----------
    x : scalar or array
        Value to evaluate.
    mu : scalar or array
        Mean of the distribution.
    sigma : scalar
        Standard deviation.
    delta : scalar
        Truncation half-width (truncates at mu ± delta).

    Returns
    -------
    log_prob : scalar or array
        Log probability density. Floor value if outside bounds.
    """
    # Check if x is within bounds
    in_bounds = jnp.abs(x - mu) <= delta

    z = delta / sigma
    # Normalization: Φ(z) - Φ(-z) = 2Φ(z) - 1 for symmetric truncation
    log_norm = jnp.log(2.0 * jax_norm.cdf(z) - 1.0)
    log_prob = jax_norm.logpdf(x, mu, sigma) - log_norm

    # For out-of-bounds, use 10-sigma floor: -0.5 * 10^2 = -50
    # log_prob_floor = -jnp.log(sigma * jnp.sqrt(2.0 * jnp.pi)) - 50.0 - log_norm

    return jnp.where(in_bounds, log_prob, -10.)


def partition_volume(halo_cat, cluster_cat, linking_length=15.0, h=1.0,
                     Om0=0.3111, verbose=True):
    """
    Partition the volume into disjoint groups using a Friends-of-Friends (FoF)
    algorithm applied to the union of halo and cluster positions.

    Two objects are linked if their three-dimensional separation is less than
    the linking length. Every cluster is ensured to be linked to its nearest
    halo, even if beyond the linking_length.

    Parameters
    ----------
    halo_cat : dict
        Dictionary containing halo properties: 'GLON', 'GLAT', 'Z'.
        GLON/GLAT in degrees, Z is CMB-frame redshift.
    cluster_cat : dict
        Dictionary containing cluster properties: 'GLON', 'GLAT', 'Z'.
        GLON/GLAT in degrees, Z is CMB-frame redshift.
    linking_length : float, optional
        Linking length in Mpc/h. Default is 15.0.
    h : float, optional
        Hubble parameter. Default is 1.0.
    Om0 : float, optional
        Matter density parameter. Default is 0.3111.
    verbose : bool, optional
        Print warnings about forced links. Default is True.

    Returns
    -------
    groups : list of dict
        List of groups, where each group is a dictionary containing:
        'halo_indices': indices of halos in the group (relative to input
        halo_cat)
        'cluster_indices': indices of clusters in the group (relative to input
        cluster_cat)
    """
    h_lon = np.asarray(halo_cat['GLON'])
    h_lat = np.asarray(halo_cat['GLAT'])
    h_z = np.asarray(halo_cat['Z'])

    c_lon = np.asarray(cluster_cat['GLON'])
    c_lat = np.asarray(cluster_cat['GLAT'])
    c_z = np.asarray(cluster_cat['Z'])

    n_halos = len(h_lon)
    n_clusters = len(c_lon)

    if n_clusters > 0 and n_halos == 0:
        raise ValueError("Cannot partition volume with clusters but no halos. "
                         "Every group must contain at least one halo.")

    h_dist = cz_to_comoving_distance(h_z * SPEED_OF_LIGHT_KMS, h=h, Om0=Om0)
    c_dist = cz_to_comoving_distance(c_z * SPEED_OF_LIGHT_KMS, h=h, Om0=Om0)

    h_uv = radec_to_cartesian(h_lon, h_lat)
    c_uv = radec_to_cartesian(c_lon, c_lat)

    h_pos = h_uv * h_dist[:, None]
    c_pos = c_uv * c_dist[:, None]

    all_pos = np.vstack([h_pos, c_pos])
    tree_all = cKDTree(all_pos)
    pairs = list(tree_all.query_pairs(r=linking_length))

    # Build adjacency and components
    n_total = n_halos + n_clusters
    if pairs:
        pairs_arr = np.array(pairs)
        row = pairs_arr[:, 0]
        col = pairs_arr[:, 1]
        data = np.ones(len(pairs_arr), dtype=int)
        adj = csr_matrix((data, (row, col)), shape=(n_total, n_total))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n_total, n_total), dtype=int)

    n_components, labels = connected_components(
        csgraph=adj, directed=False, return_labels=True
    )

    groups = []
    cluster_only = 0
    for comp in range(n_components):
        members = np.where(labels == comp)[0]
        h_mask = members < n_halos
        c_mask = ~h_mask
        h_idx = members[h_mask]
        c_idx = members[c_mask] - n_halos
        if len(h_idx) == 0 and len(c_idx) > 0:
            cluster_only += 1
        groups.append({
            'halo_indices': np.array(h_idx, dtype=int),
            'cluster_indices': np.array(c_idx, dtype=int),
        })

    if cluster_only > 0 and verbose:
        print(f"Warning: {cluster_only} groups contain clusters but no halos.")
        # Report nearest halo distances for these clusters
        tree_h = cKDTree(h_pos) if n_halos > 0 else None
        for gi, g in enumerate(groups):
            if len(g['halo_indices']) == 0 and len(g['cluster_indices']) > 0:
                print(f"  Group {gi}: {len(g['cluster_indices'])} clusters, "
                      f"{len(g['halo_indices'])} halos")
                if tree_h is None:
                    print("    No halos in catalog to compute distances.")
                    continue
                c_indices = g['cluster_indices']
                dists, h_near = tree_h.query(c_pos[c_indices], k=1)
                for ci, dist, hn in zip(c_indices, dists, h_near):
                    print(f"    Cluster {ci}: nearest halo {hn}, "
                          f"distance {dist:.2f} Mpc/h")

    # Drop groups with no halos or more clusters than halos
    filtered = []
    dropped_cluster_only = 0
    dropped_more_clusters = 0
    for g in groups:
        n_h = len(g['halo_indices'])
        n_c = len(g['cluster_indices'])
        if n_h == 0:
            dropped_cluster_only += 1
            continue
        if n_c > n_h:
            dropped_more_clusters += 1
            continue
        filtered.append(g)

    if verbose and (dropped_cluster_only > 0 or dropped_more_clusters > 0):
        print(f"Dropped {dropped_cluster_only} groups with no halos and "
              f"{dropped_more_clusters} groups with more clusters than halos.")

    return filtered


@jax.jit
def log_Y_expected(logM, alpha, beta, logM_piv):
    """Compute expected log Y from the linear scaling relation."""
    return alpha + beta * (logM - logM_piv)


@jax.jit
def f_Y(logY_lim, logY_exp, sigma_tot):
    """Selection function f_Y = P(logY_obs > logY_lim)."""
    return 0.5 * erfc((logY_lim - logY_exp) / (jnp.sqrt(2) * sigma_tot))


@jax.jit
def f_z(cz_max, cz_halo, sigma_v):
    """Redshift-space selection function f_z."""
    return 0.5 * (1 + erf((cz_max - cz_halo) / (jnp.sqrt(2) * sigma_v)))


@jax.jit
def f_genuine_mass_dependent(logM, f_genuine_high, logM_transition, width):
    """
    Mass-dependent genuineness: sigmoid transition.

    f_genuine(M) = f_genuine_high / (1 + exp(-(logM - logM_transition) / width))

    Parameters
    ----------
    logM : array
        Log halo mass.
    f_genuine_high : scalar
        Asymptotic genuineness at high mass.
    logM_transition : scalar
        Transition mass (50% of f_genuine_high).
    width : scalar
        Transition width in dex.

    Returns
    -------
    f_genuine : array
        Genuineness for each halo.
    """
    return f_genuine_high / (1.0 + jnp.exp(-(logM - logM_transition) / width))


@jax.jit
def angular_separation(uv1, uv2):
    """Compute angular separation between two unit vectors."""
    cos_theta = jnp.dot(uv1, uv2)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    return jnp.arccos(cos_theta)


def _segment_logsumexp(data, segment_ids, num_segments):
    """
    Compute logsumexp for each segment in a numerically stable way.

    Parameters
    ----------
    data : array
        Values to logsumexp.
    segment_ids : array
        Segment ID for each value (must be sorted).
    num_segments : int
        Total number of segments (must be static/compile-time constant).

    Returns
    -------
    result : array
        logsumexp for each segment, shape (num_segments,).

    Notes
    -----
    This function is called from within NumPyro's JIT-compiled model,
    so no additional JIT decorator is needed.
    """
    # Find max per segment for numerical stability
    segment_max = jax.ops.segment_max(data, segment_ids, num_segments)

    # Shift by max: data - max[segment_id]
    data_shifted = data - segment_max[segment_ids]

    # Sum exp(shifted), then add back max
    sum_exp = jax.ops.segment_sum(
        jnp.exp(data_shifted), segment_ids, num_segments)

    return segment_max + jnp.log(sum_exp)


class MatcherData:
    """
    Preprocess and prepare data for probabilistic cluster-halo matching.

    This class handles all preprocessing steps that should be done once
    before MCMC sampling.

    Parameters
    ----------
    halo_cat : dict
        Halo catalog with keys 'M', 'GLON', 'GLAT', 'Z'.
    cluster_cat : dict
        Cluster catalog with keys 'Y', 'GLON', 'GLAT', 'Z', 'eY'.
    groups : list of dict
        FoF groups from partition_volume.
    h : float, optional
        Hubble parameter. Default is 1.0.
    Om0 : float, optional
        Matter density parameter. Default is 0.3111.
    """

    def __init__(self, halo_cat, cluster_cat, groups, h=1.0, Om0=0.3111):
        self.h = h
        self.Om0 = Om0

        # Preprocess catalogs
        self.processed = self._preprocess_catalogs(halo_cat, cluster_cat)

        # Precompute pairs
        self.pair_data = self._precompute_pairs(groups)

    def _preprocess_catalogs(self, halo_cat, cluster_cat):
        """
        Preprocess catalogs: convert to JAX arrays and compute positions.

        Parameters
        ----------
        halo_cat : dict
            Halo catalog with keys 'M', 'GLON', 'GLAT', 'Z'.
        cluster_cat : dict
            Cluster catalog with keys 'Y', 'GLON', 'GLAT', 'Z', 'eY'.

        Returns
        -------
        processed : dict
            Dictionary with processed data.
        """
        # Convert to JAX arrays and precompute logM
        h_M = jnp.asarray(np.asarray(halo_cat['M'], dtype=float))
        h_logM = jnp.log10(h_M)
        h_lon = jnp.asarray(np.asarray(halo_cat['GLON'], dtype=float))
        h_lat = jnp.asarray(np.asarray(halo_cat['GLAT'], dtype=float))
        h_z = jnp.asarray(np.asarray(halo_cat['Z'], dtype=float))

        c_Y = jnp.asarray(np.asarray(cluster_cat['Y'], dtype=float))
        c_logY = jnp.log10(c_Y)
        c_lon = jnp.asarray(np.asarray(cluster_cat['GLON'], dtype=float))
        c_lat = jnp.asarray(np.asarray(cluster_cat['GLAT'], dtype=float))
        c_z = jnp.asarray(np.asarray(cluster_cat['Z'], dtype=float))
        c_sigma_Y = jnp.asarray(np.asarray(cluster_cat['eY'], dtype=float))
        c_sigma_logY = c_sigma_Y / (c_Y * jnp.log(10))

        # TODO: do something about this...
        # if jnp.any(c_logY < self.logY_lim):
        #     below = jnp.where(c_logY < self.logY_lim)[0]
        #     raise ValueError(f"Clusters below logY_lim: indices {below}, "
        #                      "ensure logY_obs >= logY_lim.")

        # Compute comoving distances (use numpy versions for now)
        h_dist = cz_to_comoving_distance(
            np.array(h_z) * SPEED_OF_LIGHT_KMS, h=self.h, Om0=self.Om0
        )
        c_dist = cz_to_comoving_distance(
            np.array(c_z) * SPEED_OF_LIGHT_KMS, h=self.h, Om0=self.Om0
        )

        # Compute unit vectors
        h_uv = radec_to_cartesian(np.array(h_lon), np.array(h_lat))
        c_uv = radec_to_cartesian(np.array(c_lon), np.array(c_lat))

        # Convert back to JAX
        h_uv = jnp.asarray(h_uv)
        c_uv = jnp.asarray(c_uv)
        h_dist = jnp.asarray(h_dist)
        c_dist = jnp.asarray(c_dist)
        h_cz = SPEED_OF_LIGHT_KMS * h_z
        c_cz = SPEED_OF_LIGHT_KMS * c_z

        return {
            'h_logM': h_logM,
            'h_uv': h_uv,
            'h_lat': h_lat,
            'h_cz': h_cz,
            'c_logY': c_logY,
            'c_sigma_logY': c_sigma_logY,
            'c_uv': c_uv,
            'c_lat': c_lat,
            'c_cz': c_cz,
            'mean_sigma_logY': jnp.mean(c_sigma_logY),
        }

    def _precompute_pairs(self, groups):
        """
        Precompute all cluster-halo pairs across all groups and permutations.

        For GPU efficiency, this creates flat arrays of all pairs that need
        likelihood evaluation, along with metadata for reduction.

        Parameters
        ----------
        groups : list of dict
            FoF groups from partition_volume.

        Returns
        -------
        pair_data : dict
            Dictionary containing:
            - obs_cluster_idx: cluster indices for observed pairs
            - obs_halo_idx: halo indices for observed pairs
            - obs_assoc_id: association ID for each observed pair
            - virt_halo_idx: halo indices for virtual pairs
            - virt_assoc_id: association ID for each virtual pair
            - assoc_sizes: number of halos per association (for logmeanexp)
            - n_groups: number of groups
        """
        obs_cluster_idx = []
        obs_halo_idx = []
        obs_assoc_id = []

        virt_halo_idx = []
        virt_assoc_id = []

        assoc_to_group = []
        assoc_sizes = []
        assoc_id_counter = 0

        for group_id, group in enumerate(tqdm(groups, desc="Pairing groups")):
            h_idx = group['halo_indices']
            c_idx = group['cluster_indices']

            n_h = len(h_idx)
            n_c = len(c_idx)

            if n_h == 0:
                continue
            if n_c > n_h:
                raise ValueError(f"Group {group_id} has more clusters "
                                 f"({n_c}) than halos ({n_h}). "
                                 "Matching requires n_c <= n_h.")

            # Generate all permutations for this group
            perms = _generate_permutations(n_h)

            # For each permutation
            for perm in perms:
                assoc_id = assoc_id_counter

                # Observed pairs: first n_c positions in perm match clusters
                for i in range(n_c):
                    obs_cluster_idx.append(c_idx[i])
                    obs_halo_idx.append(h_idx[perm[i]])
                    obs_assoc_id.append(assoc_id)

                # Virtual pairs: remaining positions are unobserved
                for i in range(n_c, n_h):
                    virt_halo_idx.append(h_idx[perm[i]])
                    virt_assoc_id.append(assoc_id)

                assoc_to_group.append(group_id)
                assoc_sizes.append(n_h)
                assoc_id_counter += 1

        # Convert to JAX arrays
        pair_data = {
            'obs_cluster_idx': jnp.array(obs_cluster_idx, dtype=jnp.int32),
            'obs_halo_idx': jnp.array(obs_halo_idx, dtype=jnp.int32),
            'obs_assoc_id': jnp.array(obs_assoc_id, dtype=jnp.int32),
            'virt_halo_idx': jnp.array(virt_halo_idx, dtype=jnp.int32),
            'virt_assoc_id': jnp.array(virt_assoc_id, dtype=jnp.int32),
            'assoc_to_group': jnp.array(assoc_to_group, dtype=jnp.int32),
            'assoc_sizes': jnp.array(assoc_sizes, dtype=jnp.int32),
            'n_assocs': assoc_id_counter,
            'n_groups': len(groups),
        }

        return pair_data


def print_group_summary(groups):
    """
    Print summary information about FoF groups.

    Parameters
    ----------
    groups : list of dict
        List of groups from partition_volume.
    """
    # Check constraint: no groups with clusters but no halos
    invalid_groups = []
    for i, group in enumerate(groups):
        n_h = len(group['halo_indices'])
        n_c = len(group['cluster_indices'])
        if n_c > 0 and n_h == 0:
            invalid_groups.append(i)
        if n_c > n_h:
            print(f"ERROR: Group {i} has more clusters ({n_c}) than halos "
                  f"({n_h}). Matching requires n_c <= n_h.")
            print(f"  Halos: {group['halo_indices']}")
            print(f"  Clusters: {group['cluster_indices']}")
            # return

    if invalid_groups:
        print(f"ERROR: Found {len(invalid_groups)} groups with clusters "
              "but no halos!")
        for i in invalid_groups:
            group = groups[i]
            print(f"  Group {i}: {len(group['halo_indices'])} halos, "
                  f"{len(group['cluster_indices'])} clusters")
        # return

    # Count groups by number of halos
    halo_counts = {}
    for group in groups:
        n_h = len(group['halo_indices'])
        halo_counts[n_h] = halo_counts.get(n_h, 0) + 1

    # Compute total pair evaluations across all associations
    total_obs_pairs = 0
    total_virt_pairs = 0
    total_associations = 0
    total_halos = sum(len(g['halo_indices']) for g in groups)
    total_clusters = sum(len(g['cluster_indices']) for g in groups)

    for group in groups:
        n_h = len(group['halo_indices'])
        n_c = len(group['cluster_indices'])
        n_assoc = factorial(n_h)
        total_associations += n_assoc
        total_obs_pairs += n_assoc * n_c
        total_virt_pairs += n_assoc * max(n_h - n_c, 0)

    print(f"Total groups: {len(groups)}")
    print(f"Total halos: {total_halos}")
    print(f"Total clusters: {total_clusters}")

    def fmt_num(n):
        return f"{n:.2e}" if n > 100_000 else f"{n}"

    print(f"\nTotal associations: {fmt_num(total_associations)}")
    print(f"Total observed-halo pair evaluations: {fmt_num(total_obs_pairs)}")
    print(f"Total virtual-halo pair evaluations: {fmt_num(total_virt_pairs)}")

    print("\nGroups by number of halos:")
    print(f"{'N_halos':<10} {'N_groups':<10}")
    print("-" * 20)
    for n_h in sorted(halo_counts.keys()):
        print(f"{n_h:<10} {halo_counts[n_h]:<10}")


def _generate_permutations(n):
    """
    Generate all n! permutations of range(n) as JAX array.

    Parameters
    ----------
    n : int
        Number of elements to permute.

    Returns
    -------
    perms : jnp.array
        Array of shape (n!, n) where each row is a permutation.
    """
    perms = list(permutations(range(n)))
    return jnp.array(perms, dtype=jnp.int32)


@dataclass
class ObservedInputs:
    c_logY: jnp.ndarray
    c_sigma_logY: jnp.ndarray
    c_uv: jnp.ndarray
    c_cz: jnp.ndarray
    c_lat: jnp.ndarray
    h_logM: jnp.ndarray
    h_uv: jnp.ndarray
    h_cz: jnp.ndarray
    obs_assoc_id: jnp.ndarray
    logM_piv: float
    logY_lim: float

    @property
    def __len__(self):
        return len(self.c_logY)


@dataclass
class VirtualInputs:
    h_logM: jnp.ndarray
    h_lat: jnp.ndarray
    h_cz: jnp.ndarray
    h_idx: jnp.ndarray
    h_unique_idx: jnp.ndarray  # Unique halo indices
    h_first_occurrence: jnp.ndarray  # First occurrence of each unique halo
    h_inverse_indices: jnp.ndarray  # Maps virtual pairs back to unique halos
    virt_assoc_id: jnp.ndarray
    assoc_to_group: jnp.ndarray
    n_assocs: int
    n_groups: int
    logM_piv: float
    cz_max: float
    logY_lim: float
    mean_sigma_logY: float

    @property
    def __len__(self):
        return len(self.h_logM)


def generate_mock_clusters(halos, alpha=0.0, beta=1.0, sigma_int=0.2,
                            sigma_theta_deg=1.0, sigma_v_kms=300.0,
                            f_det=0.5, z_max=0.05, b_cut=5.0,
                            Y_min=1e14, logY_piv=14.0, logM_piv=14.0, seed=None):
    """
    Generate mock cluster catalog from halos with observational effects.

    Applies scaling relation, selection cuts, observational scatter, and
    detection efficiency to create a realistic mock cluster catalog.

    Parameters
    ----------
    halos : dict
        Dictionary with keys 'GLON', 'GLAT', 'Z', 'M' (halo properties).
    alpha : float
        Scaling relation intercept (at logM_piv).
    beta : float
        Scaling relation slope.
    sigma_int : float
        Intrinsic scatter in log(Y).
    sigma_theta_deg : float
        Angular scatter, degrees.
    sigma_v_kms : float
        Velocity scatter, km/s.
    f_det : float
        Detection efficiency (fraction of clusters to keep).
    z_max : float
        Maximum redshift.
    b_cut : float
        Galactic latitude cut, degrees. Requires |b| > b_cut.
    Y_min : float, optional
        Minimum observable Y for detection. Applied after scatter.
    logY_piv : float
        Pivot Y for scaling relation.
    logM_piv : float
        Pivot mass for scaling relation.
    seed : int, optional
        Random seed.

    Returns
    -------
    clusters : dict
        Dictionary with observed cluster properties: 'GLON', 'GLAT', 'Z',
        'logY', 'sigma_logY'.
    halo_indices : ndarray
        Indices of parent halos for each cluster.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get all halo properties (no cuts yet)
    n_halos = len(halos['GLON'])
    glon_h = halos['GLON']
    glat_h = halos['GLAT']
    z_h = halos['Z']
    M_h = halos['M']

    # Generate log(Y) from scaling relation with scatter
    # logY = logY_piv + alpha + beta * (logM - logM_piv)
    logM_h = np.log10(M_h)
    logY_mean = logY_piv + alpha + beta * (logM_h - logM_piv)
    logY = logY_mean + np.random.normal(0, sigma_int, n_halos)

    # Convert angles to radians for scattering
    sigma_theta_rad = np.deg2rad(sigma_theta_deg)

    # Generate angular offsets (small angle approximation in tangent plane)
    dtheta = np.random.normal(0, sigma_theta_rad, n_halos)
    dphi = np.random.normal(0, sigma_theta_rad, n_halos)

    # Convert to observed glon, glat
    # For small angles: delta_glon = dphi / cos(glat), delta_glat = dtheta
    glat_c_rad = np.deg2rad(glat_h) + dtheta
    glon_c_rad = np.deg2rad(glon_h) + dphi / np.cos(np.deg2rad(glat_h))

    glon_c = np.rad2deg(glon_c_rad) % 360.0  # Wrap to [0, 360)
    glat_c = np.rad2deg(glat_c_rad)

    # Generate velocity scatter
    cz_h = z_h * SPEED_OF_LIGHT_KMS
    cz_c = cz_h + np.random.normal(0, sigma_v_kms, n_halos)
    z_c = cz_c / SPEED_OF_LIGHT_KMS

    # Convert Y to linear space for threshold cut
    Y_linear = 10**logY

    # Apply selection cuts on cluster (observed) properties
    z_mask = z_c < z_max
    b_mask = np.abs(glat_c) > b_cut
    Y_mask = Y_linear >= Y_min if Y_min is not None else np.ones(n_halos, dtype=bool)

    # Apply detection efficiency only to those passing thresholds
    selection_mask = z_mask & b_mask & Y_mask
    detected = selection_mask & (np.random.uniform(0, 1, n_halos) < f_det)

    # Get final Y and error values
    Y = Y_linear[detected]
    # Convert error from log space: eY ≈ Y * sigma_logY * ln(10)
    eY = Y * 0.1 * np.log(10)

    clusters = {
        'GLON': glon_c[detected],
        'GLAT': glat_c[detected],
        'Z': z_c[detected],
        'Y': Y,
        'eY': eY
    }

    halo_indices = np.where(detected)[0]

    return clusters, halo_indices


class ProbabilisticMatcher:
    """
    NumPyro model for probabilistic cluster-halo matching with marginalization
    over all associations. Includes a logY-logM scaling relation, positional
    likelihoods, selection effects, and virtual clusters.

    Parameters
    ----------
    cz_max : scalar
        Maximum comoving redshift for selection, km/s.
    logY_lim : scalar
        Log detection limit for clusters.
    sky_mask_interpolator : MaskedSkyInterpolator
        Precomputed interpolator for sky mask integrals. Must be initialized
        with the same b_lim used for sky masking.
    logM_piv : scalar
        Pivot mass for scaling relation.
    logY_piv : scalar, optional
        Pivot log Y for scaling relation.
    logM_min : scalar, optional
        Minimum halo mass in catalog.
    """

    def __init__(self, cz_max, logY_lim, sky_mask_interpolator,
                 logM_piv=14.0, logY_piv=None, logM_min=None):
        self.cz_max = float(cz_max)
        self.logM_piv = logM_piv
        self.logY_piv = logY_piv if logY_piv is not None else 0.0
        self.logY_lim = float(logY_lim)
        self.logY_lim_centered = self.logY_lim - self.logY_piv
        self.sky_mask_interpolator = sky_mask_interpolator
        self.logM_min = logM_min  # Minimum halo mass in catalog

    def prepare_model_inputs(self, data):
        """
        Precompute inputs needed by the model to avoid Python work during
        trace.

        Returns
        -------
        obs_inputs : ObservedInputs or None
            Prepared observed-pair inputs.
        virt_inputs : VirtualInputs or None
            Prepared virtual-pair inputs.
        """
        processed = data.processed
        pair_data = data.pair_data
        obs_inputs = self._prepare_observed_inputs(
            processed, pair_data, self.logY_piv)
        virt_inputs = self._prepare_virtual_inputs(processed, pair_data)
        return obs_inputs, virt_inputs

    def _prepare_observed_inputs(self, processed, pair_data, logY_piv):
        if pair_data['obs_cluster_idx'].size == 0:
            return None

        c_idx = pair_data['obs_cluster_idx']
        h_idx = pair_data['obs_halo_idx']

        c_logY = processed['c_logY'][c_idx] - logY_piv
        return ObservedInputs(
            c_logY=c_logY,
            c_sigma_logY=processed['c_sigma_logY'][c_idx],
            c_uv=processed['c_uv'][c_idx],
            c_cz=processed['c_cz'][c_idx],
            c_lat=processed['c_lat'][c_idx],
            h_logM=processed['h_logM'][h_idx],
            h_uv=processed['h_uv'][h_idx],
            h_cz=processed['h_cz'][h_idx],
            obs_assoc_id=pair_data['obs_assoc_id'],
            logM_piv=self.logM_piv,
            logY_lim=self.logY_lim_centered,
        )

    def _prepare_virtual_inputs(self, processed, pair_data):
        if pair_data['virt_halo_idx'].size == 0:
            return None

        h_idx = pair_data['virt_halo_idx']
        h_idx_jnp = jnp.asarray(h_idx)

        # Precompute unique halo mapping for efficient virtual likelihood
        # computation
        unique_h_idx, first_occurrence, inverse_indices = jnp.unique(
            h_idx_jnp, return_index=True, return_inverse=True
        )

        return VirtualInputs(
            h_logM=processed['h_logM'][h_idx],
            h_lat=processed['h_lat'][h_idx],
            h_cz=processed['h_cz'][h_idx],
            h_idx=h_idx_jnp,
            h_unique_idx=unique_h_idx,
            h_first_occurrence=first_occurrence,
            h_inverse_indices=inverse_indices,
            virt_assoc_id=pair_data['virt_assoc_id'],
            assoc_to_group=pair_data['assoc_to_group'],
            n_assocs=int(pair_data['n_assocs']),
            n_groups=int(pair_data['n_groups']),
            logM_piv=self.logM_piv,
            cz_max=self.cz_max,
            logY_lim=self.logY_lim_centered,
            mean_sigma_logY=float(processed['mean_sigma_logY']),
        )

    def _compute_observed_ll(self, obs_inputs, alpha, beta, sigma_int,
                             sigma_theta, sigma_v, f_genuine,
                             sigma_int_spurious, sigma_theta_spurious,
                             sigma_v_spurious):
        if obs_inputs is None:
            return jnp.array([])

        # Determine if f_genuine is per-pair or global
        f_genuine_axis = 0 if jnp.ndim(f_genuine) > 0 else None

        return jax.vmap(
            self._log_likelihood_obs,
            in_axes=(0, 0, 0, 0, 0, 0, 0,
                     None, None, None, None, None, f_genuine_axis, None, None,
                     None, None)
        )(
            obs_inputs.c_logY, obs_inputs.c_sigma_logY, obs_inputs.c_uv,
            obs_inputs.c_cz, obs_inputs.h_logM, obs_inputs.h_uv,
            obs_inputs.h_cz, alpha, beta, sigma_int, sigma_theta, sigma_v,
            f_genuine, obs_inputs.logM_piv,
            sigma_int_spurious, sigma_theta_spurious, sigma_v_spurious
        )

    def _compute_virtual_ll(self, virt_inputs, alpha, beta, sigma_int,
                            sigma_v, f_genuine, f_sky_all):
        if virt_inputs is None:
            return jnp.array([])

        # Get properties for unique halos
        h_logM_unique = virt_inputs.h_logM[virt_inputs.h_first_occurrence]
        h_cz_unique = virt_inputs.h_cz[virt_inputs.h_first_occurrence]
        f_sky_unique = f_sky_all[virt_inputs.h_unique_idx]

        # Get f_genuine for unique halos
        if jnp.ndim(f_genuine) > 0:
            # Mass-dependent: extract unique values
            f_genuine_unique = f_genuine[virt_inputs.h_first_occurrence]
            f_genuine_axis = 0
        else:
            # Global: broadcast
            f_genuine_unique = f_genuine
            f_genuine_axis = None

        # Compute virtual likelihood once per unique halo
        ll_unique = jax.vmap(
            self._log_likelihood_virtual,
            in_axes=(0, 0, None, None, None, None, None, f_genuine_axis, None, None,
                     None, 0)
        )(
            h_logM_unique,
            h_cz_unique,
            alpha, beta, sigma_int,
            sigma_v,
            virt_inputs.logY_lim, f_genuine_unique, virt_inputs.logM_piv,
            virt_inputs.cz_max,
            virt_inputs.mean_sigma_logY, f_sky_unique
        )

        # Map back to all virtual pairs using precomputed inverse indices
        return ll_unique[virt_inputs.h_inverse_indices]

    @staticmethod
    def _assoc_ll(ll_obs, ll_virt, obs_inputs, virt_inputs):
        assoc_ll = jnp.zeros(virt_inputs.n_assocs)

        if ll_obs.size:
            assoc_ll = assoc_ll + jax.ops.segment_sum(
                ll_obs, obs_inputs.obs_assoc_id, virt_inputs.n_assocs)

        if ll_virt.size:
            assoc_ll = assoc_ll + jax.ops.segment_sum(
                ll_virt, virt_inputs.virt_assoc_id, virt_inputs.n_assocs)

        return assoc_ll

    @staticmethod
    def _group_ll(assoc_ll, virt_inputs):
        group_logsumexp = _segment_logsumexp(
            assoc_ll, virt_inputs.assoc_to_group, virt_inputs.n_groups)

        assoc_counts = jax.ops.segment_sum(
            jnp.ones(virt_inputs.n_assocs),
            virt_inputs.assoc_to_group, virt_inputs.n_groups)

        return group_logsumexp - jnp.log(assoc_counts)

    @staticmethod
    @jax.jit
    def _log_likelihood_obs(logY_obs, sigma_logY, uv_c, cz_c,
                            logM_h, uv_h, cz_h, alpha, beta, sigma_int,
                            sigma_theta, sigma_v, f_genuine, logM_piv,
                            sigma_int_spurious,
                            sigma_theta_spurious, sigma_v_spurious):
        """
        Compute log likelihood for observed cluster-halo pair with spurious halo model.

        Mixture model: f_genuine * L_genuine + (1 - f_genuine) * L_spurious

        where:
        - L_genuine: likelihood if halo is a real structure with tight uncertainties
        - L_spurious: likelihood if halo is spurious with wide uncertainties (no mass scaling)

        Uses von Mises-Fisher distribution for angular separation and
        Gaussian for redshift separation.
        """
        theta_rad = angular_separation(uv_h, uv_c)

        # Genuine halo model (tight uncertainties, mass scaling)
        logY_exp_genuine = log_Y_expected(logM_h, alpha, beta, logM_piv)
        sigma_total_genuine = jnp.sqrt(sigma_logY**2 + sigma_int**2)
        log_prob_Y_genuine = jax_norm.logpdf(logY_obs, logY_exp_genuine, sigma_total_genuine)
        lp_theta_genuine = von_mises_fisher_logpdf(theta_rad, sigma_theta)
        lp_z_genuine = jax_norm.logpdf(cz_c, cz_h, sigma_v)
        ll_genuine = log_prob_Y_genuine + lp_theta_genuine + lp_z_genuine

        # Spurious halo model (wide uncertainties, no mass scaling)
        logY_exp_spurious = log_Y_expected(logM_h, alpha, 0.0, logM_piv)
        sigma_total_spurious = jnp.sqrt(sigma_logY**2 + sigma_int_spurious**2)
        log_prob_Y_spurious = jax_norm.logpdf(logY_obs, logY_exp_spurious, sigma_total_spurious)
        lp_theta_spurious = von_mises_fisher_logpdf(theta_rad, sigma_theta_spurious)
        lp_z_spurious = jax_norm.logpdf(cz_c, cz_h, sigma_v_spurious)
        # ll_spurious = log_prob_Y_spurious + lp_theta_spurious + lp_z_spurious
        ll_spurious = -6

        # Mixture model in log space
        return jnp.logaddexp(
            jnp.log(f_genuine) + ll_genuine,
            jnp.log(1.0 - f_genuine) + ll_spurious
        )

    @staticmethod
    @jax.jit
    def _log_likelihood_virtual(logM_h, cz_h, alpha, beta, sigma_int, sigma_v,
                                logY_lim, f_genuine, logM_piv, cz_max,
                                mean_sigma_logY, f_sky_val):
        """
        Compute log likelihood for virtual (unobserved) cluster with spurious halos.

        p_virtual(h) = 1 - f_genuine * p_detect

        where:
        - f_genuine: probability halo is a real structure (not spurious)
        - p_detect = f_Y * f_sky * f_z: detection probability for genuine halo

        This accounts for:
        - Spurious halos (1 - f_genuine): never produce clusters
        - Genuine halos (f_genuine): might not be detected (1 - p_detect)

        Assuming detection efficiency = 1 for genuine halos.
        """
        logY_exp = log_Y_expected(logM_h, alpha, beta, logM_piv)

        sigma_total = jnp.sqrt(mean_sigma_logY**2 + sigma_int**2)
        f_Y_val = f_Y(logY_lim, logY_exp, sigma_total)
        f_z_val = f_z(cz_max, cz_h, sigma_v)
        p_detect = f_Y_val * f_sky_val * f_z_val

        p_virtual = 1.0 - f_genuine * p_detect

        return jnp.log(p_virtual)

    def model(self, obs_inputs, virt_inputs,
              alpha_limits=(-10, 10),
              beta_limits=(-5, 5),
              sigma_int_limits=(0, 1),
              sigma_theta_deg_limits=(0.1, 7.5),
              sigma_v_limits=(10, 2500),
              sigma_int_spurious=1.0,
              sigma_theta_deg_spurious=10.0,
              sigma_v_spurious=2000.0,
              mass_dependent_genuineness=False,
              logM_transition_limits=(13.0, 15.0),
              width_limits=(0.1, 1.0)):
        """
        NumPyro probabilistic model for cluster-halo matching with spurious halos.

        Uses fully vectorized likelihood computation for GPU efficiency.
        Models BORG halo genuineness: some fraction of halos may be spurious.

        Parameters
        ----------
        obs_inputs : ObservedInputs
            Precomputed observed-pair inputs from prepare_model_inputs.
        virt_inputs : VirtualInputs
            Precomputed virtual-pair inputs from prepare_model_inputs.
        alpha_limits : tuple
            Prior limits for alpha (scaling relation intercept).
        beta_limits : tuple
            Prior limits for beta (scaling relation slope).
        sigma_int_limits : tuple
            Prior limits for intrinsic scatter (for genuine halos).
        sigma_theta_deg_limits : tuple
            Prior limits for angular uncertainty (for genuine halos), degrees.
        sigma_v_limits : tuple
            Prior limits for velocity uncertainty (for genuine halos), km/s.
        sigma_int_spurious : float
            Fixed intrinsic scatter for spurious halo model.
        sigma_theta_deg_spurious : float
            Fixed angular scatter for spurious halo model, degrees.
        sigma_v_spurious : float
            Fixed velocity scatter for spurious halo model, km/s.
        mass_dependent_genuineness : bool
            If True, model mass-dependent genuineness using sigmoid function.
        logM_transition_limits : tuple
            Prior limits for transition mass (if mass_dependent_genuineness=True).
        width_limits : tuple
            Prior limits for transition width in dex (if mass_dependent_genuineness=True).
        """
        # Priors for scaling relation
        alpha = sample('alpha', Uniform(*alpha_limits))
        beta = sample('beta', Uniform(*beta_limits))

        sigma_int = sample('sigma_int', Uniform(*sigma_int_limits))

        sigma_theta_deg = sample(
            'sigma_theta', Uniform(*sigma_theta_deg_limits))
        sigma_v = sample('sigma_v', Uniform(*sigma_v_limits))

        # Probability that a BORG halo is genuine (not spurious)
        if mass_dependent_genuineness:
            f_genuine_high = sample('f_genuine_high', Uniform(0, 1))
            logM_transition = sample('logM_transition', Uniform(*logM_transition_limits))
            width = sample('width', Uniform(*width_limits))
            # width = 0.1

            # Compute mass-dependent genuineness for observed pairs
            if obs_inputs is not None:
                f_genuine_obs = f_genuine_mass_dependent(
                    obs_inputs.h_logM, f_genuine_high, logM_transition, width)
            else:
                f_genuine_obs = None

            # Compute for all halos (for virtual likelihood)
            if virt_inputs is not None:
                f_genuine_virt_unique = f_genuine_mass_dependent(
                    virt_inputs.h_logM[virt_inputs.h_first_occurrence],
                    f_genuine_high, logM_transition, width)
                f_genuine_virt = f_genuine_virt_unique[virt_inputs.h_inverse_indices]
            else:
                f_genuine_virt = None
        else:
            # Global genuineness
            f_genuine_global = sample('f_genuine', Uniform(0, 1))
            f_genuine_obs = f_genuine_global
            f_genuine_virt = f_genuine_global

        lp = -jnp.log(sigma_v)

        # Convert sigma_theta to radians
        sigma_theta = jnp.deg2rad(sigma_theta_deg)
        sigma_theta_spurious = jnp.deg2rad(sigma_theta_deg_spurious)

        # Compute f_sky values for all halos (von Mises-Fisher integrals)
        # This is done once per MCMC iteration using the precomputed
        # interpolator
        f_sky_all = self.sky_mask_interpolator(sigma_theta_deg)

        # Observed cluster likelihoods (with spurious halo model)
        ll_obs = self._compute_observed_ll(
            obs_inputs, alpha, beta, sigma_int, sigma_theta, sigma_v,
            f_genuine_obs, sigma_int_spurious, sigma_theta_spurious,
            sigma_v_spurious)

        # Virtual cluster likelihoods
        ll_virt = self._compute_virtual_ll(
            virt_inputs, alpha, beta, sigma_int, sigma_v,
            f_genuine_virt, f_sky_all)

        assoc_ll = self._assoc_ll(ll_obs, ll_virt, obs_inputs, virt_inputs)

        group_ll = self._group_ll(assoc_ll, virt_inputs)

        total_log_lik = jnp.sum(group_ll)

        # Orphan cluster likelihood (clusters not in any group)
        # These might come from halos below M_min
        if self.logM_min is not None:
            # Count orphan clusters (not in observed pairs)
            # This should be passed in from prepare_model_inputs
            # For now, assume they have small but non-zero probability
            # n_orphans = ...  # would need to pass this in
            # p_orphan = 0.01  # or sample this
            # total_log_lik += n_orphans * jnp.log(p_orphan)
            pass  # Placeholder - need orphan count from prepare_model_inputs

        # Add to model
        factor('obs', total_log_lik + lp)
