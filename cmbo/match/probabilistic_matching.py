# Copyright (C) 2024 Richard Stiskalek
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

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from math import factorial

from ..constants import SPEED_OF_LIGHT_KMS
from ..utils.coords import cz_to_comoving_distance, radec_to_cartesian


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

    # Standard FoF links within linking_length
    pairs = list(tree_all.query_pairs(r=linking_length))

    # Force-link every cluster to its nearest halo to ensure
    # no cluster is isolated in a group without halos
    if n_halos > 0 and n_clusters > 0:
        tree_halos = cKDTree(h_pos)
        nn_distances, nn_halo_indices = tree_halos.query(c_pos, k=1)

        # Check if any forced links exceed the linking length
        beyond_linking = nn_distances > linking_length
        if np.any(beyond_linking) and verbose:
            n_beyond = np.sum(beyond_linking)
            max_dist = np.max(nn_distances[beyond_linking])
            print(f"Warning: {n_beyond}/{n_clusters} clusters force-linked "
                  f"beyond linking_length ({linking_length:.1f} Mpc/h). "
                  f"Max distance: {max_dist:.1f} Mpc/h")
            print("\nClusters not matched within linking length:")
            print(f"{'Index':<8} {'GLON':>10} {'GLAT':>10} {'z':>8} "
                  f"{'Distance':>10}")
            print("-" * 56)
            for idx in np.where(beyond_linking)[0]:
                print(f"{idx:<8} {c_lon[idx]:>10.4f} {c_lat[idx]:>10.4f} "
                      f"{c_z[idx]:>8.4f} {nn_distances[idx]:>10.2f}")
            print()

        # Clusters have global indices n_halos to n_halos+n_clusters-1
        c_global_indices = np.arange(n_halos, n_halos + n_clusters)
        forced_links = np.column_stack((nn_halo_indices, c_global_indices))

        pairs.extend(forced_links.tolist())

    n_total = n_halos + n_clusters
    if len(pairs) > 0:
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

    order = np.argsort(labels)
    sorted_labels = labels[order]
    sorted_indices = order

    unique_labels, unique_indices = np.unique(sorted_labels, return_index=True)
    split_indices = np.split(sorted_indices, unique_indices[1:])

    groups = []
    for group_indices in split_indices:
        # Indices < n_halos are halos, >= n_halos are clusters
        g_h_indices = group_indices[group_indices < n_halos]
        g_c_indices = group_indices[group_indices >= n_halos] - n_halos

        groups.append({
            'halo_indices': np.sort(g_h_indices),
            'cluster_indices': np.sort(g_c_indices)
        })

    return groups


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

    if invalid_groups:
        print(f"ERROR: Found {len(invalid_groups)} groups with clusters "
              "but no halos!")
        for i in invalid_groups:
            group = groups[i]
            print(f"  Group {i}: {len(group['halo_indices'])} halos, "
                  f"{len(group['cluster_indices'])} clusters")
        return

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
        total_virt_pairs += n_assoc * (n_h - n_c)

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
