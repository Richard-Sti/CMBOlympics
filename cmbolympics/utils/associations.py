"""Utilities for identifying halo associations across realisations."""

from dataclasses import dataclass
import numpy as np
from .coords import cartesian_icrs_to_galactic_spherical


@dataclass
class HaloAssociation:
    """Container summarising a single halo association."""

    label: int
    centroid: np.ndarray
    positions: np.ndarray
    masses: np.ndarray
    realisations: np.ndarray
    member_indices: np.ndarray
    fraction_present: float

    def keys(self):
        return [
            "label",
            "centroid",
            "positions",
            "masses",
            "realisations",
            "member_indices",
        ]

    def as_dict(self):
        return {
            "label": self.label,
            "centroid": self.centroid,
            "positions": self.positions,
            "masses": self.masses,
            "realisations": self.realisations,
            "member_indices": self.member_indices,
            "fraction_present": self.fraction_present,
        }

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def to_galactic_angular(self, center, coord_system="icrs"):
        """
        Convert positions to Galactic spherical coordinates (r, ell, b).

        Parameters
        ----------
        center
            Observer position in the same coordinate system as positions.
            Array of shape (3,).
        coord_system
            Coordinate system of the positions. Currently only "icrs"
            is supported.

        Returns
        -------
        r : ndarray
            Distances in same units as positions.
        ell : ndarray
            Galactic longitude in degrees.
        b : ndarray
            Galactic latitude in degrees.
        """
        if coord_system != "icrs":
            raise ValueError(
                f"coord_system='{coord_system}' not supported. "
                "Currently only 'icrs' is available."
            )

        return cartesian_icrs_to_galactic_spherical(self.positions, center)

    def centroid_to_galactic_angular(self, center, coord_system="icrs"):
        """
        Convert centroid to Galactic spherical coordinates (r, ell, b).

        Parameters
        ----------
        center
            Observer position in the same coordinate system as centroid.
            Array of shape (3,).
        coord_system
            Coordinate system of the centroid. Currently only "icrs"
            is supported.

        Returns
        -------
        r : float
            Distance in same units as centroid.
        ell : float
            Galactic longitude in degrees.
        b : float
            Galactic latitude in degrees.
        """
        if coord_system != "icrs":
            raise ValueError(
                f"coord_system='{coord_system}' not supported. "
                "Currently only 'icrs' is available."
            )

        r, ell, b = cartesian_icrs_to_galactic_spherical(
            self.centroid.reshape(1, 3), center
        )
        return float(r[0]), float(ell[0]), float(b[0])


def identify_halo_associations(positions, masses, eps=1.75, min_samples=9,
                               mass_sigma=0.3):
    """
    Cluster haloes from multiple realisations into physical associations.

    Parameters
    ----------
    positions
        Sequence of arrays with shape (Ni, 3) storing Cartesian coordinates for
        each realisation.
    masses
        Sequence of arrays with shape (Ni,) storing halo masses matched to
        ``positions``.
    eps
        DBSCAN linking length in comoving Mpc.
    min_samples
        Minimum number of members required to keep an association.
    mass_sigma
        Maximum allowed absolute deviation (in dex) from the association mean
        log-mass before a halo is rejected as an outlier.

    Returns
    -------
    list[HaloAssociation]
        List of surviving associations sorted by cluster label.
    """

    if len(positions) != len(masses):
        raise ValueError("positions and masses must have the same length.")

    pos_arrays = [np.asarray(pos, dtype=float) for pos in positions]
    mass_arrays = [np.asarray(mass, dtype=float) for mass in masses]

    for pos, mass in zip(pos_arrays, mass_arrays):
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("Each positions array must have shape (N, 3).")
        if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
            raise ValueError(
                "Mass array must have shape (N,) matching positions."
            )

    counts = [pos.shape[0] for pos in pos_arrays]
    if not counts or sum(counts) == 0:
        return []

    real_ids = np.concatenate(
        [np.full(count, idx, dtype=int) for idx, count in enumerate(counts)]
    )
    halo_ids = np.concatenate(
        [np.arange(count, dtype=int) for count in counts]
    )
    all_positions = np.vstack(pos_arrays)
    all_masses = np.concatenate(mass_arrays)

    finite_mask = np.isfinite(all_masses).astype(bool)
    finite_mask &= np.all(np.isfinite(all_positions), axis=1)
    if not np.any(finite_mask):
        return []

    all_positions = all_positions[finite_mask]
    all_masses = all_masses[finite_mask]
    real_ids = real_ids[finite_mask]
    halo_ids = halo_ids[finite_mask]

    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-learn is required for identify_halo_associations."
        ) from exc

    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(all_positions)

    associations = []
    for label in sorted(lab for lab in np.unique(labels) if lab >= 0):
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        if cluster_indices.size == 0:
            continue

        cluster_pos = all_positions[cluster_indices]
        cluster_mass = all_masses[cluster_indices]
        cluster_real = real_ids[cluster_indices]
        cluster_local = halo_ids[cluster_indices]

        centroid = cluster_pos.mean(axis=0)
        keep = []
        for rid in np.unique(cluster_real):
            rid_idx = np.where(cluster_real == rid)[0]
            if rid_idx.size == 1:
                keep.append(rid_idx[0])
                continue
            distances = np.linalg.norm(
                cluster_pos[rid_idx] - centroid,
                axis=1,
            )
            keep.append(rid_idx[np.argmin(distances)])

        keep = np.array(keep, dtype=int)
        cluster_indices = cluster_indices[keep]
        cluster_pos = cluster_pos[keep]
        cluster_mass = cluster_mass[keep]
        cluster_real = cluster_real[keep]
        cluster_local = cluster_local[keep]

        if cluster_pos.size == 0:
            continue

        log_mass = np.log10(cluster_mass)
        mean_log = log_mass.mean()
        mass_mask = np.abs(log_mass - mean_log) <= mass_sigma

        if not np.any(mass_mask):
            continue

        cluster_indices = cluster_indices[mass_mask]
        cluster_pos = cluster_pos[mass_mask]
        cluster_mass = cluster_mass[mass_mask]
        cluster_real = cluster_real[mass_mask]
        cluster_local = cluster_local[mass_mask]

        if cluster_pos.shape[0] < min_samples:
            continue

        centroid = cluster_pos.mean(axis=0)
        member_indices = np.column_stack((cluster_real, cluster_local))
        fraction_present = cluster_real.size / len(positions)

        associations.append(
            HaloAssociation(
                label=int(label),
                centroid=centroid,
                positions=cluster_pos,
                masses=cluster_mass,
                realisations=cluster_real,
                member_indices=member_indices,
                fraction_present=fraction_present,
            )
        )

    return associations
