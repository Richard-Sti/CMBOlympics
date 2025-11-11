"""Utilities for identifying halo associations across realisations."""

from dataclasses import dataclass, field
import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM
from tqdm import tqdm
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
    optional_data: dict = field(default=None)

    # Map signal fields (populated by compute_map_signals)
    median_galactic: tuple = field(default=None)  # (r, ell, b)
    median_theta500: float = field(default=None)
    median_signal: float = field(default=None)
    median_pval: float = field(default=None)
    halo_galactic: tuple = field(default=None)  # (r, ell, b) arrays
    halo_theta500: np.ndarray = field(default=None)
    halo_signals: np.ndarray = field(default=None)
    halo_pvals: np.ndarray = field(default=None)

    def keys(self):
        keys = [
            "label",
            "centroid",
            "positions",
            "masses",
            "realisations",
            "member_indices",
        ]
        if self.optional_data:
            keys.extend(self.optional_data.keys())
        return keys

    def as_dict(self):
        d = {
            "label": self.label,
            "centroid": self.centroid,
            "positions": self.positions,
            "masses": self.masses,
            "realisations": self.realisations,
            "member_indices": self.member_indices,
            "fraction_present": self.fraction_present,
        }
        if self.optional_data:
            d.update(self.optional_data)
        return d

    def __getitem__(self, key):
        if self.optional_data and key in self.optional_data:
            return self.optional_data[key]
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

    def to_z(self, center, Om):
        """
        Compute cosmological redshifts from comoving distances.

        Parameters
        ----------
        center : array-like
            Observer position used as the comoving-distance origin.
        Om : float
            Matter density parameter of a flat LCDM cosmology with h=1.

        Returns
        -------
        ndarray
            Redshifts corresponding to each halo position.
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a 3-vector.")

        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")

        rel = positions - center
        comoving = np.linalg.norm(rel, axis=1)

        z = np.full_like(comoving, np.nan, dtype=float)
        mask = np.isfinite(comoving) & (comoving >= 0)
        if not np.any(mask):
            return z

        cosmo = FlatLambdaCDM(H0=100.0, Om0=float(Om))
        distances = comoving * u.Mpc
        idx = np.where(mask)[0]
        for i in idx:
            dist = distances[i]
            z[i] = float(z_at_value(cosmo.comoving_distance, dist))

        return z

    def to_da(self, center, Om):
        """
        Compute angular diameter distances for halo members.

        Parameters
        ----------
        center : array-like
            Observer position used as the comoving-distance origin.
        Om : float
            Matter density parameter of a flat LCDM cosmology with h=1.

        Returns
        -------
        ndarray
            Angular diameter distances in Mpc for each halo in this
            association.
        """
        z = self.to_z(center, Om)
        da = np.full_like(z, np.nan, dtype=float)
        mask = np.isfinite(z) & (z >= 0)
        if not np.any(mask):
            return da

        cosmo = FlatLambdaCDM(H0=100.0, Om0=float(Om))
        da[mask] = cosmo.angular_diameter_distance(z[mask]).to_value(u.Mpc)

        return da

    def compute_map_signals(self, profiler, obs_pos, theta_rand, map_rand,
                            r_key="Group_R_Crit500", coord_system="icrs"):
        """
        Compute and store map signals and p-values for this association.

        Computes signals at both the median position and for each individual
        halo. Results are stored directly in the association object.

        Parameters
        ----------
        profiler
            PointingEnclosedProfile instance for measuring map signals.
        obs_pos
            Observer position (3D array) in same coordinate system as
            association positions.
        theta_rand
            Angular sizes for random signal profiles (arcmin).
        map_rand
            Map signal profiles for random pointings, shape
            (n_random, n_theta).
        r_key
            Key in optional_data for halo radii (e.g., 'Group_R_Crit500').
        coord_system
            Coordinate system of positions. Currently only "icrs".
        """
        if coord_system != "icrs":
            raise ValueError(
                f"coord_system='{coord_system}' not supported. "
                "Currently only 'icrs' is available."
            )

        if r_key not in self.optional_data:
            raise KeyError(
                f"Association {self.label} missing '{r_key}' in "
                "optional_data."
            )

        # Get galactic coordinates for all haloes
        r, ell, b = self.to_galactic_angular(obs_pos, coord_system)
        radii = self.optional_data[r_key]

        # Compute aperture sizes
        theta500 = np.rad2deg(np.arctan(radii / r)) * 60

        # Compute median position and aperture
        median_r = float(np.median(r))
        median_ell = float(np.median(ell))
        median_b = float(np.median(b))
        median_theta500 = float(np.median(theta500))

        # Measure signal at median position
        median_signal = float(profiler.get_profile(
            median_ell, median_b, median_theta500
        ))

        # Compute p-value for median
        median_pval = float(profiler.signal_to_pvalue(
            np.array([median_theta500]),
            np.array([median_signal]),
            theta_rand,
            map_rand
        )[0])

        # Measure signals for all haloes (disable progress bar)
        halo_signals = profiler.get_profiles_per_source(
            ell, b, theta500, verbose=False
        )

        # Compute p-values for all haloes
        halo_pvals = profiler.signal_to_pvalue(
            theta500, halo_signals, theta_rand, map_rand
        )

        # Store results
        self.median_galactic = (median_r, median_ell, median_b)
        self.median_theta500 = median_theta500
        self.median_signal = median_signal
        self.median_pval = median_pval
        self.halo_galactic = (r, ell, b)
        self.halo_theta500 = theta500
        self.halo_signals = halo_signals
        self.halo_pvals = halo_pvals


def compute_association_signals(associations, profiler, obs_pos,
                                theta_rand, map_rand,
                                r_key="Group_R_Crit500",
                                coord_system="icrs"):
    """
    Compute map signals for all associations in a list.

    Calls compute_map_signals() on each association, storing results
    directly in the association objects.

    Parameters
    ----------
    associations
        List of HaloAssociation objects.
    profiler
        PointingEnclosedProfile instance for measuring map signals.
    obs_pos
        Observer position (3D array).
    theta_rand
        Angular sizes for random pointings (arcmin).
    map_rand
        Map signals for random pointings, shape (n_random, n_theta).
    r_key
        Key in optional_data for halo radii.
    coord_system
        Coordinate system of positions. Currently only "icrs".
    """
    for assoc in tqdm(associations, desc="Computing association signals"):
        assoc.compute_map_signals(
            profiler, obs_pos, theta_rand, map_rand,
            r_key=r_key,
            coord_system=coord_system
        )


def identify_halo_associations(positions, masses, eps=1.75, min_samples=9,
                               mass_sigma=0.3, optional_data=None):
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
    optional_data
        Dictionary mapping key names to sequences of arrays (one per
        realisation), matched to ``positions``. These will be filtered and
        stored in the associations.

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

    # Process optional data
    opt_dict = {}
    if optional_data:
        for key, data_list in optional_data.items():
            if len(data_list) != len(positions):
                raise ValueError(
                    f"Optional key '{key}' must have same length as "
                    "positions."
                )
            arrays = [np.asarray(arr, dtype=float) for arr in data_list]
            for arr, pos in zip(arrays, pos_arrays):
                if arr.shape[0] != pos.shape[0]:
                    raise ValueError(
                        f"Optional key '{key}' arrays must match positions."
                    )
            opt_dict[key] = arrays

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

    # Concatenate optional data
    all_opt = {}
    for key, arrays in opt_dict.items():
        all_opt[key] = np.concatenate(arrays)

    finite_mask = np.isfinite(all_masses).astype(bool)
    finite_mask &= np.all(np.isfinite(all_positions), axis=1)
    if not np.any(finite_mask):
        return []

    all_positions = all_positions[finite_mask]
    all_masses = all_masses[finite_mask]
    real_ids = real_ids[finite_mask]
    halo_ids = halo_ids[finite_mask]
    for key in all_opt:
        all_opt[key] = all_opt[key][finite_mask]

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
        cluster_opt = {key: all_opt[key][cluster_indices]
                       for key in all_opt}

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
        for key in cluster_opt:
            cluster_opt[key] = cluster_opt[key][keep]

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
        for key in cluster_opt:
            cluster_opt[key] = cluster_opt[key][mass_mask]

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
                optional_data=cluster_opt if cluster_opt else None,
            )
        )

    return associations
