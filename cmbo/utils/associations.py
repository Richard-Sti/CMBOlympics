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

"""Utilities for identifying halo associations across realisations."""

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm

from ..constants import SPEED_OF_LIGHT_KMS
from .coords import (cartesian_icrs_to_galactic_spherical,
                     cartesian_to_radec,
                     comoving_distance_to_cz,
                     cz_to_comoving_distance,
                     radec_to_cartesian)


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
    velocities: np.ndarray | None = None
    optional_data: dict = field(default_factory=dict)

    # Map signal fields (populated by compute_map_signals)
    median_theta500: float = field(default=None)
    median_signal: float = field(default=None)
    median_pval: float = field(default=None)
    halo_theta500: np.ndarray = field(default=None)
    halo_signals: np.ndarray = field(default=None)
    halo_pvals: np.ndarray = field(default=None)

    def keys(self):
        """Return a list of available data fields."""
        keys = [
            "label",
            "centroid",
            "positions",
            "masses",
            "realisations",
            "member_indices",
        ]
        if self.velocities is not None:
            keys.append("velocities")
        if self.optional_data:
            keys.extend(self.optional_data.keys())
        return keys

    def as_dict(self):
        """Return a dictionary representation of the association."""
        d = {
            "label": self.label,
            "centroid": self.centroid,
            "positions": self.positions,
            "masses": self.masses,
            "realisations": self.realisations,
            "member_indices": self.member_indices,
            "fraction_present": self.fraction_present,
        }
        if self.velocities is not None:
            d["velocities"] = self.velocities
        if self.optional_data:
            d.update(self.optional_data)
        return d

    def __getitem__(self, key):
        if self.optional_data and key in self.optional_data:
            return self.optional_data[key]
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    @cached_property
    def Om0(self):
        """Matter density parameter, read from 'omega_m' in optional_data."""
        if self.optional_data and "omega_m" in self.optional_data:
            return float(self.optional_data["omega_m"])
        raise ValueError("omega_m must be present in optional_data.")

    @cached_property
    def box_size(self):
        """Simulation box size for this association."""
        if self.optional_data and "box_size" in self.optional_data:
            return float(self.optional_data["box_size"])
        raise ValueError("box_size must be present in optional_data.")

    @cached_property
    def radec(self):
        """
        Right ascension and declination for all halo members (degrees).
        """
        center = np.full(3, self.box_size / 2.0, dtype=float)
        ra, dec = cartesian_to_radec(self.positions, center=center)
        return np.column_stack((ra, dec))

    @cached_property
    def galactic_angular(self):
        """
        Galactic angular coordinates (ell, b) for all halo members.
        """
        center = np.full(3, self.box_size / 2.0, dtype=float)
        _, ell, b = cartesian_icrs_to_galactic_spherical(
            self.positions, center=center)
        return np.column_stack((ell, b))

    @cached_property
    def distance(self):
        """Comoving distances (Mpc/h) of all halo members."""
        center = np.full(3, self.box_size / 2.0, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")
        return np.linalg.norm(positions - center, axis=1)

    @cached_property
    def cosmo_redshift(self):
        """
        Cosmological redshifts for halo members assuming the observer
        sits at the box centre and Om0 from optional_data.
        """
        cz = comoving_distance_to_cz(
            self.distance, h=1.0, Om0=self.Om0)
        return cz / SPEED_OF_LIGHT_KMS

    @cached_property
    def obs_redshift(self):
        """
        Observed redshifts for halo members (including peculiar velocities).
        """
        if self.velocities is None:
            raise ValueError(
                "Velocities are required to compute the observed redshift."
            )
        runit = (self.positions - self.box_size / 2.0) / self.distance[:, None]
        Vlos = np.sum(self.velocities * runit, axis=1)
        zcosmo = self.cosmo_redshift
        return (1 + zcosmo) * (1 + Vlos / SPEED_OF_LIGHT_KMS) - 1.0

    @cached_property
    def redshift_position(self):
        """
        Reconstruct Cartesian positions (Mpc/h) from angular coordinates and
        observed redshifts.

        Uses observed redshifts (including peculiar velocities) and angular
        positions to infer comoving distances, then converts back to Cartesian
        coordinates centered on the box.
        """
        radec = self.radec
        ra = radec[:, 0]
        dec = radec[:, 1]
        unit_vectors = radec_to_cartesian(ra, dec)
        cz = self.obs_redshift * SPEED_OF_LIGHT_KMS
        distances = cz_to_comoving_distance(cz, h=1.0, Om0=self.Om0)
        center = np.full(3, self.box_size / 2.0, dtype=float)
        return (unit_vectors.T * distances).T + center

    @cached_property
    def da(self):
        """
        Angular diameter distances (Mpc) for halo members inferred from
        the cosmological redshifts.
        """
        cosmo = FlatLambdaCDM(H0=100.0, Om0=float(self.Om0))
        return cosmo.angular_diameter_distance(
            self.cosmo_redshift).to_value(u.Mpc)

    @cached_property
    def centroid_radec(self):
        """
        RA/Dec centroid coordinates (degrees) assuming the box centre observer.
        """
        center = np.full(3, self.box_size / 2.0, dtype=float)
        ra, dec = cartesian_to_radec(
            self.centroid.reshape(1, 3), center=center)
        return float(ra[0]), float(dec[0])

    @cached_property
    def centroid_galactic_angular(self):
        """
        Centroid Galactic longitude/latitude (degrees) assuming ICRS input.
        """
        center = np.full(3, self.box_size / 2.0, dtype=float)
        __, ell, b = cartesian_icrs_to_galactic_spherical(
            self.centroid.reshape(1, 3), center)
        return float(ell[0]), float(b[0])

    @cached_property
    def centroid_distance(self):
        """Centroid comoving distance from the observer (Mpc/h)."""
        rel = self.centroid - self.box_size / 2.0
        return float(np.linalg.norm(rel))

    @cached_property
    def centroid_obs_redshift(self):
        """Compute observed centroid redshift including peculiar velocity."""
        return np.mean(self.obs_redshift)

    @cached_property
    def centroid_cosmo_redshift(self):
        """Compute centroid cosmological redshift (no peculiar velocity)."""
        cz = comoving_distance_to_cz(
            self.centroid_distance, h=1.0, Om0=self.Om0)
        return cz / SPEED_OF_LIGHT_KMS

    @cached_property
    def redshift_space_centroid(self):
        """
        Redshift space centroid (Mpc/h), computed as the mean of member
        redshift space positions relative to the observer at box center.
        """
        redshift_positions = self.redshift_position
        center = np.full(3, self.box_size / 2.0, dtype=float)
        return np.mean(redshift_positions - center, axis=0)

    def compute_map_signals(self, profiler, theta_rand, map_rand,
                            r_key="Group_R_Crit500"):
        """
        Compute and store map signals and p-values for this association.

        Computes signals at both the median position and for each individual
        halo. Results are stored directly in the association object.

        Parameters
        ----------
        profiler
            PointingEnclosedProfile instance for measuring map signals.
        theta_rand
            Angular sizes for random signal profiles (arcmin).
        map_rand
            Map signal profiles for random pointings, shape
            (n_random, n_theta).
        r_key
            Key in optional_data for halo radii (e.g., 'Group_R_Crit500').
        """
        if r_key not in self.optional_data:
            raise KeyError(
                f"Association {self.label} missing '{r_key}' in "
                "optional_data."
            )

        # Get galactic coordinates and distances for all haloes
        ell_b = self.galactic_angular
        r = self.distance
        ell = ell_b[:, 0]
        b = ell_b[:, 1]
        radii = self.optional_data[r_key]

        # Compute aperture sizes
        theta500 = np.rad2deg(np.arctan(radii / r)) * 60

        # Compute median position and aperture
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
        self.median_theta500 = median_theta500
        self.median_signal = median_signal
        self.median_pval = median_pval
        self.halo_theta500 = theta500
        self.halo_signals = halo_signals
        self.halo_pvals = halo_pvals


class HaloAssociationList(list):
    """
    List-like container for HaloAssociation objects with convenience methods.

    Behaves like a regular list but provides properties for computing
    statistics across all associations.
    """

    @property
    def mean_log_mass(self):
        """Mean log mass for each association."""
        return np.array([np.log10(assoc.masses).mean() for assoc in self])

    @property
    def std_log_mass(self):
        """Standard deviation of log mass for each association."""
        return np.array([np.log10(assoc.masses).std() for assoc in self])

    @property
    def centroid_radec(self):
        """Right ascension and declination centroid pairs in degrees."""
        return np.array([assoc.centroid_radec for assoc in self])

    @property
    def centroid_cosmo_redshift(self):
        """Cosmological redshift of each association centroid."""
        return np.array([assoc.centroid_cosmo_redshift for assoc in self])

    @property
    def centroid_obs_redshift(self):
        """
        Observed redshift of each association centroid (including peculiar
        velocities).
        """
        return np.array([assoc.centroid_obs_redshift for assoc in self])

    @property
    def centroid_cartesian(self):
        """Centroid positions as Cartesian coordinates."""
        return np.array([assoc.centroid for assoc in self])

    @property
    def centroid_distance(self):
        """Centroid comoving distances (Mpc/h)."""
        return np.array([assoc.centroid_distance for assoc in self])

    @property
    def redshift_space_centroid(self):
        """Redshift space centroids relative to observer (Mpc/h)."""
        return np.array([assoc.redshift_space_centroid for assoc in self])

    @property
    def box_size(self):
        """Simulation box size inferred from the first association."""
        if not self:
            raise ValueError("Cannot determine box_size for an empty list.")

        return self[0].box_size

    @property
    def Om0(self):
        """Matter density parameter inferred from the first association."""
        if not self:
            raise ValueError("Cannot determine Om0 for an empty list.")

        return self[0].Om0


def compute_association_signals(associations, profiler, theta_rand, map_rand,
                                r_key="Group_R_Crit500"):
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
    theta_rand
        Angular sizes for random pointings (arcmin).
    map_rand
        Map signals for random pointings, shape (n_random, n_theta).
    r_key
        Key in optional_data for halo radii.
    """
    for assoc in tqdm(associations, desc="Computing association signals"):
        assoc.compute_map_signals(
            profiler, theta_rand, map_rand,
            r_key=r_key,
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
    HaloAssociationList
        List-like container of surviving associations sorted by cluster label.
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
        return HaloAssociationList()

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
        return HaloAssociationList()

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
                optional_data=cluster_opt,
            )
        )

    return HaloAssociationList(associations)


def _infer_mass_definition(mass_key):
    """Infer mass definition (e.g. 500c) from a halo mass key."""
    key = mass_key.lower()
    match = re.search(r"m\s*(\d+)([cm])", key)
    if match:
        if match.group(1) not in ("200", "500"):
            raise ValueError(
                f"Unsupported mass scale '{match.group(1)}' in '{mass_key}'.")
        if match.group(2) != "c":
            raise ValueError("Mass definition must be 'c' (critical), "
                             f"got '{match.group(2)}'.")
        return f"{match.group(1)}c"
    match = re.search(r"(crit|mean)\s*(\d+)", key)
    if match:
        if match.group(2) not in ("200", "500"):
            raise ValueError(f"Unsupported mass scale '{match.group(2)}' "
                             f"in '{mass_key}'.")
        if match.group(1) != "crit":
            raise ValueError(
                f"Mass definition must be critical, got '{match.group(1)}'.")
        return f"{match.group(2)}c"
    raise ValueError(
        f"Cannot infer mass definition from mass key '{mass_key}'. "
        "Expected patterns like 'M200c' or 'Group_M_Crit500'."
    )


def _load_simulation_halos(cfg, sim_key):
    """Return filtered halo data across all realisations."""
    import cmbo

    try:
        catalogue_cfg = cfg["halo_catalogues"][sim_key]
    except KeyError as exc:
        raise ValueError(
            f"Simulation '{sim_key}' not defined in config."
        ) from exc

    fname = catalogue_cfg["fname"]
    position_key = catalogue_cfg["position_key"]
    mass_key = catalogue_cfg["mass_key"]
    radius_key = catalogue_cfg["radius_key"]
    velocity_key = catalogue_cfg.get("velocity_key")
    box_size = float(catalogue_cfg["box_size"])
    omega_m = catalogue_cfg.get("omega_m")
    if omega_m is None:
        omega_m = catalogue_cfg.get("Om0", 0.3111)
    omega_m = float(omega_m)
    centre = np.array(catalogue_cfg.get(
        "observer_position",
        [box_size / 2.0, box_size / 2.0, box_size / 2.0],
    ), dtype=float)
    if centre.shape != (3,):
        raise ValueError("observer_position must be a 3-element sequence.")

    sim_ids = cmbo.io.list_simulations_hdf5(fname)
    if not sim_ids:
        raise ValueError(f"No simulations found in {fname}.")

    mass_floor = cfg["halo_catalogues"].get(
        "min_association_mass", 0.0
    )
    positions_all = []
    masses_all = []
    r500_all = []
    theta_all = []
    selection_all = []
    velocity_all = [] if velocity_key else None
    catalogue_sizes = []
    theta_catalogues = []
    sim_ids_kept = []

    for sim_id in sim_ids:
        reader = cmbo.io.SimulationHaloReader(fname, sim_id)
        pos = np.asarray(reader[position_key], dtype=float)
        mass = np.asarray(reader[mass_key], dtype=float)
        r500 = np.asarray(reader[radius_key], dtype=float)
        if velocity_key:
            velocity = np.asarray(reader[velocity_key], dtype=float)
        indices = np.arange(pos.shape[0], dtype=int)

        r, ell, b = cartesian_icrs_to_galactic_spherical(pos, centre)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(r500, r, out=np.zeros_like(r500), where=r > 0)
            theta_arcmin = np.rad2deg(np.arctan(ratio)) * 60.0

        mask = np.isfinite(mass)
        mask &= mass > 0
        mask &= np.isfinite(r)
        mask &= np.isfinite(theta_arcmin)
        mask &= mass >= mass_floor

        if not np.any(mask):
            continue

        positions_all.append(pos[mask])
        masses_all.append(mass[mask])
        r500_all.append(r500[mask])
        if velocity_key:
            velocity_all.append(velocity[mask])
        theta_masked = theta_arcmin[mask]
        theta_all.append(theta_masked)
        selection_all.append(indices[mask])
        catalogue_sizes.append(pos.shape[0])
        theta_catalogues.append(theta_arcmin)
        sim_ids_kept.append(str(sim_id))

    if not positions_all:
        raise ValueError(
            "No haloes survive the selection cuts across any simulation."
        )

    mass_definition = _infer_mass_definition(mass_key)

    data = {
        "positions": positions_all,
        "masses": masses_all,
        "r500": r500_all,
        "theta_arcmin": theta_all,
        "selection_indices": selection_all,
        "sim_ids": sim_ids_kept,
        "catalogue_sizes": catalogue_sizes,
        "theta_catalogues": theta_catalogues,
        "box_size": box_size,
        "centre": centre,
        "mass_definition": mass_definition,
        "omega_m": omega_m,
    }
    if velocity_key:
        data["velocities"] = velocity_all
        data["velocity_key"] = velocity_key
    return data


def _results_path(cfg, sim_key):
    analysis_cfg = cfg["analysis"]
    output_dir = Path(analysis_cfg.get("output_folder", "."))
    tag = analysis_cfg.get("output_tag")
    stem = sim_key if not tag else f"{sim_key}_{tag}"
    return (output_dir / f"{stem}.hdf5").resolve()


def _load_original_order_dataset(
    results_path,
    sim_ids,
    catalogue_sizes,
    dataset_names,
    required=True,
):
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file '{results_path}' not found. "
            "Run scripts/run_suite.py first."
        )

    if len(sim_ids) != len(catalogue_sizes):
        raise ValueError(
            "sim_ids and catalogue_sizes must have the same length."
        )

    if isinstance(dataset_names, str):
        dataset_names = (dataset_names,)
    elif not dataset_names:
        raise ValueError("dataset_names must not be empty.")

    lookups = {}
    with h5py.File(results_path, "r") as h5:
        for sim_id, n_obj in zip(sim_ids, catalogue_sizes):
            sim_id = str(sim_id)
            if sim_id not in h5:
                raise KeyError(
                    f"Simulation group '{sim_id}' missing from {results_path}."
                )
            halos = h5[sim_id]["halos"]
            try:
                catalogue_idx = np.asarray(
                    halos["catalogue_index_original"][...],
                    dtype=int,
                )
            except KeyError as exc:
                raise KeyError(
                    "Result file missing 'catalogue_index_original' dataset."
                ) from exc

            dataset = None
            dataset_name = None
            for candidate in dataset_names:
                if candidate in halos:
                    dataset = np.asarray(halos[candidate][...], dtype=float)
                    dataset_name = candidate
                    break
            if dataset is None:
                if required:
                    raise KeyError(
                        f"Dataset(s) {dataset_names} missing for simulation "
                        f"{sim_id} in {results_path}."
                    )
                continue

            if catalogue_idx.shape != dataset.shape:
                raise ValueError(
                    f"Dataset '{dataset_name}' mis-sized for "
                    f"simulation {sim_id}."
                )

            if np.any(catalogue_idx < 0) or np.any(catalogue_idx >= n_obj):
                raise ValueError(
                    f"Catalogue indices out of bounds for simulation {sim_id}."
                )

            array = np.full(n_obj, np.nan, dtype=float)
            array[catalogue_idx] = dataset
            lookups[sim_id] = array

    return lookups


def _load_pval_lookup(results_path, sim_ids, catalogue_sizes):
    return _load_original_order_dataset(
        results_path,
        sim_ids,
        catalogue_sizes,
        dataset_names="pval_original_order",
        required=True,
    )


def _load_signal_lookup(results_path, sim_ids, catalogue_sizes):
    return _load_original_order_dataset(
        results_path,
        sim_ids,
        catalogue_sizes,
        dataset_names=(
            "signal_original_order",
            "halo_signal_original",
        ),
        required=False,
    )


def _attach_per_halo_data(
    associations,
    selection_indices,
    sim_ids,
    pval_lookup,
    signal_lookup,
    theta_lookup,
):
    for assoc in associations:
        n_members = assoc.member_indices.shape[0]
        member_pvals = np.full(n_members, np.nan, dtype=float)
        member_signals = (
            np.full(n_members, np.nan, dtype=float)
            if signal_lookup
            else None
        )
        member_theta = (
            np.full(n_members, np.nan, dtype=float)
            if theta_lookup
            else None
        )
        for i, (real_idx, local_idx) in enumerate(assoc.member_indices):
            try:
                per_real_indices = selection_indices[real_idx]
            except IndexError as exc:
                raise IndexError(
                    "Association references unknown realisation index "
                    f"{real_idx}."
                ) from exc
            if local_idx >= per_real_indices.size:
                raise IndexError(
                    f"Local halo index {local_idx} out of range for "
                    f"realisation {real_idx}."
                )
            catalogue_idx = int(per_real_indices[local_idx])
            sim_id = sim_ids[real_idx]
            per_sim_lookup = pval_lookup.get(sim_id)
            if per_sim_lookup is None:
                continue
            if catalogue_idx < per_sim_lookup.size:
                member_pvals[i] = per_sim_lookup[catalogue_idx]
            if member_signals is not None:
                per_sim_signals = signal_lookup.get(sim_id)
                if (
                    per_sim_signals is not None
                    and catalogue_idx < per_sim_signals.size
                ):
                    member_signals[i] = per_sim_signals[catalogue_idx]
            if member_theta is not None:
                per_sim_theta = theta_lookup.get(sim_id)
                if (
                    per_sim_theta is not None
                    and catalogue_idx < per_sim_theta.size
                ):
                    member_theta[i] = per_sim_theta[catalogue_idx]
        assoc.halo_pvals = member_pvals
        if member_pvals.size and np.any(np.isfinite(member_pvals)):
            assoc.median_pval = float(np.nanmedian(member_pvals))
        else:
            assoc.median_pval = np.nan
        if member_signals is not None:
            assoc.halo_signals = member_signals
            if member_signals.size and np.any(np.isfinite(member_signals)):
                assoc.median_signal = float(np.nanmedian(member_signals))
            else:
                assoc.median_signal = np.nan
        if member_theta is not None:
            assoc.halo_theta500 = member_theta
            if member_theta.size and np.any(np.isfinite(member_theta)):
                assoc.median_theta500 = float(np.nanmedian(member_theta))
            else:
                assoc.median_theta500 = np.nan


def load_associations(sim_key, cfg, verbose=True):
    """
    Identify halo associations and attach per-halo data.

    Parameters
    ----------
    sim_key
        Simulation key from the config's halo_catalogues section.
    cfg
        CMBO configuration dictionary.
    verbose
        Print progress messages.

    Returns
    -------
    HaloAssociationList
        List of associations with attached p-values and signals.
    """
    halo_data = _load_simulation_halos(cfg, sim_key)
    if verbose:
        print(f"Loaded {len(halo_data['positions'])} simulation realisations.")
    catalogue_cfg = cfg["halo_catalogues"][sim_key]
    radius_key = catalogue_cfg["radius_key"]
    velocity_key = catalogue_cfg.get("velocity_key")
    optional = {
        radius_key: halo_data["r500"],
        "theta500_arcmin": halo_data["theta_arcmin"],
    }
    if velocity_key and "velocities" in halo_data:
        optional[velocity_key] = halo_data["velocities"]
    associations = identify_halo_associations(
        halo_data["positions"],
        halo_data["masses"],
        optional_data=optional,
    )
    frac_thresh = cfg["halo_catalogues"].get(
        "min_fraction_present", 0.0
    )
    if frac_thresh > 0.0 and associations:
        associations = HaloAssociationList([
            assoc for assoc in associations
            if assoc.fraction_present >= frac_thresh
        ])
    if verbose:
        print(f"Identified {len(associations)} halo associations.")
    results_path = _results_path(cfg, sim_key)
    pval_lookup = _load_pval_lookup(
        results_path,
        halo_data["sim_ids"],
        halo_data["catalogue_sizes"],
    )
    signal_lookup = _load_signal_lookup(
        results_path,
        halo_data["sim_ids"],
        halo_data["catalogue_sizes"],
    )
    if signal_lookup:
        if verbose:
            print("Loaded halo signal lookup tables from run_suite output.")
    elif verbose:
        print("Halo signal datasets were not found in run_suite output.")
    theta_lookup = {
        sim_id: np.asarray(theta, dtype=float)
        for sim_id, theta in zip(
            halo_data["sim_ids"], halo_data["theta_catalogues"]
        )
    }
    _attach_per_halo_data(
        associations,
        halo_data["selection_indices"],
        halo_data["sim_ids"],
        pval_lookup,
        signal_lookup,
        theta_lookup,
    )
    for assoc in associations:

        assoc.optional_data.setdefault("box_size", halo_data["box_size"])
        assoc.optional_data.setdefault(
            "mass_definition", halo_data["mass_definition"]
        )
        assoc.optional_data.setdefault("omega_m", halo_data["omega_m"])

        if velocity_key and velocity_key in assoc.optional_data:
            assoc.velocities = assoc.optional_data[velocity_key]

    return associations
