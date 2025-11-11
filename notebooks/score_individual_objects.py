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
"""
Helper for scoring observed clusters against digital-twin haloes.

Given a simulation key and a loaded CMBO configuration dictionary, this module
reads the halo catalogue, constructs associations, matches them to the observed
cluster catalogue, and attaches the per-halo tSZ ``p``-values measured by
``scripts/run_suite.py``. The function returns the associations (with
p-values), matching assignments, and simulation box size.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import cmbo
from cmbo.utils import (
    identify_halo_associations,
    compute_matching_matrix,
    greedy_global_matching,
    cartesian_icrs_to_galactic_spherical,
)


def _load_simulation_halos(cfg, sim_key):
    """Return filtered halo data across all realisations."""
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
    box_size = float(catalogue_cfg["box_size"])
    centre = np.full(3, box_size / 2.0, dtype=float)

    sim_ids = cmbo.io.list_simulations_hdf5(fname)
    if not sim_ids:
        raise ValueError(f"No simulations found in {fname}.")

    mass_floor = cfg["halo_catalogues"].get(
        "min_association_mass", 0.0
    )
    positions_all = []
    masses_all = []
    r500_all = []
    selection_all = []
    sim_ids_kept = []

    for sim_id in sim_ids:
        reader = cmbo.io.SimulationHaloReader(fname, sim_id)
        pos = np.asarray(reader[position_key], dtype=float)
        mass = np.asarray(reader[mass_key], dtype=float)
        r500 = np.asarray(reader[radius_key], dtype=float)
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
        selection_all.append(indices[mask])
        sim_ids_kept.append(str(sim_id))

    if not positions_all:
        raise ValueError(
            "No haloes survive the selection cuts across any simulation."
        )

    data = {
        "positions": positions_all,
        "masses": masses_all,
        "r500": r500_all,
        "selection_indices": selection_all,
        "sim_ids": sim_ids_kept,
        "box_size": box_size,
    }
    return data


def _results_path(cfg, sim_key):
    analysis_cfg = cfg["analysis"]
    output_dir = Path(analysis_cfg.get("output_folder", "."))
    tag = analysis_cfg.get("output_tag")
    stem = sim_key if not tag else f"{sim_key}_{tag}"
    return (output_dir / f"{stem}.hdf5").resolve()


def _load_pval_lookup(results_path, sim_ids):
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file '{results_path}' not found. "
            "Run scripts/run_suite.py first."
        )

    lookups = {}
    with h5py.File(results_path, "r") as h5:
        for sim_id in sim_ids:
            if str(sim_id) not in h5:
                raise KeyError(
                    f"Simulation group '{sim_id}' missing from {results_path}."
                )
            halos = h5[str(sim_id)]["halos"]
            try:
                catalogue_idx = halos["catalogue_index_original"][...]
                pvals = halos["pval_original_order"][...]
            except KeyError as exc:
                raise KeyError(
                    "Result file missing per-halo index or p-value datasets."
                ) from exc
            lookup = {
                int(idx): float(pval)
                for idx, pval in zip(catalogue_idx, pvals)
            }
            lookups[str(sim_id)] = lookup

    return lookups


def _attach_pvals_to_associations(
    associations,
    selection_indices,
    sim_ids,
    pval_lookup,
):
    for assoc in associations:
        n_members = assoc.member_indices.shape[0]
        member_pvals = np.full(n_members, np.nan, dtype=float)
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
            try:
                member_pvals[i] = pval_lookup[sim_id][catalogue_idx]
            except KeyError as exc:
                raise KeyError(
                    f"No p-value found for halo index {catalogue_idx} "
                    f"in simulation {sim_id}."
                ) from exc
        assoc.halo_pvals = member_pvals
        assoc.median_pval = (
            float(np.nanmedian(member_pvals))
            if member_pvals.size
            else np.nan
        )


def load_associations_and_matches(sim_key, cfg, verbose=True):
    """
    Return halo associations, matches, and box size with per-halo tSZ p-values.
    """
    halo_data = _load_simulation_halos(cfg, sim_key)
    if verbose:
        print(f"Loaded {len(halo_data['positions'])} simulation realisations.")
    radius_key = cfg["halo_catalogues"][sim_key]["radius_key"]
    optional = {radius_key: halo_data["r500"]}

    associations = identify_halo_associations(
        halo_data["positions"],
        halo_data["masses"],
        optional_data=optional,
    )
    if verbose:
        print(f"Identified {len(associations)} halo associations.")
    if not associations:
        raise ValueError(
            "No halo associations were found for the selected simulation."
        )

    obs_clusters = cmbo.io.load_observed_clusters(
        cfg["paths"]["observed_clusters"]
    )
    if verbose:
        print(f"Loaded {len(obs_clusters)} observed clusters.")
    pval_matrix, dist_matrix = compute_matching_matrix(
        obs_clusters,
        associations,
        halo_data["box_size"],
    )
    matches = greedy_global_matching(
        pval_matrix,
        dist_matrix,
        obs_clusters,
        associations,
        0.05,
    )

    results_path = _results_path(cfg, sim_key)
    pval_lookup = _load_pval_lookup(results_path, halo_data["sim_ids"])
    _attach_pvals_to_associations(
        associations,
        halo_data["selection_indices"],
        halo_data["sim_ids"],
        pval_lookup,
    )

    return associations, matches, halo_data["box_size"]
