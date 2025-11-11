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

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues

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
    catalogue_sizes = []
    theta_catalogues = []
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
    }
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
                    f"Dataset '{dataset_name}' mis-sized for simulation {sim_id}."  # noqa
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


def load_associations_and_matches(sim_key, cfg, verbose=True):
    """
    Return halo associations, matches, and box size with per-halo tSZ p-values.
    """
    halo_data = _load_simulation_halos(cfg, sim_key)
    if verbose:
        print(f"Loaded {len(halo_data['positions'])} simulation realisations.")
    radius_key = cfg["halo_catalogues"][sim_key]["radius_key"]
    optional = {
        radius_key: halo_data["r500"],
        "theta500_arcmin": halo_data["theta_arcmin"],
    }

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
        cfg["analysis"].get("matching_pvalue_threshold", 0.05),
    )

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
        print(
            "Halo signal datasets were not found in the run_suite output; "
            "p-values only."
        )
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

    # _annotate_theta_and_signals(
    #     associations,
    #     halo_data["centre"],
    # )

    return associations, matches, halo_data["box_size"]


def _observer_centre_from_cfg(cfg, sim_key=None):
    """
    Return observer centre vector for the requested simulation.
    """
    if sim_key is None:
        sim_key = cfg.get("analysis", {}).get("which_simulation")
        if sim_key is None:
            raise ValueError(
                "sim_key must be provided when analysis.which_simulation "
                "is not set in the config."
            )
    try:
        catalogue_cfg = cfg["halo_catalogues"][sim_key]
    except KeyError as exc:
        raise ValueError(
            f"Simulation '{sim_key}' not defined in cfg['halo_catalogues']."
        ) from exc

    box_size = float(catalogue_cfg["box_size"])
    centre = np.array(
        catalogue_cfg.get(
            "observer_position",
            [box_size / 2.0, box_size / 2.0, box_size / 2.0],
        ),
        dtype=float,
    )
    if centre.shape != (3,):
        raise ValueError("observer_position must contain 3 elements.")
    return centre


def print_cluster_scores(
    cfg,
    matches,
    obs_clusters=None,
    sim_key=None,
    observer_centre=None,
    default_pval=0.5,
    percentiles=(5, 50, 95),
):
    """
    Print per-cluster scoring summary as described in the README discipline.

    Parameters
    ----------
    cfg
        CMBO configuration dictionary. Used to reload the observed clusters
        if ``obs_clusters`` is not provided.
    matches
        Output list from ``greedy_global_matching`` where each element is
        either ``None`` (no match) or ``(association, pval, distance)``.
    obs_clusters
        Optional ``ObservedClusterCatalogue`` aligned with ``matches``.
        If omitted, the catalogue is loaded from ``cfg["paths"]``.
    sim_key
        Simulation key whose observer position should be used when converting
        association centroids to Galactic coordinates. Falls back to
        ``cfg['analysis']['which_simulation']``.
    observer_centre
        Optional override for the observer position (3-vector in Mpc/h).
    default_pval : float, optional
        Per-cluster p-value assigned when no association was matched.
    percentiles : sequence, optional
        Percentiles (0-100) of the per-halo p-value distribution to report.
    Returns
    -------
    list of dict
        Summary rows per cluster, including combined Stouffer p-values even
        though they are not displayed.
    """
    if obs_clusters is None:
        try:
            cluster_path = cfg["paths"]["observed_clusters"]
        except KeyError as exc:
            raise ValueError(
                "cfg missing 'paths.observed_clusters' and obs_clusters "
                "was not provided."
            ) from exc
        obs_clusters = cmbo.io.load_observed_clusters(cluster_path)

    names = getattr(obs_clusters, "names", None)
    if names is None:
        raise ValueError(
            "obs_clusters must expose a 'names' attribute for reporting."
        )
    if len(matches) != len(names):
        raise ValueError(
            "Number of matches does not align with observed clusters."
        )

    if observer_centre is None:
        observer_centre = _observer_centre_from_cfg(cfg, sim_key=sim_key)
    observer_centre = np.asarray(observer_centre, dtype=float)
    if observer_centre.shape != (3,):
        raise ValueError("observer_centre must have shape (3,).")

    percentiles = tuple(percentiles)
    rows = []
    perc_header = " ".join(
        f"P{int(p):02d}%".rjust(8) for p in percentiles
    ) if percentiles else ""
    base_header = (
        f"{'Cluster':<22} {'Assoc':>7} {'Frac':>6} "
        f"{'logM [Msun/h]':>14} {'Pfeifer pval':>12} "
        f"{'Dist [Mpc/h]':>13} {'ell [deg]':>10} {'b [deg]':>10}"
    )
    tsz_block = f"{'Frac p<0.05':>14}"
    if perc_header:
        tsz_block = f"{tsz_block} {perc_header}"
    if tsz_block:
        label_line = (
            " " * (len(base_header) + 1)
            + "tSZ significance".center(len(tsz_block))
        )
        print(label_line)
    header = f"{base_header} {tsz_block}"
    print(header)
    print("-" * len(header))

    tsz_cluster_pvals = []

    for idx, name in enumerate(names):
        entry = matches[idx]
        assoc_label = "-"
        frac_present = np.nan
        median_logm = np.nan
        match_p = np.nan
        combined = np.nan
        centroid_dist = np.nan
        ell_deg = np.nan
        b_deg = np.nan
        perc_vals = np.full(len(percentiles), np.nan)
        frac_low_p = np.nan
        median_tsz = np.nan

        if entry is not None:
            assoc, match_p, _ = entry
            assoc_label = getattr(assoc, "label", "NA")
            frac_present = float(
                getattr(assoc, "fraction_present", np.nan)
            )
            median_tsz = float(getattr(assoc, "median_pval", np.nan))
            masses = np.asarray(getattr(assoc, "masses", []), dtype=float)
            masses = masses[np.isfinite(masses) & (masses > 0)]
            if masses.size:
                median_logm = float(np.nanmedian(np.log10(masses)))
            if observer_centre is not None:
                try:
                    r_sim, ell_deg, b_deg = assoc.centroid_to_galactic_angular(
                        observer_centre
                    )
                    centroid_dist = float(r_sim)
                except Exception:
                    ell_deg = np.nan
                    b_deg = np.nan
                    centroid_dist = np.nan
            per_halo = np.asarray(
                getattr(assoc, "halo_pvals", []), dtype=float)
            finite = per_halo[np.isfinite(per_halo)]
            if finite.size:
                frac_low_p = float(
                    np.count_nonzero(finite < 0.05) / finite.size)
                _, combined_val = combine_pvalues(
                    finite,
                    method="stouffer",
                    nan_policy="omit",
                )
                combined = float(combined_val)
                if percentiles:
                    perc_vals = np.percentile(finite, percentiles)
            else:
                combined = np.nan

        if entry is None:
            match_display = None
        else:
            match_display = match_p if np.isfinite(match_p) else 1.0
        match_display_str = (
            f"{match_display:>12.1e}" if match_display is not None else " " * 12
        )

        row = (
            f"{name:<22} "
            f"{str(assoc_label):>7} "
            f"{frac_present:>6.2f} "
            f"{median_logm:>14.2f} "
            f"{match_display_str} "
            f"{centroid_dist:>13.3f} "
            f"{ell_deg:>10.2f} "
            f"{b_deg:>10.2f} "
            f"{frac_low_p:>14.2%}"
        )
        if percentiles:
            perc_str = " ".join(f"{val:>8.1e}" for val in perc_vals)
            row = f"{row} {perc_str}"
        print(row)
        rows.append(
            {
                "name": name,
                "association_label": assoc_label,
                "fraction_present": frac_present,
                "median_log_mass": median_logm,
                "match_p": match_p,
                "combined_p": combined,
                "distance_mpc_h": centroid_dist,
                "ell_deg": ell_deg,
                "b_deg": b_deg,
                "frac_low_p": frac_low_p,
                "median_tsz_pval": median_tsz,
                "percentiles": perc_vals.copy(),
            }
        )
        tsz_cluster_pvals.append(
            median_tsz if np.isfinite(median_tsz) else default_pval
        )

    tsz_cluster_pvals = np.asarray(tsz_cluster_pvals, dtype=float)
    if tsz_cluster_pvals.size:
        methods = (
            "fisher",
            "pearson",
            "tippett",
            "stouffer",
            "mudholkar_george",
        )
        print("\nCombined tSZ p-values across clusters:")
        for method in methods:
            _, combined_val = combine_pvalues(
                tsz_cluster_pvals,
                method=method,
            )
            print(f"  - {method:<17}: {combined_val: .3e}")


def plot_cluster_pvalue_violins(
    cfg,
    matches,
    obs_clusters=None,
    default_pval=0.5,
    sim_key=None,
    observer_centre=None,
    ax=None,
):
    """
    Plot per-cluster distributions of per-halo tSZ p-values as violins.
    """
    if obs_clusters is None:
        try:
            cluster_path = cfg["paths"]["observed_clusters"]
        except KeyError as exc:
            raise ValueError(
                "cfg missing 'paths.observed_clusters' and obs_clusters "
                "was not provided."
            ) from exc
        obs_clusters = cmbo.io.load_observed_clusters(cluster_path)

    names = getattr(obs_clusters, "names", None)
    if names is None:
        raise ValueError("obs_clusters must expose 'names'.")
    if len(matches) != len(names):
        raise ValueError("matches and observed clusters lengths differ.")
    if not names:
        raise ValueError("No observed clusters available to plot.")

    if observer_centre is None:
        observer_centre = _observer_centre_from_cfg(cfg, sim_key=sim_key)
    observer_centre = np.asarray(observer_centre, dtype=float)
    if observer_centre.shape != (3,):
        raise ValueError("observer_centre must be a 3-vector.")

    violin_data = []
    violin_positions = []
    medians = []
    median_positions = []
    blank_positions = []
    for idx, entry in enumerate(matches, start=1):
        if entry is None:
            blank_positions.append(idx)
            continue
        assoc = entry[0]
        per_halo = np.asarray(getattr(assoc, "halo_pvals", []), dtype=float)
        finite = per_halo[np.isfinite(per_halo)]
        if finite.size == 0:
            blank_positions.append(idx)
            continue
        violin_data.append(finite)
        violin_positions.append(idx)
        medians.append(float(np.median(finite)))
        median_positions.append(idx)

    with plt.style.context("science"):
        if ax is None:
            width = max(6.0, 0.5 * len(names))
            fig, ax = plt.subplots(figsize=(width, 4.0))
        else:
            fig = ax.figure

        positions = np.arange(1, len(names) + 1)
        if violin_data:
            vparts = ax.violinplot(
                violin_data,
                positions=violin_positions,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            face_color = "#4C78A8"
            for body in vparts["bodies"]:
                body.set_facecolor(face_color)
                body.set_edgecolor("black")
                body.set_alpha(0.7)
            if "cmedians" in vparts:
                vparts["cmedians"].set_color("black")
                vparts["cmedians"].set_linewidth(1.2)

        if medians:
            ax.scatter(
                median_positions,
                medians,
                color="black",
                s=15,
                zorder=3,
            )

        ax.set_xticks(positions, names, rotation=45, ha="right")
        ax.set_ylabel(r"$p_{\mathrm{tSZ}}$")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        for thresh in (0.05, 0.005):
            ax.axhline(
                thresh,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.4,
            )

    return fig, ax


def plot_pfeifer_vs_tsz(
    matches,
    default_pfeifer=1.0,
    default_tsz=0.5,
    ax=None,
):
    """
    Plot correlation between Pfeifer matching p-values and median tSZ p-values.
    """
    pfeifer_vals = []
    tsz_vals = []
    labels = []
    for idx, entry in enumerate(matches):
        if entry is None:
            pfeifer_vals.append(default_pfeifer)
            tsz_vals.append(default_tsz)
            labels.append(idx)
            continue
        assoc, match_p, _ = entry
        pfeifer_vals.append(
            match_p if np.isfinite(match_p) else default_pfeifer
        )
        tsz_vals.append(
            float(getattr(assoc, "median_pval", default_tsz))
            if np.isfinite(getattr(assoc, "median_pval", np.nan))
            else default_tsz
        )
        labels.append(idx)

    pfeifer_vals = np.clip(np.asarray(pfeifer_vals, dtype=float), 1e-6, 1.0)
    tsz_vals = np.clip(np.asarray(tsz_vals, dtype=float), 1e-6, 1.0)

    with plt.style.context("science"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.0, 4.0))
        else:
            fig = ax.figure

        ax.scatter(pfeifer_vals, tsz_vals, c="tab:blue", s=30, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\mathrm{Pfeifer}}$")
        ax.set_ylabel(r"$p_{\mathrm{tSZ,median}}$")
        ax.grid(False)
        x_min, x_max = ax.get_xlim()
        line_x = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
        ax.plot(
            line_x,
            line_x,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )

    return fig, ax
