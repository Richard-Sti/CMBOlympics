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

from collections.abc import Sequence

import cmbo
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from cmbo.match.cluster_matching import (compute_matching_matrix_obs,
                                         greedy_global_matching)
from matplotlib.lines import Line2D
from scipy.stats import combine_pvalues

plt.style.use("science")


def attach_associations_to_obs_clusters(
    obs_clusters, associations, cfg, verbose=True
):
    """Match associations to observed clusters via greedy matching."""
    if obs_clusters is None:
        raise ValueError("obs_clusters must be provided.")
    if verbose:
        print(f"Using {len(obs_clusters)} observed clusters.")
    if not associations:
        raise ValueError("No associations provided.")
    pval_matrix, dist_matrix = compute_matching_matrix_obs(
        obs_clusters,
        associations,
        box_size=None,
    )
    matches = greedy_global_matching(
        pval_matrix,
        dist_matrix,
        associations,
        obs_clusters=obs_clusters,
        threshold=cfg["analysis"].get("matching_pvalue_threshold", 0.05),
    )
    return matches


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
                    ell_deg, b_deg = assoc.centroid_galactic_angular
                    centroid_dist = float(assoc.centroid_distance)
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
            f"{match_display:>12.1e}" if match_display is not None else " " * 12  # noqa
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


def plot_cluster_pvalue_percentiles(
    cfg,
    matches,
    obs_clusters=None,
    sim_key=None,
    observer_centre=None,
    ax=None,
    suite_labels=None,
    suite_colors=None,
):
    """
    Plot per-cluster percentile summaries of per-halo tSZ p-values.

    Parameters
    ----------
    suite_colors
        Optional sequence of Matplotlib color specs, one per simulation suite.
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
    if not names:
        raise ValueError("No observed clusters available to plot.")

    if observer_centre is None:
        observer_centre = _observer_centre_from_cfg(cfg, sim_key=sim_key)
    observer_centre = np.asarray(observer_centre, dtype=float)
    if observer_centre.shape != (3,):
        raise ValueError("observer_centre must be a 3-vector.")

    def _is_match_entry(entry):
        if entry is None:
            return True
        if isinstance(entry, tuple) and entry:
            entry = entry[0]
        return hasattr(entry, "halo_pvals")

    def _normalise_matches(match_input):
        if not isinstance(match_input, Sequence) or isinstance(
            match_input, (str, bytes)
        ):
            raise ValueError("matches must be a sequence.")
        if not match_input:
            raise ValueError("matches must not be empty.")

        first = match_input[0]
        if _is_match_entry(first):
            if len(match_input) != len(names):
                raise ValueError(
                    "Single-suite matches length does not match observed "
                    "clusters."
                )
            return [match_input]

        def _is_suite(candidate):
            if not isinstance(candidate, Sequence) or isinstance(
                candidate, (str, bytes)
            ):
                return False
            if len(candidate) != len(names):
                return False
            probe = next((
                item for item in candidate if item is not None), None)
            if probe is None:
                return True
            return _is_match_entry(probe)

        if not _is_suite(first):
            raise ValueError(
                "Could not interpret matches input. Provide either a single "
                "match list aligned with the observed clusters or a sequence "
                "of such lists (one per simulation suite)."
            )

        suites = []
        for suite_idx, suite in enumerate(match_input):
            if not _is_suite(suite):
                raise ValueError(
                    f"Suite index {suite_idx} does not align with clusters."
                )
            suites.append(suite)
        return suites

    matches_by_suite = _normalise_matches(matches)
    num_suites = len(matches_by_suite)
    if suite_labels is not None:
        if len(suite_labels) != num_suites:
            raise ValueError(
                "suite_labels must match the number of match collections."
            )
    else:
        suite_labels = [f"Suite {idx + 1}" for idx in range(num_suites)]
    if suite_colors is not None:
        if not isinstance(suite_colors, Sequence) or isinstance(
            suite_colors, (str, bytes)
        ):
            raise ValueError(
                "suite_colors must be a sequence of Matplotlib color specs."
            )
        if len(suite_colors) < num_suites:
            raise ValueError(
                "suite_colors sequence must provide at least one entry per suite."  # noqa
            )
        suite_colors = list(suite_colors)

    percentile_levels = (5, 50, 95)
    stats_payloads = []
    for suite in matches_by_suite:
        positions = []
        p05 = []
        p50 = []
        p95 = []
        for idx, entry in enumerate(suite, start=1):
            if entry is None:
                continue
            assoc = entry[0]
            per_halo = np.asarray(
                getattr(assoc, "halo_pvals", []), dtype=float)
            finite = per_halo[np.isfinite(per_halo)]
            if finite.size == 0:
                continue
            positions.append(idx)
            q05, q50, q95 = np.percentile(finite, percentile_levels)
            p05.append(float(q05))
            p50.append(float(q50))
            p95.append(float(q95))
        stats_payloads.append(
            {
                "positions": np.asarray(positions, dtype=float),
                "p05": np.asarray(p05, dtype=float),
                "p50": np.asarray(p50, dtype=float),
                "p95": np.asarray(p95, dtype=float),
            }
        )

    with plt.style.context("science"):
        if ax is None:
            width = max(8.0, 0.7 * len(names))
            fig, ax = plt.subplots(figsize=(width, 4.0))
        else:
            fig = ax.figure

        positions = np.arange(1, len(names) + 1)
        offsets = np.zeros(num_suites)
        if num_suites > 1:
            offsets = np.linspace(-0.35, 0.35, num_suites)
        marker_size = 25
        if suite_colors is not None:
            colors = suite_colors
        else:
            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            default_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
            if prop_cycle is not None:
                colors = prop_cycle.by_key().get("color", default_colors)
            else:
                colors = default_colors

        legend_handles = []
        for suite_idx, payload in enumerate(stats_payloads):
            pos = payload["positions"]
            if pos.size == 0:
                continue
            color = colors[suite_idx % len(colors)]
            suite_positions = pos + offsets[suite_idx]
            medians = payload["p50"]
            yerr = np.vstack(
                [
                    medians - payload["p05"],
                    payload["p95"] - medians,
                ]
            )
            ax.errorbar(
                suite_positions,
                medians,
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                markersize=np.sqrt(marker_size),
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    label=suite_labels[suite_idx],
                )
            )

        ax.set_xticks(positions, names)
        ax.tick_params(axis="x", which="both", length=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation(50)
            label.set_rotation_mode("anchor")
        ax.set_xlim(0.5, len(names) + 0.5)
        boundaries = np.arange(1.5, len(names) + 0.5, 1.0)
        for xpos in boundaries:
            ax.axvline(
                xpos,
                color="grey",
                linestyle="-",
                linewidth=1.1,
                alpha=0.7,
                zorder=0,
            )
        ax.set_ylabel(r"$p_{\mathrm{tSZ}}$")
        if legend_handles:
            legend = ax.legend(
                handles=legend_handles,
                frameon=True,
                framealpha=1.0,
                facecolor="white",
                edgecolor="none",
                loc="upper right",
            )
            legend.get_frame().set_linewidth(0.0)
        ax.set_yscale("log")
        y_min, y_max = ax.get_ylim()
        if y_max > 1.0:
            ax.set_ylim(y_min, 1.0)
        ax.grid(False)
        for thresh in (0.05, 0.005):
            ax.axhline(
                thresh,
                color="red",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                zorder=0,
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
