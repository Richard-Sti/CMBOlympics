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
from astropy import units as u
from astropy.coordinates import SkyCoord
from cmbo.constants import SPEED_OF_LIGHT_KMS
from cmbo.match.cluster_matching import (compute_matching_matrix_obs,
                                         greedy_global_matching,
                                         )
from cmbo.utils.coords import heliocentric_to_cmb
from matplotlib.lines import Line2D
from scipy.stats import combine_pvalues

plt.style.use("science")

SIM_LABEL_NAMES = {
    "csiborg2": r"$\texttt{CB2}$",
    "CSiBORG2": r"$\texttt{CB2}$",
    "manticore": r"$\texttt{CBM}$",
    "Manticore": r"$\texttt{CBM}$",
    "Manticore-Local": r"$\texttt{CBM}$",
}


def match_obs_clusters_to_catalogue(
    obs_clusters,
    catalogue,
    ra_key="RA",
    dec_key="DEC",
    z_key="BEST_Z",
    mass_key=None,
    max_sep_arcmin=15.0,
    max_cz_diff_kms=500.0,
    convert_helio_to_cmb=False,
    prefer_massive=False,
):
    """
    Match observed clusters to an external catalogue (e.g. eRASS, Planck).

    Parameters
    ----------
    obs_clusters : ObservedClusterCatalogue
        Observed cluster catalogue.
    catalogue : dict or structured array
        External catalogue with RA, Dec, and redshift columns.
    ra_key, dec_key, z_key : str
        Column names for RA (deg), Dec (deg), and redshift in the catalogue.
    mass_key : str, optional
        Column name for mass. Required if prefer_massive=True.
    max_sep_arcmin : float
        Maximum angular separation in arcminutes.
    max_cz_diff_kms : float
        Maximum cz difference in km/s.
    convert_helio_to_cmb : bool
        If True, convert catalogue redshifts from heliocentric to CMB frame.
    prefer_massive : bool
        If True, prefer the most massive match instead of the closest.

    Returns
    -------
    matches : list
        For each observed cluster, the index into `catalogue` of the best
        match, or None if no match found.
    ang_sep : ndarray
        Angular separation in arcminutes for each match.
    delta_cz : ndarray
        cz difference in km/s for each match.
    """
    if prefer_massive and mass_key is None:
        raise ValueError("mass_key must be provided when prefer_massive=True")

    obs_ra = np.array([c.ra_deg for c in obs_clusters], dtype=float)
    obs_dec = np.array([c.dec_deg for c in obs_clusters], dtype=float)
    obs_cz = np.array([
        c.cz_cmb if c.cz_cmb is not None else np.nan
        for c in obs_clusters
    ], dtype=float)

    cat_ra = np.asarray(catalogue[ra_key], dtype=float)
    cat_dec = np.asarray(catalogue[dec_key], dtype=float)
    cat_z = np.asarray(catalogue[z_key], dtype=float)
    if convert_helio_to_cmb:
        cat_z = heliocentric_to_cmb(cat_z, cat_ra, cat_dec)
    cat_cz = cat_z * SPEED_OF_LIGHT_KMS

    obs_coord = SkyCoord(obs_ra * u.deg, obs_dec * u.deg)
    cat_coord = SkyCoord(cat_ra * u.deg, cat_dec * u.deg)

    idx_cat, idx_obs, sep2d, _ = obs_coord.search_around_sky(
        cat_coord, max_sep_arcmin * u.arcmin
    )

    n_obs = len(obs_clusters)
    matches = [None] * n_obs
    ang_sep = np.full(n_obs, np.nan, dtype=float)
    delta_cz = np.full(n_obs, np.nan, dtype=float)

    if idx_obs.size:
        ocz = obs_cz[idx_obs]
        ccz = cat_cz[idx_cat]
        valid = (
            np.isfinite(ocz)
            & np.isfinite(ccz)
            & (ocz > 0.0)
            & (ccz > 0.0)
            & (np.abs(ocz - ccz) <= max_cz_diff_kms)
        )

        idx_obs_valid = idx_obs[valid]
        idx_cat_valid = idx_cat[valid]
        sep2d_valid = sep2d[valid]
        dcz_valid = (ocz - ccz)[valid]

        if prefer_massive:
            # Loop over catalogue entries by decreasing mass
            cat_masses = np.asarray(catalogue[mass_key], dtype=float)
            matched_obs = set()
            for c_idx in np.argsort(cat_masses)[::-1]:
                # Find observed clusters matching this catalogue entry
                mask = idx_cat_valid == c_idx
                if not np.any(mask):
                    continue
                candidates_o = idx_obs_valid[mask]
                candidates_sep = sep2d_valid[mask]
                candidates_dcz = dcz_valid[mask]
                # Pick closest unmatched observed cluster
                seps_arcmin = [s.to_value(u.arcmin) for s in candidates_sep]
                for j in np.argsort(seps_arcmin):
                    o_idx = candidates_o[j]
                    if o_idx not in matched_obs:
                        matched_obs.add(o_idx)
                        matches[o_idx] = int(c_idx)
                        sep_val = candidates_sep[j].to_value(u.arcmin)
                        ang_sep[o_idx] = float(sep_val)
                        delta_cz[o_idx] = float(candidates_dcz[j])
                        break
        else:
            # Default: pick closest match for each observed cluster
            for o_idx, c_idx, sep, dz in zip(
                idx_obs_valid, idx_cat_valid, sep2d_valid, dcz_valid
            ):
                sep_arcmin = float(sep.to_value(u.arcmin))
                is_closer = sep_arcmin < ang_sep[o_idx]
                if not np.isfinite(ang_sep[o_idx]) or is_closer:
                    ang_sep[o_idx] = sep_arcmin
                    delta_cz[o_idx] = float(dz)
                    matches[o_idx] = int(c_idx)

    return matches, ang_sep, delta_cz


def print_obs_cluster_catalogue_matches(
    obs_clusters,
    catalogue,
    ra_key="RA",
    dec_key="DEC",
    z_key="BEST_Z",
    mass_key="M500",
    max_sep_arcmin=15.0,
    max_cz_diff_kms=500.0,
    convert_helio_to_cmb=False,
    prefer_massive=True,
):
    """
    Print a table of observed clusters matched to an external catalogue.

    Parameters
    ----------
    obs_clusters : ObservedClusterCatalogue
        Observed cluster catalogue.
    catalogue : dict or structured array
        External catalogue with RA, Dec, redshift, and mass columns.
    ra_key, dec_key, z_key, mass_key : str
        Column names in the catalogue.
    max_sep_arcmin : float
        Maximum angular separation in arcminutes.
    max_cz_diff_kms : float
        Maximum cz difference in km/s.
    convert_helio_to_cmb : bool
        If True, convert catalogue redshifts from heliocentric to CMB frame.
    prefer_massive : bool
        If True, most massive catalogue entries get matched first.
    """
    matches, ang_sep, delta_cz = match_obs_clusters_to_catalogue(
        obs_clusters, catalogue,
        ra_key=ra_key, dec_key=dec_key, z_key=z_key,
        mass_key=mass_key,
        max_sep_arcmin=max_sep_arcmin, max_cz_diff_kms=max_cz_diff_kms,
        convert_helio_to_cmb=convert_helio_to_cmb,
        prefer_massive=prefer_massive,
    )

    masses = np.asarray(catalogue[mass_key], dtype=float)
    names = obs_clusters.names
    galactic = obs_clusters.galactic_coordinates

    header = (
        f"{'Cluster':<22} {'ell [deg]':>10} {'b [deg]':>10} "
        f"{'Sep [arcmin]':>12} {'dcz [km/s]':>12} {'log M500':>10}"
    )
    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        ell, b = galactic[i]
        if matches[i] is not None:
            idx = matches[i]
            log_m500 = np.log10(masses[idx])
            print(
                f"{name:<22} {ell:>10.2f} {b:>10.2f} "
                f"{ang_sep[i]:>12.2f} {delta_cz[i]:>12.1f} {log_m500:>10.2f}"
            )
        else:
            print(
                f"{name:<22} {ell:>10.2f} {b:>10.2f} "
                f"{'--':>12} {'--':>12} {'--':>10}"
            )


def attach_associations_to_obs_clusters(
    obs_clusters, associations, cfg, verbose=True,
    cluster_priority=None, prioritize_hercules=True
):
    """Match associations to observed clusters via greedy matching.

    Parameters
    ----------
    obs_clusters
        ObservedClusterCatalogue instance.
    associations
        List of associations.
    cfg
        Configuration dictionary.
    verbose
        If True, print progress.
    cluster_priority
        Optional list of cluster names in priority order. Clusters earlier
        in the list are matched first when they have good p-values.
    prioritize_hercules
        If True and cluster_priority is None, prioritize matching
        "Hercules (A2147)" before "Hercules (A2151)". Default: True.
    """
    if obs_clusters is None:
        raise ValueError("obs_clusters must be provided.")
    if verbose:
        print(f"Using {len(obs_clusters)} observed clusters.")
    if not associations:
        raise ValueError("No associations provided.")

    # Default priority: A2147 before A2151
    if cluster_priority is None and prioritize_hercules:
        cluster_priority = ["Hercules (A2147)", "Hercules (A2151)"]

    pval_matrix, dist_matrix, pval_per_halo = compute_matching_matrix_obs(
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
        cluster_priority=cluster_priority,
        pval_per_halo=pval_per_halo,
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

    # Get observed cluster Cartesian positions for 3D separation
    obs_cartesian = obs_clusters.icrs_cartesian()

    # Sort by observed redshift
    redshifts = np.asarray(obs_clusters.redshifts, dtype=float)
    sort_order = np.argsort(redshifts)

    percentiles = tuple(percentiles)
    rows = []
    perc_header = " ".join(
        f"P{int(p):02d}%".rjust(8) for p in percentiles
    ) if percentiles else ""
    base_header = (
        f"{'Cluster':<22} {'z':>6} {'Assoc':>7} {'Frac':>6} "
        f"{'logM [Msun/h]':>14} {'Pfeifer pval':>12} "
        f"{'Dist [Mpc/h]':>13} {'Sep [Mpc/h]':>12} "
        f"{'ell [deg]':>10} {'b [deg]':>10} "
        f"{'Med tSZ pval':>12}"
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

    for idx in sort_order:
        name = names[idx]
        z = redshifts[idx]
        entry = matches[idx]
        assoc_label = "-"
        frac_present = np.nan
        median_logm = np.nan
        match_p = np.nan
        combined = np.nan
        centroid_dist = np.nan
        separation_3d = np.nan
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

            # Compute 3D separation in redshift space
            obs_pos = obs_cartesian[idx]
            box_size = getattr(assoc, "box_size", None)
            if box_size is not None:
                assoc_centroid = np.mean(
                    assoc.redshift_position - box_size / 2, axis=0)
                separation_3d = float(np.linalg.norm(obs_pos - assoc_centroid))

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
        if match_display is not None:
            match_display_str = f"{match_display:>12.1e}"
        else:
            match_display_str = " " * 12

        row = (
            f"{name:<22} "
            f"{z:>6.4f} "
            f"{str(assoc_label):>7} "
            f"{frac_present:>6.2f} "
            f"{median_logm:>14.2f} "
            f"{match_display_str} "
            f"{centroid_dist:>13.3f} "
            f"{separation_3d:>12.3f} "
            f"{ell_deg:>10.2f} "
            f"{b_deg:>10.2f} "
            f"{median_tsz:>12.1e} "
            f"{frac_low_p:>14.2%}"
        )
        if percentiles:
            perc_str = " ".join(f"{val:>8.1e}" for val in perc_vals)
            row = f"{row} {perc_str}"
        print(row)
        rows.append(
            {
                "name": name,
                "redshift": z,
                "association_label": assoc_label,
                "fraction_present": frac_present,
                "median_log_mass": median_logm,
                "match_p": match_p,
                "combined_p": combined,
                "distance_mpc_h": centroid_dist,
                "separation_3d_mpc_h": separation_3d,
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

    separations = np.array([r["separation_3d_mpc_h"] for r in rows])
    separations = separations[np.isfinite(separations)]
    if separations.size:
        mean_sep = np.mean(separations)
        std_sep = np.std(separations)
        print(f"\nMean 3D separation: {mean_sep:.2f} +/- {std_sep:.2f} Mpc/h")


def plot_cluster_pvalue_percentiles(
    cfg,
    matches,
    obs_clusters=None,
    sim_key=None,
    observer_centre=None,
    ax=None,
    suite_labels=None,
    suite_colors=None,
    exclude_prefixes=None,
):
    """
    Plot per-cluster percentile summaries of per-halo tSZ p-values.

    Parameters
    ----------
    suite_colors
        Optional sequence of Matplotlib color specs, one per simulation suite.
    exclude_prefixes
        Optional list of strings. Clusters whose names start with any of these
        prefixes will be excluded from the plot.
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

    # Create include mask based on exclude_prefixes
    original_names = list(names)
    if exclude_prefixes is not None:
        include_mask = np.array(
            [not any(name.startswith(prefix) for prefix in exclude_prefixes)
             for name in original_names],
            dtype=bool
        )
        if not np.any(include_mask):
            raise ValueError(
                "No clusters remain after applying exclude_prefixes.")
    else:
        include_mask = np.ones(len(original_names), dtype=bool)

    # Get redshifts and create sorting order
    redshifts = np.asarray(obs_clusters.redshifts, dtype=float)
    filtered_indices = np.where(include_mask)[0]
    filtered_redshifts = redshifts[filtered_indices]
    sort_order = np.argsort(filtered_redshifts)
    sorted_filtered_indices = filtered_indices[sort_order]

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
            if len(match_input) != len(original_names):
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
            if len(candidate) != len(original_names):
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

    # Filter and sort names for display after validation
    filtered_names = [original_names[idx] for idx in sorted_filtered_indices]
    num_suites = len(matches_by_suite)
    if suite_labels is not None:
        if len(suite_labels) != num_suites:
            raise ValueError(
                "suite_labels must match the number of match collections."
            )
        suite_labels = [SIM_LABEL_NAMES.get(lbl, lbl) for lbl in suite_labels]
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
                "suite_colors sequence must provide at least one entry "
                "per suite."
            )
        suite_colors = list(suite_colors)

    percentile_levels = (5, 50, 95)
    stats_payloads = []
    for suite in matches_by_suite:
        positions = []
        p05 = []
        p50 = []
        p95 = []
        for plot_position, orig_idx in enumerate(sorted_filtered_indices,
                                                 start=1):
            entry = suite[orig_idx]
            if entry is None:
                continue
            assoc = entry[0]
            per_halo = np.asarray(
                getattr(assoc, "halo_pvals", []), dtype=float)
            finite = per_halo[np.isfinite(per_halo)]
            if finite.size == 0:
                continue
            positions.append(plot_position)
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
            # width = max(8.0, 0.7 * len(filtered_names))
            fig, ax = plt.subplots(figsize=(9, 2.5))
        else:
            fig = ax.figure

        positions = np.arange(1, len(filtered_names) + 1)
        offsets = np.zeros(num_suites)
        if num_suites > 1:
            offsets = np.linspace(-0.2, 0.2, num_suites)
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

        ax.set_xticks(positions, filtered_names)
        ax.tick_params(axis="x", which="both", length=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation(50)
            label.set_rotation_mode("anchor")
        ax.set_xlim(0.5, len(filtered_names) + 0.5)
        boundaries = np.arange(1.5, len(filtered_names) + 0.5, 1.0)
        spine_width = ax.spines['bottom'].get_linewidth()
        for xpos in boundaries:
            ax.axvline(
                xpos,
                color="black",
                linestyle="--",
                linewidth=spine_width,
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
        for thresh in [0.05,]:
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
    percentiles=(16, 84),
    ax=None,
):
    """
    Plot correlation between LUM matching p-values and median tSZ p-values.
    """
    lum_vals = []
    lum_lo = []
    lum_hi = []
    tsz_vals = []
    tsz_lo = []
    tsz_hi = []
    labels = []
    for idx, entry in enumerate(matches):
        if entry is None:
            lum_vals.append(default_pfeifer)
            lum_lo.append(default_pfeifer)
            lum_hi.append(default_pfeifer)
            tsz_vals.append(default_tsz)
            tsz_lo.append(default_tsz)
            tsz_hi.append(default_tsz)
            labels.append(idx)
            continue
        assoc, match_p, _ = entry

        # LUM (x-axis)
        lum_median = match_p if np.isfinite(match_p) else default_pfeifer
        lum_vals.append(lum_median)

        lum_pvals = np.asarray(getattr(assoc, "lum_pvals", []), dtype=float)
        finite_lum = lum_pvals[np.isfinite(lum_pvals)]
        if len(finite_lum) >= 2:
            lo_lum, hi_lum = np.percentile(finite_lum, percentiles)
        else:
            lo_lum, hi_lum = lum_median, lum_median
        lum_lo.append(lo_lum)
        lum_hi.append(hi_lum)

        # tSZ (y-axis)
        median_pval = float(getattr(assoc, "median_pval", default_tsz))
        if not np.isfinite(median_pval):
            median_pval = default_tsz
        tsz_vals.append(median_pval)

        halo_pvals = np.asarray(
            getattr(assoc, "halo_pvals", []), dtype=float)
        finite = halo_pvals[np.isfinite(halo_pvals)]
        if len(finite) >= 2:
            lo, hi = np.percentile(finite, percentiles)
        else:
            lo, hi = median_pval, median_pval
        tsz_lo.append(lo)
        tsz_hi.append(hi)
        labels.append(idx)

    lum_vals = np.clip(np.asarray(lum_vals, dtype=float), 1e-6, 1.0)
    lum_lo = np.clip(np.asarray(lum_lo, dtype=float), 1e-6, 1.0)
    lum_hi = np.clip(np.asarray(lum_hi, dtype=float), 1e-6, 1.0)
    tsz_vals = np.clip(np.asarray(tsz_vals, dtype=float), 1e-6, 1.0)
    tsz_lo = np.clip(np.asarray(tsz_lo, dtype=float), 1e-6, 1.0)
    tsz_hi = np.clip(np.asarray(tsz_hi, dtype=float), 1e-6, 1.0)

    xerr_lo = lum_vals - lum_lo
    xerr_hi = lum_hi - lum_vals
    yerr_lo = tsz_vals - tsz_lo
    yerr_hi = tsz_hi - tsz_vals

    with plt.style.context("science"):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.errorbar(
            lum_vals, tsz_vals,
            xerr=[xerr_lo, xerr_hi],
            yerr=[yerr_lo, yerr_hi],
            fmt='o', c="#731dd8", ms=5, alpha=0.8,
            elinewidth=1.0, capsize=3,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{\rm LUM}$")
        ax.set_ylabel(r"$p_{\mathrm{tSZ}}$")
        ax.grid(False)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lower = min(xlim[0], ylim[0])
        upper = max(xlim[1], ylim[1])
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.axline(
            (1e-3, 1e-3), (1, 1),
            color="#ef476f",
            linestyle="--",
            label=r"$1$:$1$",
        )
        ax.legend(frameon=False)
        fig.tight_layout()

    return fig, ax
