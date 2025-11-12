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
"""Plotting helpers for tSZ mass-bin analyses."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from cmbo.io import TSZMassBinResults


def load_mass_bin_results(
    cfg,
    sim_key,
    include_profiles=True,
    simulation_id=None,
):
    """
    Load TSZ mass-bin results for a given simulation using the CMBO config.

    Parameters
    ----------
    cfg : dict
        Parsed CMBO configuration dictionary.
    sim_key : str
        Simulation key (e.g., ``"csiborg2"``) locating both the halo catalogue
        and the corresponding HDF5 output.
    include_profiles : bool, optional
        If ``True`` (default) the stacked profile datasets are loaded.

    simulation_id : str or int, optional
        Specific simulation group inside the results file (e.g. ``15517``).
        If omitted, all simulations are loaded.

    Returns
    -------
    TSZMassBinResults
        Results object ready for plotting utilities.
    """
    try:
        analysis_cfg = cfg["analysis"]
    except KeyError as exc:
        raise ValueError("Configuration missing 'analysis' section.") from exc

    output_dir = Path(analysis_cfg.get("output_folder", "."))
    root_path = Path(cfg.get("_root_path", "."))
    if not output_dir.is_absolute():
        output_dir = (root_path / output_dir).resolve()
    tag = analysis_cfg.get("output_tag")
    stem = sim_key if not tag else f"{sim_key}_{tag}"
    results_path = output_dir / f"{stem}.hdf5"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file '{results_path}' not found. "
            "Run the mass-bin analysis first."
        )

    if simulation_id is not None:
        simulation_id = str(simulation_id)

    return TSZMassBinResults(
        results_path,
        include_profiles=include_profiles,
        simulation=simulation_id,
    )


def _prepare_profile_block(res_obj, nbins, simulation):
    all_sims = list(res_obj.iter_simulations())
    if not all_sims:
        raise ValueError("No simulations available.")

    if simulation is None:
        sims = all_sims
    else:
        if isinstance(simulation, int):
            if simulation < 0 or simulation >= len(all_sims):
                raise IndexError(
                    f"Simulation index {simulation} is out of range "
                    f"[0, {len(all_sims) - 1}]."
                )
            sims = [all_sims[simulation]]
        else:
            sims = [(simulation, res_obj[simulation])]

    max_bins = max((len(sim.bins) for _, sim in sims), default=0)
    if max_bins == 0:
        raise ValueError("No bins found in simulations.")

    radii_norm = None
    for _, sim in sims:
        if sim.bins and sim.bins[0].radii_norm is not None:
            candidate = np.asarray(sim.bins[0].radii_norm, dtype=float)
            if candidate.size:
                radii_norm = candidate
                break
    if radii_norm is None:
        raise ValueError("No radial grid stored in the results.")
    nradii = radii_norm.size
    x_min = float(np.nanmin(radii_norm))
    x_max = float(np.nanmax(radii_norm))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Non-finite radial grid detected.")

    nsim = len(sims)
    profile_mean = np.full((nsim, max_bins, nradii), np.nan)
    profile_err = np.full((nsim, max_bins, nradii), np.nan)
    random_mean = np.full((nsim, max_bins, nradii), np.nan)
    random_err = np.full((nsim, max_bins, nradii), np.nan)
    pvalue_profile = np.full((nsim, max_bins, nradii), np.nan)
    t_fit_pvals = np.full((nsim, max_bins, nradii), np.nan)
    t_fit_sigmas = np.full((nsim, max_bins, nradii), np.nan)

    for i, (_, sim) in enumerate(sims):
        for j, bin_res in enumerate(sim.bins):
            if (
                bin_res.stacked_profile is None
                or bin_res.stacked_error is None
            ):
                continue
            profile_mean[i, j] = np.asarray(
                bin_res.stacked_profile,
                dtype=float,
            )
            profile_err[i, j] = np.asarray(
                bin_res.stacked_error,
                dtype=float,
            )
            if getattr(bin_res, "random_profile", None) is not None:
                random_mean[i, j] = np.asarray(
                    bin_res.random_profile,
                    dtype=float,
                )
            if getattr(bin_res, "random_error", None) is not None:
                random_err[i, j] = np.asarray(
                    bin_res.random_error,
                    dtype=float,
                )
            if getattr(bin_res, "p_value_profile", None) is not None:
                pvalue_profile[i, j] = np.asarray(
                    bin_res.p_value_profile,
                    dtype=float,
                )
            if getattr(bin_res, "t_fit_p_value", None) is not None:
                t_fit_pvals[i, j] = np.asarray(
                    bin_res.t_fit_p_value,
                    dtype=float,
                )
            if getattr(bin_res, "t_fit_sigma", None) is not None:
                t_fit_sigmas[i, j] = np.asarray(
                    bin_res.t_fit_sigma,
                    dtype=float,
                )

    nbins_to_plot = min(nbins, max_bins)
    bin_labels = []
    for j in range(max_bins):
        label = None
        lo_val = None
        hi_val = None
        for _, sim in sims:
            if j >= len(sim.bins):
                continue
            lo, hi = sim.bins[j].lo, sim.bins[j].hi
            lo_val = lo if lo_val is None else min(lo_val, lo)
            if hi is not None:
                hi_val = hi if hi_val is None else max(hi_val, hi)
            if hi is None:
                label = rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}}$"
            else:
                label = (
                    rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}} < {hi:.2f}$"
                )
            break
        if lo_val is not None:
            if hi_val is None:
                mass_str = rf"$\log M_{{200\mathrm{{c}}}} > {lo_val:.2f}$"
            else:
                mass_str = (
                    rf"${lo_val:.2f} < \log M_{{200\mathrm{{c}}}} "
                    rf"< {hi_val:.2f}$"
                )
        else:
            mass_str = None
        bin_labels.append((label, mass_str))

    return {
        "profile_mean": profile_mean,
        "profile_err": profile_err,
        "random_mean": random_mean,
        "random_err": random_err,
        "pvalue_profile": pvalue_profile,
        "t_fit_p_value": t_fit_pvals,
        "t_fit_sigma": t_fit_sigmas,
        "radii_norm": radii_norm,
        "x_min": x_min,
        "x_max": x_max,
        "nbins": nbins_to_plot,
        "nsim": nsim,
        "bin_labels": bin_labels,
    }


def plot_stacked_profiles(
    res,
    nbins=3,
    simulation=None,
    theta500_scale=5.0,
    row_labels=None,
):
    r"""
    Plot stacked or per-simulation y-profile summaries for leading mass bins.

    Parameters
    ----------
    res
        TSZMassBinResults instance or a sequence of them. When a sequence
        is provided, each element is plotted on a separate pair of rows.
    nbins : int, optional
        Maximum number of bins to display, default is 3.
    simulation : str, optional
        If provided, select a single simulation to plot without stacking.
    theta500_scale : float, optional
        Scaling applied to the radial normalization relative to theta_500c.
    row_labels : sequence of str, optional
        Custom labels to annotate each suite row. Must match the number of
        result objects when ``res`` is a sequence.
    """

    if hasattr(res, "iter_simulations"):
        res_list = [res]
    else:
        if not isinstance(res, Sequence) or isinstance(res, (str, bytes)):
            raise ValueError(
                "res must be a TSZMassBinResults instance or a sequence of "
                "such instances."
            )
        if not res:
            raise ValueError("res sequence must not be empty.")
        res_list = list(res)
        for idx, entry in enumerate(res_list):
            if not hasattr(entry, "iter_simulations"):
                raise ValueError(
                    f"Element {idx} of res is not a TSZMassBinResults object."
                )

    blocks = [_prepare_profile_block(item, nbins, simulation)
              for item in res_list]
    if row_labels is not None:
        if len(row_labels) != len(blocks):
            raise ValueError(
                "row_labels must match the number of result objects."
            )
    else:
        row_labels = [
            Path(getattr(item, "path", f"suite_{idx}")).stem
            for idx, item in enumerate(res_list)
        ]
    nbins_to_plot = min(block["nbins"] for block in blocks)
    if nbins_to_plot <= 0:
        raise ValueError("nbins must be positive.")

    with plt.style.context("science"):
        width = 3.2 * nbins_to_plot
        height = 4.0 * len(blocks)
        if theta500_scale == 1:
            theta_label = r"$\theta / \theta_{500\mathrm{c}}$"
        else:
            theta_label = (
                r"$\theta / ("
                f"{theta500_scale:g}"
                r"\,\theta_{500\mathrm{c}})$"
            )
        height_ratios = np.tile([3.0, 1.5], len(blocks))
        fig, axes = plt.subplots(
            2 * len(blocks),
            nbins_to_plot,
            figsize=(width, height),
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if nbins_to_plot == 1:
            axes = axes.reshape(2 * len(blocks), 1)
        axes = axes.reshape(len(blocks), 2, nbins_to_plot)

        lw = plt.rcParams["lines.linewidth"]
        any_tfit_used = False
        for block_idx, (block, row_label) in enumerate(zip(blocks, row_labels)):
            profile_axes = axes[block_idx, 0]
            sig_axes = axes[block_idx, 1]
            radii_norm = block["radii_norm"]
            x_min = block["x_min"]
            x_max = block["x_max"]
            nsim = block["nsim"]

            for j_bin in range(nbins_to_plot):
                ax = profile_axes[j_bin]
                ax_sig = sig_axes[j_bin]
                have_bin = ~np.isnan(block["profile_mean"][:, j_bin, 0])
                if not np.any(have_bin):
                    ax.set_axis_off()
                    ax_sig.set_axis_off()
                    continue

                pm = block["profile_mean"][have_bin, j_bin]
                pe = block["profile_err"][have_bin, j_bin]
                rm = block["random_mean"][have_bin, j_bin]
                re = block["random_err"][have_bin, j_bin]
                pv = block["t_fit_p_value"][have_bin, j_bin]
                using_tfit_panel = np.isfinite(pv).any()
                if using_tfit_panel:
                    any_tfit_used = True
                else:
                    pv = block["pvalue_profile"][have_bin, j_bin]

                if simulation is None or nsim > 1:
                    stack_mean = np.nanmean(pm, axis=0)
                    stack_err = np.nanmean(pe, axis=0)
                    if not np.isfinite(stack_err).any():
                        stack_err = np.zeros_like(stack_mean)
                    label_data = (
                        "Data" if (j_bin == 0 and block_idx == 0) else None
                    )
                    ax.plot(
                        radii_norm,
                        stack_mean,
                        lw=lw,
                        color="C0",
                        label=label_data,
                    )
                    ax.fill_between(
                        radii_norm,
                        stack_mean - stack_err,
                        stack_mean + stack_err,
                        color="C0",
                        alpha=0.20,
                        linewidth=0,
                    )
                else:
                    mean_val = pm[0]
                    err_val = pe[0]
                    label_data = (
                        "Data" if (j_bin == 0 and block_idx == 0) else None
                    )
                    ax.plot(
                        radii_norm,
                        mean_val,
                        lw=lw,
                        color="C0",
                        label=label_data,
                    )
                    ax.fill_between(
                        radii_norm,
                        mean_val - err_val,
                        mean_val + err_val,
                        color="C0",
                        alpha=0.25,
                        linewidth=0,
                    )

                if np.isfinite(rm).any():
                    rand_mean = np.nanmean(rm, axis=0)
                    rand_spread = np.sqrt(np.nanmean(re**2, axis=0))
                    label_rand = (
                        "Random" if (j_bin == 0 and block_idx == 0) else None
                    )
                    ax.plot(
                        radii_norm,
                        rand_mean,
                        lw=lw * 0.8,
                        ls="--",
                        color="0.35",
                        label=label_rand,
                    )
                    ax.fill_between(
                        radii_norm,
                        rand_mean - rand_spread,
                        rand_mean + rand_spread,
                        color="0.6",
                        alpha=0.20,
                        linewidth=0,
                    )

                label_text, _ = block["bin_labels"][j_bin]
                if label_text:
                    ax.text(
                        0.5,
                        1.02,
                        label_text,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                    )

                ax.axhline(0.0, lw=lw * 0.7, alpha=0.6, color="k")
                ax.set_xlim(x_min, x_max)

                ax_sig.set_xlim(x_min, x_max)
                ax_sig.set_yscale("log")
                ax_sig.axhline(0.05, color="k", lw=lw)
                multi_sim = pv.shape[0] > 1
                if simulation is None or nsim > 1:
                    stack_pv = np.nanmedian(pv, axis=0)
                else:
                    stack_pv = pv[0]
                valid = np.isfinite(stack_pv) & (stack_pv > 0)
                if np.any(valid):
                    ax_sig.plot(
                        radii_norm[valid],
                        stack_pv[valid],
                        color="C3",
                        lw=lw * 0.9,
                    )
                    if multi_sim:
                        pv_clean = pv.copy()
                        pv_clean[~np.isfinite(pv_clean) | (pv_clean <= 0)] = np.nan
                        low = np.nanpercentile(pv_clean, 16.0, axis=0)
                        high = np.nanpercentile(pv_clean, 84.0, axis=0)
                        band = (
                            np.isfinite(low)
                            & np.isfinite(high)
                            & (low > 0)
                            & (high > 0)
                        )
                        if np.any(band):
                            ax_sig.fill_between(
                                radii_norm[band],
                                low[band],
                                high[band],
                                color="C3",
                                alpha=0.2,
                                linewidth=0,
                            )
                    ymin = stack_pv[valid].min()
                    ymax = stack_pv[valid].max()
                    if ymin == ymax:
                        ymin *= 0.8
                        ymax *= 1.2
                    ymin = min(ymin, 0.05 * 0.8)
                    ymax = max(ymax, 0.05 * 1.2)
                    ax_sig.set_ylim(ymin, ymax)
                else:
                    ax_sig.set_axis_off()
                    continue

            profile_axes[0].set_ylabel(
                r"$\langle y(<\theta) \rangle$ (mean enclosed)")
            if block_idx == 0:
                profile_axes[0].legend(frameon=False, loc="upper right")
            y_label = (
                r"$p_{t}(\theta)$" if any_tfit_used else r"$p_{\rm KS}(\theta)$"
            )
            sig_axes[0].set_ylabel(y_label)
            for ax_sig in sig_axes:
                ax_sig.set_xlabel(theta_label)

            if row_label:
                right_ax = profile_axes[-1]
                right_ax.text(
                    1.02,
                    0.5,
                    row_label,
                    transform=right_ax.transAxes,
                    ha="left",
                    va="center",
                    rotation=90,
                    fontsize=10,
                )

    plt.close()

    return fig, axes


def plot_cutout_maps(
    res,
    nbins=3,
    mode="stack",
    simulation=None,
    circle_color="#3b82f6",
    circle_lw=1.0,
    cmap="coolwarm",
    theta500_scale=5.0,
    row_labels=None,
):
    r"""
    Plot stacked or per-simulation cutout mean maps for selected bins.

    Parameters
    ----------
    theta500_scale : float, optional
        Scaling applied to the angular normalization relative to theta_500c.
    row_labels : sequence of str, optional
        Row annotations applied when multiple results are provided.
    """

    mode = mode.lower()
    if mode not in {"stack", "single"}:
        raise ValueError("mode must be 'stack' or 'single'.")

    def _prepare_block(res_obj):
        all_sims = list(res_obj.iter_simulations())
        if not all_sims:
            raise ValueError("No simulations available.")

        max_bins = max((len(sim.bins) for _, sim in all_sims), default=0)
        if max_bins == 0:
            raise ValueError("No cutouts found in simulations.")

        shapes = [None] * max_bins
        extents = [None] * max_bins
        for j in range(max_bins):
            for _, sim in all_sims:
                if j >= len(sim.bins):
                    continue
                cm = getattr(sim.bins[j], "cutout_mean", None)
                if cm is not None:
                    shapes[j] = cm.shape
                    extents[j] = getattr(sim.bins[j], "cutout_extent", None)
                    break

        ny, nx = next((shape for shape in shapes if shape is not None), (0, 0))
        if ny == 0 or nx == 0:
            raise ValueError("No cutout imagery available.")

        nsim = len(all_sims)
        cutouts = np.full((nsim, max_bins, ny, nx), np.nan, dtype=float)
        for i, (_, sim) in enumerate(all_sims):
            for j, bin_res in enumerate(sim.bins):
                cm = getattr(bin_res, "cutout_mean", None)
                if cm is None:
                    continue
                arr = np.asarray(cm, dtype=float)
                if arr.shape != (ny, nx):
                    raise ValueError("Cutout shapes do not match.")
                cutouts[i, j] = arr

        sel_idx = None
        if mode == "single":
            if simulation is None:
                raise ValueError("Provide simulation when mode='single'.")
            if isinstance(simulation, int):
                if simulation < 0 or simulation >= nsim:
                    raise IndexError(
                        f"Simulation index {simulation} outside [0, {nsim - 1}]."
                    )
                sel_idx = simulation
            else:
                match = None
                for idx, (name, _) in enumerate(all_sims):
                    if name == simulation:
                        match = idx
                        break
                if match is None:
                    names = [name for name, _ in all_sims]
                    raise KeyError(
                        f"Simulation '{simulation}' not found. "
                        f"Available: {names}"
                    )
                sel_idx = match

        nbins_plot = min(max_bins, nbins)
        images = []
        labels = []
        extent_list = []
        for j in range(nbins_plot):
            if mode == "stack":
                have = ~np.isnan(cutouts[:, j, 0, 0])
                if not np.any(have):
                    images.append(None)
                else:
                    images.append(np.nanmean(cutouts[have, j], axis=0))
            else:
                if sel_idx is None or np.isnan(cutouts[sel_idx, j, 0, 0]):
                    images.append(None)
                else:
                    images.append(cutouts[sel_idx, j])

            label = None
            for _, sim in all_sims:
                if j < len(sim.bins):
                    lo, hi = sim.bins[j].lo, sim.bins[j].hi
                    if hi is None:
                        label = rf"$\log M_{{200\mathrm{{c}}}} > {lo:.2f}$"
                    else:
                        label = (
                            rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}} "
                            rf"< {hi:.2f}$"
                        )
                    break
            labels.append(label)
            extent_list.append(extents[j])

        return {
            "images": images,
            "shape": (ny, nx),
            "extents": extent_list,
            "labels": labels,
            "nbins": nbins_plot,
            "path": getattr(res_obj, "path", None),
        }

    if hasattr(res, "iter_simulations"):
        res_list = [res]
    else:
        if not isinstance(res, Sequence) or isinstance(res, (str, bytes)):
            raise ValueError(
                "res must be a TSZMassBinResults instance or a sequence."
            )
        if not res:
            raise ValueError("res sequence must not be empty.")
        res_list = list(res)
        for idx, entry in enumerate(res_list):
            if not hasattr(entry, "iter_simulations"):
                raise ValueError(
                    f"Element {idx} of res is not a TSZMassBinResults object."
                )

    blocks = [_prepare_block(item) for item in res_list]
    nbins_plot = min(block["nbins"] for block in blocks)
    if nbins_plot <= 0:
        raise ValueError("nbins must be positive.")

    if row_labels is not None:
        if len(row_labels) != len(blocks):
            raise ValueError("row_labels must match number of result objects.")
    else:
        row_labels = [
            Path(block["path"]).stem if block["path"] else f"suite_{idx}"
            for idx, block in enumerate(blocks)
        ]

    if theta500_scale == 1:
        theta_x_label = r"$\theta_x / \theta_{500\mathrm{c}}$"
        theta_y_label = r"$\theta_y / \theta_{500\mathrm{c}}$"
    else:
        factor = f"{theta500_scale:g}"
        theta_x_label = (
            r"$\theta_x / ("
            f"{factor}"
            r"\,\theta_{500\mathrm{c}})$"
        )
        theta_y_label = (
            r"$\theta_y / ("
            f"{factor}"
            r"\,\theta_{500\mathrm{c}})$"
        )

    with plt.style.context("science"):
        fig, axes = plt.subplots(
            len(blocks),
            nbins_plot,
            figsize=(4.8 * nbins_plot, 4.3 * len(blocks)),
            squeeze=False,
        )

        n_rows = len(blocks)
        for row_idx, (block, label_text) in enumerate(zip(blocks, row_labels)):
            ny, nx = block["shape"]
            for j_bin in range(nbins_plot):
                ax = axes[row_idx, j_bin]
                img = block["images"][j_bin]
                if img is None:
                    ax.set_visible(False)
                    continue

                scale = np.nanpercentile(np.abs(img), 99.0)
                if not np.isfinite(scale) or scale == 0:
                    scale = np.nanmax(np.abs(img))
                if not np.isfinite(scale) or scale == 0:
                    vmin, vmax = -1.0, 1.0
                else:
                    vmin, vmax = -scale, scale

                extent = block["extents"][j_bin]
                im = ax.imshow(
                    img,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    interpolation="nearest",
                )
                ax.set_aspect("equal")

                cross_color = circle_color or "#3b82f6"
                if extent is not None:
                    centre = (0.0, 0.0)
                    radius = 1.0
                else:
                    cy = (ny - 1) / 2.0
                    cx = (nx - 1) / 2.0
                    radius = min(nx, ny) / 2.0
                    centre = (cx, cy)
                circle = Circle(
                    centre,
                    radius=radius,
                    edgecolor=cross_color,
                    facecolor="none",
                    lw=circle_lw,
                    ls="--",
                )
                ax.add_patch(circle)
                cross_half = radius * 0.95
                hline = Line2D(
                    [centre[0] - cross_half, centre[0] + cross_half],
                    [centre[1], centre[1]],
                    color=cross_color,
                    lw=circle_lw,
                    linestyle="--",
                )
                vline = Line2D(
                    [centre[0], centre[0]],
                    [centre[1] - cross_half, centre[1] + cross_half],
                    color=cross_color,
                    lw=circle_lw,
                    linestyle="--",
                )
                hline.set_clip_path(circle)
                vline.set_clip_path(circle)
                ax.add_line(hline)
                ax.add_line(vline)

                bin_label = block["labels"][j_bin]
                if bin_label:
                    ax.text(
                        0.5,
                        1.02,
                        bin_label,
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )

                if row_idx == len(blocks) - 1:
                    ax.set_xlabel(theta_x_label)
                else:
                    ax.set_xlabel("")
                if j_bin == 0:
                    ax.set_ylabel(theta_y_label)
                else:
                    ax.set_ylabel("")

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label(r"$y$", fontsize=9)

            if label_text:
                row_pos = (n_rows - row_idx - 0.5) / n_rows
                fig.text(
                    1.01,
                    row_pos,
                    label_text,
                    rotation=90,
                    ha="left",
                    va="center",
                    fontsize=10,
                    clip_on=False,
                )

        fig.tight_layout()

    plt.close()

    return fig, axes


__all__ = [
    "plot_stacked_profiles",
    "plot_stacked_profiles_overlay",
    "plot_cutout_maps",
    "load_mass_bin_results",
]


def plot_stacked_profiles_overlay(
    res,
    nbins=3,
    simulation=None,
    theta500_scale=5.0,
    suite_labels=None,
    colors=None,
):
    """
    Plot per-bin stacked profiles from multiple suites within a single row,
    overlaying each suite as a band (no random-control or p-value panels).

    Parameters
    ----------
    res
        Single ``TSZMassBinResults`` or a sequence of them.
    nbins : int, optional
        Maximum number of mass bins to display (per suite).
    simulation : str or int, optional
        Restrict plotting to a particular simulation entry within each suite.
    theta500_scale : float, optional
        Radial normalization factor relative to :math:`\\theta_{500c}`.
    suite_labels : sequence of str, optional
        Labels to use in the legend for each suite. Defaults to the stem of
        the underlying results file.
    colors : sequence, optional
        Matplotlib-compatible color specifications applied per suite.
    """
    if hasattr(res, "iter_simulations"):
        res_list = [res]
    else:
        if not isinstance(res, Sequence) or isinstance(res, (str, bytes)):
            raise ValueError("res must be a TSZMassBinResults instance or a sequence.")
        if not res:
            raise ValueError("res sequence must not be empty.")
        for idx, entry in enumerate(res):
            if not hasattr(entry, "iter_simulations"):
                raise ValueError(f"Element {idx} of res is not a TSZMassBinResults.")
        res_list = list(res)

    blocks = [_prepare_profile_block(item, nbins, simulation)
              for item in res_list]

    nbins_to_plot = min(block["nbins"] for block in blocks)
    if nbins_to_plot <= 0:
        raise ValueError("nbins must be positive.")

    nsuites = len(blocks)
    if suite_labels is not None:
        if len(suite_labels) != nsuites:
            raise ValueError("suite_labels must match the number of suites.")
    else:
        suite_labels = [
            Path(getattr(item, "path", f"suite_{idx}")).stem
            for idx, item in enumerate(res_list)
        ]

    if colors is not None:
        if len(colors) < nsuites:
            raise ValueError("Need at least one color per suite.")
        color_cycle = list(colors)
    else:
        prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
        default_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
        if prop_cycle is not None:
            color_cycle = prop_cycle.by_key().get("color", default_colors)
        else:
            color_cycle = default_colors
    if not color_cycle:
        raise ValueError("No colors available for plotting.")

    with plt.style.context("science"):
        if theta500_scale == 1:
            theta_label = r"$\theta / \theta_{500\mathrm{c}}$"
        else:
            theta_label = (
                r"$\theta / ("
                f"{theta500_scale:g}"
                r"\,\theta_{500\mathrm{c}})$"
            )
        width = 3.2 * nbins_to_plot
        fig, axes = plt.subplots(
            1,
            nbins_to_plot,
            figsize=(width, 3.6),
            constrained_layout=True,
        )
        if nbins_to_plot == 1:
            axes = [axes]
        lw = plt.rcParams["lines.linewidth"]

        x_min = min(block["x_min"] for block in blocks)
        x_max = max(block["x_max"] for block in blocks)

        for j_bin in range(nbins_to_plot):
            ax = axes[j_bin]

            for suite_idx, block in enumerate(blocks):
                color = color_cycle[suite_idx % len(color_cycle)]
                have_bin = ~np.isnan(block["profile_mean"][:, j_bin, 0])
                if not np.any(have_bin):
                    continue
                radii_norm = block["radii_norm"]
                pm = block["profile_mean"][have_bin, j_bin]
                pe = block["profile_err"][have_bin, j_bin]

                multi_sim = (simulation is None) or (block["nsim"] > 1)
                if multi_sim:
                    stack_mean = np.nanmean(pm, axis=0)
                    stack_err = np.nanmean(pe, axis=0)
                    if not np.isfinite(stack_err).any():
                        stack_err = np.zeros_like(stack_mean)
                else:
                    stack_mean = pm[0]
                    stack_err = pe[0]

                _, mass_label = block["bin_labels"][j_bin]
                if mass_label:
                    legend_label = f"{suite_labels[suite_idx]}: {mass_label}"
                else:
                    legend_label = suite_labels[suite_idx]
                label_data = legend_label
                ax.plot(
                    radii_norm,
                    stack_mean,
                    lw=lw,
                    color=color,
                    label=label_data,
                )
                ax.fill_between(
                    radii_norm,
                    stack_mean - stack_err,
                    stack_mean + stack_err,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )

            ax.axhline(0.0, lw=lw * 0.7, alpha=0.6, color="k")
            ax.set_xlim(x_min, x_max)
            ax.set_xlabel(theta_label)
            ax.legend(frameon=False, loc="upper right", fontsize=9)

        axes[0].set_ylabel(r"$\langle y(<\theta) \rangle$ (mean enclosed)")

    plt.close()
    return fig, axes
