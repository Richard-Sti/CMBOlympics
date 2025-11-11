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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_stacked_profiles(
    res,
    nbins=3,
    simulation=None,
    theta500_scale=5.0,
):
    r"""
    Plot stacked or per-simulation y-profile summaries for leading mass bins.

    Parameters
    ----------
    res
        TSZMassBinResults instance.
    nbins : int, optional
        Maximum number of bins to display, default is 3.
    dpi : int, optional
        Figure DPI used when saving, default is 300.
    simulation : str, optional
        If provided, select a single simulation to plot without stacking.
    theta500_scale : float, optional
        Scaling applied to the radial normalization relative to theta_500c.
    """

    all_sims = list(res.iter_simulations())
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
            sims = [(simulation, res[simulation])]

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

    nbins_to_plot = min(nbins, max_bins)
    with plt.style.context("science"):
        width = 3.2 * nbins_to_plot
        height = 4.0
        if theta500_scale == 1:
            theta_label = r"$\theta / \theta_{500\mathrm{c}}$"
        else:
            theta_label = (
                r"$\theta / ("
                f"{theta500_scale:g}"
                r"\,\theta_{500\mathrm{c}})$"
            )
        fig, axes = plt.subplots(
            2,
            nbins_to_plot,
            figsize=(width, height),
            sharex="col",
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
        if nbins_to_plot == 1:
            axes = axes.reshape(2, 1)
        profile_axes = axes[0]
        sig_axes = axes[1]

        lw = plt.rcParams["lines.linewidth"]
        for j_bin in range(nbins_to_plot):
            ax = profile_axes[j_bin]
            ax_sig = sig_axes[j_bin]
            have_bin = ~np.isnan(profile_mean[:, j_bin, 0])
            if not np.any(have_bin):
                ax.set_axis_off()
                ax_sig.set_axis_off()
                continue

            pm = profile_mean[have_bin, j_bin]
            pe = profile_err[have_bin, j_bin]
            rm = random_mean[have_bin, j_bin]
            re = random_err[have_bin, j_bin]
            pv = pvalue_profile[have_bin, j_bin]

            if simulation is None or nsim > 1:
                stack_mean = np.nanmean(pm, axis=0)
                stack_err = np.nanmean(pe, axis=0)
                if not np.isfinite(stack_err).any():
                    stack_err = np.zeros_like(stack_mean)
                label_data = "Data" if j_bin == 0 else None
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
                label_data = "Data" if j_bin == 0 else None
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
                label_rand = "Random" if j_bin == 0 else None
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

            ref_idx = np.where(have_bin)[0][0]
            bin_ref = sims[ref_idx][1].bins[j_bin]
            lo, hi = bin_ref.lo, bin_ref.hi
            if hi is None:
                label = rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}}$"
            else:
                label = rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}} < {hi:.2f}$"
            ax.text(
                0.5,
                1.02,
                label,
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
        profile_axes[0].legend(frameon=False, loc="upper right")
        for ax_sig in sig_axes:
            ax_sig.set_xlabel(theta_label)
        sig_axes[0].set_ylabel(r"$p_{\rm KS}(\theta)$")

    plt.close()

    return fig, axes


def plot_cutout_maps(
    res,
    nbins=3,
    mode="stack",
    simulation=None,
    circle_color="k",
    circle_lw=1.0,
    cmap="coolwarm",
    theta500_scale=5.0,
):
    r"""
    Plot stacked or per-simulation cutout mean maps for selected bins.

    Parameters
    ----------
    theta500_scale : float, optional
        Scaling applied to the angular normalization relative to theta_500c.
    """

    mode = mode.lower()
    if mode not in {"stack", "single"}:
        raise ValueError("mode must be 'stack' or 'single'.")

    all_sims = list(res.iter_simulations())
    if not all_sims:
        raise ValueError("No simulations available.")

    max_bins = max((len(sim.bins) for _, sim in all_sims), default=0)
    if max_bins == 0:
        raise ValueError("No cutouts found in simulations.")

    nbins_plot = min(max_bins, nbins)
    if nbins_plot <= 0:
        raise ValueError("nbins must be positive.")

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

    sim_idx = None
    if mode == "single":
        if simulation is None:
            raise ValueError("Provide simulation when mode='single'.")
        if isinstance(simulation, int):
            if simulation < 0 or simulation >= nsim:
                raise IndexError(
                    f"Simulation index {simulation} outside [0, {nsim - 1}]."
                )
            sim_idx = simulation
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
            sim_idx = match

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
            1,
            nbins_plot,
            figsize=(4.8 * nbins_plot, 4.3),
            sharey=True,
        )
        if nbins_plot == 1:
            axes = [axes]

        for j_bin, ax in enumerate(axes[:nbins_plot]):
            if mode == "stack":
                have = ~np.isnan(cutouts[:, j_bin, 0, 0])
                if not np.any(have):
                    ax.set_visible(False)
                    continue
                img = np.nanmean(cutouts[have, j_bin], axis=0)
            else:
                if np.isnan(cutouts[sim_idx, j_bin, 0, 0]):
                    ax.set_visible(False)
                    continue
                img = cutouts[sim_idx, j_bin]

            scale = np.nanpercentile(np.abs(img), 99.0)
            if not np.isfinite(scale) or scale == 0:
                scale = np.nanmax(np.abs(img))
            if not np.isfinite(scale) or scale == 0:
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = -scale, scale

            extent = extents[j_bin]
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

            if extent is not None:
                circle = Circle(
                    (0.0, 0.0),
                    radius=1.0,
                    edgecolor=circle_color,
                    facecolor="none",
                    lw=circle_lw,
                    ls="--",
                )
            else:
                cy = (ny - 1) / 2.0
                cx = (nx - 1) / 2.0
                radius = min(nx, ny) / 2.0
                circle = Circle(
                    (cx, cy),
                    radius=radius,
                    edgecolor=circle_color,
                    facecolor="none",
                    lw=circle_lw,
                    ls="--",
                )
            ax.add_patch(circle)

            bin_ref = None
            for _, sim in all_sims:
                if j_bin < len(sim.bins):
                    bin_ref = sim.bins[j_bin]
                    break
            if bin_ref is not None:
                lo, hi = bin_ref.lo, bin_ref.hi
                if hi is None:
                    label = rf"$\log M_{{200\mathrm{{c}}}} > {lo:.2f}$"
                else:
                    label = (
                        rf"${lo:.2f} < \log M_{{200\mathrm{{c}}}} "
                        rf"< {hi:.2f}$"
                    )
                ax.text(
                    0.5,
                    1.02,
                    label,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            ax.set_xlabel(theta_x_label)
            if j_bin == 0:
                ax.set_ylabel(theta_y_label)
            else:
                ax.set_ylabel("")

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(r"$y$", fontsize=9)

        fig.tight_layout()

    plt.close()

    return fig, axes


__all__ = [
    "plot_stacked_profiles",
    "plot_cutout_maps",
]
