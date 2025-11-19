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
"""Plotting helpers for cluster cutouts."""

import astropy.units as u
import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord

from cmbo.utils import E_z

ARCMIN2_TO_SR = (np.pi / (180.0 * 60.0))**2
PLANCK_H = 0.7
LOG10 = np.log(10.0)


def extract_match_field(obs_clusters, match_attr, key):
    """
    Return an array with a field from Planck/MCXC/eRASS matches.

    Parameters
    ----------
    obs_clusters
        Iterable of ObservedCluster objects (or catalogue) with match data.
    match_attr : str
        Attribute name on ``ObservedCluster`` storing the match dictionary
        (e.g. ``\"planck_match\"``).
    key : str
        Field to extract from the match dictionary (e.g. ``\"M500\"``).

    Returns
    -------
    numpy.ndarray
        One-dimensional float array; entries are NaN when the requested match
        is missing for a cluster or when the key is absent/invalid.
    """
    if obs_clusters is None:
        raise ValueError("obs_clusters cannot be None.")
    clusters = list(obs_clusters)
    values = np.full(len(clusters), np.nan, dtype=float)
    for idx, cluster in enumerate(clusters):
        match = getattr(cluster, match_attr, None)
        if not match:
            continue
        if isinstance(match, dict):
            value = match.get(key, np.nan)
        else:
            value = getattr(match, key, np.nan)
        try:
            values[idx] = float(value)
        except (TypeError, ValueError):
            values[idx] = np.nan
    return values


def match_angular_separation(obs_clusters, match_attr_a, match_attr_b):
    """
    Angular offsets between two match catalogues.

    Returns an array (arcmin) of separations between the coordinates stored in
    ``match_attr_a`` and ``match_attr_b`` for each ObservedCluster. Missing
    matches or invalid coordinates yield NaN entries.
    """
    if obs_clusters is None:
        raise ValueError("obs_clusters cannot be None.")

    def _get(match, key):
        if match is None:
            return np.nan
        if isinstance(match, dict):
            return match.get(key, np.nan)
        return getattr(match, key, np.nan)

    clusters = list(obs_clusters)
    offsets = np.full(len(clusters), np.nan, dtype=float)
    for idx, cluster in enumerate(clusters):
        match_a = getattr(cluster, match_attr_a, None)
        match_b = getattr(cluster, match_attr_b, None)
        if not match_a or not match_b:
            continue
        ra_a = float(_get(match_a, "RA"))
        dec_a = float(_get(match_a, "DEC"))
        ra_b = float(_get(match_b, "RA"))
        dec_b = float(_get(match_b, "DEC"))
        if not (
            np.isfinite(ra_a) and np.isfinite(dec_a)
            and np.isfinite(ra_b) and np.isfinite(dec_b)
        ):
            continue
        coord_a = SkyCoord(ra_a * u.deg, dec_a * u.deg, frame="icrs")
        coord_b = SkyCoord(ra_b * u.deg, dec_b * u.deg, frame="icrs")
        offsets[idx] = coord_a.separation(coord_b).to_value(u.arcmin)
    return offsets


def estimate_mass_ratio_log(log_a, log_err_a, log_b, log_err_b,
                            n_samples=50000, rng=None):
    """
    Return Monte-Carlo estimate of <M_B / M_A> using log-normal sampling.
    """
    log_a = np.asarray(log_a, dtype=float)
    log_b = np.asarray(log_b, dtype=float)
    err_a = np.clip(np.asarray(log_err_a, dtype=float), 1e-6, None)
    err_b = np.clip(np.asarray(log_err_b, dtype=float), 1e-6, None)

    if not (
        log_a.shape == log_b.shape
        == err_a.shape
        == err_b.shape
    ):
        raise ValueError(
            "log_a/log_b and their errors must share the same shape.")

    if rng is None:
        rng = np.random.default_rng()

    draws_a = rng.normal(
        log_a[:, None],
        err_a[:, None],
        size=(log_a.size, n_samples),
    )
    draws_b = rng.normal(
        log_b[:, None],
        err_b[:, None],
        size=(log_b.size, n_samples),
    )
    log_ratio = draws_b - draws_a
    ratios = 10.0 ** log_ratio
    sample_means = ratios.mean(axis=0)
    return sample_means.mean(), sample_means.std(ddof=1)


def plot_mass_y_scaling(matches, obs_clusters, Om, sim_label="Manticore"):
    """
    Plot log10(M) vs log10(E(z)^(-2/3) Y500 D_A^2) and the Planck vs
    Manticore mass comparison beneath it.

    Parameters
    ----------
    matches
        Iterable whose first element per entry is the matched HaloAssociation.
    obs_clusters
        ObservedClusterCatalogue with ``planck_match`` metadata.
    Om
        Matter density parameter for flat LCDM (h = 1).
    sim_label : str, optional
        Name describing the association-based masses (default: "Manticore").

    Returns
    -------
    fig, axes
        Figure and axes array (upper: Y scaling, lower: mass comparison).
    data : dict
        Dictionary containing the left panel data: x, xerr, y, yerr.
    """
    if obs_clusters is None:
        raise ValueError("obs_clusters must be provided.")
    if len(obs_clusters) != len(matches):
        raise ValueError("obs_clusters and matches must have the same length.")
    if not np.isfinite(Om):
        raise ValueError("Om must be finite.")

    cosmo = FlatLambdaCDM(H0=70, Om0=float(Om))

    x_vals, x_errs = [], []
    y_vals, y_errs = [], []
    y_mass_vals = []
    y_mass_err_low = []
    y_mass_err_up = []
    labels = []

    for cluster, match in zip(obs_clusters, matches):
        if match is None:
            continue

        planck_match = getattr(cluster, "planck_match", None)
        if not planck_match:
            continue

        y5r500_arcmin2 = float(planck_match.get("y5r500", np.nan))
        y5r500_err_arcmin2 = float(planck_match.get("y5r500_err", np.nan))
        z_planck = float(planck_match.get("redshift", np.nan))

        # Convert Y5R500 to Y500 using spherical profile assumption
        y_arcmin2 = y5r500_arcmin2 / 1.81 * 1e-3
        y_err_arcmin2 = y5r500_err_arcmin2 / 1.81 * 1e-3

        if not (
            np.isfinite(y_arcmin2)
            and np.isfinite(y_err_arcmin2)
            and np.isfinite(z_planck)
        ):
            continue
        if y_arcmin2 <= 0 or y_err_arcmin2 <= 0 or z_planck < 0:
            continue

        if isinstance(match, (tuple, list)):
            if not match:
                continue
            assoc = match[0]
        else:
            assoc = match

        masses = getattr(assoc, "masses", None)
        if masses is None:
            continue

        masses = np.asarray(masses, dtype=float)
        mask = np.isfinite(masses) & (masses > 0)
        if not np.any(mask):
            continue

        log_mass = np.log10(masses[mask])

        da_mpc = cosmo.angular_diameter_distance(z_planck).value
        Ez = E_z(z_planck, Om)
        conversion = ARCMIN2_TO_SR * (da_mpc**2)
        y_phys = y_arcmin2 * conversion
        y_phys_err = y_err_arcmin2 * conversion
        Ez_factor = Ez**(-2.0 / 3.0)
        y_scaled = y_phys * Ez_factor
        y_scaled_err = y_phys_err * Ez_factor

        msz = float(planck_match.get("M500", np.nan))
        msz_err = float(planck_match.get("M500_err", np.nan))

        if not (
            np.isfinite(y_phys)
            and np.isfinite(y_phys_err)
            and np.isfinite(Ez_factor)
            and np.isfinite(msz)
            and np.isfinite(msz_err)
        ):
            continue

        if (
            y_scaled <= 0
            or y_scaled_err <= 0
            or msz <= 0
            or msz_err <= 0
        ):
            continue

        msz_lower = msz - msz_err
        msz_upper = msz + msz_err
        if msz_lower <= 0 or msz_upper <= 0:
            continue

        mass_scale = PLANCK_H
        log_msz = np.log10(msz * mass_scale)
        err_low_log = (np.log10(msz * mass_scale)
                       - np.log10(msz_lower * mass_scale))
        err_up_log = (np.log10(msz_upper * mass_scale)
                      - np.log10(msz * mass_scale))

        x_vals.append(np.mean(log_mass))
        x_errs.append(np.std(log_mass))
        y_vals.append(np.log10(y_scaled))
        y_errs.append(y_scaled_err / (y_scaled * LOG10))
        y_mass_vals.append(log_msz)
        y_mass_err_low.append(err_low_log)
        y_mass_err_up.append(err_up_log)
        labels.append(cluster.name)

    if not x_vals:
        raise ValueError("No matched clusters with finite Planck data.")

    x = np.asarray(x_vals)
    xerr = np.asarray(x_errs)
    y = np.asarray(y_vals)
    yerr = np.asarray(y_errs)
    y_mass = np.asarray(y_mass_vals)
    y_mass_err_low = np.asarray(y_mass_err_low)
    y_mass_err_up = np.asarray(y_mass_err_up)
    labels = np.asarray(labels, dtype=object)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(xerr)
        & np.isfinite(yerr)
        & np.isfinite(y_mass)
        & np.isfinite(y_mass_err_low)
        & np.isfinite(y_mass_err_up)
    )
    x = x[mask]
    xerr = xerr[mask]
    y = y[mask]
    yerr = yerr[mask]
    y_mass = y_mass[mask]
    y_mass_err_low = y_mass_err_low[mask]
    y_mass_err_up = y_mass_err_up[mask]
    labels = labels[mask]

    if x.size == 0:
        raise ValueError("All cluster entries failed the quality cuts.")

    slope = 5.0 / 3.0

    sim_label_tex = sim_label.replace("_", r"\_").replace(" ", r"\ ")

    with plt.style.context("science"):
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 4), constrained_layout=True
        )
        ax = axes[0]
        ax_mass = axes[1]

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color="C0",
        )
        anchor_y = (np.nanmedian(x), np.nanmedian(y))
        ax.axline(
            anchor_y,
            slope=slope,
            color="k",
            linestyle="--",
            label=r"$E(z)^{-2/3}Y_{500}D_A^2 \propto M^{5/3}$",
        )
        ax.set_xlabel(r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]$")
        ax.set_ylabel(r"$\log E(z)^{-2/3}Y_{500}D_A^2\,[\mathrm{Mpc}^2]$")  # noqa
        ax.legend(loc="lower right")
        for xi, yi, name in zip(x, y, labels):
            ax.annotate(
                name,
                xy=(xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax_mass.errorbar(
            x,
            y_mass,
            xerr=xerr,
            yerr=np.vstack((y_mass_err_low, y_mass_err_up)),
            fmt="o",
            color="C0",
        )
        anchor_mass = (np.nanmedian(x), np.nanmedian(x))
        ax_mass.axline(
            anchor_mass,
            slope=1.0,
            color="k",
            linestyle="--",
            label=rf"$M_{{500}}^{{\mathrm{{Planck}}}} = M_{{500}}^{{\mathrm{{{sim_label_tex}}}}}$",  # noqa
        )
        ax_mass.set_xlabel(
            r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]\ (\mathrm{" + sim_label_tex + "})$"  # noqa
        )
        ax_mass.set_ylabel(
            r"$\log M_{500}^{\mathrm{Planck}}\,[h^{-1}M_\odot]$")
        ax_mass.legend(loc="lower right")
        for xi, yi, name in zip(x, y_mass, labels):
            ax_mass.annotate(
                name,
                xy=(xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        y_mass_err_sym = 0.5 * (y_mass_err_low + y_mass_err_up)
        ratio_mean, ratio_std = estimate_mass_ratio_log(
            x, xerr, y_mass, y_mass_err_sym
        )
        ratio_label = (
            rf"$\langle M_{{\mathrm{{Planck}}}} / M_{{\mathrm{{{sim_label_tex}}}}} \rangle = "  # noqa
            f"{ratio_mean:.2f} \\pm {ratio_std:.2f}$"
        )
        ax_mass.text(
            0.02,
            0.95,
            ratio_label,
            transform=ax_mass.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "0.8",
                "alpha": 0.75,
            },
        )

    plt.close()

    return fig, axes


def plot_match_mass_comparison(
    matches, obs_clusters, match_attr="planck_match", field="M500",
    field_err="M500_err", mass_key="masses", sim_label="Manticore",
):
    """
    Compare simulation mass estimates to external catalogue masses.

    Parameters
    ----------
    matches
        Iterable (same length as ``obs_clusters``) with HaloAssociation data.
    obs_clusters
        ObservedClusterCatalogue containing Planck/MCXC/eRASS matches.
    match_attr : str, optional
        Name of the match attribute to compare against (default: planck).
    field : str, optional
        Key within the match dict supplying the mass estimate (default: M500).
    field_err : str, optional
        Key supplying the symmetric uncertainty (default: M500_err). Set to
        ``None`` to skip match errors.
    mass_key : str, optional
        Attribute on association providing the simulation masses (default:
        ``masses``). If it is a sequence, its mean/std are used.
    sim_label : str, optional
        Label describing the simulation masses (default: "Manticore").
    """
    if obs_clusters is None or matches is None:
        raise ValueError("obs_clusters and matches must be provided.")
    if len(obs_clusters) != len(matches):
        raise ValueError("obs_clusters and matches must have equal length.")

    log_mass_sim = []
    log_mass_sim_err = []
    log_mass_match = []
    log_mass_match_err = []
    cluster_names = []

    for cluster, assoc in zip(obs_clusters, matches):
        if assoc is None:
            continue
        if isinstance(assoc, (tuple, list)):
            assoc = assoc[0] if assoc else None
        if assoc is None:
            continue
        masses = getattr(assoc, mass_key, None)
        if masses is None:
            continue
        masses = np.asarray(masses, dtype=float)
        mask = np.isfinite(masses) & (masses > 0)
        if not np.any(mask):
            continue
        log_masses = np.log10(masses[mask])
        mean_log_mass = float(np.mean(log_masses))
        std_log_mass = float(np.std(log_masses))
        match = getattr(cluster, match_attr, None)
        if not match:
            continue
        match_mass = match.get(field, np.nan)
        match_err = match.get(field_err, np.nan) if field_err else np.nan
        if not np.isfinite(match_mass) or match_mass <= 0:
            continue
        log_match_mass = float(np.log10(match_mass * PLANCK_H))
        log_match_err = np.nan
        if field_err and np.isfinite(match_err) and match_err > 0:
            log_match_err = float(match_err / (match_mass * LOG10))
        log_mass_sim.append(mean_log_mass)
        log_mass_sim_err.append(std_log_mass)
        log_mass_match.append(log_match_mass)
        log_mass_match_err.append(log_match_err)
        cluster_names.append(cluster.name)

    if not log_mass_sim:
        raise ValueError("No overlapping clusters with finite masses.")

    log_mass_sim = np.asarray(log_mass_sim, dtype=float)
    log_mass_sim_err = np.asarray(log_mass_sim_err, dtype=float)
    log_mass_match = np.asarray(log_mass_match, dtype=float)
    log_mass_match_err = np.asarray(log_mass_match_err, dtype=float)
    cluster_names = np.asarray(cluster_names, dtype=object)

    mask = (
        np.isfinite(log_mass_sim)
        & np.isfinite(log_mass_match)
        & np.isfinite(log_mass_sim_err)
    )
    log_mass_sim = log_mass_sim[mask]
    log_mass_sim_err = log_mass_sim_err[mask]
    log_mass_match = log_mass_match[mask]
    log_mass_match_err = log_mass_match_err[mask]
    cluster_names = cluster_names[mask]

    if log_mass_sim.size == 0:
        raise ValueError("All cluster entries failed the mass quality cuts.")

    sim_label_tex = sim_label.replace("_", r"\_").replace(" ", r"\ ")
    match_label = match_attr.replace("_match", "").replace("_", r"\_")

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.errorbar(
            log_mass_sim,
            log_mass_match,
            xerr=log_mass_sim_err,
            yerr=log_mass_match_err if field_err else None,
            fmt="o",
            color="C0",
        )
        anchor = (np.nanmedian(log_mass_sim), np.nanmedian(log_mass_sim))
        ax.axline(anchor, slope=1.0, color="k", linestyle="--", label="1:1")
        ax.set_xlabel(
            r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]\ (\mathrm{"
            + sim_label_tex + "})$"
        )
        ax.set_ylabel(
            r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]\ (\mathrm{"
            + match_label + "})$"
        )
        for xi, yi, name in zip(log_mass_sim, log_mass_match, cluster_names):
            ax.annotate(
                name,
                xy=(xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.legend(loc="lower right")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lo = min(xmin, ymin)
        hi = max(xmax, ymax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    plt.close()
    return fig, ax


def tangent_offsets_arcmin(ell_deg, b_deg, ellc_deg, bc_deg):
    """
    Compute tangent-plane offsets in arcminutes from a Galactic center.

    Parameters
    ----------
    ell_deg
        Galactic longitude(s) in degrees.
    b_deg
        Galactic latitude(s) in degrees.
    ellc_deg
        Center Galactic longitude in degrees.
    bc_deg
        Center Galactic latitude in degrees.

    Returns
    -------
    x_arcmin : ndarray
        Longitude offset in arcminutes.
    y_arcmin : ndarray
        Latitude offset in arcminutes.
    """
    c = SkyCoord(
        l=np.asarray(ell_deg) * u.deg,
        b=np.asarray(b_deg) * u.deg,
        frame="galactic",
    )
    ctr = SkyCoord(
        l=float(ellc_deg) * u.deg,
        b=float(bc_deg) * u.deg,
        frame="galactic",
    )
    off = c.transform_to(ctr.skyoffset_frame())
    x = off.lon.wrap_at(180 * u.deg).to_value(u.arcmin)
    y = off.lat.to_value(u.arcmin)
    return x, y


def plot_cluster_cutout(cutout, extent, ell, b, ellc, bc,
                        output_path=None, dpi=450, cmap=None,
                        cbar_label=r"$y$"):
    """
    Plot a cluster cutout with halo positions overlaid.

    Parameters
    ----------
    cutout
        2D array of tSZ signal or other map data.
    extent
        Image extent in arcminutes [xmin, xmax, ymin, ymax].
    ell
        Galactic longitude(s) of halos in degrees.
    b
        Galactic latitude(s) of halos in degrees.
    ellc
        Center Galactic longitude in degrees.
    bc
        Center Galactic latitude in degrees.
    output_path : str or pathlib.Path, optional
        When provided, save the figure to this location.
    dpi : int, optional
        Figure DPI used when saving, default is 450.
    cmap : str, optional
        Colormap name, default is cmasher.fusion_r.
    cbar_label : str, optional
        Label for the colorbar, default is "$y$" (Compton-y parameter).

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    if cmap is None:
        cmap = cmr.fusion_r

    x_arcmin, y_arcmin = tangent_offsets_arcmin(ell, b, ellc, bc)

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            cutout,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
        )
        ax.scatter(
            x_arcmin,
            y_arcmin,
            s=16,
            c="w",
            edgecolor="k",
            lw=0.5,
            zorder=3,
        )
        ax.scatter(0.0, 0.0, marker="+", c="r", s=64, zorder=4)

        ax.set_xlabel(r"$\xi\,[\mathrm{arcmin}]$")
        ax.set_ylabel(r"$\eta\,[\mathrm{arcmin}]$")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
        fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi)

    return fig, ax


def _plot_cutout_on_axis(ax, obs_cluster, association=None,
                         obs_pos=None, zoom_arcmin=None,
                         cmap=None, show_legend=True,
                         show_xlabel=True, show_ylabel=True):
    """
    Plot cluster cutout on a given axis (helper function).

    Parameters
    ----------
    ax
        Matplotlib axis to plot on.
    obs_cluster
        ObservedCluster instance with map_fit populated.
    association
        Optional HaloAssociation instance.
    obs_pos
        Observer position (3D array). Required if association provided.
    zoom_arcmin
        Optional zoom level in arcminutes.
    cmap
        Colormap name, default is cmasher.fusion_r.
    show_legend
        Whether to show the legend. Default: True.
    show_xlabel
        Whether to show x-axis label. Default: True.
    show_ylabel
        Whether to show y-axis label. Default: True.

    Returns
    -------
    im
        The image object from imshow.
    """
    if cmap is None:
        cmap = cmr.fusion_r

    cutout = obs_cluster.map_fit['cutout']
    extent = obs_cluster.map_fit['extent']
    ellc = obs_cluster.map_fit['ell']
    bc = obs_cluster.map_fit['b']

    # Get original observed position
    ell_obs, b_obs = obs_cluster.galactic_coordinates

    # Convert to tangent plane offsets
    x_obs, y_obs = tangent_offsets_arcmin(ell_obs, b_obs, ellc, bc)

    im = ax.imshow(
        cutout,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
    )

    # Mark the refined center (at origin)
    ax.scatter(0.0, 0.0, marker="o", c="r", s=48, zorder=4,
               edgecolor="k", lw=1, label="Refined center")

    # Mark the original observed position
    ax.scatter(x_obs, y_obs, marker="o", c="green", s=48, zorder=4,
               edgecolor="k", lw=1, label="Observed position")

    # Mark association halo positions if provided
    if association is not None:
        gal = association.galactic_angular
        ell_halos, b_halos = gal[:, 0], gal[:, 1]
        x_halos, y_halos = tangent_offsets_arcmin(
            ell_halos, b_halos, ellc, bc
        )
        ax.scatter(x_halos, y_halos, marker="o", c="w", s=16,
                   edgecolor="k", lw=0.5, zorder=3,
                   label="Association")

    if show_xlabel:
        ax.set_xlabel(r"$\xi\,[\mathrm{arcmin}]$")
    if show_ylabel:
        ax.set_ylabel(r"$\eta\,[\mathrm{arcmin}]$")
    # Replace spaces with LaTeX spaces for proper rendering
    name_latex = obs_cluster.name.replace(" ", r"\ ")
    ax.set_title(rf"$\mathrm{{{name_latex}}}$")

    if show_legend:
        ax.legend(loc="upper right", frameon=True, fancybox=True,
                  shadow=True, framealpha=0.95)

    # Apply zoom if requested
    if zoom_arcmin is not None:
        ax.set_xlim(-zoom_arcmin, zoom_arcmin)
        ax.set_ylim(-zoom_arcmin, zoom_arcmin)

    return im


def plot_observed_cluster_cutout(obs_cluster, association=None,
                                 obs_pos=None, zoom_arcmin=None,
                                 output_path=None, dpi=450,
                                 cmap=None, cbar_label=r"$y$"):
    """
    Plot cutout for an observed cluster with original position marked.

    Parameters
    ----------
    obs_cluster
        ObservedCluster instance with map_fit populated.
    association
        Optional HaloAssociation instance. If provided, plots halo
        positions from the association on the cutout.
    obs_pos
        Observer position (3D array). Required if association is
        provided. Used to convert association positions to galactic
        coordinates.
    zoom_arcmin : float, optional
        If provided, zoom in by setting axis limits to
        [-zoom_arcmin, zoom_arcmin] for both x and y axes.
        Default: None (uses full cutout extent).
    output_path : str or pathlib.Path, optional
        When provided, save the figure to this location.
    dpi : int, optional
        Figure DPI used when saving, default is 450.
    cmap : str, optional
        Colormap name, default is cmasher.fusion_r.
    cbar_label : str, optional
        Label for the colorbar, default is "$y$" (Compton-y parameter).

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    if obs_cluster.map_fit is None:
        raise ValueError(
            f"Cluster {obs_cluster.name} has no map_fit. "
            "Run find_centers_observed_clusters first."
        )

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6, 5))

        im = _plot_cutout_on_axis(
            ax, obs_cluster, association=association, obs_pos=obs_pos,
            zoom_arcmin=zoom_arcmin, cmap=cmap, show_legend=False
        )

        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046, aspect=20)
        cbar.set_label(cbar_label)
        fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi)

    plt.close()

    return fig, ax


def plot_observed_cluster_grid(obs_clusters, matches, boxsize,
                               ncols=4, zoom_arcmin=None,
                               cmap=None, cbar_label=r"$y$",
                               show_legend=False):
    """
    Plot multiple observed clusters in a grid layout.

    Parameters
    ----------
    obs_clusters
        ObservedClusterCatalogue instance with map_fit populated.
    matches
        List of tuples (association, min_pval, distance) or None for
        each cluster. Length must match number of clusters. Clusters
        with None matches are skipped.
    boxsize
        Simulation box size for computing observer position.
    ncols
        Number of columns in the grid. Default: 4.
    zoom_arcmin
        Optional zoom level in arcminutes. Default: None.
    cmap
        Colormap name, default is cmasher.fusion_r.
    cbar_label
        Label for the colorbar, default is "$y$".
    show_legend
        Whether to show legend on each panel. Default: False.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array.
    """
    # Filter out unmatched clusters
    matched_indices = [
        k for k in range(len(obs_clusters))
        if matches[k] is not None
    ]

    n_matched = len(matched_indices)
    if n_matched == 0:
        raise ValueError("No matched clusters to plot.")

    print(f"Plotting {n_matched}/{len(obs_clusters)} matched clusters")
    skipped = [
        k for k in range(len(obs_clusters))
        if matches[k] is None
    ]
    if skipped:
        skipped_names = [obs_clusters.names[k] for k in skipped]
        print(f"Skipping {len(skipped)} unmatched: {skipped_names}")

    nrows = int(np.ceil(n_matched / ncols))

    obs_pos = np.full((3,), boxsize / 2)

    with plt.style.context("science"):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False
        )

        # Increase spacing between subplots
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        for plot_idx, cluster_idx in enumerate(matched_indices):
            row = plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]

            obs = obs_clusters[cluster_idx]
            assoc, min_pval, d = matches[cluster_idx]

            # Determine if labels should be shown
            is_bottom_row = (row == nrows - 1)
            is_leftmost_col = (col == 0)

            # Plot on this axis
            im = _plot_cutout_on_axis(
                ax, obs, association=assoc, obs_pos=obs_pos,
                zoom_arcmin=zoom_arcmin, cmap=cmap,
                show_legend=show_legend,
                show_xlabel=is_bottom_row,
                show_ylabel=is_leftmost_col
            )

            # Add colorbar to this panel (no label)
            fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046, aspect=20)

        # Remove unused subplots
        for plot_idx in range(n_matched, nrows * ncols):
            row = plot_idx // ncols
            col = plot_idx % ncols
            fig.delaxes(axes[row, col])

        # Reduce whitespace
        fig.tight_layout(pad=0.3)

    return fig, axes


def plot_mass_comparison(matches_a, matches_b, obs_clusters,
                         labels=("A", "B")):
    """
    Compare mean masses from two match catalogues.

    Parameters
    ----------
    matches_a, matches_b
        Sequences aligned with obs_clusters. Only entries with valid
        HaloAssociation masses in both are plotted.
    obs_clusters
        ObservedClusterCatalogue providing cluster names.
    labels : tuple of str, optional
        Names for the two data sets (used in axis labels).

    Returns
    -------
    fig, axes
        Figure containing the scatter comparison and a text panel.
    """
    if len(matches_a) != len(matches_b):
        raise ValueError("matches_a and matches_b must have the same length.")
    if len(matches_a) != len(obs_clusters):
        raise ValueError("Matches must align with obs_clusters.")

    def _stats(match):
        assoc = match[0] if isinstance(match, (tuple, list)) else match
        masses = getattr(assoc, "masses", None)
        if masses is None:
            return None
        masses = np.asarray(masses, dtype=float)
        mask = np.isfinite(masses) & (masses > 0)
        if not np.any(mask):
            return None
        sel = masses[mask]
        return np.mean(sel), np.std(sel)

    means_a, stds_a, means_b, stds_b, names = [], [], [], [], []
    for cluster, ma, mb in zip(obs_clusters, matches_a, matches_b):
        if ma is None or mb is None:
            continue
        stats_a = _stats(ma)
        stats_b = _stats(mb)
        if stats_a is None or stats_b is None:
            continue
        means_a.append(stats_a[0])
        stds_a.append(stats_a[1])
        means_b.append(stats_b[0])
        stds_b.append(stats_b[1])
        names.append(cluster.name)

    if not means_a:
        raise ValueError("No overlapping clusters with valid masses.")

    means_a = np.array(means_a)
    stds_a = np.array(stds_a)
    means_b = np.array(means_b)
    stds_b = np.array(stds_b)

    log_a = np.log10(means_a)
    log_b = np.log10(means_b)
    log_err_a = stds_a / (means_a * LOG10)
    log_err_b = stds_b / (means_b * LOG10)

    with plt.style.context("science"):
        fig, ax_scatter = plt.subplots(figsize=(6, 4), constrained_layout=True)

        ax_scatter.errorbar(
            log_a,
            log_b,
            xerr=log_err_a,
            yerr=log_err_b,
            fmt="o",
            color="C0",
            alpha=0.85,
        )
        anchor = (np.median(log_a), np.median(log_a))
        ax_scatter.axline(
            anchor,
            slope=1.0,
            color="k",
            linestyle="--",
            label="1:1",
        )
        ax_scatter.set_xlabel(
            r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]\ (\mathrm{" + labels[0] + "})$"  # noqa
        )
        ax_scatter.set_ylabel(
            r"$\log M_{500\mathrm{c}}\,[h^{-1}M_\odot]\ (\mathrm{" + labels[1] + "})$"  # noqa
        )
        for xi, yi, name in zip(log_a, log_b, names):
            ax_scatter.annotate(
                name,
                xy=(xi, yi),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=6,
                bbox={"boxstyle": "round,pad=0.15",
                      "fc": "white", "ec": "none", "alpha": 0.4},
            )
        ax_scatter.legend(loc="lower right")
        ratio_mean, ratio_std = estimate_mass_ratio_log(
            log_a, log_err_a, log_b, log_err_b
        )
        ratio_label = (
            r"$\langle M_{\mathrm{" + labels[1] + r"}} / "
            r"M_{\mathrm{" + labels[0] + r"}} \rangle = "
            f"{ratio_mean:.2f} \\pm {ratio_std:.2f}$"
        )
        ax_scatter.text(
            0.02,
            0.95,
            ratio_label,
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "0.8",
                "alpha": 0.75,
            },
        )

    return fig, ax_scatter


__all__ = [
    "extract_match_field",
    "match_angular_separation",
    "tangent_offsets_arcmin",
    "plot_mass_y_scaling",
    "plot_match_mass_comparison",
    "plot_mass_comparison",
    "plot_cluster_cutout",
    "plot_observed_cluster_cutout",
    "plot_observed_cluster_grid",
]
