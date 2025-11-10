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
from astropy.coordinates import SkyCoord


def plot_mass_y_scaling(matches, obs_clusters=None):
    """
    Plot log10(M) vs log10(Y_corrected) for matched clusters.

    Parameters
    ----------
    matches : sequence
        Iterable of matches, where each entry is a tuple whose first element
        is a HaloAssociation with per-halo masses and Y_corrected data.
    obs_clusters : ObservedClusterCatalogue, optional
        Catalogue containing observed clusters. If provided, each data point
        is annotated with the corresponding cluster name.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis with the scaling plot.
    """
    if obs_clusters is not None and len(obs_clusters) != len(matches):
        raise ValueError(
            "obs_clusters and matches must have the same length."
        )

    x, xerr = [], []
    y, yerr = [], []
    names = [] if obs_clusters is not None else None

    for idx, match in enumerate(matches):
        if match is None:
            continue

        if isinstance(match, (tuple, list)):
            if not match:
                continue
            assoc = match[0]
        else:
            assoc = match

        if assoc is None:
            continue

        if (
            assoc.optional_data is None
            or "Y_corrected" not in assoc.optional_data
        ):
            raise KeyError(
                "Association missing 'Y_corrected'. "
                "Run measure_mass_matched_cluster first."
            )

        masses = np.asarray(assoc.masses, dtype=float)
        y_corr = np.asarray(assoc.optional_data["Y_corrected"], dtype=float)

        mask = (
            np.isfinite(masses)
            & np.isfinite(y_corr)
            & (masses > 0)
            & (y_corr > 0)
        )
        if not np.any(mask):
            continue

        logM = np.log10(masses[mask])
        logY = np.log10(y_corr[mask])

        x.append(np.mean(logM))
        xerr.append(np.std(logM))
        y.append(np.mean(logY))
        yerr.append(np.std(logY))
        if obs_clusters is not None:
            names.append(obs_clusters.names[idx])

    if not x:
        raise ValueError("No valid mass-Y pairs to plot.")

    x = np.asarray(x)
    xerr = np.asarray(xerr)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(xerr)
        & np.isfinite(yerr)
    )
    if not np.any(mask):
        raise ValueError("All mass-Y entries are non-finite.")

    x = x[mask]
    xerr = xerr[mask]
    y = y[mask]
    yerr = yerr[mask]
    if names is not None:
        names = np.asarray(names, dtype=object)[mask]

    if not np.any((xerr > 0) & (yerr > 0)):
        raise ValueError("Mass or Y scatter is zero for all associations.")

    slope = 5.0 / 3.0
    x0 = np.nanmedian(x)
    y0 = np.nanmedian(y)

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = y0 + slope * (x_line - x0)

    with plt.style.context("science"):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color="C0",
            label="Data",
        )
        ax.plot(
            x_line,
            y_line,
            "r--",
            label=r"$Y \propto M^{5/3}$",
        )
        ax.set_xlabel(r"$\log_{10}(M_{500\mathrm{c}}\,[M_\odot])$")
        ax.set_ylabel(r"$\log_{10}(Y_{500})$")
        ax.legend()
        if names is not None:
            for xi, yi, label in zip(x, y, names):
                ax.annotate(
                    label,
                    xy=(xi, yi),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
        fig.tight_layout()

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
        if obs_pos is None:
            raise ValueError(
                "obs_pos must be provided when association is given."
            )
        r, ell_halos, b_halos = association.to_galactic_angular(obs_pos)
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


__all__ = [
    "tangent_offsets_arcmin",
    "plot_mass_y_scaling",
    "plot_cluster_cutout",
    "plot_observed_cluster_cutout",
    "plot_observed_cluster_grid",
]
