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
                        output_path=None, dpi=450, cmap=None):
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

        ax.imshow(
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

        fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=dpi)

    return fig, ax


__all__ = [
    "tangent_offsets_arcmin",
    "plot_cluster_cutout",
]
