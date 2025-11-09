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
"""Mass-matched photometry for individual halo associations."""

from __future__ import annotations

import numpy as np

from ..utils import E_z

ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


def measure_mass_matched_cluster(
    obs_cluster,
    assoc,
    obs_pos,
    Om,
    source_scale=5.0,
    bkg_inner_scale=5.0,
):
    r"""
    Measure background-subtracted Y and self-similar corrections.

    Parameters
    ----------
    obs_cluster
        ObservedCluster with map_fit populated (cutout + extent).
    assoc
        HaloAssociation matched to the observed cluster. Must contain
        per-halo `Group_R_Crit500` entries in ``optional_data``.
    obs_pos : array-like, shape (3,)
        Observer position in the same coordinate system as ``assoc`` positions.
    Om : float
        Present-day matter density parameter (flat LCDM, h=1).
    source_scale : float, optional
        Source aperture radius in units of ``theta_500``. Default: 5.
    bkg_inner_scale : float, optional
        Inner radius of the background annulus in units of ``theta_500``.
        Default: 5.
    Returns
    -------
    ndarray
        Array of per-halo ``Y_corrected`` values, also stored in
        ``assoc.optional_data['Y_corrected']``.
    """
    if obs_cluster.map_fit is None:
        raise ValueError("Observed cluster is missing map_fit information.")

    map_fit = obs_cluster.map_fit
    required_keys = {
        "radii_arcmin",
        "signal_profile",
        "background_profile",
        "radii_background_arcmin",
    }
    missing = required_keys - map_fit.keys()
    if missing:
        raise KeyError(
            f"map_fit missing required keys: {sorted(missing)}"
        )

    if (
        assoc.optional_data is None
        or "Group_R_Crit500" not in assoc.optional_data
    ):
        raise KeyError(
            "Association optional_data must include 'Group_R_Crit500'."
        )

    r500 = np.asarray(assoc.optional_data["Group_R_Crit500"], dtype=float)
    da = np.asarray(assoc.to_da(obs_pos, Om), dtype=float)
    z_arr = np.asarray(assoc.to_z(obs_pos, Om), dtype=float)

    valid = (
        np.isfinite(r500)
        & np.isfinite(da)
        & np.isfinite(z_arr)
        & (r500 > 0)
        & (da > 0)
        & (z_arr >= 0)
    )
    if not np.all(valid):
        raise ValueError("Halos with non-finite R500, D_A, or z detected.")

    theta500_arcmin = np.rad2deg(np.arctan(r500 / da)) * 60.0

    source_radius = source_scale * theta500_arcmin
    bkg_radius = bkg_inner_scale * theta500_arcmin

    radii_signal = np.asarray(map_fit["radii_arcmin"], dtype=float)
    profile_signal = np.asarray(map_fit["signal_profile"], dtype=float)
    signal_value = _interp_profile(
        radii_signal, profile_signal, source_radius
    )

    radii_background = np.asarray(
        map_fit["radii_background_arcmin"], dtype=float)
    profile_background = np.asarray(map_fit["background_profile"], dtype=float)
    y_bkg = _interp_profile(
        radii_background, profile_background, bkg_radius
    )

    if not np.all(np.isfinite(signal_value) & np.isfinite(y_bkg)):
        raise ValueError("Profiles cannot be evaluated at requested radii.")

    source_radius_rad = source_radius * ARCMIN_TO_RAD
    omega_aperture = np.pi * source_radius_rad**2
    y_sz = omega_aperture * (signal_value - y_bkg)

    Y_500 = y_sz * (da**2)
    Ez = E_z(z_arr, Om)
    Y_corrected = Y_500 * Ez**(-2.0 / 3.0)

    if assoc.optional_data is None:
        assoc.optional_data = {}
    assoc.optional_data["Y_corrected"] = Y_corrected

    return Y_corrected


def _interp_profile(radii_arcmin, values, target_arcmin):
    radii = np.asarray(radii_arcmin, dtype=float)
    vals = np.asarray(values, dtype=float)
    targets = np.asarray(target_arcmin, dtype=float)

    if radii.ndim != 1 or vals.ndim != 1 or radii.size != vals.size:
        raise ValueError("Profiles must be 1D arrays of equal length.")
    if not np.all(np.isfinite(radii)):
        raise ValueError("radii_arcmin contains non-finite values.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("Profile values contain non-finite entries.")
    if targets.ndim != 1:
        targets = targets.ravel()

    out = np.full_like(targets, np.nan, dtype=float)
    mask = (
        np.isfinite(targets)
        & (targets >= radii[0])
        & (targets <= radii[-1])
    )
    if np.any(mask):
        out[mask] = np.interp(targets[mask], radii, vals)

    if np.isscalar(target_arcmin):
        return float(out[0])
    return out
