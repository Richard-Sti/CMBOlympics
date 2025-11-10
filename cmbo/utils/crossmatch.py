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
"""Angular + redshift cross-match helpers."""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


def crossmatch_planck_catalog(
    obs_clusters,
    planck_catalog,
    max_sep_arcmin=3.0,
    max_delta_z=0.005,
):
    """
    Cross-match observed clusters to the Planck PSZ2 union catalogue.

    Parameters
    ----------
    obs_clusters
        ObservedClusterCatalogue with RA/Dec and cz data.
    planck_catalog : dict
        Output of ``read_Planck_cluster_catalog``.
    max_sep_arcmin : float, optional
        Maximum angular separation allowed between matches.
    max_delta_z : float or None, optional
        Maximum |z_obs - z_planck|. Set to None to skip the redshift check.

    Side effects
    ------------
    Annotates each ``ObservedCluster`` with a ``planck_match`` dictionary
    containing the Planck match metadata. Unmatched clusters receive NaNs
    in that dictionary, and their names are printed to stdout.

    Returns
    -------
    None
    """
    obs_ra = np.array(
        [cluster.ra_deg for cluster in obs_clusters], dtype=float
        )
    obs_dec = np.array(
        [cluster.dec_deg for cluster in obs_clusters], dtype=float
    )
    obs_z = obs_clusters.redshifts

    planck_ra = np.asarray(planck_catalog["ra_deg"], dtype=float)
    planck_dec = np.asarray(planck_catalog["dec_deg"], dtype=float)
    planck_z = np.asarray(planck_catalog.get("redshift"), dtype=float)
    planck_y = np.asarray(planck_catalog["y5r500"], dtype=float)
    planck_yerr = np.asarray(planck_catalog["y5r500_err"], dtype=float)
    planck_msz = np.asarray(planck_catalog["msz"], dtype=float)
    planck_msz_err_up = np.asarray(planck_catalog["msz_err_up"], dtype=float)
    planck_msz_err_low = np.asarray(planck_catalog["msz_err_low"], dtype=float)
    planck_wise_flag = np.asarray(
        planck_catalog.get("wise_flag"), dtype=float
    )
    planck_validation = np.asarray(
        planck_catalog.get("validation"), dtype=float
    )

    obs_coord = SkyCoord(obs_ra * u.deg, obs_dec * u.deg)
    planck_coord = SkyCoord(planck_ra * u.deg, planck_dec * u.deg)

    idx_planck, idx_obs, sep2d, _ = obs_coord.search_around_sky(
        planck_coord,
        max_sep_arcmin * u.arcmin,
    )

    if max_delta_z is None:
        redshift_mask = np.ones_like(idx_obs, dtype=bool)
        dz = np.full(idx_obs.size, np.nan, dtype=float)
    else:
        dz = np.abs(obs_z[idx_obs] - planck_z[idx_planck])
        redshift_mask = np.isfinite(dz) & (dz <= float(max_delta_z))

    valid = redshift_mask
    matches = []
    if np.any(valid):
        obs_names = obs_clusters.names
        planck_names = planck_catalog["name"]
        best_by_obs: dict[int, dict] = {}
        for obs_idx, planck_idx, sep, delta_z in zip(
            idx_obs[valid], idx_planck[valid], sep2d[valid], dz[valid]
        ):
            sep_arcmin = float(sep.to_value(u.arcmin))
            record = {
                "obs_index": int(obs_idx),
                "planck_index": int(planck_idx),
                "obs_name": obs_names[obs_idx],
                "planck_name": planck_names[planck_idx],
                "separation_arcmin": sep_arcmin,
                "delta_z": float(delta_z),
                "y5r500": float(planck_y[planck_idx]),
                "y5r500_err": float(planck_yerr[planck_idx]),
                "msz": float(planck_msz[planck_idx]),
                "msz_err_up": float(planck_msz_err_up[planck_idx]),
                "msz_err_low": float(planck_msz_err_low[planck_idx]),
                "redshift": float(planck_z[planck_idx]),
                "wise_flag": (
                    float(planck_wise_flag[planck_idx])
                    if planck_wise_flag.size else np.nan
                ),
                "validation": (
                    float(planck_validation[planck_idx])
                    if planck_validation.size else np.nan
                ),
            }
            if obs_idx not in best_by_obs or (
                sep_arcmin < best_by_obs[obs_idx]["separation_arcmin"]
            ):
                best_by_obs[obs_idx] = record

        matches = list(best_by_obs.values())

    matched_obs_indices = np.sort(
        np.array([m["obs_index"] for m in matches], dtype=int),
    )

    unmatched_obs = np.setdiff1d(
        np.arange(len(obs_clusters), dtype=int),
        matched_obs_indices,
        assume_unique=True,
    )

    if unmatched_obs.size:
        skipped = [obs_clusters.names[idx] for idx in unmatched_obs]
        print(
            "No Planck match for {n} observed clusters: {names}".format(
                n=unmatched_obs.size,
                names=skipped,
            )
        )

    for idx in unmatched_obs:
        obs_clusters.clusters[idx].planck_match = {
            "planck_index": np.nan,
            "planck_name": None,
            "separation_arcmin": np.nan,
            "delta_z": np.nan,
            "y5r500": np.nan,
            "y5r500_err": np.nan,
            "msz": np.nan,
            "msz_err_up": np.nan,
            "msz_err_low": np.nan,
            "redshift": np.nan,
            "wise_flag": np.nan,
            "validation": np.nan,
        }

    for match in matches:
        obs_idx = match["obs_index"]
        planck_idx = match["planck_index"]
        obs_clusters.clusters[obs_idx].planck_match = {
            "planck_index": planck_idx,
            "planck_name": match["planck_name"],
            "separation_arcmin": match["separation_arcmin"],
            "delta_z": match["delta_z"],
            "y5r500": match["y5r500"],
            "y5r500_err": match["y5r500_err"],
            "msz": match["msz"],
            "msz_err_up": match["msz_err_up"],
            "msz_err_low": match["msz_err_low"],
            "redshift": match["redshift"],
            "wise_flag": match["wise_flag"],
            "validation": match["validation"],
        }

    return None
