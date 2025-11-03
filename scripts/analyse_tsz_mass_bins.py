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
"""Analyse tSZ profiles in mass bins."""

import sys

import cmbolympics
import numpy as np
from cmbolympics.utils import (build_mass_bins,
                               cartesian_icrs_to_galactic_spherical, fprint)
from scipy.stats import ks_2samp

try:
    import tomllib as _toml_loader  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "tomli is required to read the configuration file "
            "(install with 'pip install tomli')."
        ) from exc

tomli = _toml_loader


def load_config(config_path):
    with open(config_path, "rb") as fh:
        config = tomli.load(fh)

    return config


def load_halo_catalogue(cfg):
    which_simulation = cfg["analysis"]["which_simulation"]
    cfg_sim = cfg["halo_catalogues"][which_simulation]

    if which_simulation == "csiborg2":
        reader = cmbolympics.io.SimulationHaloReader(
            cfg_sim["fname"], nsim=cfg_sim["nsim"])
        center = np.full(3, cfg_sim["box_size"] / 2.0, dtype=float)
        fprint(f"Using centre at {center} Mpc/h for {which_simulation}.")

        pos = reader["Coordinates"]
        mass = reader["Group_M_Crit200"]
        rad = reader["Group_R_Crit200"]

        r, ell, b = cartesian_icrs_to_galactic_spherical(pos, center)
    else:
        raise ValueError(f"Unknown simulation '{which_simulation}'")

    theta = np.rad2deg(np.arctan(rad / r)) * 60

    cuts = cfg["halo_cuts"]
    mask = np.isfinite(mass)
    mask &= (r >= cuts["r_min"]) & (r <= cuts["r_max"])
    mask &= (np.abs(b) >= cuts["b_min"])
    mask &= (theta >= cuts["theta_min"]) & (theta <= cuts["theta_max"])
    mask &= (mass >= cuts["mass_min"]) & (mass <= cuts["mass_max"])
    fprint(f"Selected {np.sum(mask)} haloes after applying cuts "
           f"from {mass.size}.")

    return {"r": r[mask],
            "ell": ell[mask],
            "b": b[mask],
            "mass": mass[mask],
            "theta": theta[mask],
            }


def load_profiler(cfg):
    kwargs = cfg["input_map"]
    y_map = cmbolympics.io.read_Planck_comptonSZ(kwargs["signal_map"])
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(
        f"Loaded y-map {kwargs['signal_map']}: mean={mu:.3e}, std={std:.3e}"
        )

    y_map = cmbolympics.utils.smooth_map_gaussian(
        y_map,
        fwhm_arcmin=kwargs["smooth_fwhm"],
    )
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Smoothed y-map with FWHM {kwargs['smooth_fwhm']}: "
           f"mean={mu:.3e}, std={std:.3e}")

    profiler = cmbolympics.corr.PointingEnclosedProfile(
        y_map,
        n_jobs=cfg["analysis"]["n_jobs"],
        fwhm_arcmin=kwargs["smooth_fwhm"],
    )

    radii_stack = np.linspace(
        cfg["analysis"]["stack_rmin"],
        cfg["analysis"]["stack_rmax"],
        cfg["analysis"]["stack_n"],
    )

    return profiler, radii_stack


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)
    fprint(f"loaded config from {config_path}")

    halos = load_halo_catalogue(cfg)

    # Compute aperture size and log-mass.
    halos["aperture"] = cfg["analysis"]["aperture_scale"] * halos["theta"]
    halos["log_mass"] = np.log10(halos["mass"])

    edges, medians = build_mass_bins(
        halos["mass"],
        step=cfg["mass_bins"]["mass_step"],
        top_counts=cfg["mass_bins"]["top_counts"],
        verbose=cfg["mass_bins"]["verbose_bins"],
    )

    profiler, radii_stack = load_profiler(cfg)
    fprint("initialised profile extractor.")

    theta_rand, tsz_rand = cmbolympics.io.read_from_hdf5(
        cfg["input_map"]["random_pointing"],
        "theta_rand",
        "tsz_rand",
    )
    fprint(f"loaded random pointing pool from "
           f"{cfg['input_map']['random_pointing']} "
           f"which has shape {tsz_rand.shape}.")

    rng = np.random.default_rng(cfg["analysis"]["seed"])
    subtract_bg = cfg["analysis"]["subtract_background"]

    signal = profiler.get_profiles_per_source(
        halos["ell"],
        halos["b"],
        halos["aperture"],
        subtract_background=subtract_bg,
    )

    pool_samples = cfg["analysis"]["random_pool_samples"]
    if pool_samples is not None and pool_samples <= 0:
        pool_samples = None

    results = []
    for (lo, hi), median in zip(edges, medians):
        bin_mask = halos["log_mass"] >= lo
        if hi is not None:
            bin_mask &= halos["log_mass"] < hi

        if not np.any(bin_mask):
            fprint(f"No haloes in mass bin [{lo:.2f}, "
                   f"{'∞' if hi is None else f'{hi:.2f}'}). Skipping.")
            continue

        pval_data, pval_rand = cmbolympics.corr.empirical_pvalues_by_theta(
            halos["aperture"][bin_mask],
            signal[bin_mask],
            theta_rand,
            tsz_rand,
            random_pool_samples=pool_samples,
            random_theta_samples=cfg["analysis"]["random_theta_samples"],
            rng=rng,
        )

        print(f"Computed empirical p-values for {lo:.2f} < log M < {hi}.")

        ks_stat, ks_p = ks_2samp(pval_data, pval_rand)

        stack = profiler.stack_normalized_profiles(
            halos["ell"][bin_mask],
            halos["b"][bin_mask],
            halos["aperture"][bin_mask],
            radii_stack,
            subtract_background=subtract_bg,
            n_boot=cfg["analysis"]["bootstrap"],
            seed=cfg["analysis"]["seed"],
            random_profile_pool=tsz_rand,
            random_pool_radii=theta_rand,
            random_pool_samples=pool_samples,
        )
        stacked_profile, stacked_err, rand_mean, rand_err = stack

        hi_str = "∞" if hi is None else f"{hi:.2f}"
        cnt = bin_mask.sum()
        msg = (
            f"log M ∈ [{lo:.2f}, {hi_str}): N={cnt}, "
            f"median log M={median:.2f}, KS p-value={ks_p:.3e}"
        )
        fprint(msg)

        results.append(
            {
                "lo": lo,
                "hi": hi,
                "median_log_mass": median,
                "count": cnt,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "pval_data": pval_data,
                "pval_rand": pval_rand,
                "stacked_profile": stacked_profile,
                "stacked_error": stacked_err,
                "random_profile": rand_mean,
                "random_error": rand_err,
                "radii_norm": radii_stack,
            }
        )

    if not results:
        fprint("No bins contained haloes after filtering")


if __name__ == "__main__":
    main()
