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

from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import ks_2samp

import cmbolympics
from cmbolympics.utils import (
    build_mass_bins,
    cartesian_icrs_to_galactic_spherical,
    fprint,
)

try:
    import tomllib as _toml_loader  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "tomli is required to read the configuration file "
            "(install with 'pip install tomli')."
        ) from exc

tomli = _toml_loader


def load_config(path):
    """Return the raw configuration dictionary loaded from a TOML file."""

    with open(path, "rb") as fh:
        return tomli.load(fh)


def load_halo_catalogue(cfg, nsim):
    """Load halo phase-space information and apply selection cuts."""

    analysis_cfg = cfg["analysis"]
    sim_key = analysis_cfg["which_simulation"]
    catalogue_cfg = cfg["halo_catalogues"][sim_key]

    if sim_key == "csiborg2":
        reader = cmbolympics.io.SimulationHaloReader(
            catalogue_cfg["fname"],
            nsim=nsim,
        )
        centre = np.full(3, catalogue_cfg["box_size"] / 2.0, dtype=float)
        fprint(
            f"Using centre at {centre} Mpc/h for {sim_key} "
            f"(simulation {nsim})."
            )
        pos = reader["Coordinates"]
        mass = reader["Group_M_Crit200"]
        r200 = reader["Group_R_Crit200"]
        r, ell, b = cartesian_icrs_to_galactic_spherical(pos, centre)
    else:
        raise ValueError(f"Unknown simulation '{sim_key}'")

    theta_arcmin = np.rad2deg(np.arctan(r200 / r)) * 60
    cuts = cfg["halo_cuts"]

    mask = np.isfinite(mass)
    mask &= mass > 0
    mask &= np.isfinite(r)
    mask &= r >= cuts["r_min"]
    r_max = cuts.get("r_max")
    if r_max is not None:
        mask &= r <= r_max
    mask &= np.abs(b) >= cuts["b_min"]
    mask &= theta_arcmin >= cuts["theta_min"]
    theta_max = cuts.get("theta_max")
    if theta_max is not None:
        mask &= theta_arcmin <= theta_max
    mask &= mass >= cuts["mass_min"]
    mass_max = cuts.get("mass_max")
    if mass_max is not None:
        mask &= mass <= mass_max

    if not np.any(mask):
        raise ValueError(
            f"No haloes passed the selection cuts for simulation {nsim}."
        )

    selected = int(np.sum(mask))
    fprint(f"Selected {selected} haloes after applying cuts from {mass.size}.")

    return {
        "r": r[mask],
        "ell": ell[mask],
        "b": b[mask],
        "mass": mass[mask],
        "theta": theta_arcmin[mask],
    }


def load_profiler(cfg):
    """Initialise the profile extractor, sampling radii, and y-map."""

    map_cfg = cfg["input_map"]
    analysis_cfg = cfg["analysis"]

    y_map = cmbolympics.io.read_Planck_comptonSZ(map_cfg["signal_map"])
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(
        f"Loaded y-map {map_cfg['signal_map']}: mean={mu:.3e}, std={std:.3e}"
    )
    y_map = cmbolympics.utils.smooth_map_gaussian(
        y_map,
        fwhm_arcmin=map_cfg["smooth_fwhm"],
    )
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(
        f"Smoothed y-map with FWHM {map_cfg['smooth_fwhm']}: "
        f"mean={mu:.3e}, std={std:.3e}"
    )

    profiler = cmbolympics.corr.PointingEnclosedProfile(
        y_map,
        n_jobs=analysis_cfg["n_jobs"],
        fwhm_arcmin=map_cfg["smooth_fwhm"],
    )

    radii_stack = np.linspace(
        analysis_cfg["stack_rmin"],
        analysis_cfg["stack_rmax"],
        analysis_cfg["stack_n"],
    )

    return profiler, radii_stack, y_map


def save_results_hdf5(path, simulations):
    """Persist analysis outputs for one or more simulations to HDF5."""

    with h5py.File(path, "w") as h5:
        for sim_name, data in simulations.items():
            sim_group = h5.create_group(str(sim_name))

            halo_group = sim_group.create_group("halos")
            for key, values in data["halos"].items():
                halo_group.create_dataset(key, data=np.asarray(values))
            halo_group.create_dataset(
                "pval_data", data=np.asarray(data["halo_pvals"]))
            if data.get("halo_bins") is not None:
                halo_group.create_dataset(
                    "bin_index", data=np.asarray(data["halo_bins"]))
            if data.get("original_index") is not None:
                halo_group.create_dataset(
                    "original_index", data=np.asarray(data["original_index"]))

            bin_results = data.get("results") or []
            if not bin_results:
                continue

            bin_group = sim_group.create_group("halos_binned")
            for idx, entry in enumerate(bin_results):
                grp = bin_group.create_group(f"bin_{idx:03d}")
                grp.attrs["lo"] = entry["lo"]
                grp.attrs["hi"] = (
                    entry["hi"] if entry["hi"] is not None else np.nan
                )
                grp.attrs["median_log_mass"] = entry["median_log_mass"]
                grp.attrs["count"] = entry["count"]
                grp.attrs["ks_stat"] = entry["ks_stat"]
                grp.attrs["ks_p"] = entry["ks_p"]

                grp.create_dataset(
                    "pval_data",
                    data=np.asarray(entry["pval_data"]),
                )
                grp.create_dataset(
                    "pval_rand",
                    data=np.asarray(entry["pval_rand"]),
                )
                grp.create_dataset(
                    "stacked_profile",
                    data=np.asarray(entry["stacked_profile"]),
                )
                grp.create_dataset(
                    "stacked_error",
                    data=np.asarray(entry["stacked_error"]),
                )
                grp.create_dataset(
                    "random_profile",
                    data=np.asarray(entry["random_profile"]),
                )
                grp.create_dataset(
                    "random_error",
                    data=np.asarray(entry["random_error"]),
                )
                grp.create_dataset(
                    "radii_norm",
                    data=np.asarray(entry["radii_norm"]),
                )

                if entry["cutout_mean"] is not None:
                    grp.create_dataset(
                        "cutout_mean",
                        data=np.asarray(entry["cutout_mean"]),
                    )
                if entry["cutout_extent"] is not None:
                    grp.create_dataset(
                        "cutout_extent",
                        data=np.asarray(entry["cutout_extent"], dtype=float),
                    )
                if entry["cutout_random_mean"] is not None:
                    grp.create_dataset(
                        "cutout_random_mean",
                        data=np.asarray(entry["cutout_random_mean"]),
                    )


def process_simulation(cfg, sim_id, profiler, radii_stack, theta_rand,
                       tsz_rand, cutout_extractor, cutout_params, sim_index):
    """Run the full analysis pipeline for a single simulation ID."""

    analysis_cfg = cfg["analysis"]
    mass_cfg = cfg["mass_bins"]

    halos = load_halo_catalogue(cfg, sim_id)
    halos["aperture"] = analysis_cfg["aperture_scale"] * halos["theta"]
    halos["log_mass"] = np.log10(halos["mass"])

    edges, medians = build_mass_bins(
        halos["mass"],
        step=mass_cfg["mass_step"],
        top_counts=mass_cfg["top_counts"],
        verbose=mass_cfg["verbose_bins"],
    )

    rng_seed = analysis_cfg["seed"]
    if rng_seed is not None:
        rng_seed += sim_index
    rng = np.random.default_rng(rng_seed)

    subtract_bg = analysis_cfg["subtract_background"]
    signal = profiler.get_profiles_per_source(
        halos["ell"],
        halos["b"],
        halos["aperture"],
        subtract_background=subtract_bg,
    )

    pool_samples = analysis_cfg["random_pool_samples"]
    if pool_samples is not None and pool_samples <= 0:
        pool_samples = None

    size_scale = cutout_params["size_scale"]
    random_samples = cutout_params["random_samples"]
    random_abs_b_min = cutout_params["random_abs_b_min"]
    base_cutout_seed = cutout_params["random_seed"]
    if base_cutout_seed is not None:
        base_cutout_seed += sim_index
    max_cutout_bins = cutout_params["max_bins"]

    results = []
    halo_pval_data = np.full(halos["mass"].shape, np.nan, dtype=float)
    halo_bin_index = np.full(halos["mass"].shape, -1, dtype=int)

    for bin_idx, ((lo, hi), median) in enumerate(zip(edges, medians)):
        mask = halos["log_mass"] >= lo
        if hi is not None:
            mask &= halos["log_mass"] < hi

        if not np.any(mask):
            fprint(
                f"[Sim {sim_id}] No haloes in mass bin"
                f" [{lo:.2f}, {'∞' if hi is None else f'{hi:.2f}'}). Skipping."
            )
            continue

        fprint(
            f"[Sim {sim_id}] [Bin {bin_idx}] Selecting haloes for "
            f"{lo:.2f} ≤ log M < {hi if hi is not None else '∞'}: "
            f"{mask.sum()} candidates."
        )

        pval_data, pval_rand = cmbolympics.corr.empirical_pvalues_by_theta(
            halos["aperture"][mask],
            signal[mask],
            theta_rand,
            tsz_rand,
            random_pool_samples=pool_samples,
            random_theta_samples=analysis_cfg["random_theta_samples"],
            rng=rng,
        )

        fprint(
            f"[Sim {sim_id}] [Bin {bin_idx}] Computed empirical p-values "
            f"for {lo:.2f} ≤ log M < {hi if hi is not None else '∞'}."
        )

        ks_stat, ks_p = ks_2samp(pval_data, pval_rand)

        stack = profiler.stack_normalized_profiles(
            halos["ell"][mask],
            halos["b"][mask],
            halos["aperture"][mask],
            radii_stack,
            subtract_background=subtract_bg,
            n_boot=analysis_cfg["bootstrap"],
            seed=analysis_cfg["seed"],
            random_profile_pool=tsz_rand,
            random_pool_radii=theta_rand,
            random_pool_samples=pool_samples,
        )
        stacked_profile, stacked_err, rand_mean, rand_err = stack

        fprint(
            f"[Sim {sim_id}] [Bin {bin_idx}] Generated stacked profiles "
            "and summary statistics."
        )

        cutout_mean = None
        cutout_random_mean = None
        cutout_extent = None
        if max_cutout_bins is None or bin_idx < max_cutout_bins:
            cutout_sizes = size_scale * halos["theta"][mask]
            _, cutout_mean, cutout_extent = cutout_extractor.stack_cutouts(
                halos["ell"][mask],
                halos["b"][mask],
                cutout_sizes,
                halos["theta"][mask],
            )
            fprint(
                f"[Sim {sim_id}] [Bin {bin_idx}] Computed 2D cutout stack "
                f"on a {cutout_extractor.nbins}x{cutout_extractor.nbins} grid."
            )

            if random_samples:
                rand_seed = (base_cutout_seed + bin_idx
                             if base_cutout_seed is not None else None)
                _, cutout_random_mean, _ = \
                    cutout_extractor.stack_random_cutouts(
                        halos["theta"][mask],
                        n_stack=random_samples,
                        size_factor=size_scale,
                        abs_b_min=random_abs_b_min,
                        seed=rand_seed,
                    )
                fprint(
                    f"[Sim {sim_id}] [Bin {bin_idx}] Computed random 2D "
                    f"cutout stack with {random_samples} samples."
                )

        indices = np.nonzero(mask)[0]
        halo_pval_data[indices] = pval_data
        halo_bin_index[indices] = bin_idx

        hi_str = "∞" if hi is None else f"{hi:.2f}"
        fprint(
            f"[Sim {sim_id}] log M ∈ [{lo:.2f}, {hi_str}): N={mask.sum()}, "
            f"median log M={median:.2f}, KS p-value={ks_p:.3e}"
        )

        results.append(
            {
                "lo": lo,
                "hi": hi,
                "median_log_mass": median,
                "count": mask.sum(),
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "pval_data": pval_data,
                "pval_rand": pval_rand,
                "stacked_profile": stacked_profile,
                "stacked_error": stacked_err,
                "random_profile": rand_mean,
                "random_error": rand_err,
                "radii_norm": radii_stack,
                "cutout_mean": cutout_mean,
                "cutout_random_mean": cutout_random_mean,
                "cutout_extent": cutout_extent,
            }
        )

    mass_order = np.argsort(-np.asarray(halos["mass"]))
    halos_sorted = {
        key: np.asarray(values)[mass_order]
        for key, values in halos.items()
    }

    return {
        "halos": halos_sorted,
        "halo_pvals": halo_pval_data[mass_order],
        "halo_bins": halo_bin_index[mass_order],
        "original_index": mass_order,
        "results": results,
    }


def determine_simulations(catalogue_cfg, requested):
    """Return the list of simulation identifiers to process."""

    if isinstance(requested, str) and requested.lower() == "all":
        sims = cmbolympics.io.list_simulations_hdf5(catalogue_cfg["fname"])
        sims = sims[:2]

        fprint(f"Processing simulations {sims}.")

        return [str(sim) for sim in sims]

    if requested is None:
        raise ValueError(
            "nsim must be provided in halo_catalogues when not using 'all'."
        )

    return [str(requested)]


def main():
    """Entry-point for the command-line script."""

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    fprint(f"Loaded config from {config_path}")

    analysis_cfg = cfg["analysis"]
    map_cfg = cfg["input_map"]

    profiler, radii_stack, y_map = load_profiler(cfg)
    fprint("Initialised profile extractor.")

    cutout_pixels = analysis_cfg.get("cutout_pixels", 301)
    cutout_nbins = analysis_cfg.get("cutout_nbins", 128)
    cutout_halfsize = analysis_cfg.get("cutout_grid_halfsize")
    cutout_extractor = cmbolympics.corr.Pointing2DCutout(
        y_map,
        npix=cutout_pixels,
        nbins=cutout_nbins,
        grid_halfsize=cutout_halfsize,
    )

    cutout_params = {
        "size_scale": analysis_cfg.get("cutout_size_scale", 5.0),
        "random_samples": analysis_cfg.get("cutout_random_samples"),
        "random_abs_b_min": analysis_cfg.get("cutout_random_abs_b_min", 10.0),
        "random_seed": analysis_cfg.get(
            "cutout_random_seed",
            analysis_cfg["seed"],
        ),
        "max_bins": analysis_cfg.get("cutout_max_bins", 2),
    }
    if (
        cutout_params["random_samples"] is not None
        and cutout_params["random_samples"] <= 0
    ):
        cutout_params["random_samples"] = None

    theta_rand, tsz_rand = cmbolympics.io.read_from_hdf5(
        map_cfg["random_pointing"],
        "theta_rand",
        "tsz_rand",
    )
    fprint(
        f"Loaded random pointing pool from {map_cfg['random_pointing']} "
        f"with shape {tsz_rand.shape}."
    )

    sim_key = analysis_cfg["which_simulation"]
    catalogue_cfg = cfg["halo_catalogues"][sim_key]
    sim_ids = determine_simulations(catalogue_cfg, catalogue_cfg.get("nsim"))

    results_by_sim = {}
    for idx, sim_id in enumerate(sim_ids):
        fprint(f"Processing simulation {sim_id}")
        results_by_sim[sim_id] = process_simulation(
            cfg,
            sim_id,
            profiler,
            radii_stack,
            theta_rand,
            tsz_rand,
            cutout_extractor,
            cutout_params,
            idx,
        )

    if not results_by_sim:
        fprint("No simulations produced results; aborting save.")
        return

    output_dir = Path(analysis_cfg.get("output_folder", ".")).expanduser()
    secondary_tag = analysis_cfg.get("output_tag")
    stem = sim_key if not secondary_tag else f"{sim_key}_{secondary_tag}"
    output_path = (output_dir / f"{stem}.hdf5").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_results_hdf5(output_path, results_by_sim)
    fprint(f"Wrote outputs to {output_path}")


if __name__ == "__main__":
    main()
