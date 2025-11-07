#!/usr/bin/env python3
"""
Generate random tSZ signal and background profiles from the Planck NILC
Compton-y map.
"""

from pathlib import Path

import numpy as np

import cmbo
from cmbo.utils import fprint

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
CONFIG_PATH = Path(__file__).with_name("config.toml")


def _resolve_root_path(cfg):
    paths_cfg = cfg.get("paths", {})
    root_value = paths_cfg.get("root")
    if root_value is None:
        return Path(__file__).resolve().parents[1]

    root_path = Path(root_value).expanduser()
    if not root_path.is_absolute():
        root_path = (Path(__file__).resolve().parent / root_path).resolve()
    return root_path


def _resolve_with_root(root_path, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root_path / path
    return str(path)


def load_config_sections(config_path):
    with open(config_path, "rb") as fh:
        cfg = tomli.load(fh)

    root_path = _resolve_root_path(cfg)

    try:
        map_cfg = cfg["input_map"]
    except KeyError as exc:
        raise SystemExit(
            f"'input_map' section missing from {config_path}."
        ) from exc
    try:
        rand_cfg = cfg["random_profiles"]
    except KeyError as exc:
        raise SystemExit(
            f"'random_profiles' section missing from {config_path}."
        ) from exc
    try:
        runtime_cfg = cfg["runtime"]
    except KeyError as exc:
        raise SystemExit(
            f"'runtime' section missing from {config_path}."
        ) from exc

    map_required = ("signal_map", "random_pointing")
    missing_map = [key for key in map_required if key not in map_cfg]
    if missing_map:
        names = ", ".join(missing_map)
        raise SystemExit(
            f"Missing keys [{names}] in 'input_map' section of {config_path}."
        )

    rand_required = [
        "theta_min",
        "theta_max",
        "n_theta",
        "n_points",
        "abs_b_min",
        "fwhm_arcmin",
        "seed",
    ]
    missing_rand = [key for key in rand_required if key not in rand_cfg]
    if missing_rand:
        names = ", ".join(missing_rand)
        raise SystemExit(
            f"Missing keys [{names}] in 'random_profiles' section of "
            f"{config_path}."
        )

    if "n_jobs" not in runtime_cfg:
        raise SystemExit(
            f"'runtime.n_jobs' missing from {config_path}."
        )

    map_cfg = dict(map_cfg)
    map_cfg["signal_map"] = _resolve_with_root(root_path, map_cfg["signal_map"])
    map_cfg["random_pointing"] = _resolve_with_root(
        root_path, map_cfg["random_pointing"]
    )

    return map_cfg, rand_cfg, runtime_cfg


def main():
    map_cfg, rand_cfg, runtime_cfg = load_config_sections(CONFIG_PATH)

    planck_map = map_cfg["signal_map"]
    output_path = map_cfg["random_pointing"]
    theta_min = rand_cfg["theta_min"]
    theta_max = rand_cfg["theta_max"]
    n_theta = rand_cfg["n_theta"]
    theta_bg_min = rand_cfg.get("theta_background_min")
    theta_bg_max = rand_cfg.get("theta_background_max")
    n_points = rand_cfg["n_points"]
    abs_b_min = rand_cfg["abs_b_min"]
    fwhm_arcmin = rand_cfg["fwhm_arcmin"]
    seed = rand_cfg["seed"]
    n_jobs = runtime_cfg["n_jobs"]

    if theta_bg_min is None:
        theta_bg_min = theta_min
    if theta_bg_max is None:
        theta_bg_max = theta_max

    y_map = cmbo.io.read_Planck_comptonSZ(planck_map)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Loaded map {planck_map}: mean={mu:.3e}, std={std:.3e}")
    y_map = cmbo.utils.smooth_map_gaussian(
        y_map, fwhm_arcmin=fwhm_arcmin)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Smoothed map: mean={mu:.3e}, std={std:.3e}")

    profiler = cmbo.corr.PointingEnclosedProfile(
        y_map, n_jobs=n_jobs, fwhm_arcmin=fwhm_arcmin)

    theta_rand = np.linspace(theta_min, theta_max, n_theta)
    theta_rand_bg = np.linspace(theta_bg_min, theta_bg_max, n_theta)

    fprint(f"Signal radii: [{theta_min}, {theta_max}] arcmin")
    fprint(f"Background radii: [{theta_bg_min}, {theta_bg_max}] arcmin")

    tsz_rand_signal, tsz_rand_background = profiler.get_random_profiles(
        theta_rand,
        n_points=n_points,
        abs_b_min=abs_b_min,
        radii_background_arcmin=theta_rand_bg,
        seed=seed,
    )

    fprint(
        f"Random profiles generated with {n_points} pointings "
        f"over {n_theta} apertures.")

    cmbo.io.dump_to_hdf5(
        output_path,
        theta_rand=theta_rand,
        tsz_rand_signal=tsz_rand_signal,
        theta_rand_bg=theta_rand_bg,
        tsz_rand_background=tsz_rand_background,
    )
    fprint(f"Wrote random profiles to {output_path}")


if __name__ == "__main__":
    main()
