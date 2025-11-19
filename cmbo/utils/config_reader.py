"""
Utilities for loading the CMBO configuration files and resolving their paths.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib as _toml_loader  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "tomli is required to read CMBO configuration files "
            "(install with 'pip install tomli')."
        ) from exc

toml_loader = _toml_loader


def load_config(path):
    """
    Return the configuration dictionary loaded from a TOML file, resolving
    all paths relative to the configured root. The resolved root directory is
    stored under ``cfg["_root_path"]``.
    """
    with open(path, "rb") as fh:
        cfg = toml_loader.load(fh)

    root_path = apply_root_to_config_paths(cfg)
    cfg["_root_path"] = str(root_path)
    return cfg


def _resolve_root_path(cfg):
    """
    Return the absolute root directory for resolving relative paths.
    """
    paths_cfg = cfg.get("paths", {})
    root_value = paths_cfg.get("root")
    if root_value is None:
        return Path(__file__).resolve().parents[2]

    root_path = Path(root_value).expanduser()
    if not root_path.is_absolute():
        root_path = (Path(__file__).resolve().parents[2] / root_path).resolve()
    return root_path


def _resolve_with_root(root_path, value):
    """
    Resolve a file path relative to the configured root directory.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root_path / path
    return str(path)


def apply_root_to_config_paths(cfg):
    """
    Resolve all known file paths in the config using the root directory.
    """
    root_path = _resolve_root_path(cfg)

    paths_cfg = cfg.setdefault("paths", {})
    if "observed_clusters" in paths_cfg:
        paths_cfg["observed_clusters"] = _resolve_with_root(
            root_path, paths_cfg["observed_clusters"]
        )
    else:
        paths_cfg["observed_clusters"] = str(
            (root_path / "data" / "observed_cluster_masses.toml").resolve()
        )

    for key in ("MCXC_catalogue", "Planck_tSZ_catalogue", "eRASS_catalogue"):
        if key in paths_cfg:
            paths_cfg[key] = _resolve_with_root(root_path, paths_cfg[key])

    map_cfg = cfg.get("input_map", {})
    for key in ("signal_map", "random_pointing"):
        if key in map_cfg:
            map_cfg[key] = _resolve_with_root(root_path, map_cfg[key])

    analysis_cfg = cfg.get("analysis", {})
    if "output_folder" in analysis_cfg:
        analysis_cfg["output_folder"] = _resolve_with_root(
            root_path, analysis_cfg["output_folder"]
        )

    mass_scoring_cfg = cfg.get("mass_scoring", {})
    if "output_dir" in mass_scoring_cfg:
        mass_scoring_cfg["output_dir"] = _resolve_with_root(
            root_path, mass_scoring_cfg["output_dir"]
        )

    for catalogue in cfg.get("halo_catalogues", {}).values():
        if not isinstance(catalogue, dict):
            continue
        if "fname" in catalogue:
            catalogue["fname"] = _resolve_with_root(
                root_path, catalogue["fname"])

    return root_path
