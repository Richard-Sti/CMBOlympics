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
"""
Aggregate matched fractions from mass_scoring summary files and summarize
luminosity and mass–mass fits.
"""

import argparse
from pathlib import Path

import cmbo
import h5py
import numpy as np


def parse_filename(path):
    parts = path.stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name}")
    sim = parts[0]
    catalogue = parts[-2]
    return sim, catalogue


def parse_counts(path):
    matched = total = None
    with path.open() as f:
        for line in f:
            if line.startswith("Number matched:"):
                tokens = line.split()
                matched, total = (
                    tokens[2].split("/")[0], tokens[2].split("/")[1])
                matched = int(matched)
                total = int(total)
                break
    if matched is None or total is None:
        raise ValueError(f"Could not find matched counts in {path}")
    return matched, total, matched / total if total > 0 else 0.0


def format_table(fractions):
    sims = sorted({k[0] for k in fractions})
    catalogues = sorted({k[1] for k in fractions})
    header = ["Simulation"] + [f"{cat} (matched/total)" for cat in catalogues]
    col_widths = [max(len(header[0]), max(len(sim) for sim in sims))]
    for cat in catalogues:
        max_len = len(f"{cat} (matched/total)")
        for sim in sims:
            m, t, frac = fractions.get((sim, cat), (None, None, None))
            if m is not None and t is not None:
                max_len = max(max_len, len(f"{m}/{t} ({frac:.3f})"))
        col_widths.append(max_len)

    def fmt_row(values):
        return "  ".join(val.ljust(w) for val, w in zip(values, col_widths))

    rows = [fmt_row(header)]
    for sim in sims:
        cells = [sim]
        for cat in catalogues:
            entry = fractions.get((sim, cat))
            if entry is None:
                cells.append("n/a")
            else:
                m, t, frac = entry
                cells.append(f"{m}/{t} ({frac:.3f})")
        rows.append(fmt_row(cells))
    return "\n".join(rows)


def load_output_root(config_path):
    cfg = cmbo.utils.load_config(str(config_path))
    try:
        return Path(cfg["mass_scoring"]["output_dir"])
    except KeyError as exc:
        raise KeyError(
            "Config missing 'mass_scoring.output_dir' needed to "
            "locate results."
        ) from exc


def collect_match_fractions(root):
    summary_files = sorted(root.glob("*_summary.txt"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found under {root}")
    fractions = {}
    for path in summary_files:
        sim, catalogue = parse_filename(path)
        m, t, frac = parse_counts(path)
        fractions[(sim, catalogue)] = (m, t, frac)
    return fractions


def collect_spearman(root):
    summary_files = sorted(root.glob("*_summary.txt"))
    out = []
    for path in summary_files:
        parts = path.stem.split("_")
        if len(parts) < 5:
            continue
        sim, threshold, yvar, catalogue = (
            parts[0], parts[1], parts[-3], parts[-2])
        spearman = None
        spearman_err = None
        with path.open() as f:
            for line in f:
                if line.startswith("Spearman:"):
                    tokens = line.strip().split()
                    try:
                        spearman = float(tokens[1])
                        spearman_err = float(tokens[3])
                    except (IndexError, ValueError):
                        pass
                    break
        if spearman is None or spearman_err is None:
            continue
        out.append({
            "sim": sim,
            "catalogue": catalogue,
            "threshold": threshold,
            "yvar": yvar,
            "spearman": spearman,
            "spearman_err": spearman_err,
        })
    return out


def collect_tension(root, yvar_filter):
    summary_files = sorted(root.glob("*_summary.txt"))
    tensions = {}
    for path in summary_files:
        parts = path.stem.split("_")
        if len(parts) < 5:
            continue
        sim, threshold, yvar, catalogue = (
            parts[0], parts[1], parts[-3], parts[-2])
        if yvar != yvar_filter:
            continue
        sigma_val = None
        with path.open() as f:
            for line in f:
                if line.startswith("Slope=5/3"):
                    tokens = line.strip().split()
                    try:
                        sigma_val = float(tokens[-1].split("=")[-1])
                    except (IndexError, ValueError):
                        sigma_val = None
                    break
        if sigma_val is not None:
            tensions[(sim, catalogue, threshold)] = sigma_val
    return tensions


def collect_mass_tensions(root):
    summary_files = sorted(root.glob("*_summary.txt"))
    tensions = {}
    for path in summary_files:
        parts = path.stem.split("_")
        if len(parts) < 5:
            continue
        sim, threshold, yvar, catalogue = (
            parts[0], parts[1], parts[-3], parts[-2])
        if yvar != "M500":
            continue
        sig_2d = sig_slope = sig_intercept = np.nan
        with path.open() as f:
            for line in f:
                if "Slope=1, Intercept=0 (2D)" in line:
                    if "sigma=" in line:
                        try:
                            sig_2d = float(line.split("sigma=")[-1])
                        except ValueError:
                            sig_2d = np.nan
                elif line.startswith("Slope=1 (1D):"):
                    if "sigma=" in line:
                        try:
                            sig_slope = float(line.split("sigma=")[-1])
                        except ValueError:
                            sig_slope = np.nan
                elif line.startswith("Intercept=0 (1D):"):
                    if "sigma=" in line:
                        try:
                            sig_intercept = float(line.split("sigma=")[-1])
                        except ValueError:
                            sig_intercept = np.nan
        tensions[(sim, catalogue, threshold)] = {
            "2d": sig_2d,
            "slope": sig_slope,
            "intercept": sig_intercept
        }
    return tensions


def collect_luminosity_stats(root):
    sample_files = sorted(root.glob("*_samples.hdf5"))
    results = []
    skipped = 0
    spearman_map = {}
    for rec in collect_spearman(root):
        spearman_map[(rec["sim"], rec["catalogue"], rec["threshold"],
                      rec["yvar"])] = (rec["spearman"], rec["spearman_err"])
    tension_L500 = collect_tension(root, "L500")
    tension_YSZ = collect_tension(root, "YSZ")
    for path in sample_files:
        parts = path.stem.split("_")
        if len(parts) < 5:
            continue
        sim, threshold, yvar, catalogue = (
            parts[0], parts[1], parts[-3], parts[-2])
        if yvar not in {"L500", "YSZ"}:
            continue
        with h5py.File(path, "r") as f:
            mcmc = f["mcmc"]
            if ("slope" not in mcmc or "intercept" not in mcmc
                    or "sig" not in mcmc):
                skipped += 1
                continue
            slope = np.asarray(mcmc["slope"], dtype=float)
            intercept = np.asarray(mcmc["intercept"], dtype=float)
            sig = np.asarray(mcmc["sig"], dtype=float)
        results.append({
            "sim": sim,
            "catalogue": catalogue,
            "threshold": threshold,
            "yvar": yvar,
            "slope_mean": float(slope.mean()),
            "slope_std": float(slope.std()),
            "intercept_mean": float(intercept.mean()),
            "intercept_std": float(intercept.std()),
            "sig_mean": float(sig.mean()),
            "sig_std": float(sig.std()),
            "spearman": spearman_map.get(
                (sim, catalogue, threshold, yvar), (np.nan, np.nan)
            )[0],
            "spearman_err": spearman_map.get(
                (sim, catalogue, threshold, yvar), (np.nan, np.nan)
            )[1],
            "tension_sigma": tension_L500.get(
                (sim, catalogue, threshold), np.nan
            ) if yvar == "L500" else tension_YSZ.get(
                (sim, catalogue, threshold), np.nan
            ),
        })
    return results, skipped


def collect_mass_stats(root):
    sample_files = sorted(root.glob("*_samples.hdf5"))
    tensions = collect_mass_tensions(root)
    results = []
    for path in sample_files:
        parts = path.stem.split("_")
        if len(parts) < 5:
            continue
        sim, threshold, yvar, catalogue = (
            parts[0], parts[1], parts[-3], parts[-2])
        if yvar != "M500":
            continue
        with h5py.File(path, "r") as f:
            mcmc = f["mcmc"]
            if ("slope" not in mcmc or "intercept" not in mcmc
                    or "sig" not in mcmc):
                continue
            slope = np.asarray(mcmc["slope"], dtype=float)
            intercept = np.asarray(mcmc["intercept"], dtype=float)
            sig = np.asarray(mcmc["sig"], dtype=float)
        results.append({
            "sim": sim,
            "catalogue": catalogue,
            "threshold": threshold,
            "slope_mean": float(slope.mean()),
            "slope_std": float(slope.std()),
            "intercept_mean": float(intercept.mean()),
            "intercept_std": float(intercept.std()),
            "sig_mean": float(sig.mean()),
            "sig_std": float(sig.std()),
            "tension_2d": tensions.get(
                (sim, catalogue, threshold), {}).get("2d", np.nan),
            "tension_slope": tensions.get(
                (sim, catalogue, threshold), {}).get("slope", np.nan),
            "tension_intercept": tensions.get(
                (sim, catalogue, threshold), {}).get("intercept", np.nan),
        })
    return results


def format_luminosity_table(results, yvar_filter):
    rows = [r for r in results if r["yvar"] == yvar_filter]
    if not rows:
        return ""
    label = "Luminosity" if yvar_filter == "L500" else "tSZ"
    target_slope = "target m ≈ 5/3"
    catalogues = sorted({r["catalogue"] for r in rows})
    col_names = ["Simulation", "Threshold",
                 "m_mean", "m_std",
                 "c_mean", "c_std",
                 "sig_mean", "sig_std",
                 "spearman", "spearman_err",
                 "tension_theory_slope"]
    col_widths = [max(len(cn), 10) for cn in col_names]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    lines = []
    for cat in catalogues:
        cat_rows = [r for r in rows if r["catalogue"] == cat]
        if not cat_rows:
            continue
        lines.append(f"\n{label} relations ({cat}, mean ± std for m, "
                     f"c, sigma_int; Spearman) [{target_slope}]:")
        lines.append(fmt_row(col_names))
        for row in cat_rows:
            vals = [
                row["sim"],
                row["threshold"],
                f"{row['slope_mean']:.3f}",
                f"{row['slope_std']:.3f}",
                f"{row['intercept_mean']:.3f}",
                f"{row['intercept_std']:.3f}",
                f"{row['sig_mean']:.3f}",
                f"{row['sig_std']:.3f}",
                f"{row['spearman']:.3f}"
                if np.isfinite(row['spearman']) else "n/a",
                f"{row['spearman_err']:.3f}"
                if np.isfinite(row['spearman_err']) else "n/a",
                f"{row['tension_sigma']:.3f}"
                if np.isfinite(row['tension_sigma']) else "n/a",
            ]
            lines.append(fmt_row(vals))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize matched fractions and luminosity fits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config file for locating output_dir.",
    )
    args = parser.parse_args()

    root = load_output_root(args.config)
    lines = []

    fractions = collect_match_fractions(root)
    lines.append(
        "Matched fraction (matched / total) per simulation and catalogue:")
    lines.append(format_table(fractions))

    lum_results, skipped = collect_luminosity_stats(root)
    lum_L = format_luminosity_table(lum_results, "L500")
    lum_Y = format_luminosity_table(lum_results, "YSZ")
    if lum_L:
        lines.append(lum_L)
    if lum_Y:
        lines.append(lum_Y)

    mass_results = collect_mass_stats(root)
    if mass_results:
        lines.append("\nMass–mass relations (mean ± std for m, c, sigma_int; "
                     "tensions to 1:1 for joint m,c and marginals):")
        col_names = ["Simulation", "Threshold",
                     "m_mean", "m_std",
                     "c_mean", "c_std",
                     "sig_mean", "sig_std",
                     "tension_1to1", "tension_m", "tension_c"]
        col_widths = [max(len(cn), 10) for cn in col_names]

        def fmt_row(vals):
            return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

        catalogues = sorted({r["catalogue"] for r in mass_results})
        for cat in catalogues:
            cat_rows = [r for r in mass_results if r["catalogue"] == cat]
            if not cat_rows:
                continue
            lines.append(f"\nCatalogue: {cat}")
            lines.append(fmt_row(col_names))
            for row in cat_rows:
                vals = [
                    row["sim"],
                    row["threshold"],
                    f"{row['slope_mean']:.3f}",
                    f"{row['slope_std']:.3f}",
                    f"{row['intercept_mean']:.3f}",
                    f"{row['intercept_std']:.3f}",
                    f"{row['sig_mean']:.3f}",
                    f"{row['sig_std']:.3f}",
                    f"{row['tension_2d']:.3f}"
                    if np.isfinite(row["tension_2d"]) else "n/a",
                    f"{row['tension_slope']:.3f}"
                    if np.isfinite(row["tension_slope"]) else "n/a",
                    f"{row['tension_intercept']:.3f}"
                    if np.isfinite(row["tension_intercept"]) else "n/a",
                ]
                lines.append(fmt_row(vals))

    output = "\n".join(lines)
    print(output)
    outfile = root / "summary_overview.txt"
    outfile.write_text(output)


if __name__ == "__main__":
    main()
