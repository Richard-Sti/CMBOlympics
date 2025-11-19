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
Score mass calibration by matching observed cluster catalogues to simulation
associations and performing linear regression.
"""

import argparse
import shutil
import sys
from io import StringIO
from pathlib import Path

import cmbo
import h5py
import matplotlib.pyplot as plt
import numpy as np
from cmbo.utils import fprint

SIM_DISPLAY_NAMES = {
    "csiborg1": "CSiBORG1",
    "csiborg2": "CSiBORG2",
    "manticore": "Manticore"
}

CATALOGUE_DISPLAY_NAMES = {
    "planck": "Planck",
    "mcxc": "MCXC",
    "erass": "eROSITA"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match observed clusters to associations and fit "
                    "mass relations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to configuration file (default: scripts/config.toml)"
    )
    return parser.parse_args()


def load_catalogue(catalogue_name, cfg):
    catalogue_loaders = {
        "planck": lambda: cmbo.io.read_Planck_cluster_catalog(
            cfg['paths']['Planck_tSZ_catalogue']
        ),
        "mcxc": lambda: cmbo.io.load_mcxc_catalogue(
            cfg['paths']['MCXC_catalogue']
        ),
        "erass": lambda: cmbo.io.load_erass_catalogue(
            cfg['paths']['eRASS_catalogue']
        )
    }
    if catalogue_name not in catalogue_loaders:
        raise ValueError(f"Unknown catalogue: {catalogue_name}")
    return catalogue_loaders[catalogue_name]()


def match_catalogue_to_associations(catalogue_name, data, associations,
                                    **kwargs):
    match_functions = {
        "planck": cmbo.match.match_planck_catalog_to_associations,
        "mcxc": cmbo.match.match_mcxc_catalog_to_associations,
        "erass": cmbo.match.match_erass_catalog_to_associations
    }
    if catalogue_name not in match_functions:
        raise ValueError(f"Unknown catalogue: {catalogue_name}")
    return match_functions[catalogue_name](data, associations, **kwargs)


def get_y_variables(catalogue_name):
    """Return list of y-variables to process for a given catalogue."""
    if catalogue_name == "planck":
        return ["M500", "YSZ"]
    elif catalogue_name in ["mcxc", "erass"]:
        return ["M500", "L500"]
    else:
        return ["M500"]


def extract_y_data(catalogue_matched, y_variable, catalogue_name, h):
    """Extract y-axis data and labels based on variable type."""
    cat_display = CATALOGUE_DISPLAY_NAMES.get(
        catalogue_name, catalogue_name.upper())

    if y_variable == "M500":
        y = np.log10(catalogue_matched["M500"] * h)
        yerr = (catalogue_matched["eM500"] /
                (catalogue_matched["M500"] * np.log(10)))
        y_label = (rf'$\log M_{{\rm {cat_display}}} ~ '
                   rf'[h^{{-1}}\,M_\odot]$')
        y_description = (f'log10(M_{{obs}}) for {catalogue_name} '
                         f'catalogue [h^-1 M_sun]')
        yerr_description = 'Uncertainty in log10(M_obs) from catalogue errors'
    elif y_variable == "L500":
        y = np.log10(catalogue_matched["L500"])
        yerr = (catalogue_matched["eL500"] /
                (catalogue_matched["L500"] * np.log(10)))
        y_label = (rf'$\log L_{{\rm {cat_display}}} ~ '
                   rf'[10^{{44}}\,\mathrm{{erg\,s}}^{{-1}}]$')
        y_description = (f'log10(L500) for {catalogue_name} catalogue '
                         f'[10^44 erg/s]')
        yerr_description = 'Uncertainty in log10(L500) from catalogue errors'
    elif y_variable == "YSZ":
        y = np.log10(catalogue_matched["Y500_scaled"])
        yerr = (catalogue_matched["eY500_scaled"] /
                (catalogue_matched["Y500_scaled"] * np.log(10)))
        y_label = (rf'$\log Y_{{500,{{\rm {cat_display}}}}} ~ '
                   rf'[\mathrm{{arcmin}}^2]$')
        y_description = (f'log10(Y500_scaled) for {catalogue_name} '
                         f'catalogue [arcmin^2]')
        yerr_description = 'Uncertainty in log10(Y500_scaled) from catalogue errors'  # noqa
    else:
        raise ValueError(f"Unknown y_variable: {y_variable}")

    return y, yerr, y_label, y_description, yerr_description


def compute_significance(fitter, y_variable):
    """Compute significance test based on variable type."""
    if y_variable == "M500":
        return fitter.get_slope_intercept_significance([1, 0])
    elif y_variable in ["L500", "YSZ"]:
        return fitter.get_slope_significance(5/3)
    else:
        raise ValueError(f"Unknown y_variable: {y_variable}")


def save_mcmc_samples(result_dict, output_path, data_dict=None):
    """Save MCMC samples and input data to HDF5 file."""
    output_path = Path(output_path)
    with h5py.File(output_path, 'w') as f:
        mcmc_grp = f.create_group('mcmc')
        for key, value in result_dict.items():
            if key == 'samples':
                continue
            if isinstance(value, np.ndarray):
                mcmc_grp.create_dataset(key, data=value)
            elif isinstance(value, (int, float)):
                mcmc_grp.attrs[key] = value

        if data_dict is not None:
            data_grp = f.create_group('data')
            for key in ['x', 'xerr', 'y', 'yerr', 'pvals', 'distances']:
                if key in data_dict:
                    data_grp.create_dataset(key, data=data_dict[key])
            if 'metadata' in data_dict:
                for key, value in data_dict['metadata'].items():
                    data_grp.attrs[key] = value


def save_fit_summary(fit_output, corr_results, sig_result, y_variable,
                     output_path):
    """Save fit summary, correlations, and significance to text file."""
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MCMC FIT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(fit_output)

        f.write("\n" + "="*80 + "\n")
        f.write("CORRELATION COEFFICIENTS\n")
        f.write("="*80 + "\n")
        f.write(f"Pearson:  {corr_results['pearson']:.3f} +/- "
                f"{corr_results['pearson_err']:.3f}\n")
        f.write(f"Spearman: {corr_results['spearman']:.3f} +/- "
                f"{corr_results['spearman_err']:.3f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SIGNIFICANCE TESTS\n")
        f.write("="*80 + "\n")
        if y_variable == "M500":
            f.write(f"Slope=1, Intercept=0: p={sig_result[0]:.4f}, "
                    f"p={sig_result[1]:.4f}\n")
        elif y_variable in ["L500", "YSZ"]:
            f.write(f"Slope=5/3: p={sig_result[0]:.4f}, "
                    f"sigma={sig_result[1]:.2f}\n")


def save_plots(fitter, x, y, xerr, yerr, y_label, y_variable, sim_key,
               output_dir, base_name):
    """Save scatter and corner plots."""
    sim_display = SIM_DISPLAY_NAMES.get(sim_key, sim_key.capitalize())

    add_one_to_one = (y_variable == "M500")
    fig, ax = fitter.plot_fit(x, y, xerr, yerr, add_one_to_one=add_one_to_one)
    ax.set_xlabel(rf'$\log M_{{\rm {sim_display}}} ~ '
                  rf'[h^{{-1}}\,M_\odot]$')
    ax.set_ylabel(y_label)
    fig.savefig(output_dir / f"{base_name}_scatter.png", dpi=450,
                bbox_inches='tight')
    plt.close(fig)

    if y_variable in ["L500", "YSZ"]:
        truths = [5/3, None, None]
    else:
        truths = [1., 0, None]
    fig_corner = fitter.plot_corner(truths=truths)
    fig_corner.savefig(output_dir / f"{base_name}_corner.png", dpi=450,
                       bbox_inches='tight')
    plt.close(fig_corner)


def process_combination(sim_key, catalogue_name, match_threshold, y_variable,
                        cfg, mass_cfg):
    """Process a single simulation-catalogue-threshold-variable combination."""
    fprint(f"Processing: {sim_key} vs {catalogue_name} "
           f"({y_variable}, threshold={match_threshold})")

    # Load data and perform matching
    data = load_catalogue(catalogue_name, cfg)
    associations = cmbo.utils.load_associations(sim_key, cfg, verbose=False)

    result = match_catalogue_to_associations(
        catalogue_name, data, associations,
        match_threshold=match_threshold,
        mass_preference_threshold=mass_cfg.get("mass_preference_threshold"),  # noqa
        use_median_mass=mass_cfg.get("use_median_mass", True),
        matching_method=mass_cfg.get("matching_method", "greedy"),
        z_max=mass_cfg["z_max"],
        m500_min=mass_cfg["m500_min"],
        verbose=False,
    )
    (catalogue_matched, assoc_matched, pvals, distances,
     n_matched, n_total) = result

    if len(catalogue_matched) == 0:
        fprint("  -> No matches found, skipping.")
        return

    # Prepare x-axis data
    x = np.array(assoc_matched.mean_log_mass)
    xerr = np.array(assoc_matched.std_log_mass)

    # Prepare y-axis data
    h = mass_cfg.get("h", 0.68)
    y, yerr, y_label, y_description, yerr_description = extract_y_data(
        catalogue_matched, y_variable, catalogue_name, h
    )

    # Apply mask
    mask = x < 30
    if mask.sum() < 3:
        fprint(f"  -> Only {mask.sum()} points available, skipping.")
        return

    # Fit regression model
    if y_variable == "L500":
        y_pivot = mass_cfg.get("y_pivot_L500", 44.0)
    elif y_variable == "YSZ":
        y_pivot = mass_cfg.get("y_pivot_YSZ", -4.0)
    else:
        y_pivot = mass_cfg.get("y_pivot_M500", 14.0)

    fitter = cmbo.utils.LinearRoxyFitter()

    # Capture stdout during fit to get NumPyro summary
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        res = fitter.fit(
            x[mask], y[mask], xerr[mask], yerr[mask],
            method="mnr",
            x_pivot=mass_cfg.get("x_pivot", 14.0),
            y_pivot=y_pivot,
            nwarm=mass_cfg.get("mcmc_warmup", 1000),
            nsamp=mass_cfg.get("mcmc_samples", 10000),
            num_chains=mass_cfg.get("mcmc_chains", 1),
        )
        fit_output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Compute statistics
    corr_results = cmbo.utils.correlation_with_errors(
        x[mask], y[mask], xerr[mask], yerr[mask],
        n_samples=10000, verbose=False
    )
    sig_result = compute_significance(fitter, y_variable)

    # Save outputs
    output_dir = mass_cfg.get("output_dir")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        threshold_str = f"p{match_threshold:.3f}".replace(".", "_")
        base_name = f"{sim_key}_{threshold_str}_{y_variable}"

        data_dict = {
            'x': x[mask], 'xerr': xerr[mask],
            'y': y[mask], 'yerr': yerr[mask],
            'pvals': pvals, 'distances': distances,
            'metadata': {
                'x_description': (f'log10(M_{{sim}}) for {sim_key} '
                                  f'associations [h^-1 M_sun]'),
                'xerr_description': ('Standard deviation of log10(M_sim) '
                                     'across realisations'),
                'y_description': y_description,
                'yerr_description': yerr_description,
                'pvals_description': ('Matching p-values between observed '
                                      'clusters and associations'),
                'distances_description': ('3D comoving distances between '
                                          'matched clusters [h^-1 Mpc]'),
                'sim_key': sim_key,
                'catalogue': catalogue_name,
                'y_variable': y_variable,
                'match_threshold': match_threshold,
                'n_matched': n_matched,
                'n_total': n_total,
                'match_fraction': n_matched / n_total if n_total > 0 else 0.0,
                'h': h,
                'x_pivot': mass_cfg.get("x_pivot", 14.0),
                'y_pivot': y_pivot,
            }
        }

        save_mcmc_samples(res, output_dir / f"{base_name}_samples.hdf5",
                          data_dict)
        save_fit_summary(fit_output, corr_results, sig_result, y_variable,
                         output_dir / f"{base_name}_summary.txt")
        save_plots(fitter, x[mask], y[mask], xerr[mask], yerr[mask],
                   y_label, y_variable, sim_key, output_dir, base_name)

    fprint(f"  -> Completed ({n_matched}/{n_total} matched "
           f"[{100*n_matched/n_total:.1f}%], {mask.sum()} points fitted)")


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = cmbo.utils.load_config(str(config_path))
    try:
        mass_cfg = cfg["mass_scoring"]
    except KeyError as exc:
        raise KeyError(
            "Missing 'mass_scoring' section in config file. "
            "See config.toml for the required parameters."
        ) from exc

    # simulations = ["csiborg1", "csiborg2", "manticore"]
    # catalogues = ["planck", "mcxc", "erass"]
    simulations = ["manticore"]
    catalogues = ["erass"]

    match_thresholds = mass_cfg["match_threshold"]
    if not isinstance(match_thresholds, list):
        match_thresholds = [match_thresholds]

    # Clean output directory if requested
    output_dir = mass_cfg.get("output_dir")
    clear_output_dir = mass_cfg.get("clear_output_dir", True)
    if output_dir is not None:
        output_dir = Path(output_dir)
        if clear_output_dir and output_dir.exists():
            fprint(f"Removing existing files in {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total combinations
    total_combinations = 0
    for cat in catalogues:
        total_combinations += (len(simulations) * len(get_y_variables(cat))
                               * len(match_thresholds))

    fprint(f"Processing {total_combinations} combinations")
    fprint("")

    completed = 0
    for sim_key in simulations:
        for catalogue_name in catalogues:
            for y_variable in get_y_variables(catalogue_name):
                for threshold in match_thresholds:
                    try:
                        process_combination(sim_key, catalogue_name, threshold,
                                            y_variable, cfg, mass_cfg)
                        completed += 1
                    except Exception as exc:
                        fprint(f"  -> ERROR: {exc}")
                        continue

    fprint(f"\nCompleted {completed}/{total_combinations} combinations "
           f"successfully.")


if __name__ == "__main__":
    main()
