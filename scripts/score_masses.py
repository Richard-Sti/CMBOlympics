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

import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
import cmbo
import h5py
import numpy as np
from cmbo.utils import fprint

plt.style.use('science')

SIM_DISPLAY_NAMES = {
    "csiborg1": "CSiBORG1",
    "csiborg2": "CSiBORG2",
    "manticore": "Manticore-Local"
}

SIM_LABEL_NAMES = {
    "csiborg2": r"\texttt{CB2}",
    "manticore": r"\texttt{CBM}",
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

    if y_variable == "M500":
        y = np.log10(catalogue_matched["M500"] * h)
        yerr = (catalogue_matched["eM500"] /
                (catalogue_matched["M500"] * np.log(10)))
        if catalogue_name in ["mcxc", "erass"]:
            y_label = (r'$\log M^{\rm X-ray}_{500\mathrm{c}}'
                       r' ~ [h^{-1}\,M_\odot]$')
        else:
            y_label = (r'$\log M^{\rm tSZ}_{500\mathrm{c}}'
                       r' ~ [h^{-1}\,M_\odot]$')
        y_description = (f'log10(M_{{obs}}) for {catalogue_name} '
                         f'catalogue [h^-1 M_sun]')
        yerr_description = 'Uncertainty in log10(M_obs) from catalogue errors'
    elif y_variable == "L500":
        y = np.log10(catalogue_matched["L500"])
        yerr = (catalogue_matched["eL500"] /
                (catalogue_matched["L500"] * np.log(10)))
        y_label = (r'$\log L^{\rm X-ray}_{500\mathrm{c}} ~ '
                   r'[10^{44}\,\mathrm{erg\,s}^{-1}]$')
        y_description = (f'log10(L500) for {catalogue_name} catalogue '
                         f'[10^44 erg/s]')
        yerr_description = 'Uncertainty in log10(L500) from catalogue errors'
    elif y_variable == "YSZ":
        y = np.log10(catalogue_matched["Y500_scaled"])
        yerr = (catalogue_matched["eY500_scaled"] /
                (catalogue_matched["Y500_scaled"] * np.log(10)))
        y_label = r'$\log Y^{\rm tSZ}_{500\mathrm{c}} ~ [\mathrm{arcmin}^2]$'
        y_description = (f'log10(Y500_scaled) for {catalogue_name} '
                         f'catalogue [arcmin^2]')
        yerr_description = 'Uncertainty in log10(Y500_scaled) from catalogue errors'  # noqa
    else:
        raise ValueError(f"Unknown y_variable: {y_variable}")

    return y, yerr, y_label, y_description, yerr_description


def compute_significance(fitter, y_variable):
    """Compute significance test based on variable type."""
    if y_variable == "M500":
        joint = fitter.get_slope_intercept_significance([1, 0])
        slope_1d = fitter.get_param_significance("slope", 1.0)
        intercept_1d = fitter.get_param_significance("intercept", 0.0)
        return joint, slope_1d, intercept_1d
    elif y_variable == "L500":
        # Self-similar slope for soft-band X-ray luminosity is 4/3
        slope_1d = fitter.get_param_significance("slope", 4/3)
        return None, slope_1d, None
    elif y_variable == "YSZ":
        # Self-similar slope for integrated Compton-y is 5/3
        slope_1d = fitter.get_param_significance("slope", 5/3)
        return None, slope_1d, None
    else:
        raise ValueError(f"Unknown y_variable: {y_variable}")


def save_mcmc_samples(result_dict, output_path, data_dict=None):
    """Save MCMC samples and input data to HDF5 file."""
    output_path = Path(output_path)

    with h5py.File(output_path, 'w') as f:
        mcmc_grp = f.create_group('mcmc')
        for key, value in result_dict.items():
            mcmc_grp.create_dataset(key, data=np.asarray(value))

        if data_dict is not None:
            data_grp = f.create_group('data')
            for key in ['x', 'xerr', 'y', 'yerr', 'z_obs', 'pvals',
                        'distances']:
                if key in data_dict:
                    data_grp.create_dataset(key, data=data_dict[key])
            if 'metadata' in data_dict:
                for key, value in data_dict['metadata'].items():
                    data_grp.attrs[key] = value


def save_fit_summary(fit_output, corr_results, sig_result, y_variable,
                     output_path, n_matched, n_total):
    """Save fit summary, correlations, and significance to text file."""
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MCMC FIT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(fit_output)

        f.write("\nNumber matched: "
                f"{n_matched}/{n_total} "
                f"({100*n_matched/n_total:.1f}%)\n")

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
        joint, slope_sig, intercept_sig = sig_result
        if y_variable == "M500" and joint is not None:
            f.write(f"Slope=1, Intercept=0 (2D): p={joint[0]:.4f}, "
                    f"sigma={joint[1]:.4f}\n")
            f.write(f"Slope=1 (1D): p={slope_sig[0]:.4f}, "
                    f"sigma={slope_sig[1]:.4f}\n")
            f.write(f"Intercept=0 (1D): p={intercept_sig[0]:.4f}, "
                    f"sigma={intercept_sig[1]:.4f}\n")
        elif y_variable == "L500":
            f.write(f"Slope=4/3 (1D): p={slope_sig[0]:.4f}, "
                    f"sigma={slope_sig[1]:.2f}\n")
        elif y_variable == "YSZ":
            f.write(f"Slope=5/3 (1D): p={slope_sig[0]:.4f}, "
                    f"sigma={slope_sig[1]:.2f}\n")


def compute_matching_fractions(catalogue_data, matches, h, bin_edges=None,
                               nbins=5):
    """
    Compute matching fractions in mass bins.

    Parameters
    ----------
    catalogue_data : dict
        Catalogue with M500 field.
    matches : list
        Match results (None for unmatched clusters).
    h : float
        Hubble parameter for mass conversion.
    bin_edges : array, optional
        Pre-computed bin edges. If None, computed from percentiles.
    nbins : int
        Number of percentile-spaced mass bins (used if bin_edges is None).

    Returns
    -------
    bin_centres, fractions, errors, bin_edges, n_matched, n_total
    """
    logM500 = np.log10(catalogue_data["M500"] * h)
    is_matched = np.array([m is not None for m in matches])

    # Create percentile-spaced bins if not provided
    if bin_edges is None:
        percentiles = np.linspace(0, 100, nbins + 1)
        bin_edges = np.percentile(logM500, percentiles)

    nbins = len(bin_edges) - 1

    # Compute matching fraction in each bin
    bin_centres = []
    fractions = []
    errors = []
    for i in range(nbins):
        mask = (logM500 >= bin_edges[i]) & (logM500 < bin_edges[i + 1])
        if i == nbins - 1:  # Include right edge for last bin
            mask = (logM500 >= bin_edges[i]) & (logM500 <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            bin_centres.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
            fractions.append(np.nan)
            errors.append(np.nan)
            continue

        n_matched_bin = is_matched[mask].sum()
        frac = n_matched_bin / n_bin

        # Poisson error: sqrt(n_matched) / n_total
        if n_matched_bin > 0:
            err = np.sqrt(n_matched_bin) / n_bin
        else:
            err = 1 / n_bin

        bin_centres.append(np.mean(logM500[mask]))
        fractions.append(frac)
        errors.append(err)

    return (np.array(bin_centres), np.array(fractions), np.array(errors),
            bin_edges, np.sum(is_matched), len(is_matched))


def plot_matching_fraction_combined(matching_data, output_path,
                                    catalogue_name, bin_edges=None):
    """
    Plot matching fractions for multiple simulations on a single plot.

    Parameters
    ----------
    matching_data : dict
        Dictionary mapping sim_key to (bin_centres, fractions, errors,
        n_matched, n_total).
    output_path : Path
        Where to save the figure.
    catalogue_name : str
        Name of the catalogue for labeling.
    bin_edges : array, optional
        Bin edges to draw as vertical lines.
    """
    fig, ax = plt.subplots()

    n_sims = len(matching_data)
    offsets = np.linspace(-0.005, 0.005, n_sims) if n_sims > 1 else [0]

    for i, (sim_key, (centres, fracs, errs, n_matched, n_total)) in \
            enumerate(matching_data.items()):
        sim_label = SIM_LABEL_NAMES.get(sim_key, sim_key)
        label = rf'${sim_label}$ ({n_matched}/{n_total})'
        # Clip lower error bar at 0
        err_lower = np.minimum(errs, fracs)
        err_upper = errs
        ax.errorbar(centres + offsets[i], fracs, yerr=[err_lower, err_upper],
                    fmt='o', capsize=3, markersize=5, label=label)

    if catalogue_name in ["mcxc", "erass"]:
        ax.set_xlabel(r'$\log M^{\rm X-ray}_{500\mathrm{c}}'
                      r' ~ [h^{-1}\,M_\odot]$')
    else:
        ax.set_xlabel(r'$\log M^{\rm tSZ}_{500\mathrm{c}}'
                      r' ~ [h^{-1}\,M_\odot]$')
    ax.set_ylabel(r'$\mathrm{Matched~fraction}$')

    # Set ylim based on data
    all_fracs = []
    all_errs = []
    for centres, fracs, errs, _, _ in matching_data.values():
        valid = ~np.isnan(fracs)
        all_fracs.extend(fracs[valid])
        all_errs.extend(errs[valid])
    if all_fracs:
        upper = np.array(all_fracs) + np.array(all_errs)
        ymax = max(1.0, np.max(upper) * 1.05)
    else:
        ymax = 1.0
    ax.set_ylim(0, ymax)

    # Draw bin edges as vertical dashed lines
    if bin_edges is not None:
        lw = ax.spines['bottom'].get_linewidth()
        for edge in bin_edges:
            ax.axvline(edge, color='k', ls='--', lw=lw, zorder=0)

    ax.legend(fontsize=8)
    cat_display = CATALOGUE_DISPLAY_NAMES.get(catalogue_name, catalogue_name)
    ax.set_title(rf'$\mathrm{{{cat_display}}}$')

    fig.tight_layout()
    fig.savefig(output_path, dpi=450, bbox_inches='tight', format='pdf')
    plt.close(fig)


def save_plots(fitter, x, y, xerr, yerr, y_label, y_variable, sim_key,
               output_dir, base_name):
    """Save scatter and corner plots."""
    sim_label = SIM_LABEL_NAMES.get(sim_key, sim_key.capitalize())

    # Create scatter plot
    add_one_to_one = (y_variable == "M500")
    fig_scatter, ax_scatter = plt.subplots()
    fig_scatter, ax_scatter = fitter.plot_fit(
        x, y, xerr, yerr,
        ax=ax_scatter,
        add_one_to_one=add_one_to_one
    )

    # Add theoretical slope lines for L500 and YSZ
    if y_variable == "L500":
        x_mean, y_mean = np.mean(x), np.mean(y)
        ax_scatter.axline((x_mean, y_mean), slope=4/3, ls='--', color='C3',
                          lw=1.5, label=r'$m = 4/3$', zorder=0)
        ax_scatter.legend()
    elif y_variable == "YSZ":
        x_mean, y_mean = np.mean(x), np.mean(y)
        ax_scatter.axline((x_mean, y_mean), slope=5/3, ls='--', color='C3',
                          lw=1.5, label=r'$m = 5/3$', zorder=0)
        ax_scatter.legend()

    ax_scatter.set_xlabel(rf'$\log M^{{{sim_label}}}_{{500c}} ~ '
                          rf'[h^{{-1}}\,M_\odot]$')
    ax_scatter.set_ylabel(y_label)
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / f"{base_name}_scatter.pdf", dpi=450,
                        bbox_inches='tight')
    plt.close(fig_scatter)

    # Create corner plot
    if y_variable in ["L500", "YSZ"]:
        truths = [5/3, None, None]
    else:
        truths = [1., 0, None]
    fig_corner = plt.figure()
    fig_corner = fitter.plot_corner(fig_corner, truths=truths)
    fig_corner.savefig(output_dir / f"{base_name}_corner.pdf", dpi=450,
                       bbox_inches='tight')
    plt.close(fig_corner)


def process_combination(sim_key, catalogue_name, y_variable, cfg, mass_cfg):
    """Process a single simulation-catalogue-variable combination."""
    fprint(f"Processing: {sim_key} vs {catalogue_name} ({y_variable})")

    # Load data and perform matching
    data = load_catalogue(catalogue_name, cfg)
    associations = cmbo.utils.load_associations(
        sim_key,
        cfg,
        verbose=False,
        remove_near_target=mass_cfg.get("remove_near_target", False),
        target_ra_deg=mass_cfg.get("target_ra_deg", 201.989583),
        target_dec_deg=mass_cfg.get("target_dec_deg", -31.5025),
        target_cz_kms=mass_cfg.get("target_cz_kms", 14784.0),
        max_sep_arcmin=mass_cfg.get("remove_max_sep_arcmin", 180.0),
        max_cz_diff_kms=mass_cfg.get("remove_max_cz_diff_kms", 500.0),
    )

    result = match_catalogue_to_associations(
        catalogue_name, data, associations,
        max_angular_sep=mass_cfg.get("max_angular_sep", 30.0),
        max_delta_cz=mass_cfg.get("max_delta_cz", 500.0),
        median_halo_tsz_pval_max=mass_cfg.get("median_halo_tsz_pval_max"),
        use_median_halo_tsz_pval=mass_cfg.get(
            "use_median_halo_tsz_pval", False),
        min_member_fraction=mass_cfg.get(
            "min_member_fraction", 0.5),
        z_max=mass_cfg["z_max"],
        m500_min=mass_cfg["m500_min"],
        verbose=False,
    )
    (catalogue_matched, assoc_matched, pvals, distances,
     n_matched, n_total, matched_mask, filtered_data) = result

    # Compute matching fraction data for M500 variable
    matching_result = None
    if y_variable == "M500":
        h = mass_cfg.get("h", 0.68)
        matches_full = [True if m else None for m in matched_mask]
        matching_result = (filtered_data, matches_full, h)

    if len(catalogue_matched) == 0:
        fprint("  -> No matches found, skipping.")
        return matching_result

    # Sort by observed redshift
    z_obs = assoc_matched.centroid_obs_redshift
    sort_idx = np.argsort(z_obs)
    catalogue_matched = catalogue_matched[sort_idx]
    assoc_matched = assoc_matched[sort_idx]
    pvals = pvals[sort_idx]
    distances = distances[sort_idx]
    z_obs = z_obs[sort_idx]

    # Ensure resampled masses are available and of uniform length
    max_len = max(len(assoc.masses) for assoc in assoc_matched)
    assoc_matched.resample_masses(max_len)

    # Prepare x-axis data from resampled halo masses
    x_samples = np.log10(np.asarray(assoc_matched.resampled_halo_masses,
                                    dtype=float))
    x = np.mean(x_samples, axis=1)
    xerr = np.std(x_samples, axis=1)

    # Prepare y-axis data
    h = mass_cfg.get("h", 0.68)
    y, yerr, y_label, y_description, yerr_description = extract_y_data(
        catalogue_matched, y_variable, catalogue_name, h
    )

    # Apply mask
    mask = x < 30
    if mask.sum() < 3:
        msg = f"Only {mask.sum()} points available, skipping."
        fprint(f"  -> {msg}")
        output_dir = mass_cfg.get("output_dir")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = f"{sim_key}_classical_{y_variable}_{catalogue_name}"
            skip_path = output_dir / f"{base_name}_skipped.txt"
            skip_path.write_text(
                f"{msg}\n"
                f"n_matched={n_matched}, n_total={n_total}, "
                f"sim={sim_key}, catalogue={catalogue_name}, "
                f"y_variable={y_variable}\n"
            )
        return

    # Fit regression model
    if y_variable == "L500":
        y_pivot = mass_cfg.get("y_pivot_L500", 44.0)
    elif y_variable == "YSZ":
        y_pivot = mass_cfg.get("y_pivot_YSZ", -4.0)
    else:
        y_pivot = mass_cfg.get("y_pivot_M500", 14.0)

    fitter = cmbo.utils.MarginalizedLinearFitter()
    # Capture stdout during fit to get NumPyro summary
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        res = fitter.fit(
            x_samples[mask], y[mask], yerr[mask],
            x_pivot=mass_cfg.get("x_pivot", 14.0),
            y_pivot=y_pivot,
            nwarm=mass_cfg.get("mcmc_warmup", 1000),
            nsamp=mass_cfg.get("mcmc_samples", 10000),
            num_chains=mass_cfg.get("mcmc_chains", 1),
        )
        fitter.print_summary()
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

        base_name = f"{sim_key}_classical_{y_variable}_{catalogue_name}"

        data_dict = {
            'x': x[mask], 'xerr': xerr[mask],
            'y': y[mask], 'yerr': yerr[mask],
            'z_obs': z_obs[mask],
            'pvals': pvals, 'distances': distances,
            'metadata': {
                'x_description': (f'log10(M_{{sim}}) for {sim_key} '
                                  f'associations [h^-1 M_sun]'),
                'xerr_description': ('Standard deviation of log10(M_sim) '
                                     'across realisations'),
                'y_description': y_description,
                'yerr_description': yerr_description,
                'z_obs_description': 'Observed redshift (sorted ascending)',
                'pvals_description': ('Matching p-values between observed '
                                      'clusters and associations'),
                'distances_description': ('3D comoving distances between '
                                          'matched clusters [h^-1 Mpc]'),
                'sim_key': sim_key,
                'catalogue': catalogue_name,
                'y_variable': y_variable,
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
        save_fit_summary(
            fit_output,
            corr_results,
            sig_result,
            y_variable,
            output_dir / f"{base_name}_summary.txt",
            n_matched,
            n_total
        )
        save_plots(fitter, x[mask], y[mask], xerr[mask], yerr[mask],
                   y_label, y_variable, sim_key, output_dir, base_name)

    fprint(f"  -> Completed ({n_matched}/{n_total} matched "
           f"[{100*n_matched/n_total:.1f}%], {mask.sum()} points fitted)")

    return matching_result


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

    simulations = ["csiborg2", "manticore"]
    catalogues = ["planck", "erass"]
    # simulations = ["manticore"]
    # catalogues = ["erass"]

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
        total_combinations += (len(simulations) * len(get_y_variables(cat)))

    fprint(f"Processing {total_combinations} combinations")
    fprint("")

    # Collect matching data for combined plots: {catalogue: {sim: data}}
    matching_raw_data = {cat: {} for cat in catalogues}

    completed = 0
    for sim_key in simulations:
        for catalogue_name in catalogues:
            for y_variable in get_y_variables(catalogue_name):
                result = process_combination(sim_key, catalogue_name,
                                             y_variable, cfg, mass_cfg)
                # Store M500 matching data
                if y_variable == "M500" and result is not None:
                    matching_raw_data[catalogue_name][sim_key] = result
                completed += 1

    # Create combined matching fraction plots for each catalogue
    output_dir = mass_cfg.get("output_dir")
    if output_dir is not None:
        output_dir = Path(output_dir)
        for catalogue_name in catalogues:
            raw_data = matching_raw_data[catalogue_name]
            if not raw_data:
                continue

            # Compute bin edges from first simulation's data
            first_sim = list(raw_data.keys())[0]
            filtered_data, matches, h = raw_data[first_sim]
            _, _, _, bin_edges, _, _ = compute_matching_fractions(
                filtered_data, matches, h, nbins=5)

            # Compute matching fractions for all simulations
            matching_data = {}
            for sim_key, (fdata, matches, h) in raw_data.items():
                centres, fracs, errs, _, n_matched, n_total = \
                    compute_matching_fractions(fdata, matches, h,
                                               bin_edges=bin_edges)
                matching_data[sim_key] = (centres, fracs, errs,
                                          n_matched, n_total)

            plot_path = output_dir / f"{catalogue_name}_matching_fraction.pdf"
            fprint(f"Saving combined matching plot to {plot_path}")
            plot_matching_fraction_combined(matching_data, plot_path,
                                            catalogue_name, bin_edges)

    fprint(f"\nCompleted {completed}/{total_combinations} combinations "
           f"successfully.")


if __name__ == "__main__":
    main()
