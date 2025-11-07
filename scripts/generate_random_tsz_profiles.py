#!/usr/bin/env python3
"""
Generate random tSZ signal and background profiles from the Planck NILC
Compton-y map.
"""

import argparse

import numpy as np

import cmbo
from cmbo.utils import fprint


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Generate random tSZ signal and background profiles "
                     "and dump them to HDF5.")
    )
    parser.add_argument(
        "--planck-map",
        default="../data/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits",
        help="Path to the Planck NILC Compton-y FITS file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("HDF5 file where (theta_rand, tsz_rand_signal, "
              "tsz_rand_background) will be stored."),
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=1.0,
        help="Minimum aperture radius in arcmin (default: 1.0).",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=601,
        help="Maximum aperture radius in arcmin (default: 601.0).",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=501,
        help="Number of aperture radii to sample (default: 501).",
    )
    parser.add_argument(
        "--theta-background-min",
        type=float,
        default=None,
        help="Minimum background radius in arcmin (default: theta-min).",
    )
    parser.add_argument(
        "--theta-background-max",
        type=float,
        default=None,
        help="Maximum background radius in arcmin (default: theta-max).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10_000,
        help="Number of random sky pointings (default: 10 000).",
    )
    parser.add_argument(
        "--abs-b-min",
        type=float,
        default=10.0,
        help="Minimum absolute Galactic latitude in degrees (default: 10).",
    )
    parser.add_argument(
        "--fwhm-arcmin",
        type=float,
        default=9.6,
        help="Gaussian FWHM used to smooth the y-map before sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the pointing generator (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for profile measurement (default: -1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    if output_path is None:
        base = args.planck_map
        if base.endswith('.fits'):
            output_path = base[:-5] + '_RAND_POINTING.hdf5'
        else:
            output_path = base + '_RAND_POINTING.hdf5'

    y_map = cmbo.io.read_Planck_comptonSZ(args.planck_map)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Loaded map {args.planck_map}: mean={mu:.3e}, std={std:.3e}")
    y_map = cmbo.utils.smooth_map_gaussian(
        y_map, fwhm_arcmin=args.fwhm_arcmin)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Smoothed map: mean={mu:.3e}, std={std:.3e}")

    profiler = cmbo.corr.PointingEnclosedProfile(
        y_map, n_jobs=args.n_jobs, fwhm_arcmin=args.fwhm_arcmin)

    theta_rand = np.linspace(args.theta_min, args.theta_max, args.n_theta)

    theta_bg_min = (args.theta_background_min if args.theta_background_min
                    is not None else args.theta_min)
    theta_bg_max = (args.theta_background_max if args.theta_background_max
                    is not None else args.theta_max)
    theta_rand_bg = np.linspace(theta_bg_min, theta_bg_max, args.n_theta)

    fprint(f"Signal radii: [{args.theta_min}, {args.theta_max}] arcmin")
    fprint(f"Background radii: [{theta_bg_min}, {theta_bg_max}] arcmin")

    tsz_rand_signal, tsz_rand_background = profiler.get_random_profiles(
        theta_rand,
        n_points=args.n_points,
        abs_b_min=args.abs_b_min,
        radii_background_arcmin=theta_rand_bg,
        seed=args.seed,
    )

    fprint(
        f"Random profiles generated with {args.n_points} pointings "
        f"over {args.n_theta} apertures.")

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
