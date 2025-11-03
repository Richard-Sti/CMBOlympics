#!/usr/bin/env python3
"""Generate random tSZ profiles from the Planck NILC Compton-y map."""

import argparse

import numpy as np

import cmbolympics
from cmbolympics.utils import fprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random tSZ profiles and dump them to HDF5."
    )
    parser.add_argument(
        "--planck-map",
        default="../data/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits",
        help="Path to the Planck NILC Compton-y FITS file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HDF5 file where (theta_rand, tsz_rand) will be stored.",
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
    parser.add_argument(
        "--subtract-background",
        action="store_true",
        help="Subtract annulus background when measuring profiles.",
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

    y_map = cmbolympics.io.read_Planck_comptonSZ(args.planck_map)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Loaded map {args.planck_map}: mean={mu:.3e}, std={std:.3e}")
    y_map = cmbolympics.utils.smooth_map_gaussian(
        y_map, fwhm_arcmin=args.fwhm_arcmin)
    mu, std = np.nanmean(y_map), np.nanstd(y_map)
    fprint(f"Smoothed map: mean={mu:.3e}, std={std:.3e}")

    profiler = cmbolympics.corr.PointingEnclosedProfile(
        y_map, n_jobs=args.n_jobs, fwhm_arcmin=args.fwhm_arcmin)

    theta_rand = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    tsz_rand = profiler.get_random_profiles(
        theta_rand,
        n_points=args.n_points,
        abs_b_min=args.abs_b_min,
        subtract_background=args.subtract_background,
        seed=args.seed,
    )

    fprint(
        f"Random profiles generated with {args.n_points} pointings "
        f"over {args.n_theta} apertures.")

    cmbolympics.io.dump_to_hdf5(
        output_path,
        theta_rand=theta_rand,
        tsz_rand=tsz_rand,
    )
    fprint(f"Wrote random profiles to {output_path}")


if __name__ == "__main__":
    main()
