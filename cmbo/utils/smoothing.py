"""Smoothing utilities for HEALPix maps."""

import healpy as hp
import numpy as np


def smooth_map_gaussian(m, fwhm_arcmin):
    """
    Return a Gaussian-smoothed copy of a HEALPix map.

    Parameters
    ----------
    m : array_like
        Input map in RING ordering.
    fwhm_arcmin : float
        Full width at half maximum of the Gaussian kernel in arcminutes.

    Returns
    -------
    ndarray
        Smoothed map.
    """
    map_arr = np.asarray(m, dtype=float)
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    return hp.sphtfunc.smoothing(map_arr, fwhm=fwhm_rad)
