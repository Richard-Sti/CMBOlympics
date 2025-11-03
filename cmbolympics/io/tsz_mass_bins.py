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
"""Loader utilities for tSZ mass-bin analysis outputs."""

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass
class BinResult:
    """Container for a single mass-bin entry."""

    name: str
    lo: float
    hi: Optional[float]
    log_median_mass: float
    count: int
    ks_stat: float
    ks_p: float
    pval_data: np.ndarray
    pval_rand: np.ndarray
    stacked_profile: Optional[np.ndarray] = None
    stacked_error: Optional[np.ndarray] = None
    random_profile: Optional[np.ndarray] = None
    random_error: Optional[np.ndarray] = None
    radii_norm: Optional[np.ndarray] = None
    cutout_mean: Optional[np.ndarray] = None
    cutout_random_mean: Optional[np.ndarray] = None
    cutout_extent: Optional[np.ndarray] = None

    @property
    def has_profiles(self) -> bool:
        return self.stacked_profile is not None

    def __getitem__(self, key):
        """Allow dict-like access to attributes."""
        return getattr(self, key)

    def keys(self):
        """Return the available attribute names."""
        return list(self.__dict__.keys())

    def as_dict(self):
        """Return a shallow copy of the underlying data."""
        return self.__dict__.copy()


class HaloResults:
    """Container for halo-level diagnostics."""

    def __init__(self, arrays):
        self._data = dict(arrays)
        for key, value in self._data.items():
            setattr(self, key, value)

    def keys(self):
        """Return dataset names stored for haloes."""
        return list(self._data.keys())

    def as_dict(self):
        """Return halo arrays as a plain dictionary."""
        return dict(self._data)

    def __getitem__(self, item):
        return self._data[item]


class TSZMassBinResults:
    """Load and expose data stored by ``analyse_tsz_mass_bins.py``."""

    def __init__(self, path, include_profiles=True):
        self.path = path
        self.include_profiles = include_profiles
        self.halos = {}
        self.bins = []
        self._load()

    def _load(self):
        with h5py.File(self.path, "r") as h5:
            halo_group = h5["halos"]
            halo_data = {key: halo_group[key][...]
                         for key in halo_group.keys()}
            self.halos = HaloResults(halo_data)

            bin_group = h5["halos_binned"]
            bins: list[BinResult] = []
            for name in sorted(bin_group.keys()):
                subgrp = bin_group[name]
                entry_kwargs = dict(
                    name=name,
                    lo=float(subgrp.attrs["lo"]),
                    hi=self._normalise_hi(subgrp.attrs["hi"]),
                    log_median_mass=float(subgrp.attrs["median_log_mass"]),
                    count=int(subgrp.attrs["count"]),
                    ks_stat=float(subgrp.attrs["ks_stat"]),
                    ks_p=float(subgrp.attrs["ks_p"]),
                    pval_data=subgrp["pval_data"][...],
                    pval_rand=subgrp["pval_rand"][...],
                )
                if self.include_profiles:
                    entry_kwargs.update(
                        stacked_profile=subgrp["stacked_profile"][...],
                        stacked_error=subgrp["stacked_error"][...],
                        random_profile=subgrp["random_profile"][...],
                        random_error=subgrp["random_error"][...],
                        radii_norm=subgrp["radii_norm"][...],
                    )
                if "cutout_mean" in subgrp:
                    entry_kwargs["cutout_mean"] = subgrp["cutout_mean"][...]
                if "cutout_random_mean" in subgrp:
                    entry_kwargs["cutout_random_mean"] = subgrp["cutout_random_mean"][...]
                if "cutout_extent" in subgrp:
                    entry_kwargs["cutout_extent"] = subgrp["cutout_extent"][...]
                bins.append(BinResult(**entry_kwargs))
            self.bins = bins

    @staticmethod
    def _normalise_hi(value):
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            pass
        return value

    def as_dict(self):
        """Return the loaded content as a simple dictionary."""
        return {
            "halos": self.halos.as_dict(),
            "bins": [bin_result.__dict__.copy() for bin_result in self.bins],
        }

    def bin_keys(self):
        """Return the attribute keys present for a representative mass bin."""
        if not self.bins:
            return []
        sample = self.bins[0]
        return sorted(list(sample.__dict__.keys()))
