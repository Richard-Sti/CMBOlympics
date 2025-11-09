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
from typing import Dict, Optional

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
    pval_data: np.ndarray
    stacked_profile: Optional[np.ndarray] = None
    stacked_error: Optional[np.ndarray] = None
    random_profile: Optional[np.ndarray] = None
    random_error: Optional[np.ndarray] = None
    radii_norm: Optional[np.ndarray] = None
    individual_profiles: Optional[np.ndarray] = None
    random_profiles: Optional[np.ndarray] = None
    cutout_mean: Optional[np.ndarray] = None
    cutout_random_mean: Optional[np.ndarray] = None
    cutout_extent: Optional[np.ndarray] = None
    p_value_profile: Optional[np.ndarray] = None
    sigma_profile: Optional[np.ndarray] = None
    t_fit_p_value: Optional[np.ndarray] = None
    t_fit_sigma: Optional[np.ndarray] = None

    @property
    def has_profiles(self):
        return self.stacked_profile is not None

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return list(self.__dict__.keys())

    def as_dict(self):
        return self.__dict__.copy()


class HaloResults:
    """Container for halo-level diagnostics."""

    def __init__(self, arrays):
        self._data = dict(arrays)
        for key, value in self._data.items():
            setattr(self, key, value)

    def keys(self):
        return list(self._data.keys())

    def as_dict(self):
        return dict(self._data)

    def __getitem__(self, item):
        return self._data[item]


@dataclass
class SimulationResult:
    """Results for a single simulation realisation."""

    name: str
    halos: HaloResults
    bins: list[BinResult]

    def bin_keys(self):
        if not self.bins:
            return []
        return sorted(self.bins[0].keys())

    def as_dict(self):
        return {
            "halos": self.halos.as_dict(),
            "bins": [bin_res.as_dict() for bin_res in self.bins],
        }


class TSZMassBinResults:
    """Load and expose data stored by ``analyse_tsz_mass_bins.py``."""

    def __init__(self, path, include_profiles=True, simulation=None):
        self.path = path
        self.include_profiles = include_profiles
        self._simulations: Dict[str, SimulationResult] = {}
        self._load()

        if simulation is not None:
            if simulation not in self._simulations:
                available = list(self._simulations)
                msg = (
                    "Simulation '{name}' not found. Available: "
                    "{available}".format(
                        name=simulation,
                        available=available,
                    )
                )
                raise KeyError(msg)
            self._default = simulation
        else:
            self._default = next(iter(self._simulations), None)

    # ------------------------------------------------------------------
    # Mapping-style helpers
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self._simulations[key]

    def __iter__(self):
        return iter(self._simulations)

    def __len__(self):  # pragma: no cover - trivial
        return len(self._simulations)

    def keys(self):
        return list(self._simulations.keys())

    def items(self):
        return list(self._simulations.items())

    def values(self):
        return list(self._simulations.values())

    def iter_simulations(self):
        """Yield ``(name, SimulationResult)`` pairs."""

        for name in sorted(self._simulations):
            yield name, self._simulations[name]

    @property
    def simulation_count(self):
        """Number of simulations loaded."""

        return len(self._simulations)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def default_simulation(self):
        return self._default

    @property
    def halos(self):
        if len(self._simulations) != 1 and self._default is None:
            msg = (
                "Multiple simulations loaded; use results['sim_id']."
            )
            raise ValueError(msg)
        key = self._default or next(iter(self._simulations))
        return self._simulations[key].halos

    @property
    def bins(self):
        if len(self._simulations) != 1 and self._default is None:
            msg = (
                "Multiple simulations loaded; use results['sim_id']."
            )
            raise ValueError(msg)
        key = self._default or next(iter(self._simulations))
        return self._simulations[key].bins

    def bin_keys(self, simulation=None):
        if simulation is None:
            if len(self._simulations) != 1 and self._default is None:
                msg = (
                    "Multiple simulations available; pass a simulation name."
                )
                raise ValueError(msg)
            simulation = self._default or next(iter(self._simulations))
        return self._simulations[simulation].bin_keys()

    def as_dict(self):
        return {name: sim.as_dict() for name, sim in self._simulations.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self):
        with h5py.File(self.path, "r") as h5:
            if "halos" in h5:  # legacy layout without simulation grouping
                self._simulations["default"] = self._parse_simulation(
                    h5,
                    "default",
                )
                return

            for name in sorted(h5.keys()):
                self._simulations[name] = self._parse_simulation(
                    h5[name],
                    name,
                )

    def _parse_simulation(self, group, name):
        if "halos" not in group:
            raise KeyError(f"Group '{name}' missing 'halos' dataset")

        halo_group = group["halos"]
        halo_data = {key: halo_group[key][...] for key in halo_group.keys()}
        halos = HaloResults(halo_data)

        bins: list[BinResult] = []
        if "halos_binned" in group:
            for bin_name in sorted(group["halos_binned"].keys()):
                subgrp = group["halos_binned"][bin_name]
                entry_kwargs = dict(
                    name=bin_name,
                    lo=float(subgrp.attrs["lo"]),
                    hi=self._normalise_hi(subgrp.attrs["hi"]),
                    log_median_mass=float(subgrp.attrs["median_log_mass"]),
                    count=int(subgrp.attrs["count"]),
                    pval_data=subgrp["pval_data"][...],
                )
                if self.include_profiles and "stacked_profile" in subgrp:
                    # Load standard deviation fields
                    if "stacked_error" in subgrp:
                        entry_kwargs.update(
                            stacked_profile=subgrp["stacked_profile"][...],
                            stacked_error=subgrp["stacked_error"][...],
                            random_profile=subgrp["random_profile"][...],
                            random_error=subgrp["random_error"][...],
                            radii_norm=subgrp["radii_norm"][...],
                        )
                    # Backward compatibility: old files with percentile-based fields
                    elif "stacked_low" in subgrp and "stacked_high" in subgrp:
                        # For backward compatibility, use the midpoint of low/high as error
                        stacked_low = subgrp["stacked_low"][...]
                        stacked_high = subgrp["stacked_high"][...]
                        stacked_err = (stacked_high - stacked_low) / 2.0
                        random_low = subgrp["random_low"][...]
                        random_high = subgrp["random_high"][...]
                        random_err = (random_high - random_low) / 2.0
                        entry_kwargs.update(
                            stacked_profile=subgrp["stacked_profile"][...],
                            stacked_error=stacked_err,
                            random_profile=subgrp["random_profile"][...],
                            random_error=random_err,
                            radii_norm=subgrp["radii_norm"][...],
                        )
                if self.include_profiles and "individual_profiles" in subgrp:
                    entry_kwargs["individual_profiles"] = subgrp[
                        "individual_profiles"
                    ][...]
                if self.include_profiles and "random_profiles" in subgrp:
                    entry_kwargs["random_profiles"] = subgrp[
                        "random_profiles"
                    ][...]
                if "cutout_mean" in subgrp:
                    entry_kwargs["cutout_mean"] = subgrp["cutout_mean"][...]
                if "cutout_random_mean" in subgrp:
                    entry_kwargs["cutout_random_mean"] = subgrp[
                        "cutout_random_mean"
                    ][...]
                if "cutout_extent" in subgrp:
                    entry_kwargs["cutout_extent"] = subgrp[
                        "cutout_extent"
                    ][...]
                if "p_value_profile" in subgrp:
                    entry_kwargs["p_value_profile"] = subgrp[
                        "p_value_profile"
                    ][...]
                if "sigma_profile" in subgrp:
                    entry_kwargs["sigma_profile"] = subgrp[
                        "sigma_profile"
                    ][...]
                if "t_fit_p_value" in subgrp:
                    entry_kwargs["t_fit_p_value"] = subgrp[
                        "t_fit_p_value"
                    ][...]
                if "t_fit_sigma" in subgrp:
                    entry_kwargs["t_fit_sigma"] = subgrp[
                        "t_fit_sigma"
                    ][...]
                bins.append(BinResult(**entry_kwargs))

        return SimulationResult(name=name, halos=halos, bins=bins)

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
