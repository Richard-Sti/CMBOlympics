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
"""FoF halo catalogue readers for HDF5 outputs.

Provides access to Friends-of-Friends halo catalogues with optional axis
flipping and consistent mass unit conversion.
"""

import numpy as np
from h5py import File

from ..utils.logging import fprint


class FoFHaloReader:
    """Reader exposing FoF halo fields through dictionary-style access."""

    def __init__(self, fname, flip_xz=False):
        """Initialise the FoF halo reader.

        Parameters
        ----------
        fname : str or path-like
            Path to the FoF halo HDF5 file.
        flip_xz : bool, optional
            If True, swap the x and z axes for positions and velocities, by
            default False.
        """
        self.fname = fname
        self.flip_xz = flip_xz

    def __getitem__(self, name):
        """Return the requested FoF field as a NumPy array.

        Handles optional x/z axis flipping for `GroupPos` and `GroupVel`, and
        converts `GroupMass` values to solar masses divided by h.
        """
        with File(self.fname, "r") as f:
            grp = f["Group"]
            x = grp[name][...]
            if name == "GroupPos" and self.flip_xz:
                fprint(f"Flipping x and z coordinates for {self.fname}.")
                x[:, [0, 2]] = x[:, [2, 0]]
            elif name == "GroupVel" and self.flip_xz:
                fprint(f"Flipping x and z velocities for {self.fname}.")
                x[:, [0, 2]] = x[:, [2, 0]]
            elif name == "GroupMass":
                x *= 1e10  # convert to Msun/h

        return x


class SimulationHaloReader:
    """
    Reader for multi-realisation halo catalogues stored under HDF5 groups.

    The file is expected to contain one group per simulation realisation,
    with datasets such as ``Coordinates`` and ``Group_M_Crit200`` nested
    underneath.

    Parameters
    ----------
    fname : str or path-like
        Path to the HDF5 catalogue.
    nsim : str or int
        Identifier of the simulation realisation (group name in the file).
    """

    def __init__(self, fname, nsim):
        self.fname = fname
        self.nsim = str(nsim)

        with File(self.fname, "r") as f:
            if self.nsim not in f:
                raise KeyError(
                    f"Simulation '{self.nsim}' not found in "
                    f"'{self.fname}'. Available: {list(f.keys())}"
                )
            self._fields = tuple(sorted(f[self.nsim].keys()))

    @property
    def fields(self):
        """Return the dataset names available for the simulation."""
        return self._fields

    def __contains__(self, name):
        return name in self._fields

    def __getitem__(self, name):
        """Return a dataset for the selected simulation."""
        if name not in self:
            raise KeyError(
                f"Field '{name}' not found for simulation "
                f"'{self.nsim}'. Available: {self._fields}"
            )

        with File(self.fname, "r") as f:
            return f[self.nsim][name][...]


def list_simulations_hdf5(fname):
    """
    Return the available simulation identifiers stored in the catalogue.

    Parameters
    ----------
    fname : str or path-like
        Path to the HDF5 catalogue.

    Returns
    -------
    tuple of int
        Sorted tuple of group identifiers present in the file.
    """
    with File(fname, "r") as f:
        return tuple(sorted(int(key) for key in f.keys()))


def load_halo_positions_masses(fname, position_key, mass_key,
                               nsim="all", r_max=400.0, mass_min=5.0e13,
                               optional_keys=None):
    """Return filtered halo positions and masses via `SimulationHaloReader`.

    The function supports catalogues that follow the layout expected by
    `SimulationHaloReader`, i.e. top-level groups keyed by simulation ID with
    datasets such as coordinates and masses stored directly under each group.

    Parameters
    ----------
    optional_keys : list of str, optional
        Additional keys to load from the catalogue alongside positions and
        masses. These will be filtered with the same mask and returned.

    Returns
    -------
    positions_all : list of ndarray
        Filtered halo positions for each simulation.
    masses_all : list of ndarray
        Filtered halo masses for each simulation.
    optional_data : dict of lists, optional
        If optional_keys is provided, returns a dictionary mapping each key
        to a list of arrays (one per simulation).
    """

    if nsim == "all":
        sim_ids = list_simulations_hdf5(fname)
    else:
        sim_ids = (nsim,)

    positions_all = []
    masses_all = []
    optional_data = {key: [] for key in (optional_keys or [])}

    for idx in range(len(sim_ids)):
        sim_id = sim_ids[idx]
        reader = SimulationHaloReader(fname, sim_id)

        if position_key not in reader or mass_key not in reader:
            raise KeyError(
                f"Simulation {sim_id} missing '{position_key}' "
                f"or '{mass_key}'. Available: {reader.fields}"
            )

        pos = np.asarray(reader[position_key], dtype=float)
        mass = np.asarray(reader[mass_key], dtype=float)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"Positions for simulation {sim_id} must have shape (N, 3)."
            )
        if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
            raise ValueError(
                f"Masses for simulation {sim_id} must match positions."
            )

        # Load optional keys
        opt_arrays = {}
        for key in (optional_keys or []):
            if key not in reader:
                raise KeyError(
                    f"Simulation {sim_id} missing optional key '{key}'. "
                    f"Available: {reader.fields}"
                )
            arr = np.asarray(reader[key], dtype=float)
            if arr.shape[0] != pos.shape[0]:
                raise ValueError(
                    f"Optional key '{key}' for simulation {sim_id} "
                    f"must match positions."
                )
            opt_arrays[key] = arr

        box_size = np.max(pos, axis=0) - np.min(pos, axis=0)
        centre = np.min(pos, axis=0) + box_size / 2.0
        r = np.linalg.norm(pos - centre, axis=1)
        mask = np.isfinite(r)
        mask &= np.isfinite(mass)
        mask &= r <= r_max
        mask &= mass >= mass_min

        positions_all.append(pos[mask])
        masses_all.append(mass[mask])
        for key in optional_data:
            optional_data[key].append(opt_arrays[key][mask])

    if optional_keys:
        return positions_all, masses_all, optional_data
    return positions_all, masses_all
