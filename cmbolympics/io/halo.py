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

from h5py import File

from ..utils import fprint


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
