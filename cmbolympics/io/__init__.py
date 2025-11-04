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

from .cmb import (  # noqa
    read_Planck_comptonSZ,  # noqa
)

from .snapshot import (  # noqa
    Gadget4Reader,  # noqa
)

from .halo import (  # noqa
    FoFHaloReader,  # noqa
    SimulationHaloReader,  # noqa
    list_simulations_hdf5,  # noqa
    load_halo_positions_masses,  # noqa
)

from .tsz_mass_bins import (  # noqa
    TSZMassBinResults,  # noqa
)

from h5py import File


def dump_to_hdf5(fname, **kwargs):
    """Shortcut to dump multiple arrays to an HDF5 file."""
    with File(fname, "w") as f:
        for key, val in kwargs.items():
            f.create_dataset(key, data=val)


def read_from_hdf5(fname, *args):
    """Shortcut to read multiple arrays from an HDF5 file."""
    out = []
    with File(fname, "r") as f:
        for key in args:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in file '{fname}'. "
                               f"Available keys: {list(f.keys())}")
            out.append(f[key][...])

    if len(args) == 1:
        return out[0]

    return out
