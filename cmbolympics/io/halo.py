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
"""Various halo catalogue readers."""

from h5py import File

from ..utils import fprint


class FoFHaloReader:
    """FoF halo catalogue reader."""

    def __init__(self, fname, flip_xz=False):
        self.fname = fname
        self.flip_xz = flip_xz

    def __getitem__(self, name):
        with File(self.fname, "r") as f:
            grp = f["Group"]
            print(grp.keys())
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
