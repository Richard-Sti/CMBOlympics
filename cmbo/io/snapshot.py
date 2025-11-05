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
"""Particle snapshot readers for Gadget-4 HDF5 outputs."""

from h5py import File
import numpy as np


class Gadget4Reader:
    """Minimal reader for Gadget-4 HDF5 snapshots."""

    def __init__(self, fname, flip_xz=False):
        """Store metadata about the snapshot file to read.

        Parameters
        ----------
        fname : str or path-like
            Path to the Gadget-4 HDF5 snapshot.
        flip_xz : bool, optional
            If True, swap the x and z axes when reading coordinates,
            by default False.
        """
        self.fname = fname
        self.flip_xz = flip_xz

    @property
    def header(self):
        """Return a small subset of the snapshot header attributes."""
        with File(self.fname, "r") as f:
            h = f["Header"].attrs

            keys = ["BoxSize", "Time", "Redshift"]
            return {k: (float(h[k]) if k in h else np.nan) for k in keys}

    @property
    def part_types(self):
        """List the particle type groups present in the file."""
        with File(self.fname, "r") as f:
            return sorted([k for k in f.keys() if k.startswith("PartType")])

    def load_positions(self, part_types=None, chunk=None, concat=False,
                       dtype=np.float32):
        """Load particle coordinates for the requested types.

        Parameters
        ----------
        part_types : str or iterable of str, optional
            Particle type labels (e.g., ``PartType0``). None loads all.
        chunk : int, optional
            Read data in blocks of this many rows to limit memory usage.
        concat : bool, optional
            If True, return a single stacked array instead of a dict.
        dtype : np.dtype, optional
            Target dtype for the returned arrays.

        Returns
        -------
        dict[str, np.ndarray] or np.ndarray
            Mapping of particle type to coordinate array, or a concatenated
            array if ``concat`` is True.
        """
        # Normalize part_types to a list
        if part_types is None:
            part_types = self.part_types
        elif isinstance(part_types, str):
            part_types = [part_types]
        else:
            part_types = list(part_types)

        # Filter only valid types
        valid = set(self.part_types)
        part_types = [pt for pt in part_types if pt in valid]
        if not part_types:
            raise ValueError(
                f"No valid part types. Available: {sorted(valid)}")

        if chunk is not None and chunk <= 0:
            chunk = None

        pos = {}
        with File(self.fname, "r") as f:
            for pt in part_types:
                ds = f[f"{pt}/Coordinates"]
                n = ds.shape[0]
                if chunk is None or chunk >= n:
                    arr = ds[...].astype(dtype, copy=False)
                else:
                    arr = np.empty((n, 3), dtype=dtype)
                    for i0 in range(0, n, chunk):
                        i1 = min(i0 + chunk, n)
                        arr[i0:i1] = ds[i0:i1]
                if self.flip_xz:
                    arr[:, [0, 2]] = arr[:, [2, 0]]

                pos[pt] = arr

        if concat:
            # Return only a single concatenated array
            return np.vstack([pos[pt] for pt in part_types])

        return pos
