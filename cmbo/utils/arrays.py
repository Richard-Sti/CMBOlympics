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

"""Array manipulation utilities."""

from __future__ import annotations

import numpy as np


def mask_structured_array(data, mask):
    """
    Apply a boolean mask to a structured array or dict-like object.

    Parameters
    ----------
    data : mapping or structured array
        Dictionary-like object or structured NumPy array.
    mask : ndarray
        Boolean mask to apply.

    Returns
    -------
    filtered : structured ndarray
        New structured array containing only the selected entries.
    """
    if isinstance(data, np.ndarray) and data.dtype.names is not None:
        return data[mask]

    filtered_arrays = {key: np.asarray(val)[mask]
                       for key, val in data.items()}

    dtype = [(key, val.dtype) for key, val in filtered_arrays.items()]
    n = len(next(iter(filtered_arrays.values())))
    filtered = np.empty(n, dtype=dtype)
    for key, val in filtered_arrays.items():
        filtered[key] = val

    return filtered
