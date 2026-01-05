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


def add_field(data, field_name, values, dtype=None):
    """
    Add a new field to a structured array.

    Parameters
    ----------
    data : structured ndarray
        Input structured array.
    field_name : str
        Name of the new field to add.
    values : array-like
        Values for the new field (must match length of data).
    dtype : dtype, optional
        Data type for the new field. If None, inferred from values.

    Returns
    -------
    new_data : structured ndarray
        New structured array with the additional field.
    """
    values = np.asarray(values)
    if values.shape[0] != data.shape[0]:
        raise ValueError(
            f"Length of values ({values.shape[0]}) must match data length "
            f"({data.shape[0]})."
        )

    if dtype is None:
        dtype = values.dtype

    new_dtype = data.dtype.descr + [(field_name, dtype)]
    new_data = np.empty(data.shape, dtype=new_dtype)

    for name in data.dtype.names:
        new_data[name] = data[name]
    new_data[field_name] = values

    return new_data


def rename_field(data, old_name, new_name):
    """
    Rename a field in a structured array.

    Parameters
    ----------
    data : structured ndarray
        Input structured array.
    old_name : str
        Current name of the field to rename.
    new_name : str
        New name for the field.

    Returns
    -------
    new_data : structured ndarray
        New structured array with the renamed field.
    """
    if old_name not in data.dtype.names:
        raise ValueError(f"Field '{old_name}' not found in array.")

    new_dtype = [(new_name if name == old_name else name, data.dtype[name])
                 for name in data.dtype.names]
    new_data = np.empty(data.shape, dtype=new_dtype)

    for old, new in zip(data.dtype.names, new_data.dtype.names):
        new_data[new] = data[old]

    return new_data


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
