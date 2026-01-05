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
"""Readers for observed cluster catalogues stored as TOML files."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import SPEED_OF_LIGHT_KMS
from ..utils.coords import (cz_to_comoving_distance, radec_to_cartesian,
                            radec_to_galactic)

try:  # pragma: no cover - runtime import fallback for Python <3.11
    import tomllib as _toml_loader
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        _toml_loader = None

toml_loader = _toml_loader
if toml_loader is None:  # pragma: no cover - enforced via dependency list
    raise ImportError(
        "tomli is required to load observed cluster catalogues "
        "(install with 'pip install tomli')."
    )


@dataclass(slots=True)
class ObservedCluster:
    """Container describing a single observed cluster."""

    identifier: str
    name: str
    ra_deg: float
    dec_deg: float
    cz_cmb: float | None
    cz_cmb_err: float | None
    map_fit: dict | None = None
    planck_match: dict | None = None
    mcxc_match: dict | None = None
    erass_match: dict | None = None

    @property
    def galactic_coordinates(self):
        """Galactic longitude and latitude in degrees."""
        ell, b = radec_to_galactic(
            np.array([self.ra_deg]), np.array([self.dec_deg])
        )
        return float(ell[0]), float(b[0])

    def as_dict(self):
        """Return a serialisable view of the cluster."""
        return {
            "identifier": self.identifier,
            "name": self.name,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "cz_cmb": self.cz_cmb,
            "cz_cmb_err": self.cz_cmb_err,
            "map_fit": self.map_fit,
            "planck_match": self.planck_match,
            "mcxc_match": self.mcxc_match,
            "erass_match": self.erass_match,
        }


class ObservedClusterCatalogue:
    """Sequence-like container of observed clusters."""

    def __init__(self, clusters):
        self._clusters = tuple(clusters)

    def __len__(self):
        return len(self._clusters)

    def __iter__(self):
        return iter(self._clusters)

    def __getitem__(self, item):
        return self._clusters[item]

    @property
    def clusters(self):
        return self._clusters

    @property
    def galactic_coordinates(self):
        """Galactic longitudes and latitudes in degrees."""
        if not self._clusters:
            return np.empty((0, 2), dtype=float)

        ra = np.array([cluster.ra_deg for cluster in self._clusters],
                      dtype=float)
        dec = np.array([cluster.dec_deg for cluster in self._clusters],
                       dtype=float)
        ell, b = radec_to_galactic(ra, dec)
        return np.stack((ell, b), axis=-1)

    @property
    def redshifts(self):
        """CMB-frame redshifts derived from `cz_cmb`."""
        if not self._clusters:
            return np.empty(0, dtype=float)

        cz = np.array([
            cluster.cz_cmb if cluster.cz_cmb is not None else np.nan
            for cluster in self._clusters
        ], dtype=float)
        return cz / SPEED_OF_LIGHT_KMS

    @property
    def names(self):
        """List of cluster names."""
        return [cluster.name for cluster in self._clusters]

    def galactic_cartesian(self, h=1.0, Om0=0.3111):
        """
        Return Galactic Cartesian positions in Mpc/h using cz-derived
        distances.
        """
        if not self._clusters:
            return np.empty((0, 3), dtype=float)

        coords = self.galactic_coordinates
        ell_rad = np.deg2rad(coords[:, 0])
        b_rad = np.deg2rad(coords[:, 1])

        cz = np.array([
            cluster.cz_cmb if cluster.cz_cmb is not None else np.nan
            for cluster in self._clusters
        ], dtype=float)
        r = cz_to_comoving_distance(cz, h, Om0)

        x = r * np.cos(b_rad) * np.cos(ell_rad)
        y = r * np.cos(b_rad) * np.sin(ell_rad)
        z = r * np.sin(b_rad)

        return np.stack((x, y, z), axis=-1)

    def icrs_cartesian(self, h=1.0, Om0=0.3111):
        """
        Return ICRS Cartesian positions in Mpc/h using cz-derived distances.
        """
        if not self._clusters:
            return np.empty((0, 3), dtype=float)

        ra = np.array([cluster.ra_deg for cluster in self._clusters],
                      dtype=float)
        dec = np.array([cluster.dec_deg for cluster in self._clusters],
                       dtype=float)

        unit_vec = radec_to_cartesian(ra, dec)

        cz = np.array([
            cluster.cz_cmb if cluster.cz_cmb is not None else np.nan
            for cluster in self._clusters
        ], dtype=float)
        r = cz_to_comoving_distance(cz, h, Om0)

        return (unit_vec.T * r).T


def _load_catalogue(path: Path):
    raw_text = path.read_text(encoding="utf-8")
    try:
        return toml_loader.loads(raw_text)
    except Exception:
        fixed = re.sub(r"(?<=\d)\.(?=[eE][+-]?\d)", ".0", raw_text)
        return toml_loader.loads(fixed)


def load_observed_clusters(fname, skip_names=None, verbose=True):
    """Return the full observed cluster catalogue."""
    path = Path(fname)
    raw = _load_catalogue(path)
    skip_set = None
    if skip_names:
        skip_iter = (
            [skip_names] if isinstance(skip_names, str) else skip_names
        )
        skip_set = {str(name).lower() for name in skip_iter}

    clusters = []
    for identifier, entries in raw.items():
        if not isinstance(entries, dict):
            continue

        details = entries.get("details", {})
        if "ra" not in details or "dec" not in details:
            continue

        name = details.get("name", identifier)
        if skip_set and name.lower() in skip_set:
            if verbose:
                print(f"Skipping observed cluster '{name}'.")
            continue
        ra = float(details["ra"])
        dec = float(details["dec"])
        cz_cmb = details.get("cz_cmb")
        cz_cmb_err = details.get("cz_cmb_err")

        clusters.append(
            ObservedCluster(
                identifier=identifier,
                name=name,
                ra_deg=float(ra),
                dec_deg=float(dec),
                cz_cmb=float(cz_cmb) if cz_cmb is not None else None,
                cz_cmb_err=(
                    float(cz_cmb_err) if cz_cmb_err is not None else None
                ),
            )
        )

    if not clusters:
        raise ValueError(
            f"No clusters with sky coordinates found in '{fname}'.")

    return ObservedClusterCatalogue(clusters)
