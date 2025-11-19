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
"""Coordinate transformations."""

import astropy.units as u
import numpy as np
from astropy.coordinates import (ICRS, CartesianRepresentation, Galactic,
                                 SkyCoord, SphericalRepresentation)
from astropy.cosmology import FlatLambdaCDM, z_at_value
from tqdm import trange

from ..constants import SPEED_OF_LIGHT_KMS


def cartesian_to_r_theta_phi(x, y, z, center=[0.0, 0.0, 0.0]):
    """Convert Cartesian coordinates to spherical (r, theta, phi)."""
    x0, y0, z0 = center

    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    theta = np.arccos(np.clip((z - z0) / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(y - y0, x - x0), 2.0 * np.pi)

    return r, theta, phi


def cartesian_to_radec(pos, center=[0.0, 0.0, 0.0]):
    """Convert Cartesian coordinates to right ascension and declination."""
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must have shape (N, 3).")

    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,).")

    __, theta, phi = cartesian_to_r_theta_phi(
        pos[:, 0], pos[:, 1], pos[:, 2], center=center
    )
    ra = np.degrees(phi)
    dec = np.degrees(np.pi / 2 - theta)
    return ra, dec


def radec_to_galactic(ra_deg, dec_deg):
    """Convert equatorial coordinates to Galactic longitude and latitude."""
    ra, dec = np.broadcast_arrays(np.asarray(ra_deg, dtype=float),
                                  np.asarray(dec_deg, dtype=float))
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
    ell = coord.l.to_value(u.deg)
    b = coord.b.to_value(u.deg)
    return ell, b


def galactic_to_radec(l_deg, b_deg):
    """Convert Galactic longitude and latitude to equatorial coordinates."""
    ell, b = np.broadcast_arrays(np.asarray(l_deg, dtype=float),
                                 np.asarray(b_deg, dtype=float))
    coord = SkyCoord(l=ell * u.deg, b=b * u.deg, frame="galactic").icrs
    ra = coord.ra.to_value(u.deg)
    dec = coord.dec.to_value(u.deg)
    return ra, dec


def cz_to_comoving_distance(cz, h=1.0, Om0=0.3111):
    """Convert CMB-frame velocity (km/s) to comoving distance (Mpc/h)."""
    scalar_input = np.isscalar(cz)
    cz_arr = np.array(cz, dtype=float, ndmin=1)
    out = np.full_like(cz_arr, np.nan, dtype=float)

    mask = np.isfinite(cz_arr)
    if not np.any(mask):
        return float(out[0]) if scalar_input else out

    cosmo_obj = FlatLambdaCDM(H0=100.0 * h, Om0=Om0)
    redshift = cz_arr[mask] / SPEED_OF_LIGHT_KMS
    distance = cosmo_obj.comoving_distance(redshift).value * cosmo_obj.h

    out[mask] = distance
    if scalar_input:
        return float(out[0])
    return out


def comoving_distance_to_cz(distance, h=1.0, Om0=0.3111):
    """
    Convert comoving distance (in Mpc/h if h = 1) to CMB-frame
    velocity (km/s).
    """
    scalar_input = np.isscalar(distance)
    dist_arr = np.array(distance, dtype=float, ndmin=1)

    cosmo_obj = FlatLambdaCDM(H0=100.0 * h, Om0=Om0)
    dist_mpc = (dist_arr / h) * u.Mpc

    # z_at_value supports array inputs in recent Astropy versions
    redshifts = z_at_value(cosmo_obj.comoving_distance, dist_mpc)
    out = redshifts * SPEED_OF_LIGHT_KMS

    if scalar_input:
        return float(out[0])
    return out.value


def radec_to_cartesian(ra_deg, dec_deg):
    """
    Convert RA, Dec to a Cartesian unit vector.

    Parameters
    ----------
    ra_deg, dec_deg : array_like
        RA, Dec (in degrees).

    Returns
    -------
    pos : ndarray
        Cartesian unit vectors of shape (N, 3).
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.stack([x, y, z], axis=-1)


def cartesian_icrs_to_galactic_spherical(pos, center):
    """
    Convert ICRS Cartesian coordinates to Galactic spherical coordinates
    (r, ell, b) about a fixed pivot `center` (observer).
    """
    pos_q = u.Quantity(pos, copy=False)
    cen_q = u.Quantity(center, copy=False)

    # Broadcast and shift to the chosen center.
    rel = pos_q - cen_q  # shape (..., 3)

    rep = CartesianRepresentation(rel[..., 0], rel[..., 1], rel[..., 2])
    icrs = ICRS(rep)
    gal = icrs.transform_to(Galactic())
    sph = gal.represent_as(SphericalRepresentation)

    ell = sph.lon.to(u.deg).value
    b = sph.lat.to(u.deg).value
    r = sph.distance.value

    return r, ell, b


def cartesian_icrs_to_galactic(pos, center, chunk=None):
    """
    Rotate ICRS Cartesian coordinates to Galactic Cartesian coordinates
    about a fixed pivot `center` (observer). The pivot remains numerically
    unchanged:
        x_gal = R @ (x_icrs - center) + center
    """
    c = np.asarray(center, dtype=float)
    if c.shape != (3,):
        raise ValueError("center must have shape (3,)")

    # --- Build correct rotation matrix R such that v_gal = R @ v_icrs ---
    icrs = SkyCoord(
        x=[1.0, 0.0, 0.0] * u.one,
        y=[0.0, 1.0, 0.0] * u.one,
        z=[0.0, 0.0, 1.0] * u.one,
        representation_type="cartesian",
        frame="icrs",
        )
    gal = icrs.galactic.cartesian

    # Rows are component names; columns are images of ICRS basis vectors:
    # column 0 = ex' = (x0,y0,z0)^T, etc.
    R = np.vstack([gal.x.value, gal.y.value, gal.z.value])  # shape (3,3)

    def _apply(block):
        return (R @ (block - c).T).T + c

    if chunk is None or pos.shape[0] <= (chunk or 0):
        out = _apply(pos)
    else:
        out = np.empty_like(pos)
        n = pos.shape[0]
        iter_kwargs = {"desc": "Rotating coordinates",
                       "disable": not n >= chunk}
        for i in trange(0, n, chunk, **iter_kwargs):
            sl = slice(i, min(i + chunk, n))
            out[sl] = _apply(pos[sl])

    return out


def build_mass_bins(mass, step=0.2, top_counts=(10, 100, 1000),
                    verbose=True):
    """Return log-mass bin edges and medians."""
    mass = np.asarray(mass, dtype=float)
    if mass.size == 0:
        return [], []
    if np.any(mass <= 0):
        raise ValueError("mass must contain only positive values")

    logm = np.log10(mass)
    order = np.argsort(logm)[::-1]

    bins = []
    consumed = 0
    prev_lo = None

    for rank_idx, count in enumerate(top_counts):
        if count <= 0 or consumed >= logm.size:
            continue
        end = min(consumed + count, logm.size)
        idx = order[consumed:end]
        if idx.size == 0:
            break
        lo = float(np.min(logm[idx]))
        hi = None if rank_idx == 0 else prev_lo
        label = f"top{idx.size}" if rank_idx == 0 else f"next{idx.size}"
        bins.append({
            'lo': lo,
            'hi': hi,
            'idx': idx,
            'label': label,
        })
        prev_lo = lo
        consumed = end

    start = float(bins[-1]['lo']) if bins else float(np.max(logm))
    min_logm = float(np.min(logm))
    hi = start

    while True:
        lo = hi - step
        if lo <= min_logm:
            lo = min_logm
            idx = np.where((logm >= lo) & (logm <= hi))[0]
            bins.append({
                'lo': lo,
                'hi': float(hi),
                'idx': idx,
                'label': None,
            })
            break
        idx = np.where((logm >= lo) & (logm < hi))[0]
        bins.append({
            'lo': float(lo),
            'hi': float(hi),
            'idx': idx,
            'label': None,
        })
        hi = lo

    edges = []
    medians = []
    for entry in bins:
        idx = entry['idx']
        cnt = int(idx.size)
        med = float(np.median(logm[idx])) if cnt > 0 else float('nan')
        lo = entry['lo']
        hi = entry['hi']
        label = entry['label']
        tag = f"{label}: " if label else ""

        if hi is None:
            if verbose:
                msg = (
                    f"{tag}log M ∈ [{lo:.2f}, ∞): "
                    f"{cnt} halos, median log M = {med:.2f}"
                )
                print(msg)
            edges.append([lo, None])
        else:
            if verbose:
                msg = (
                    f"{tag}log M ∈ [{lo:.2f}, {hi:.2f}): "
                    f"{cnt} halos, median log M = {med:.2f}"
                )
                print(msg)
            edges.append([lo, hi])
        medians.append(med)

    return edges, medians
