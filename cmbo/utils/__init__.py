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

from .coords import (  # noqa: F401
    build_mass_bins,
    cartesian_icrs_to_galactic,
    cartesian_icrs_to_galactic_spherical,
    cartesian_to_r_theta_phi,
    cz_to_comoving_distance,
    radec_to_galactic,
    galactic_to_radec,
)
from .associations import (  # noqa: F401
    HaloAssociation,
    identify_halo_associations,
    compute_association_signals,
)
from .cluster_matching import (  # noqa: F401
    compute_matching_matrix,
    greedy_global_matching,
)
from .crossmatch import crossmatch_planck_catalog  # noqa: F401
from .pfeifer import MatchingProbability  # noqa: F401
from .smoothing import smooth_map_gaussian  # noqa: F401
from .config_reader import (  # noqa: F401
    load_config,
    apply_root_to_config_paths,
)

from datetime import datetime
import numpy as np
from scipy.stats import norm


def fprint(*args, verbose=True, **kwargs):
    """Prints a message with a timestamp prepended."""
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]
        print(f"{timestamp}", *args, **kwargs)


def pvalue_to_sigma(pval):
    """Convert a one-sided p-value to a Gaussian sigma significance."""
    return norm.isf(pval)


def E_z(z, Om):
    """
    Dimensionless Hubble parameter E(z) for flat LCDM with matter and dark
    energy.

    Parameters
    ----------
    z : array-like or float
        Redshift(s) where E(z) is evaluated.
    Om : float
        Matter density parameter today. Dark energy density is 1 - Om.

    Returns
    -------
    array-like or float
        E(z) = sqrt(Om (1 + z)^3 + (1 - Om)). Scalars return floats.
    """
    if not np.isfinite(Om):
        raise ValueError("Om must be finite.")
    Ol = 1.0 - Om
    z_arr = np.asarray(z, dtype=float)
    Ez = np.sqrt(Om * (1.0 + z_arr)**3 + Ol)
    return float(Ez) if Ez.ndim == 0 else Ez
