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

"""Matching utilities (associations, crossmatches, Pfeifer probabilities)."""

from .cluster_matching import (  # noqa: F401
    compute_matching_matrix_cartesian,
    compute_matching_matrix_obs,
    greedy_global_matching,
)
from .crossmatch import (  # noqa: F401
    crossmatch_planck_catalog,
    crossmatch_mcxc,
    crossmatch_erass,
)
from .pfeifer import MatchingProbability  # noqa: F401
from .assignments import (  # noqa: F401
    match_catalogue_to_associations,
    match_planck_catalog_to_associations,
    match_mcxc_catalog_to_associations,
    match_erass_catalog_to_associations,
)
from .probabilistic_matching import partition_volume, print_group_summary  # noqa: F401
