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

# from .probabilistic_matching import (  # noqa: F401
#     fof_tessellation,
#     generate_valid_assignments,
#     compute_gaussian_probabilities,
#     compute_group_probabilities,
#     AssociationModel,
#     ScalingRelationModel,
# )
