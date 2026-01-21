"""
Geospatial site visualization and ML modules.

Subpackages:
- services: Data loading, training, distance calculations
- core: Core geospatial utilities (spatial ops, visualization)
- dooh_ml: DOOH ML models (similarity, causal, classifier)
- utils: Debug and utility functions
"""

# Re-export commonly used functions from services
from .services.interstate_distance import (
    distance_to_nearest_interstate,
    batch_distance_to_interstate,
    preload_highway_data,
)
from .services.nearest_site import calculate_nearest_site_distances
from .services.epa_walkability import (
    get_walkability_score,
    batch_walkability_scores,
    build_walkability_index,
    preload_walkability_data,
)

__all__ = [
    "distance_to_nearest_interstate",
    "batch_distance_to_interstate",
    "preload_highway_data",
    "calculate_nearest_site_distances",
    "get_walkability_score",
    "batch_walkability_scores",
    "build_walkability_index",
    "preload_walkability_data",
]
