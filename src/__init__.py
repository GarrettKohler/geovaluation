"""
Geospatial distance calculation modules.

This package provides functions for:
- Calculating distance to nearest US Interstate highway
- Finding nearest neighboring sites
- EPA walkability score lookups
"""

from .interstate_distance import (
    distance_to_nearest_interstate,
    batch_distance_to_interstate,
    preload_highway_data,
)
from .nearest_site import calculate_nearest_site_distances
from .epa_walkability import (
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
