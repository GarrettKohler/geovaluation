"""
Geospatial site visualization and ML modules.

Subpackages:
- services: Data loading, training, distance calculations
"""

# Re-export commonly used functions from services
from .services.nearest_site import calculate_nearest_site_distances
from .services.epa_walkability import (
    get_walkability_score,
    batch_walkability_scores,
    build_walkability_index,
    preload_walkability_data,
)

__all__ = [
    "calculate_nearest_site_distances",
    "get_walkability_score",
    "batch_walkability_scores",
    "build_walkability_index",
    "preload_walkability_data",
]
