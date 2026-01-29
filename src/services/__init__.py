"""
Service modules for the geospatial application.

Contains:
- data_service: Data loading and caching
- training_service: ML model training with GPU acceleration
- nearest_site: Spatial indexing and nearest neighbor queries
- epa_walkability: EPA walkability score lookups
"""

from .data_service import (
    load_sites,
    load_revenue_metrics,
    load_site_details,
    get_filter_options,
    get_filtered_site_ids,
    get_site_details_for_display,
    preload_all_data,
    CATEGORICAL_FIELDS,
)

from .training_service import (
    get_system_info,
    start_training,
    stop_training,
    get_training_status,
    stream_training_progress,
)
