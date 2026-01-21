"""
Geospatial data analysis and visualization toolkit.

This package provides tools for loading, joining, and visualizing
geospatial data with support for arbitrary datasets.

Modules:
    datasource: Data loading abstraction for various geospatial formats
    spatial_ops: Spatial join and overlay operations
    visualizer: Map visualization and choropleth rendering
    config: Pre-defined dataset configurations

Example:
    >>> from geospatial import (
    ...     load_dataset,
    ...     spatial_join,
    ...     plot_map,
    ...     get_dataset_config
    ... )
    >>>
    >>> # Load data
    >>> walkability = load_dataset(
    ...     path="./data/walkability.gdb",
    ...     id_column="GEOID10",
    ...     value_column="NatWalkInd",
    ...     layer="NationalWalkabilityIndex"
    ... )
    >>>
    >>> # Create visualization
    >>> fig, ax = plot_map(
    ...     data=walkability,
    ...     value_column="NatWalkInd",
    ...     title="Walkability Index"
    ... )
"""

from .datasource import (
    DatasetConfig,
    DataSource,
    FileDataSource,
    BoundaryDataSource,
    ValueDataSource,
    load_dataset,
)

from .spatial_ops import (
    AggregationMethod,
    JoinResult,
    SpatialJoiner,
    spatial_join,
)

from .visualizer import (
    ColorScale,
    MapStyle,
    LayerConfig,
    MapVisualizer,
    plot_map,
    compare_maps,
)

from .config import (
    DatasetRegistry,
    registry,
    get_dataset_config,
    register_dataset,
    list_datasets,
    WALKABILITY_INDEX,
    ZCTA_2020,
    CENSUS_TRACTS_2020,
    COUNTIES_2020,
    STATES_2020,
)

__version__ = "1.0.0"

__all__ = [
    # Data loading
    "DatasetConfig",
    "DataSource",
    "FileDataSource",
    "BoundaryDataSource",
    "ValueDataSource",
    "load_dataset",
    # Spatial operations
    "AggregationMethod",
    "JoinResult",
    "SpatialJoiner",
    "spatial_join",
    # Visualization
    "ColorScale",
    "MapStyle",
    "LayerConfig",
    "MapVisualizer",
    "plot_map",
    "compare_maps",
    # Configuration
    "DatasetRegistry",
    "registry",
    "get_dataset_config",
    "register_dataset",
    "list_datasets",
    # Pre-defined configs
    "WALKABILITY_INDEX",
    "ZCTA_2020",
    "CENSUS_TRACTS_2020",
    "COUNTIES_2020",
    "STATES_2020",
]
