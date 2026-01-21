"""
ZIP code mapping and visualization for geospatial datasets.

This module demonstrates how to use the geospatial toolkit to map
arbitrary datasets to ZIP code boundaries and create visualizations.

The refactored architecture supports:
- Loading any geospatial dataset format
- Mapping data to any boundary type (ZIP, county, tract, etc.)
- Customizable aggregation methods
- Flexible map visualization

Example usage:
    >>> from zip_code_mapping import ZipCodeMapper
    >>>
    >>> # Create mapper with custom dataset
    >>> mapper = ZipCodeMapper()
    >>> mapper.load_boundaries("path/to/zipcodes.shp", "ZIP_CODE")
    >>> mapper.load_values("path/to/data.gdb", "GEOID", "value_col", layer="MyLayer")
    >>>
    >>> # Map and visualize
    >>> result = mapper.map_to_boundaries(zip_codes=["10001", "10002"])
    >>> mapper.visualize(result, title="My Data by ZIP Code")
"""

from typing import List, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from datasource import (
    DatasetConfig,
    FileDataSource,
    BoundaryDataSource,
    ValueDataSource,
    load_dataset,
)
from spatial_ops import (
    AggregationMethod,
    JoinResult,
    SpatialJoiner,
    spatial_join,
)
from visualizer import (
    ColorScale,
    MapStyle,
    MapVisualizer,
    plot_map,
)
from config import (
    get_dataset_config,
    register_dataset,
    WALKABILITY_INDEX,
    ZCTA_2020,
)


class ZipCodeMapper:
    """Maps arbitrary datasets to ZIP code boundaries.

    This class provides a high-level interface for:
    1. Loading boundary data (ZIP codes or other regions)
    2. Loading value data to be mapped
    3. Performing spatial joins with customizable aggregation
    4. Creating visualizations

    Attributes:
        boundaries: GeoDataFrame containing boundary geometries
        values: GeoDataFrame containing value data
        boundary_config: Configuration for boundary dataset
        value_config: Configuration for value dataset
    """

    def __init__(self):
        """Initialize an empty mapper."""
        self.boundaries: Optional[gpd.GeoDataFrame] = None
        self.values: Optional[gpd.GeoDataFrame] = None
        self.boundary_config: Optional[DatasetConfig] = None
        self.value_config: Optional[DatasetConfig] = None
        self._visualizer = MapVisualizer()

    def load_boundaries(
        self,
        path: str,
        id_column: str,
        layer: Optional[str] = None,
        name: Optional[str] = None
    ) -> "ZipCodeMapper":
        """Load boundary data (ZIP codes, counties, etc.).

        Args:
            path: Path to the boundary data file
            id_column: Column containing boundary identifiers
            layer: Layer name for multi-layer formats
            name: Human-readable name for the dataset

        Returns:
            Self for method chaining

        Example:
            >>> mapper = ZipCodeMapper()
            >>> mapper.load_boundaries(
            ...     "data/zip_codes.shp",
            ...     id_column="ZCTA5CE20"
            ... )
        """
        self.boundary_config = DatasetConfig(
            path=path,
            id_column=id_column,
            layer=layer,
            name=name or "Boundaries"
        )
        source = FileDataSource(self.boundary_config)
        self.boundaries = source.load()
        return self

    def load_boundaries_from_config(
        self,
        config_name: str
    ) -> "ZipCodeMapper":
        """Load boundaries using a registered configuration.

        Args:
            config_name: Name of registered dataset configuration

        Returns:
            Self for method chaining

        Example:
            >>> mapper = ZipCodeMapper()
            >>> mapper.load_boundaries_from_config("zcta")
        """
        self.boundary_config = get_dataset_config(config_name)
        source = FileDataSource(self.boundary_config)
        self.boundaries = source.load()
        return self

    def load_values(
        self,
        path: str,
        id_column: str,
        value_column: str,
        layer: Optional[str] = None,
        name: Optional[str] = None
    ) -> "ZipCodeMapper":
        """Load value data to be mapped to boundaries.

        Args:
            path: Path to the value data file
            id_column: Column containing unique identifiers
            value_column: Column containing values to aggregate
            layer: Layer name for multi-layer formats
            name: Human-readable name for the dataset

        Returns:
            Self for method chaining

        Example:
            >>> mapper.load_values(
            ...     "data/walkability.gdb",
            ...     id_column="GEOID10",
            ...     value_column="NatWalkInd",
            ...     layer="NationalWalkabilityIndex"
            ... )
        """
        self.value_config = DatasetConfig(
            path=path,
            id_column=id_column,
            value_column=value_column,
            layer=layer,
            name=name or "Values"
        )
        source = FileDataSource(self.value_config)
        self.values = source.load()
        return self

    def load_values_from_config(
        self,
        config_name: str
    ) -> "ZipCodeMapper":
        """Load values using a registered configuration.

        Args:
            config_name: Name of registered dataset configuration

        Returns:
            Self for method chaining

        Example:
            >>> mapper = ZipCodeMapper()
            >>> mapper.load_values_from_config("walkability")
        """
        self.value_config = get_dataset_config(config_name)
        source = FileDataSource(self.value_config)
        self.values = source.load()
        return self

    def map_to_boundaries(
        self,
        boundary_ids: Optional[List[str]] = None,
        aggregation: Union[str, AggregationMethod] = "mean",
        include_mapping: bool = False
    ) -> JoinResult:
        """Map value data to boundaries and aggregate.

        Args:
            boundary_ids: Optional list of boundary IDs to include
            aggregation: Aggregation method ("mean", "sum", "count", etc.)
            include_mapping: Whether to include ID mapping in results

        Returns:
            JoinResult containing aggregated data

        Raises:
            ValueError: If boundaries or values not loaded
        """
        if self.boundaries is None:
            raise ValueError("Boundaries not loaded. Call load_boundaries() first.")
        if self.values is None:
            raise ValueError("Values not loaded. Call load_values() first.")

        # Convert string to enum if needed
        if isinstance(aggregation, str):
            aggregation = AggregationMethod(aggregation)

        joiner = SpatialJoiner(self.boundaries, self.boundary_config)
        return joiner.join(
            self.values,
            self.value_config,
            filter_boundaries=boundary_ids,
            aggregation=aggregation,
            include_mapping=include_mapping
        )

    def visualize(
        self,
        result: JoinResult,
        title: Optional[str] = None,
        colormap: Union[str, ColorScale] = ColorScale.VIRIDIS,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """Create a map visualization of the results.

        Args:
            result: JoinResult from map_to_boundaries()
            title: Map title
            colormap: Color scale for the visualization
            figsize: Figure size in inches
            save_path: Optional path to save the figure

        Returns:
            Tuple of (Figure, Axes)
        """
        style = MapStyle(
            colormap=colormap,
            title=title,
            legend_label=result.value_column,
            figsize=figsize
        )

        fig, ax = self._visualizer.plot(
            result.data,
            result.value_column,
            style
        )

        if save_path:
            self._visualizer.save(save_path, fig)

        return fig, ax


# Backward compatibility function
def get_zip_block_mapping(zip_codes: List[str]) -> gpd.GeoDataFrame:
    """Map census blocks to ZIP codes and calculate average walkability.

    This function provides backward compatibility with the original API.
    For new code, use the ZipCodeMapper class instead.

    Args:
        zip_codes: List of ZIP codes to analyze

    Returns:
        DataFrame indexed by ZIP code with block mappings and walkability averages

    Example:
        >>> result = get_zip_block_mapping(["48201", "48202"])
        >>> print(result["NatWalkIndAvg"])
    """
    mapper = ZipCodeMapper()

    # Load ZCTA boundaries
    mapper.load_boundaries(
        path=ZCTA_2020.path,
        id_column=ZCTA_2020.id_column
    )

    # Load walkability data
    mapper.load_values(
        path=WALKABILITY_INDEX.path,
        id_column=WALKABILITY_INDEX.id_column,
        value_column=WALKABILITY_INDEX.value_column,
        layer=WALKABILITY_INDEX.layer
    )

    # Perform mapping
    result = mapper.map_to_boundaries(
        boundary_ids=zip_codes,
        aggregation=AggregationMethod.MEAN,
        include_mapping=True
    )

    # Format output for backward compatibility
    output = result.mapping.copy()
    output = output.rename(columns={
        WALKABILITY_INDEX.id_column: "GEOID",
        ZCTA_2020.id_column: "ZIP"
    })
    output = output.groupby("ZIP").agg(list)

    # Add average walkability
    value_col = f"{WALKABILITY_INDEX.value_column}_{AggregationMethod.MEAN.value}"
    zip_values = result.data.set_index(ZCTA_2020.id_column)[value_col]
    output["NatWalkIndAvg"] = output.index.map(zip_values)

    return output


# Example usage
if __name__ == "__main__":
    import sys

    print("Geospatial Mapping Toolkit")
    print("=" * 50)
    print()
    print("Example usage with ZipCodeMapper:")
    print()
    print("""
    # Create mapper
    mapper = ZipCodeMapper()

    # Load boundaries (ZIP codes)
    mapper.load_boundaries(
        path="data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp",
        id_column="ZCTA5CE20"
    )

    # Load value data (walkability)
    mapper.load_values(
        path="data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Map to specific ZIP codes
    result = mapper.map_to_boundaries(
        boundary_ids=["48201", "48202", "48203"],
        aggregation="mean"
    )

    # Visualize
    fig, ax = mapper.visualize(
        result,
        title="Walkability by ZIP Code",
        colormap="YlGnBu"
    )

    plt.show()
    """)

    print()
    print("Or use the convenience functions:")
    print()
    print("""
    from zip_code_mapping import get_zip_block_mapping
    from visualizer import plot_map

    # Quick mapping (backward compatible)
    mapping = get_zip_block_mapping(["48201", "48202"])

    # Or use plot_map for any GeoDataFrame
    fig, ax = plot_map(
        data=my_geodataframe,
        value_column="my_value",
        title="My Map"
    )
    """)
