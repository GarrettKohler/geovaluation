"""
Spatial operations for joining and overlaying geospatial datasets.

This module provides flexible spatial join operations that can work
with arbitrary datasets and configurable column mappings.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union

import geopandas as gpd
import pandas as pd

from datasource import DatasetConfig, DataSource


class AggregationMethod(Enum):
    """Supported aggregation methods for spatial joins."""
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"


@dataclass
class JoinResult:
    """Result of a spatial join operation.

    Attributes:
        data: GeoDataFrame containing the joined/aggregated data
        boundary_id_column: Column name for boundary identifiers
        value_column: Column name for the aggregated values
        source_id_column: Column name for source data identifiers
        mapping: DataFrame mapping source IDs to boundary IDs
    """
    data: gpd.GeoDataFrame
    boundary_id_column: str
    value_column: str
    source_id_column: str
    mapping: Optional[pd.DataFrame] = None


class SpatialJoiner:
    """Performs spatial join operations between datasets.

    This class handles the process of:
    1. Aligning coordinate reference systems
    2. Finding spatial intersections
    3. Calculating overlap areas
    4. Assigning source features to boundaries
    5. Aggregating values by boundary
    """

    def __init__(
        self,
        boundary_data: Union[gpd.GeoDataFrame, DataSource],
        boundary_config: Optional[DatasetConfig] = None
    ):
        """Initialize the spatial joiner.

        Args:
            boundary_data: GeoDataFrame or DataSource containing boundary regions
            boundary_config: Configuration for the boundary dataset (required if
                            boundary_data is a GeoDataFrame)
        """
        if isinstance(boundary_data, DataSource):
            self.boundaries = boundary_data.load()
            self.boundary_config = boundary_data.get_config()
        else:
            if boundary_config is None:
                raise ValueError(
                    "boundary_config is required when passing a GeoDataFrame"
                )
            self.boundaries = boundary_data
            self.boundary_config = boundary_config

    def join(
        self,
        source_data: Union[gpd.GeoDataFrame, DataSource],
        source_config: Optional[DatasetConfig] = None,
        filter_boundaries: Optional[List[str]] = None,
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        custom_aggregator: Optional[Callable] = None,
        include_mapping: bool = False
    ) -> JoinResult:
        """Join source data to boundaries and aggregate values.

        Args:
            source_data: GeoDataFrame or DataSource containing data to join
            source_config: Configuration for source dataset (required if
                          source_data is a GeoDataFrame)
            filter_boundaries: Optional list of boundary IDs to include
            aggregation: Method to use for aggregating values
            custom_aggregator: Optional custom function for aggregation
            include_mapping: Whether to include the ID mapping in results

        Returns:
            JoinResult containing aggregated data by boundary
        """
        # Get source data and config
        if isinstance(source_data, DataSource):
            source_gdf = source_data.load()
            src_config = source_data.get_config()
        else:
            if source_config is None:
                raise ValueError(
                    "source_config is required when passing a GeoDataFrame"
                )
            source_gdf = source_data
            src_config = source_config

        # Filter boundaries if specified
        boundaries = self._filter_boundaries(filter_boundaries)

        # Align CRS
        source_aligned = source_gdf.to_crs(boundaries.crs)

        # Find intersecting features
        intersecting = self._find_intersections(source_aligned, boundaries)

        # Perform overlay and calculate areas
        overlay_result = self._calculate_overlay(
            intersecting,
            boundaries,
            src_config.id_column,
            self.boundary_config.id_column
        )

        # Assign features to boundaries based on maximum overlap
        mapping_df = self._assign_to_boundaries(
            overlay_result,
            src_config.id_column,
            self.boundary_config.id_column
        )

        # Aggregate values
        aggregated = self._aggregate_values(
            mapping_df,
            source_gdf,
            src_config,
            aggregation,
            custom_aggregator
        )

        # Join with boundary geometries
        result_gdf = boundaries.merge(
            aggregated,
            left_on=self.boundary_config.id_column,
            right_index=True,
            how='inner'
        )

        # Determine the value column name
        if src_config.value_column:
            value_col = f"{src_config.value_column}_{aggregation.value}"
        else:
            value_col = f"count_{src_config.id_column}"

        return JoinResult(
            data=result_gdf,
            boundary_id_column=self.boundary_config.id_column,
            value_column=value_col,
            source_id_column=src_config.id_column,
            mapping=mapping_df if include_mapping else None
        )

    def _filter_boundaries(
        self,
        filter_ids: Optional[List[str]]
    ) -> gpd.GeoDataFrame:
        """Filter boundaries to specified IDs."""
        if filter_ids is None:
            return self.boundaries

        id_col = self.boundary_config.id_column
        return self.boundaries[self.boundaries[id_col].isin(filter_ids)]

    def _find_intersections(
        self,
        source: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Find source features that intersect with any boundary."""
        boundary_union = boundaries.unary_union
        return source[source.intersects(boundary_union)]

    def _calculate_overlay(
        self,
        source: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        source_id_col: str,
        boundary_id_col: str
    ) -> gpd.GeoDataFrame:
        """Calculate overlay and area of intersections."""
        overlay = gpd.overlay(source, boundaries, how="intersection")
        overlay["_overlap_area"] = overlay.geometry.area
        return overlay

    def _assign_to_boundaries(
        self,
        overlay: gpd.GeoDataFrame,
        source_id_col: str,
        boundary_id_col: str
    ) -> pd.DataFrame:
        """Assign each source feature to the boundary with maximum overlap."""
        # Create pivot table of overlap areas
        pivot = overlay.pivot_table(
            index=source_id_col,
            columns=boundary_id_col,
            values="_overlap_area",
            fill_value=0
        )

        # Find boundary with maximum overlap for each source feature
        max_boundary = pivot.idxmax(axis=1)

        # Create mapping dataframe
        mapping = max_boundary.to_frame().reset_index()
        mapping.columns = [source_id_col, boundary_id_col]

        return mapping

    def _aggregate_values(
        self,
        mapping: pd.DataFrame,
        source: gpd.GeoDataFrame,
        source_config: DatasetConfig,
        aggregation: AggregationMethod,
        custom_aggregator: Optional[Callable]
    ) -> pd.DataFrame:
        """Aggregate source values by boundary."""
        source_id_col = source_config.id_column
        boundary_id_col = self.boundary_config.id_column
        value_col = source_config.value_column

        # Group source IDs by boundary
        grouped = mapping.groupby(boundary_id_col)[source_id_col].agg(list)

        if custom_aggregator is not None:
            # Use custom aggregation function
            def apply_custom(source_ids):
                subset = source[source[source_id_col].isin(source_ids)]
                return custom_aggregator(subset)

            result = grouped.apply(apply_custom)
            result_df = result.to_frame(name="custom_value")

        elif value_col is not None:
            # Aggregate the value column
            def apply_aggregation(source_ids):
                subset = source[source[source_id_col].isin(source_ids)]
                values = subset[value_col]

                if aggregation == AggregationMethod.MEAN:
                    return values.mean()
                elif aggregation == AggregationMethod.SUM:
                    return values.sum()
                elif aggregation == AggregationMethod.COUNT:
                    return len(values)
                elif aggregation == AggregationMethod.MIN:
                    return values.min()
                elif aggregation == AggregationMethod.MAX:
                    return values.max()
                elif aggregation == AggregationMethod.MEDIAN:
                    return values.median()
                elif aggregation == AggregationMethod.FIRST:
                    return values.iloc[0] if len(values) > 0 else None
                elif aggregation == AggregationMethod.LAST:
                    return values.iloc[-1] if len(values) > 0 else None

            result = grouped.apply(apply_aggregation)
            result_df = result.to_frame(name=f"{value_col}_{aggregation.value}")

            # Also include count
            count = grouped.apply(len)
            result_df[f"count_{source_id_col}"] = count

        else:
            # No value column, just count
            count = grouped.apply(len)
            result_df = count.to_frame(name=f"count_{source_id_col}")

        # Add list of source IDs
        result_df[f"{source_id_col}_list"] = grouped

        return result_df


def spatial_join(
    boundaries: gpd.GeoDataFrame,
    boundary_id_column: str,
    source: gpd.GeoDataFrame,
    source_id_column: str,
    value_column: Optional[str] = None,
    filter_boundaries: Optional[List[str]] = None,
    aggregation: str = "mean"
) -> JoinResult:
    """Convenience function for performing spatial joins.

    Args:
        boundaries: GeoDataFrame containing boundary regions
        boundary_id_column: Column name for boundary identifiers
        source: GeoDataFrame containing source data
        source_id_column: Column name for source identifiers
        value_column: Optional column to aggregate
        filter_boundaries: Optional list of boundary IDs to include
        aggregation: Aggregation method ("mean", "sum", "count", etc.)

    Returns:
        JoinResult with aggregated data

    Example:
        >>> result = spatial_join(
        ...     boundaries=zip_codes,
        ...     boundary_id_column="ZCTA5CE20",
        ...     source=walkability,
        ...     source_id_column="GEOID10",
        ...     value_column="NatWalkInd",
        ...     filter_boundaries=["48201", "48202"],
        ...     aggregation="mean"
        ... )
    """
    boundary_config = DatasetConfig(
        path="",
        id_column=boundary_id_column
    )

    source_config = DatasetConfig(
        path="",
        id_column=source_id_column,
        value_column=value_column
    )

    agg_method = AggregationMethod(aggregation)

    joiner = SpatialJoiner(boundaries, boundary_config)
    return joiner.join(
        source,
        source_config,
        filter_boundaries=filter_boundaries,
        aggregation=agg_method
    )
