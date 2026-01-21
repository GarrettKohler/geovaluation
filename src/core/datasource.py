"""
Data source abstraction for loading geospatial datasets.

This module provides flexible data loading for various geospatial formats
including Shapefiles, GeoDatabase, GeoJSON, and more.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union

import geopandas as gpd
import fiona


@dataclass
class DatasetConfig:
    """Configuration for a geospatial dataset.

    Attributes:
        path: Path to the data file
        id_column: Column name containing unique identifiers
        value_column: Column name containing the values to visualize/aggregate
        geometry_column: Column name for geometry (default: 'geometry')
        layer: Layer name for multi-layer formats like GeoDatabase
        name: Human-readable name for the dataset
    """
    path: str
    id_column: str
    value_column: Optional[str] = None
    geometry_column: str = "geometry"
    layer: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = Path(self.path).stem


class DataSource(ABC):
    """Abstract base class for geospatial data sources."""

    @abstractmethod
    def load(self) -> gpd.GeoDataFrame:
        """Load and return the geospatial data."""
        pass

    @abstractmethod
    def get_config(self) -> DatasetConfig:
        """Return the dataset configuration."""
        pass


class FileDataSource(DataSource):
    """Data source that loads from a file path.

    Supports various geospatial formats:
    - Shapefile (.shp)
    - GeoDatabase (.gdb)
    - GeoJSON (.geojson, .json)
    - GeoPackage (.gpkg)
    - And other formats supported by GeoPandas/Fiona
    """

    def __init__(self, config: DatasetConfig):
        """Initialize the data source.

        Args:
            config: Dataset configuration specifying path and column mappings
        """
        self.config = config
        self._data: Optional[gpd.GeoDataFrame] = None

    def load(self) -> gpd.GeoDataFrame:
        """Load the geospatial data from file.

        Returns:
            GeoDataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the specified layer doesn't exist
        """
        if self._data is not None:
            return self._data

        path = Path(self.config.path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Handle GeoDatabase with layer selection
        if path.suffix.lower() == '.gdb':
            self._data = self._load_geodatabase(path)
        else:
            self._data = gpd.read_file(str(path))

        self._validate_columns()
        return self._data

    def _load_geodatabase(self, path: Path) -> gpd.GeoDataFrame:
        """Load data from a GeoDatabase file.

        Args:
            path: Path to the .gdb file

        Returns:
            GeoDataFrame from the specified layer
        """
        available_layers = fiona.listlayers(str(path))

        if self.config.layer is None:
            if len(available_layers) == 1:
                self.config.layer = available_layers[0]
            else:
                raise ValueError(
                    f"GeoDatabase has multiple layers: {available_layers}. "
                    "Please specify a layer in the config."
                )

        if self.config.layer not in available_layers:
            raise ValueError(
                f"Layer '{self.config.layer}' not found. "
                f"Available layers: {available_layers}"
            )

        return gpd.read_file(str(path), layer=self.config.layer)

    def _validate_columns(self):
        """Validate that required columns exist in the loaded data."""
        if self._data is None:
            return

        missing = []
        if self.config.id_column not in self._data.columns:
            missing.append(f"id_column: {self.config.id_column}")

        if self.config.value_column and self.config.value_column not in self._data.columns:
            missing.append(f"value_column: {self.config.value_column}")

        if missing:
            available = list(self._data.columns)
            raise ValueError(
                f"Missing columns in dataset: {missing}. "
                f"Available columns: {available}"
            )

    def get_config(self) -> DatasetConfig:
        """Return the dataset configuration."""
        return self.config

    def list_layers(self) -> List[str]:
        """List available layers for GeoDatabase files.

        Returns:
            List of layer names, or empty list for non-GDB files
        """
        path = Path(self.config.path)
        if path.suffix.lower() == '.gdb':
            return fiona.listlayers(str(path))
        return []

    def get_columns(self) -> List[str]:
        """Get list of available columns after loading data.

        Returns:
            List of column names in the dataset
        """
        if self._data is None:
            self.load()
        return list(self._data.columns)


class BoundaryDataSource(FileDataSource):
    """Specialized data source for boundary/region data (ZIP codes, counties, etc.).

    This is a convenience class that provides common defaults for
    boundary datasets used in spatial joins.
    """

    def __init__(
        self,
        path: str,
        id_column: str,
        name: Optional[str] = None,
        layer: Optional[str] = None
    ):
        """Initialize boundary data source.

        Args:
            path: Path to the boundary data file
            id_column: Column containing boundary identifiers (e.g., ZIP code)
            name: Human-readable name for the dataset
            layer: Layer name for multi-layer formats
        """
        config = DatasetConfig(
            path=path,
            id_column=id_column,
            value_column=None,
            layer=layer,
            name=name or "Boundaries"
        )
        super().__init__(config)


class ValueDataSource(FileDataSource):
    """Specialized data source for value/metric data to be mapped.

    This class is for datasets containing values that will be
    visualized or aggregated onto boundary regions.
    """

    def __init__(
        self,
        path: str,
        id_column: str,
        value_column: str,
        name: Optional[str] = None,
        layer: Optional[str] = None
    ):
        """Initialize value data source.

        Args:
            path: Path to the data file
            id_column: Column containing unique identifiers
            value_column: Column containing values to map/aggregate
            name: Human-readable name for the dataset
            layer: Layer name for multi-layer formats
        """
        config = DatasetConfig(
            path=path,
            id_column=id_column,
            value_column=value_column,
            layer=layer,
            name=name or "Values"
        )
        super().__init__(config)


def load_dataset(
    path: str,
    id_column: str,
    value_column: Optional[str] = None,
    layer: Optional[str] = None,
    name: Optional[str] = None
) -> gpd.GeoDataFrame:
    """Convenience function to quickly load a geospatial dataset.

    Args:
        path: Path to the data file
        id_column: Column containing unique identifiers
        value_column: Optional column containing values to visualize
        layer: Layer name for multi-layer formats like GeoDatabase
        name: Human-readable name for the dataset

    Returns:
        Loaded GeoDataFrame

    Example:
        >>> gdf = load_dataset(
        ...     path="data/walkability.gdb",
        ...     id_column="GEOID10",
        ...     value_column="NatWalkInd",
        ...     layer="NationalWalkabilityIndex"
        ... )
    """
    config = DatasetConfig(
        path=path,
        id_column=id_column,
        value_column=value_column,
        layer=layer,
        name=name
    )
    source = FileDataSource(config)
    return source.load()
