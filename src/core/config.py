"""
Configuration presets for common geospatial datasets.

This module provides pre-configured dataset definitions for commonly used
geospatial data sources, making it easy to load and work with standard datasets.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from datasource import DatasetConfig


# Pre-defined dataset configurations
WALKABILITY_INDEX = DatasetConfig(
    path="./data/WalkabilityIndex/Natl_WI.gdb",
    id_column="GEOID10",
    value_column="NatWalkInd",
    layer="NationalWalkabilityIndex",
    name="National Walkability Index"
)

ZCTA_2020 = DatasetConfig(
    path="./data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp",
    id_column="ZCTA5CE20",
    value_column=None,
    name="ZIP Code Tabulation Areas (2020)"
)

# Census tract boundaries
CENSUS_TRACTS_2020 = DatasetConfig(
    path="./data/tl_2020_us_tract/tl_2020_us_tract.shp",
    id_column="GEOID",
    value_column=None,
    name="Census Tracts (2020)"
)

# County boundaries
COUNTIES_2020 = DatasetConfig(
    path="./data/tl_2020_us_county/tl_2020_us_county.shp",
    id_column="GEOID",
    value_column=None,
    name="US Counties (2020)"
)

# State boundaries
STATES_2020 = DatasetConfig(
    path="./data/tl_2020_us_state/tl_2020_us_state.shp",
    id_column="STATEFP",
    value_column=None,
    name="US States (2020)"
)


@dataclass
class DatasetRegistry:
    """Registry for managing dataset configurations.

    This class provides a central location for storing and retrieving
    dataset configurations, making it easy to switch between datasets.
    """

    def __init__(self):
        """Initialize with pre-defined datasets."""
        self._datasets: Dict[str, DatasetConfig] = {}
        self._load_defaults()

    def _load_defaults(self):
        """Load default dataset configurations."""
        self.register("walkability", WALKABILITY_INDEX)
        self.register("zcta", ZCTA_2020)
        self.register("zip_codes", ZCTA_2020)  # Alias
        self.register("census_tracts", CENSUS_TRACTS_2020)
        self.register("counties", COUNTIES_2020)
        self.register("states", STATES_2020)

    def register(self, name: str, config: DatasetConfig):
        """Register a dataset configuration.

        Args:
            name: Unique identifier for the dataset
            config: Dataset configuration
        """
        self._datasets[name.lower()] = config

    def get(self, name: str) -> DatasetConfig:
        """Retrieve a dataset configuration.

        Args:
            name: Dataset identifier

        Returns:
            Dataset configuration

        Raises:
            KeyError: If dataset not found
        """
        name = name.lower()
        if name not in self._datasets:
            available = list(self._datasets.keys())
            raise KeyError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return self._datasets[name]

    def list_datasets(self) -> Dict[str, str]:
        """List all registered datasets.

        Returns:
            Dictionary mapping dataset names to their descriptions
        """
        return {
            name: config.name or config.path
            for name, config in self._datasets.items()
        }

    def create_config(
        self,
        path: str,
        id_column: str,
        value_column: Optional[str] = None,
        layer: Optional[str] = None,
        name: Optional[str] = None,
        register_as: Optional[str] = None
    ) -> DatasetConfig:
        """Create and optionally register a new dataset configuration.

        Args:
            path: Path to the data file
            id_column: Column containing unique identifiers
            value_column: Column containing values to visualize
            layer: Layer name for multi-layer formats
            name: Human-readable dataset name
            register_as: If provided, register the config with this name

        Returns:
            New DatasetConfig instance
        """
        config = DatasetConfig(
            path=path,
            id_column=id_column,
            value_column=value_column,
            layer=layer,
            name=name
        )

        if register_as:
            self.register(register_as, config)

        return config


# Global registry instance
registry = DatasetRegistry()


def get_dataset_config(name: str) -> DatasetConfig:
    """Get a dataset configuration from the global registry.

    Args:
        name: Dataset identifier

    Returns:
        Dataset configuration

    Example:
        >>> config = get_dataset_config("walkability")
        >>> print(config.value_column)
        NatWalkInd
    """
    return registry.get(name)


def register_dataset(name: str, config: DatasetConfig):
    """Register a dataset in the global registry.

    Args:
        name: Unique identifier
        config: Dataset configuration

    Example:
        >>> my_config = DatasetConfig(
        ...     path="./my_data.shp",
        ...     id_column="id",
        ...     value_column="score"
        ... )
        >>> register_dataset("my_data", my_config)
    """
    registry.register(name, config)


def list_datasets() -> Dict[str, str]:
    """List all datasets in the global registry.

    Returns:
        Dictionary mapping dataset names to descriptions
    """
    return registry.list_datasets()
