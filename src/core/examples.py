"""
Examples demonstrating the geospatial mapping toolkit.

This module provides example code showing how to:
1. Load and visualize arbitrary datasets
2. Perform spatial joins with different aggregations
3. Create choropleth maps
4. Compare multiple datasets
"""

import geopandas as gpd

from datasource import (
    DatasetConfig,
    FileDataSource,
    load_dataset,
)
from spatial_ops import (
    AggregationMethod,
    SpatialJoiner,
    spatial_join,
)
from visualizer import (
    ColorScale,
    MapStyle,
    MapVisualizer,
    plot_map,
    compare_maps,
)
from config import (
    register_dataset,
    get_dataset_config,
    list_datasets,
    WALKABILITY_INDEX,
    ZCTA_2020,
)
from zip_code_mapping import ZipCodeMapper


def example_basic_visualization():
    """Example: Basic map visualization of any GeoDataFrame.

    This shows how to quickly visualize any GeoDataFrame with values.
    """
    print("Example: Basic Visualization")
    print("-" * 40)

    # Load any dataset
    data = load_dataset(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Quick plot with convenience function
    fig, ax = plot_map(
        data=data,
        value_column="NatWalkInd",
        title="National Walkability Index by Census Block",
        colormap="YlGnBu"
    )

    print("Created visualization of walkability data")
    return fig, ax


def example_custom_dataset():
    """Example: Working with a custom dataset.

    Shows how to register and use your own dataset configuration.
    """
    print("Example: Custom Dataset Registration")
    print("-" * 40)

    # Define your dataset configuration
    my_config = DatasetConfig(
        path="./my_data/population.shp",
        id_column="TRACT_ID",
        value_column="POPULATION",
        name="Population by Tract"
    )

    # Register it for reuse
    register_dataset("population", my_config)

    # List all available datasets
    datasets = list_datasets()
    print("Available datasets:")
    for name, description in datasets.items():
        print(f"  - {name}: {description}")

    # Later, retrieve and use it
    # config = get_dataset_config("population")
    # data = FileDataSource(config).load()

    return datasets


def example_spatial_join():
    """Example: Spatial join with different aggregations.

    Shows how to join value data to boundaries with various
    aggregation methods.
    """
    print("Example: Spatial Join Operations")
    print("-" * 40)

    # Load boundary and value data
    boundaries = load_dataset(
        path="./data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp",
        id_column="ZCTA5CE20"
    )

    values = load_dataset(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Use the convenience function for quick joins
    result_mean = spatial_join(
        boundaries=boundaries,
        boundary_id_column="ZCTA5CE20",
        source=values,
        source_id_column="GEOID10",
        value_column="NatWalkInd",
        filter_boundaries=["48201", "48202", "48203"],
        aggregation="mean"
    )

    print(f"Mean walkability result columns: {list(result_mean.data.columns)}")
    print(f"Value column: {result_mean.value_column}")

    # Or use different aggregations
    result_sum = spatial_join(
        boundaries=boundaries,
        boundary_id_column="ZCTA5CE20",
        source=values,
        source_id_column="GEOID10",
        value_column="NatWalkInd",
        aggregation="sum"
    )

    return result_mean, result_sum


def example_zip_code_mapper():
    """Example: Using ZipCodeMapper for end-to-end workflow.

    Shows the high-level API for mapping any data to ZIP codes.
    """
    print("Example: ZipCodeMapper Workflow")
    print("-" * 40)

    mapper = ZipCodeMapper()

    # Load ZIP code boundaries
    mapper.load_boundaries(
        path="./data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp",
        id_column="ZCTA5CE20"
    )

    # Load value data
    mapper.load_values(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Map to specific ZIP codes
    result = mapper.map_to_boundaries(
        boundary_ids=["48201", "48202", "48203", "48204", "48205"],
        aggregation="mean"
    )

    print(f"Mapped {len(result.data)} ZIP codes")
    print(f"Result columns: {list(result.data.columns)}")

    # Visualize
    fig, ax = mapper.visualize(
        result,
        title="Walkability by ZIP Code (Detroit Area)",
        colormap=ColorScale.YELLOW_GREEN_BLUE
    )

    return mapper, result, fig, ax


def example_compare_datasets():
    """Example: Comparing multiple datasets side by side.

    Shows how to create a grid of maps for comparison.
    """
    print("Example: Dataset Comparison")
    print("-" * 40)

    # Assume we have multiple aggregated datasets
    # In practice, these would come from different sources or aggregations

    mapper = ZipCodeMapper()
    mapper.load_boundaries(
        path="./data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp",
        id_column="ZCTA5CE20"
    )
    mapper.load_values(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    zip_codes = ["48201", "48202", "48203", "48204", "48205"]

    # Get results with different aggregations
    result_mean = mapper.map_to_boundaries(zip_codes, aggregation="mean")
    result_max = mapper.map_to_boundaries(zip_codes, aggregation="max")
    result_min = mapper.map_to_boundaries(zip_codes, aggregation="min")

    # Create comparison grid
    datasets = [
        (result_mean.data, result_mean.value_column, "Mean Walkability"),
        (result_max.data, result_max.value_column.replace("mean", "max"), "Max Walkability"),
        (result_min.data, result_min.value_column.replace("mean", "min"), "Min Walkability"),
    ]

    # Note: compare_maps expects the value column to exist
    # This is a conceptual example

    print("Created comparison datasets for mean, max, and min aggregations")
    return datasets


def example_custom_styling():
    """Example: Custom map styling options.

    Shows how to customize the appearance of map visualizations.
    """
    print("Example: Custom Styling")
    print("-" * 40)

    data = load_dataset(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Create custom style
    style = MapStyle(
        colormap=ColorScale.SPECTRAL,
        edge_color="white",
        edge_width=0.3,
        alpha=0.9,
        missing_color="#cccccc",
        figsize=(15, 10),
        title="National Walkability Index",
        legend=True,
        legend_label="Walkability Score"
    )

    # Create visualization with custom style
    viz = MapVisualizer(style)
    fig, ax = viz.plot(data, "NatWalkInd")

    print("Created map with custom styling")
    return fig, ax


def example_multi_layer_map():
    """Example: Multi-layer map with boundaries and values.

    Shows how to overlay multiple layers on a single map.
    """
    print("Example: Multi-Layer Map")
    print("-" * 40)

    # Load boundary layer (e.g., state outlines)
    # boundaries = load_dataset(
    #     path="./data/states.shp",
    #     id_column="STATE_FIPS"
    # )

    # Load value layer
    values = load_dataset(
        path="./data/WalkabilityIndex/Natl_WI.gdb",
        id_column="GEOID10",
        value_column="NatWalkInd",
        layer="NationalWalkabilityIndex"
    )

    # Create visualizer and add layers
    viz = MapVisualizer()

    # Add value layer (bottom)
    viz.add_layer(
        data=values,
        value_column="NatWalkInd",
        style=MapStyle(colormap=ColorScale.VIRIDIS, alpha=0.8),
        zorder=1
    )

    # Could add boundary layer on top
    # viz.add_layer(
    #     data=boundaries,
    #     style=MapStyle(edge_color="black", edge_width=1.5, missing_color="none"),
    #     zorder=2
    # )

    # Render all layers
    fig, ax = viz.plot()

    print("Created multi-layer map")
    return fig, ax


def example_flexible_mapping():
    """Example: Mapping any dataset to any boundary type.

    Shows the flexibility of the architecture for different use cases.
    """
    print("Example: Flexible Mapping Scenarios")
    print("-" * 40)

    # Scenario 1: Map income data to counties
    print("\nScenario 1: Income data -> Counties")
    print("  income_mapper = ZipCodeMapper()")
    print("  income_mapper.load_boundaries('counties.shp', 'COUNTY_FIPS')")
    print("  income_mapper.load_values('income.csv', 'TRACT_ID', 'MEDIAN_INCOME')")
    print("  result = income_mapper.map_to_boundaries(aggregation='median')")

    # Scenario 2: Map crime data to neighborhoods
    print("\nScenario 2: Crime data -> Neighborhoods")
    print("  crime_mapper = ZipCodeMapper()")
    print("  crime_mapper.load_boundaries('neighborhoods.geojson', 'HOOD_ID')")
    print("  crime_mapper.load_values('crimes.shp', 'INCIDENT_ID', 'SEVERITY')")
    print("  result = crime_mapper.map_to_boundaries(aggregation='sum')")

    # Scenario 3: Map environmental data to census tracts
    print("\nScenario 3: Pollution data -> Census Tracts")
    print("  env_mapper = ZipCodeMapper()")
    print("  env_mapper.load_boundaries('census_tracts.shp', 'GEOID')")
    print("  env_mapper.load_values('pollution.gdb', 'SITE_ID', 'PM25_LEVEL', layer='AirQuality')")
    print("  result = env_mapper.map_to_boundaries(aggregation='max')")

    print("\nThe architecture supports any combination of:")
    print("  - Boundary types: ZIP, county, tract, state, custom polygons")
    print("  - Value sources: shapefiles, GeoDatabase, GeoJSON, GeoPackage")
    print("  - Aggregations: mean, sum, count, min, max, median, custom")


if __name__ == "__main__":
    print("=" * 60)
    print("Geospatial Mapping Toolkit - Examples")
    print("=" * 60)
    print()

    # Run conceptual examples (no actual data needed)
    example_custom_dataset()
    print()

    example_flexible_mapping()
    print()

    print("=" * 60)
    print("To run examples with actual data, ensure the data files exist")
    print("in the ./data directory, then uncomment the relevant examples.")
    print("=" * 60)
