# API Reference

Complete API documentation for the Geospatial Distance Calculator library.

## Table of Contents

- [interstate_distance Module](#interstate_distance-module)
  - [distance_to_nearest_interstate](#distance_to_nearest_interstate)
  - [batch_distance_to_interstate](#batch_distance_to_interstate)
  - [preload_highway_data](#preload_highway_data)
- [nearest_site Module](#nearest_site-module)
  - [calculate_nearest_site_distances](#calculate_nearest_site_distances)
- [Constants](#constants)

---

## interstate_distance Module

Calculate distance from coordinates to the nearest US Interstate highway.

This module downloads and caches Census Bureau TIGER/Line Primary Roads data, filters to Interstate highways only, and uses EPSG:5070 projection for accurate distance calculations.

### Module Constants

```python
TIGER_PRIMARY_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
METERS_PER_MILE = 1609.344
```

---

### distance_to_nearest_interstate

Calculate the distance from a single point to the nearest US Interstate highway.

```python
def distance_to_nearest_interstate(
    latitude: float,
    longitude: float
) -> dict
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `latitude` | `float` | Latitude in decimal degrees (WGS84). Valid range: -90 to 90. |
| `longitude` | `float` | Longitude in decimal degrees (WGS84). Valid range: -180 to 180. |

#### Returns

| Key | Type | Description |
|-----|------|-------------|
| `distance_miles` | `float` | Distance to nearest Interstate in miles |
| `distance_meters` | `float` | Distance to nearest Interstate in meters |
| `nearest_highway` | `str` | Full name of the nearest Interstate (e.g., "I- 95", "I- 80") |

#### Example

```python
from interstate_distance import distance_to_nearest_interstate

# San Francisco coordinates
result = distance_to_nearest_interstate(37.7749, -122.4194)

print(f"Distance: {result['distance_miles']:.2f} miles")
print(f"Distance: {result['distance_meters']:.0f} meters")
print(f"Nearest: {result['nearest_highway']}")
```

Output:
```
Distance: 2.45 miles
Distance: 3943 meters
Nearest: I- 80
```

#### Notes

- The first call will download highway data (~15MB) from the Census Bureau
- Subsequent calls use cached data for faster response
- Uses STRtree spatial indexing for efficient nearest-neighbor queries
- Coordinates are internally projected to EPSG:5070 for accurate distance calculation

---

### batch_distance_to_interstate

Calculate distance to nearest Interstate for multiple points efficiently.

```python
def batch_distance_to_interstate(
    points_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> gpd.GeoDataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points_df` | `pd.DataFrame` | *required* | DataFrame containing latitude and longitude columns |
| `lat_col` | `str` | `'latitude'` | Name of the column containing latitude values |
| `lon_col` | `str` | `'longitude'` | Name of the column containing longitude values |

#### Returns

Returns a `gpd.GeoDataFrame` with all original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `distance_to_interstate_m` | `float` | Distance in meters |
| `distance_to_interstate_mi` | `float` | Distance in miles |
| `nearest_interstate` | `str` | Name of nearest Interstate highway |
| `geometry` | `Point` | Point geometry in WGS84 (EPSG:4326) |

#### Example

```python
import pandas as pd
from interstate_distance import batch_distance_to_interstate

# Create sample data
sites = pd.DataFrame({
    'site_id': ['A', 'B', 'C'],
    'Latitude': [37.7749, 39.7392, 40.7128],
    'Longitude': [-122.4194, -104.9903, -74.0060]
})

# Calculate distances
result = batch_distance_to_interstate(
    sites,
    lat_col='Latitude',
    lon_col='Longitude'
)

# View results
print(result[['site_id', 'nearest_interstate', 'distance_to_interstate_mi']])
```

Output:
```
  site_id nearest_interstate  distance_to_interstate_mi
0       A              I- 80                       2.45
1       B             I- 25                       0.89
2       C             I- 95                       1.23
```

#### Notes

- More efficient than calling `distance_to_nearest_interstate` repeatedly
- Uses GeoPandas `sjoin_nearest` for optimized spatial join operations
- Preserves all original DataFrame columns in the output

---

### preload_highway_data

Pre-load highway data into memory for faster subsequent queries.

```python
def preload_highway_data() -> None
```

#### Parameters

None

#### Returns

None. Prints status messages to stdout.

#### Example

```python
from interstate_distance import preload_highway_data

# Call at application startup
preload_highway_data()
# Output: Loading US Interstate highway data from Census Bureau...
#         Loaded 12,345 Interstate highway segments
#         Highway data pre-loaded and indexed.
```

#### Notes

- Call this at application startup to avoid delays on first distance query
- Downloads and caches highway data
- Builds spatial index (STRtree) for efficient queries
- Subsequent calls to distance functions will use cached data

---

## nearest_site Module

Calculate distance from each site to its nearest neighboring site.

This module processes a CSV file of site locations and calculates the distance to each site's nearest neighbor using KDTree spatial indexing.

### Module Constants

```python
PROJECT_DIR = Path(__file__).parent
INPUT_FILE = PROJECT_DIR / "Sites - Base Data Set.csv"
OUTPUT_DIR = PROJECT_DIR / "distance_results"
METERS_PER_MILE = 1609.344
```

---

### calculate_nearest_site_distances

Calculate distance from each site to its nearest neighboring site.

```python
def calculate_nearest_site_distances(
    input_file: Path = INPUT_FILE
) -> pd.DataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | `Path` | `INPUT_FILE` | Path to CSV file containing site data. Must have columns: `GTVID`, `Latitude`, `Longitude` |

#### Returns

Returns a `pd.DataFrame` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `GTVID` | `str` | Site identifier |
| `Latitude` | `float` | Original latitude (WGS84) |
| `Longitude` | `float` | Original longitude (WGS84) |
| `nearest_site` | `str` | GTVID of the nearest neighboring site |
| `nearest_site_lat` | `float` | Latitude of the nearest site |
| `nearest_site_lon` | `float` | Longitude of the nearest site |
| `nearest_site_distance_m` | `float` | Distance to nearest site in meters |
| `nearest_site_distance_mi` | `float` | Distance to nearest site in miles |

#### Input File Format

The input CSV must contain:

| Column | Required | Description |
|--------|----------|-------------|
| `GTVID` | Yes | Unique site identifier |
| `Latitude` | Yes | Latitude in decimal degrees (WGS84) |
| `Longitude` | Yes | Longitude in decimal degrees (WGS84) |

#### Example

```python
from pathlib import Path
from nearest_site import calculate_nearest_site_distances

# Use default input file
result = calculate_nearest_site_distances()

# Or specify a custom input file
result = calculate_nearest_site_distances(
    input_file=Path("/path/to/custom_sites.csv")
)

# View results
print(result[['GTVID', 'nearest_site', 'nearest_site_distance_mi']].head())
```

Output:
```
   GTVID nearest_site  nearest_site_distance_mi
0  S0001        S0042                      0.35
1  S0002        S0089                      1.24
2  S0003        S0015                      0.78
3  S0004        S0067                      2.15
4  S0005        S0023                      0.56
```

#### Algorithm

1. Load sites from CSV file
2. Convert to GeoDataFrame with point geometries
3. Project coordinates to EPSG:5070 for accurate distance calculation
4. Build KDTree spatial index from projected coordinates
5. Query for k=3 nearest neighbors (self, nearest, 2nd nearest)
6. Extract nearest neighbor (index 1, since index 0 is self)
7. Return DataFrame with distances

#### Notes

- Uses `scipy.spatial.cKDTree` for O(n log n) nearest neighbor queries
- Coordinates are projected to EPSG:5070 before distance calculation
- Automatically removes unnamed columns from input CSV
- The nearest neighbor excludes the point itself

---

## Constants

### Unit Conversion

```python
METERS_PER_MILE = 1609.344
```

Used for converting distances between meters and miles.

### Data Sources

```python
TIGER_PRIMARY_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
```

Census Bureau TIGER/Line Primary Roads shapefile URL (2024 edition).

---

## Coordinate Reference Systems

### EPSG:4326 (WGS84)

- **Usage**: Input and output coordinates
- **Description**: World Geodetic System 1984, the standard for GPS coordinates
- **Units**: Decimal degrees

### EPSG:5070 (NAD83 / Conus Albers)

- **Usage**: Internal distance calculations
- **Description**: NAD83 / Conus Albers Equal Area Conic projection
- **Units**: Meters
- **Best for**: Continental United States measurements

The library automatically handles projection conversions. All input should be in WGS84 (standard GPS coordinates), and all output coordinates are returned in WGS84.

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'Latitude'` | Column name mismatch | Specify correct column names in `lat_col`/`lon_col` parameters |
| `ConnectionError` | Cannot download Census data | Check internet connection; data is cached after first download |
| `FileNotFoundError` | Input CSV not found | Verify the path to your input file |

### Example Error Handling

```python
from interstate_distance import batch_distance_to_interstate
import pandas as pd

try:
    result = batch_distance_to_interstate(
        df,
        lat_col='lat',  # Use your column name
        lon_col='lng'   # Use your column name
    )
except KeyError as e:
    print(f"Column not found: {e}")
    print(f"Available columns: {df.columns.tolist()}")
```

---

## Performance Considerations

### Memory Usage

- Highway data: ~50-100 MB in memory after loading
- Spatial index: Additional ~20-50 MB for STRtree
- KDTree: Scales with number of input points

### Processing Speed

| Operation | Typical Performance |
|-----------|---------------------|
| First query (with download) | 30-60 seconds |
| Subsequent single queries | <100 ms |
| Batch processing (1000 points) | 2-5 seconds |
| Batch processing (100,000 points) | 30-60 seconds |

### Optimization Tips

1. Call `preload_highway_data()` at application startup
2. Use `batch_distance_to_interstate()` for multiple points
3. For very large datasets, consider chunking into batches of 50,000-100,000 points
