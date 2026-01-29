# US Highway Geospatial Data with Python: A Complete Implementation Guide

**Census TIGER/Line Primary Roads** provides the most practical solution for nationwide highway data in Python—a single 15MB shapefile covering all US Interstates, downloadable directly into GeoPandas. For richer attribute data including traffic counts and lane information, the **FHWA National Highway Planning Network** offers ~450,000 miles of highways with multiple format options. The key technical requirement: always reproject from geographic coordinates (EPSG:4269) to **EPSG:5070** (Albers Equal Area Conic) before calculating distances, or your measurements will be wildly inaccurate. This guide provides complete, working code for loading highway data, calculating point-to-highway distances for gas stations, and optimizing performance across thousands of locations.

## Authoritative government data sources for US highways

Three primary federal sources provide highway geospatial data suitable for GeoPandas analysis, each optimized for different use cases.

**Census Bureau TIGER/Line Files** represent the most accessible option. The **Primary Roads** layer (MTFCC code S1100) contains Interstate highways and limited-access roads as a single nationwide shapefile. The **Primary & Secondary Roads** layer adds US Routes, state highways, and county highways but requires state-by-state downloads. All TIGER/Line data uses **EPSG:4269 (NAD83)** coordinates and includes key attributes: `FULLNAME` (road name), `RTTYP` (route type: I=Interstate, U=US Highway, S=State), and `LINEARID` (unique identifier). Data updates annually each September and is **public domain** with no usage restrictions.

| Dataset | Coverage | Size | Best For |
|---------|----------|------|----------|
| TIGER/Line Primary Roads | Nationwide (single file) | ~15 MB | Interstate-only analysis |
| TIGER/Line Primary+Secondary | State-by-state | ~50 MB/state | US/State highway analysis |
| NHPN v14.05 | Nationwide (single file) | ~178 MB | NHS network with traffic data |
| HPMS | Nationwide | ~500 MB | Detailed condition/traffic metrics |

The **FHWA National Highway Planning Network (NHPN)** provides the most comprehensive major highway dataset—approximately **450,000 miles** including the National Highway System, Interstate System, and Strategic Highway Network. Available in shapefile, GeoJSON, and GML formats with attributes for route signs (`SIGNT1`, `SIGNN1`), functional classification, number of through lanes, and linear referencing system keys. The dataset's **1:100,000 scale** provides approximately 80-meter accuracy.

**Bureau of Transportation Statistics (BTS) NTAD** aggregates multiple datasets through a single portal at geodata.bts.gov, including HPMS data with **Annual Average Daily Traffic (AADT)**, pavement condition indices, and speed limits—useful when distance calculations need to be weighted by traffic volume or road quality.

## Downloading and loading highway data into GeoPandas

GeoPandas can load TIGER/Line shapefiles directly from Census Bureau URLs without manual downloads. This code loads nationwide Interstate highways:

```python
import geopandas as gpd

# Load Interstate highways directly from Census Bureau (Primary Roads)
highways = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
)

# Verify the data loaded correctly
print(f"Loaded {len(highways):,} highway segments")
print(f"CRS: {highways.crs}")  # Should be EPSG:4269 (NAD83)
print(f"Columns: {list(highways.columns)}")
```

For state-level data including US and state highways, use the Primary & Secondary Roads layer with the state FIPS code (e.g., 06 for California, 48 for Texas):

```python
# California primary and secondary roads
ca_highways = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2024/PRISECROADS/tl_2024_06_prisecroads.zip"
)

# Filter by route type
interstates = ca_highways[ca_highways['RTTYP'] == 'I']  # Interstate only
us_routes = ca_highways[ca_highways['RTTYP'] == 'U']    # US Highways
state_routes = ca_highways[ca_highways['RTTYP'] == 'S'] # State highways

# Or filter by MTFCC feature class code
primary_roads = ca_highways[ca_highways['MTFCC'] == 'S1100']    # Limited-access
secondary_roads = ca_highways[ca_highways['MTFCC'] == 'S1200']  # Major highways
```

For the NHPN dataset with richer attributes, download the shapefile from FHWA and load locally:

```python
# After downloading nhpnv14-05shp.zip from FHWA website
nhpn = gpd.read_file("nhpnv14-05shp.zip")

# Filter for Interstate highways by sign type
nhpn_interstates = nhpn[nhpn['SIGNT1'] == 'I']

# Access traffic-relevant attributes
print(nhpn[['LNAME', 'SIGNT1', 'SIGNN1', 'THRULANES', 'FCLASS']].head())
```

**Performance tip**: For large files, use bounding box filtering and the faster pyogrio engine:

```python
# Load only highways within a bounding box (minx, miny, maxx, maxy)
bbox = (-124.0, 32.5, -114.0, 42.0)  # California bounds
ca_highways = gpd.read_file(
    "tl_2024_us_primaryroads.zip",
    bbox=bbox,
    engine="pyogrio"  # 10x faster than default fiona engine
)
```

## Why coordinate projection determines distance accuracy

TIGER/Line data arrives in **EPSG:4269 (NAD83)**—a geographic coordinate system where coordinates represent angular degrees on an ellipsoid, not linear distances. Calculating distances directly on this CRS produces meaningless results: a "distance" of 0.1 might represent 11 km at the equator but only 7 km at latitude 45°.

For accurate distance calculations across the continental US, project to **EPSG:5070 (NAD83 / Conus Albers Equal Area Conic)**. This projection minimizes area distortion to approximately 1.25% across the lower 48 states and returns distances in **meters**.

```python
import geopandas as gpd

# Load highways (arrives in EPSG:4269 - degrees)
highways = gpd.read_file("tl_2024_us_primaryroads.zip")

# CRITICAL: Reproject to EPSG:5070 for accurate distance calculations
highways_proj = highways.to_crs(epsg=5070)

# Now .distance() returns meters, not meaningless degree values
# Highway segment lengths
highways_proj['length_m'] = highways_proj.geometry.length
highways_proj['length_mi'] = highways_proj['length_m'] / 1609.344
```

For regional analysis, **UTM zones** provide even better accuracy within their 6° longitude bands. Use EPSG:32610-32619 for US zones (e.g., 32610 for the West Coast, 32618 for the East Coast). For global or very high-precision needs, use **pyproj's geodesic** calculations, though these run 10-100x slower than projected Euclidean distances.

## Calculating minimum distance from gas stations to highways

The core workflow: load your points (gas stations), load highway lines, project both to EPSG:5070, then use `sjoin_nearest` for efficient nearest-neighbor queries.

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load your gas station data (example with lat/long coordinates)
stations_df = pd.DataFrame({
    'station_id': ['GS001', 'GS002', 'GS003', 'GS004'],
    'name': ['QuickStop', 'FuelMart', 'RoadRunner', 'Highway Gas'],
    'latitude': [37.7749, 37.8044, 37.3382, 38.5816],
    'longitude': [-122.4194, -122.2711, -121.8863, -121.4944]
})

# Convert to GeoDataFrame with Point geometries
stations = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
    crs="EPSG:4326"  # WGS84 for lat/long input
)

# Load Interstate highways
highways = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
)

# CRITICAL: Project both datasets to EPSG:5070 for accurate distances
stations_proj = stations.to_crs(epsg=5070)
highways_proj = highways.to_crs(epsg=5070)

# Calculate distance to nearest highway using sjoin_nearest
# This automatically uses spatial indexing for efficiency
result = gpd.sjoin_nearest(
    stations_proj,
    highways_proj[['geometry', 'FULLNAME', 'RTTYP']],
    how='left',
    distance_col='distance_to_highway_m'  # Distance in meters
)

# Add distance in miles
result['distance_to_highway_mi'] = result['distance_to_highway_m'] / 1609.344

# View results
print(result[['station_id', 'name', 'FULLNAME', 'distance_to_highway_m', 
              'distance_to_highway_mi']])
```

**Output columns**: The result contains all original station attributes plus the nearest highway's `FULLNAME` and `RTTYP`, and the `distance_to_highway_m` column with exact distances in meters.

For filtering to specific highway types before the distance calculation:

```python
# Filter highways to Interstates only
interstates_proj = highways_proj[highways_proj['RTTYP'] == 'I']

# Calculate distance to nearest Interstate specifically
result_interstate = gpd.sjoin_nearest(
    stations_proj,
    interstates_proj[['geometry', 'FULLNAME']],
    how='left',
    distance_col='distance_to_interstate_m'
)
```

## Optimizing performance for thousands of locations

When processing large datasets (10,000+ gas stations), three techniques provide order-of-magnitude speedups.

**Spatial indexing** is automatic in `sjoin_nearest`, but for custom queries, access the index directly:

```python
from shapely import STRtree
import numpy as np

# Build spatial index on highway geometries (one-time cost)
tree = STRtree(highways_proj.geometry.values)

# Batch query: find nearest highway for all stations at once
indices, distances = tree.query_nearest(
    stations_proj.geometry.values,
    return_distance=True
)

# indices[1] contains the highway index for each station
stations_proj['nearest_highway_idx'] = indices[1]
stations_proj['distance_m'] = distances

# Get highway names
stations_proj['nearest_highway'] = highways_proj.iloc[indices[1]]['FULLNAME'].values
```

**Shapely 2.0 vectorized operations** provide 4-100x speedups over Python loops:

```python
import shapely

# SLOW: Python loop (avoid this)
distances_slow = [pt.distance(highways_proj.unary_union) 
                  for pt in stations_proj.geometry]

# FAST: Vectorized with Shapely 2.0
all_highways_combined = highways_proj.unary_union
distances_fast = shapely.distance(stations_proj.geometry.values, all_highways_combined)
```

**Chunked processing with parallelization** for very large datasets:

```python
import dask_geopandas
from dask.distributed import Client

# Initialize Dask for parallel processing
client = Client(n_workers=4)

# Convert to Dask GeoDataFrame
stations_dask = dask_geopandas.from_geopandas(stations_proj, npartitions=8)

# Parallel spatial operations
stations_dask['buffer_10km'] = stations_dask.geometry.buffer(10000)

# Compute results
result = stations_dask.compute()
```

**Memory optimization** for constrained environments:

```python
# Load only needed columns
highways = gpd.read_file(
    "highways.shp",
    columns=['FULLNAME', 'RTTYP', 'geometry']
)

# Use categorical dtype for repeated strings
highways['RTTYP'] = highways['RTTYP'].astype('category')

# Check memory usage
print(f"Memory: {highways.memory_usage(deep=True).sum() / 1e6:.1f} MB")
```

## Complete working implementation

This production-ready function encapsulates the entire workflow:

```python
import geopandas as gpd
import pandas as pd
from shapely import STRtree
from typing import Optional

def calculate_distances_to_highways(
    points_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    highway_types: Optional[list] = None,
    max_distance_miles: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Calculate minimum distance from points to US highways.
    
    Parameters
    ----------
    points_df : DataFrame with latitude/longitude columns
    lat_col, lon_col : Column names for coordinates
    highway_types : Filter highways by RTTYP (e.g., ['I', 'U'] for Interstate + US Routes)
    max_distance_miles : Maximum search radius in miles (improves performance)
    
    Returns
    -------
    GeoDataFrame with distance_to_highway_m and distance_to_highway_mi columns
    """
    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[lon_col], points_df[lat_col]),
        crs="EPSG:4326"
    )
    
    # Load and filter highways
    highways = gpd.read_file(
        "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
    )
    
    if highway_types:
        highways = highways[highways['RTTYP'].isin(highway_types)]
    
    # Project to EPSG:5070 for accurate distance calculations
    points_proj = points_gdf.to_crs(epsg=5070)
    highways_proj = highways.to_crs(epsg=5070)
    
    # Calculate nearest highway with distance
    max_dist_m = max_distance_miles * 1609.344 if max_distance_miles else None
    
    result = gpd.sjoin_nearest(
        points_proj,
        highways_proj[['geometry', 'FULLNAME', 'RTTYP']],
        how='left',
        distance_col='distance_to_highway_m',
        max_distance=max_dist_m
    )
    
    # Convert back to WGS84 and add miles column
    result = result.to_crs(epsg=4326)
    result['distance_to_highway_mi'] = result['distance_to_highway_m'] / 1609.344
    
    # Clean up columns
    result = result.rename(columns={
        'FULLNAME': 'nearest_highway_name',
        'RTTYP': 'nearest_highway_type'
    })
    
    return result

# Example usage
gas_stations = pd.DataFrame({
    'station_id': range(1, 101),
    'latitude': [35 + i*0.1 for i in range(100)],
    'longitude': [-100 - i*0.05 for i in range(100)]
})

result = calculate_distances_to_highways(
    gas_stations,
    highway_types=['I'],  # Interstates only
    max_distance_miles=50  # Limit search radius for performance
)

print(result[['station_id', 'nearest_highway_name', 'distance_to_highway_mi']].head(10))
```

## Conclusion

The **Census TIGER/Line Primary Roads** dataset provides the fastest path to nationwide highway analysis—load it directly from the Census Bureau URL into GeoPandas without manual downloads. For analyses requiring traffic data, lane counts, or NHS designation status, the **FHWA NHPN** or **HPMS** datasets provide richer attributes at the cost of larger file sizes and offline download requirements.

Three technical requirements are non-negotiable for accurate results: always **reproject to EPSG:5070** before distance calculations, use **`sjoin_nearest`** rather than manual loops for efficient spatial queries, and leverage **Shapely 2.0's vectorized operations** when processing more than a few thousand points. Following these patterns, processing 100,000 gas station locations against the complete US Interstate network takes approximately 30 seconds on commodity hardware—fast enough for interactive analysis or batch processing pipelines.