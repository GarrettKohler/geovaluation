"""
Calculate distance from coordinates to the nearest US Interstate highway.

Uses Census Bureau TIGER/Line Primary Roads data and projects to EPSG:5070
(NAD83 / Conus Albers Equal Area Conic) for accurate distance calculations.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely import STRtree
from typing import Optional, Tuple
from pathlib import Path

# Module-level cache for highway data
_highways_cache: Optional[gpd.GeoDataFrame] = None
_highways_tree: Optional[STRtree] = None

TIGER_PRIMARY_ROADS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
)

METERS_PER_MILE = 1609.344


def _load_highways(force_reload: bool = False) -> gpd.GeoDataFrame:
    """
    Load and cache US Interstate highways from Census TIGER/Line data.

    Downloads the Primary Roads shapefile, filters to Interstates only,
    and projects to EPSG:5070 for accurate distance calculations.
    """
    global _highways_cache

    if _highways_cache is not None and not force_reload:
        return _highways_cache

    print("Loading US Interstate highway data from Census Bureau...")
    highways = gpd.read_file(TIGER_PRIMARY_ROADS_URL, engine="pyogrio")

    # Filter to Interstate highways only (RTTYP = 'I')
    interstates = highways[highways['RTTYP'] == 'I'].copy()

    # Project to EPSG:5070 for accurate distance calculations in meters
    interstates_proj = interstates.to_crs(epsg=5070)

    print(f"Loaded {len(interstates_proj):,} Interstate highway segments")
    _highways_cache = interstates_proj

    return _highways_cache


def _get_spatial_index() -> Tuple[STRtree, gpd.GeoDataFrame]:
    """Get or build the spatial index for highway geometries."""
    global _highways_tree

    highways = _load_highways()

    if _highways_tree is None:
        _highways_tree = STRtree(highways.geometry.values)

    return _highways_tree, highways


def distance_to_nearest_interstate(
    latitude: float,
    longitude: float,
    include_nearest_point: bool = False,
    include_highway_segment: bool = False,
    segment_length_meters: float = 500.0
) -> dict:
    """
    Calculate the distance from a point to the nearest US Interstate highway.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees (WGS84)
    longitude : float
        Longitude in decimal degrees (WGS84)
    include_nearest_point : bool
        If True, include the coordinates of the nearest point on the highway
    include_highway_segment : bool
        If True, include coordinates of a highway segment near the connection point
    segment_length_meters : float
        Length of highway segment to return (default 500m, ~0.3 miles)

    Returns
    -------
    dict
        Dictionary containing:
        - distance_miles: Distance to nearest Interstate in miles
        - distance_meters: Distance to nearest Interstate in meters
        - nearest_highway: Name of the nearest Interstate (e.g., "I- 95")
        - nearest_point_lat: (if include_nearest_point) Latitude of nearest point on highway
        - nearest_point_lon: (if include_nearest_point) Longitude of nearest point on highway
        - highway_segment: (if include_highway_segment) List of [lat, lon] coords for segment

    Example
    -------
    >>> result = distance_to_nearest_interstate(37.7749, -122.4194)
    >>> print(f"Distance: {result['distance_miles']:.2f} miles to {result['nearest_highway']}")
    """
    from shapely.ops import nearest_points, substring

    # Create a point and project to EPSG:5070
    point = gpd.GeoDataFrame(
        geometry=[Point(longitude, latitude)],
        crs="EPSG:4326"
    ).to_crs(epsg=5070)

    point_geom = point.geometry.values[0]

    # Get spatial index and highways
    tree, highways = _get_spatial_index()

    # Find nearest highway
    nearest_idx = tree.nearest(point_geom)
    nearest_highway = highways.iloc[nearest_idx]

    # Calculate distance
    distance_m = point_geom.distance(nearest_highway.geometry)
    distance_mi = distance_m / METERS_PER_MILE

    result = {
        "distance_miles": distance_mi,
        "distance_meters": distance_m,
        "nearest_highway": nearest_highway['FULLNAME']
    }

    if include_nearest_point or include_highway_segment:
        # Get the actual nearest point on the highway geometry
        _, highway_point = nearest_points(point_geom, nearest_highway.geometry)

        if include_nearest_point:
            # Convert back to WGS84
            highway_point_gdf = gpd.GeoDataFrame(
                geometry=[highway_point], crs="EPSG:5070"
            ).to_crs(epsg=4326)
            nearest_pt = highway_point_gdf.geometry.values[0]
            result["nearest_point_lat"] = nearest_pt.y
            result["nearest_point_lon"] = nearest_pt.x

        if include_highway_segment:
            # Get a segment of the highway around the connection point
            highway_line = nearest_highway.geometry
            total_length = highway_line.length

            # Find the distance along the line to the nearest point
            dist_along = highway_line.project(highway_point)

            # Calculate start and end distances for the segment
            half_segment = segment_length_meters / 2
            start_dist = max(0, dist_along - half_segment)
            end_dist = min(total_length, dist_along + half_segment)

            # Extract the segment using substring
            segment = substring(highway_line, start_dist, end_dist)

            # Convert segment to WGS84 coordinates
            segment_gdf = gpd.GeoDataFrame(
                geometry=[segment], crs="EPSG:5070"
            ).to_crs(epsg=4326)
            segment_wgs84 = segment_gdf.geometry.values[0]

            # Extract coordinates as list of [lat, lon] pairs
            if segment_wgs84.geom_type == 'LineString':
                coords = [[coord[1], coord[0]] for coord in segment_wgs84.coords]
            else:
                # Handle MultiLineString if it occurs
                coords = []
                for part in segment_wgs84.geoms:
                    coords.extend([[coord[1], coord[0]] for coord in part.coords])

            result["highway_segment"] = coords

    return result


def batch_distance_to_interstate(
    points_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> gpd.GeoDataFrame:
    """
    Calculate distance to nearest Interstate for multiple points.

    More efficient than calling distance_to_nearest_interstate repeatedly
    for large datasets.

    Parameters
    ----------
    points_df : pd.DataFrame
        DataFrame with latitude and longitude columns
    lat_col : str
        Name of the latitude column (default: 'latitude')
    lon_col : str
        Name of the longitude column (default: 'longitude')

    Returns
    -------
    gpd.GeoDataFrame
        Original data with added columns:
        - distance_to_interstate_m: Distance in meters
        - distance_to_interstate_mi: Distance in miles
        - nearest_interstate: Name of nearest Interstate
    """
    # Load highways
    highways = _load_highways()

    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        points_df.copy(),
        geometry=gpd.points_from_xy(points_df[lon_col], points_df[lat_col]),
        crs="EPSG:4326"
    )

    # Project to EPSG:5070
    points_proj = points_gdf.to_crs(epsg=5070)

    # Use sjoin_nearest for efficient batch processing
    result = gpd.sjoin_nearest(
        points_proj,
        highways[['geometry', 'FULLNAME']],
        how='left',
        distance_col='distance_to_interstate_m'
    )

    # Add miles column and rename
    result['distance_to_interstate_mi'] = result['distance_to_interstate_m'] / METERS_PER_MILE
    result = result.rename(columns={'FULLNAME': 'nearest_interstate'})

    # Convert back to WGS84 and drop index column from join
    result = result.to_crs(epsg=4326)
    if 'index_right' in result.columns:
        result = result.drop(columns=['index_right'])

    return result


def preload_highway_data() -> None:
    """
    Pre-load highway data into memory.

    Call this at application startup to avoid delay on first distance query.
    """
    _load_highways()
    _get_spatial_index()
    print("Highway data pre-loaded and indexed.")


if __name__ == "__main__":
    # Simple demo
    print("Testing distance to nearest Interstate...")

    # San Francisco coordinates
    result = distance_to_nearest_interstate(37.7749, -122.4194)
    print(f"\nSan Francisco, CA:")
    print(f"  Distance: {result['distance_miles']:.2f} miles")
    print(f"  Nearest: {result['nearest_highway']}")

    # Denver coordinates
    result = distance_to_nearest_interstate(39.7392, -104.9903)
    print(f"\nDenver, CO:")
    print(f"  Distance: {result['distance_miles']:.2f} miles")
    print(f"  Nearest: {result['nearest_highway']}")
