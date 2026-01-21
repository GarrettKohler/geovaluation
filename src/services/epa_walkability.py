"""
EPA Walkability Score Lookup
Uses EPA Smart Location Database joined with Census block group shapefiles
to return walkability scores for any lat/long coordinate.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Union
from shapely.geometry import Point

# Cache directory for downloaded data
CACHE_DIR = Path.home() / ".cache" / "epa_walkability"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# EPA Smart Location Database URL (Version 3.0, 2021)
EPA_SLD_URL = "https://edg.epa.gov/EPADataCommons/public/OA/EPA_SmartLocationDatabase_V3_Jan_2021_Final.csv"

# Census TIGER/Line block groups (2020)
# Note: You'll need to download by state
CENSUS_TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_{state_fips}_bg.zip"


def download_epa_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download and cache the EPA Smart Location Database.

    Args:
        force_refresh: If True, re-download even if cached

    Returns:
        DataFrame with EPA walkability scores by block group
    """
    cache_file = CACHE_DIR / "epa_sld.csv"

    if cache_file.exists() and not force_refresh:
        print(f"Loading EPA data from cache: {cache_file}")
        return pd.read_csv(
            cache_file,
            converters={'GEOID10': str, 'GEOID20': str},
            low_memory=False
        )

    print("Downloading EPA Smart Location Database (~220MB CSV)...")
    print(f"From: {EPA_SLD_URL}")

    # Download CSV directly with proper GEOID handling
    # Use converters to ensure GEOIDs are read as strings (not scientific notation)
    df = pd.read_csv(
        EPA_SLD_URL,
        converters={'GEOID10': str, 'GEOID20': str},
        low_memory=False
    )

    print(f"Downloaded {len(df):,} block groups")

    # Cache it
    df.to_csv(cache_file, index=False)
    print(f"Cached EPA data to: {cache_file}")

    return df


def download_census_blockgroups(state_fips: Union[str, list],
                                  force_refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Download Census block group shapefiles for specified state(s).

    Args:
        state_fips: Two-digit state FIPS code(s), e.g., "06" for CA, or list of codes
        force_refresh: If True, re-download even if cached

    Returns:
        GeoDataFrame with block group geometries
    """
    if isinstance(state_fips, str):
        state_fips = [state_fips]

    all_blockgroups = []

    for fips in state_fips:
        cache_file = CACHE_DIR / f"blockgroups_{fips}.gpkg"

        if cache_file.exists() and not force_refresh:
            print(f"Loading block groups for state {fips} from cache")
            gdf = gpd.read_file(cache_file)
        else:
            url = CENSUS_TIGER_URL.format(state_fips=fips)
            print(f"Downloading block groups for state {fips}...")
            print(f"From: {url}")

            # GeoPandas can read directly from zip URL
            gdf = gpd.read_file(url)

            # Cache it
            gdf.to_file(cache_file, driver="GPKG")
            print(f"Cached to: {cache_file}")

        all_blockgroups.append(gdf)

    # Combine all states
    combined = gpd.GeoDataFrame(pd.concat(all_blockgroups, ignore_index=True))

    return combined


def build_walkability_index(state_fips: Union[str, list] = None,
                            force_refresh: bool = False) -> gpd.GeoDataFrame:
    """
    Build complete walkability index by joining EPA data with Census geometries.

    Args:
        state_fips: State(s) to include. If None, downloads all states (very large!)
        force_refresh: If True, re-download all data

    Returns:
        GeoDataFrame with walkability scores and geometries
    """
    # Download EPA data
    epa_data = download_epa_data(force_refresh=force_refresh)

    print(f"\nLoaded {len(epa_data)} block groups from EPA database")

    # Download Census geometries
    if state_fips is None:
        raise ValueError("Must specify state_fips. Downloading all US states not recommended due to size.")

    blockgroups = download_census_blockgroups(state_fips, force_refresh=force_refresh)

    print(f"Loaded {len(blockgroups)} block group geometries")

    # Join on GEOID
    # Use GEOID20 from EPA to match with 2020 Census TIGER boundaries
    epa_data = epa_data.rename(columns={'GEOID20': 'GEOID'})

    # Merge
    walkability_gdf = blockgroups.merge(
        epa_data,
        on='GEOID',
        how='inner'
    )

    print(f"\nSuccessfully joined {len(walkability_gdf)} block groups")

    # Select key columns for walkability
    key_cols = [
        'GEOID',           # Block group ID
        'NatWalkInd',      # National Walkability Index (primary score)
        'D3B',             # Street intersection density
        'D2A_EPHHM',       # Employment + household entropy
        'D1B',             # Employment density
        'D4A',             # Distance to transit
        'geometry'
    ]

    # Only keep columns that exist
    available_cols = [c for c in key_cols if c in walkability_gdf.columns]
    walkability_gdf = walkability_gdf[available_cols]

    # Ensure proper CRS (should be EPSG:4269, convert to 4326 for lat/long)
    if walkability_gdf.crs is None:
        walkability_gdf.set_crs("EPSG:4269", inplace=True)

    walkability_gdf = walkability_gdf.to_crs("EPSG:4326")

    return walkability_gdf


def get_walkability_score(lat: float,
                          lon: float,
                          walkability_gdf: gpd.GeoDataFrame) -> dict:
    """
    Get EPA walkability score for a specific lat/long coordinate.

    Args:
        lat: Latitude
        lon: Longitude
        walkability_gdf: Pre-loaded walkability GeoDataFrame from build_walkability_index()

    Returns:
        Dictionary with walkability scores and metadata
    """
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame([{'geometry': point}], crs="EPSG:4326")

    # Spatial join to find containing block group
    result = gpd.sjoin(
        point_gdf,
        walkability_gdf,
        how='left',
        predicate='within'
    )

    if result.empty or pd.isna(result.iloc[0].get('GEOID')):
        return {
            'latitude': lat,
            'longitude': lon,
            'found': False,
            'message': 'No block group found for this location'
        }

    row = result.iloc[0]

    return {
        'latitude': lat,
        'longitude': lon,
        'found': True,
        'geoid': row.get('GEOID'),
        'national_walkability_index': row.get('NatWalkInd'),
        'intersection_density': row.get('D3B'),
        'employment_household_entropy': row.get('D2A_EPHHM'),
        'employment_density': row.get('D1B'),
        'distance_to_transit': row.get('D4A')
    }


def batch_walkability_scores(df: pd.DataFrame,
                             lat_col: str = 'latitude',
                             lon_col: str = 'longitude',
                             state_fips: Union[str, list] = None) -> pd.DataFrame:
    """
    Calculate walkability scores for a batch of coordinates.

    Args:
        df: DataFrame with latitude/longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        state_fips: State(s) to load data for

    Returns:
        Original DataFrame with walkability score columns added
    """
    # Build walkability index
    walkability_gdf = build_walkability_index(state_fips=state_fips)

    # Create points from coordinates
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Spatial join
    result = gpd.sjoin(
        points_gdf,
        walkability_gdf,
        how='left',
        predicate='within'
    )

    # Rename columns for clarity
    rename_map = {
        'NatWalkInd': 'walkability_index',
        'D3B': 'intersection_density',
        'D2A_EPHHM': 'employment_entropy',
        'D1B': 'employment_density',
        'D4A': 'distance_to_transit'
    }

    result = result.rename(columns=rename_map)

    # Drop geometry column for final output
    if 'geometry' in result.columns:
        result = result.drop(columns=['geometry'])

    return result


# Preload function for initial setup
def preload_walkability_data(state_fips: Union[str, list]):
    """
    Pre-download and cache all required walkability data.
    Call this once before running batch operations.
    """
    print("Pre-loading EPA walkability data...")
    build_walkability_index(state_fips=state_fips)
    print("✓ Walkability data ready!")


if __name__ == "__main__":
    # Example usage

    # Option 1: Single point lookup
    print("Example 1: Single coordinate lookup")
    print("-" * 60)

    # Load data for California (FIPS code 06)
    walkability_gdf = build_walkability_index(state_fips="06")

    # San Francisco coordinates
    score = get_walkability_score(37.7749, -122.4194, walkability_gdf)
    print(f"San Francisco walkability: {score}")

    # Option 2: Batch processing
    print("\nExample 2: Batch processing")
    print("-" * 60)

    sample_locations = pd.DataFrame({
        'location': ['San Francisco', 'Los Angeles', 'San Diego'],
        'latitude': [37.7749, 34.0522, 32.7157],
        'longitude': [-122.4194, -118.2437, -117.1611]
    })

    results = batch_walkability_scores(sample_locations, state_fips="06")
    print(results[['location', 'latitude', 'longitude', 'walkability_index']])
