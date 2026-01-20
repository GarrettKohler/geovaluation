"""
Calculate distance from each site to its nearest neighboring site.

Uses EPSG:5070 (NAD83 / Conus Albers Equal Area Conic) for accurate distance calculations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "input" / "Sites - Base Data Set.csv"
OUTPUT_DIR = PROJECT_ROOT / "distance_results"

METERS_PER_MILE = 1609.344


def calculate_nearest_site_distances(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """
    Calculate distance from each site to its nearest neighboring site.

    Parameters
    ----------
    input_file : Path
        Path to CSV file with GTVID, Latitude, Longitude columns

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - GTVID: Site ID
        - Latitude, Longitude: Original coordinates
        - nearest_site: GTVID of nearest neighbor
        - nearest_site_distance_mi: Distance in miles
        - nearest_site_distance_m: Distance in meters
    """
    # Load sites data
    sites = pd.read_csv(input_file)
    sites = sites.loc[:, ~sites.columns.str.contains('^Unnamed')]

    print(f"Loaded {len(sites)} sites")

    # Convert to GeoDataFrame
    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites['Longitude'], sites['Latitude']),
        crs="EPSG:4326"
    )

    # Project to EPSG:5070 for accurate distance calculations
    sites_proj = sites_gdf.to_crs(epsg=5070).reset_index(drop=True)

    # Extract coordinates for KDTree
    coords = np.array([(geom.x, geom.y) for geom in sites_proj.geometry])

    # Build KDTree and query for 2 nearest neighbors (self + nearest)
    print("Building spatial index and finding nearest neighbors...")
    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=3)

    # indices[:, 0] is self, indices[:, 1] is nearest neighbor
    nearest_indices = indices[:, 1]
    nearest_distances_m = distances[:, 1]

    # Build result DataFrame
    result = pd.DataFrame({
        'GTVID': sites_proj['GTVID'].values,
        'Latitude': sites_proj['Latitude'].values,
        'Longitude': sites_proj['Longitude'].values,
        'nearest_site': sites_proj['GTVID'].iloc[nearest_indices].values,
        'nearest_site_lat': sites_proj['Latitude'].iloc[nearest_indices].values,
        'nearest_site_lon': sites_proj['Longitude'].iloc[nearest_indices].values,
        'nearest_site_distance_m': nearest_distances_m,
        'nearest_site_distance_mi': nearest_distances_m / METERS_PER_MILE
    })

    return result


if __name__ == "__main__":
    print("Nearest Site Distance Calculator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Calculate distances
    print("\nCalculating distances to nearest neighboring site...")
    result = calculate_nearest_site_distances()

    print(f"\nResults (first 10):")
    display_cols = ['GTVID', 'nearest_site', 'nearest_site_distance_mi']
    print(result[display_cols].head(10).to_string(index=False))

    # Save to CSV
    output_path = OUTPUT_DIR / "nearest_site_distances.csv"
    result.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total sites: {len(result)}")
    print(f"Min distance: {result['nearest_site_distance_mi'].min():.2f} miles")
    print(f"Max distance: {result['nearest_site_distance_mi'].max():.2f} miles")
    print(f"Mean distance: {result['nearest_site_distance_mi'].mean():.2f} miles")
    print(f"Median distance: {result['nearest_site_distance_mi'].median():.2f} miles")

    print("\n" + "=" * 60)
    print("Complete!")
