"""
Demo script showing how to use the interstate_distance module.
Uses the Sites - Base Data Set.csv as input.

Run from project root: python -m scripts.demo
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.interstate_distance import (
    batch_distance_to_interstate,
    preload_highway_data
)

INPUT_FILE = PROJECT_ROOT / "data" / "input" / "Sites - Base Data Set.csv"
OUTPUT_DIR = PROJECT_ROOT / "distance_results"


def calculate_site_distances():
    """Calculate distance to nearest Interstate for all sites."""
    print("=" * 60)
    print("SITE DISTANCE CALCULATION")
    print("=" * 60)

    # Load sites data
    sites = pd.read_csv(INPUT_FILE)

    # Clean up columns (remove empty unnamed columns)
    sites = sites.loc[:, ~sites.columns.str.contains('^Unnamed')]

    print(f"\nLoaded {len(sites)} sites from {INPUT_FILE.name}")
    print(f"\nFirst 5 sites:")
    print(sites.head().to_string(index=False))

    # Calculate distances
    print("\nCalculating distances to nearest Interstate...")
    result = batch_distance_to_interstate(
        sites,
        lat_col='Latitude',
        lon_col='Longitude'
    )

    # Select output columns
    output_cols = ['GTVID', 'Latitude', 'Longitude', 'nearest_interstate', 'distance_to_interstate_mi']
    output_df = result[output_cols].copy()

    print(f"\nResults (first 10):")
    print(output_df.head(10).to_string(index=False))

    # Save to CSV
    output_path = OUTPUT_DIR / "site_interstate_distances.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total sites: {len(output_df)}")
    print(f"Min distance: {output_df['distance_to_interstate_mi'].min():.2f} miles")
    print(f"Max distance: {output_df['distance_to_interstate_mi'].max():.2f} miles")
    print(f"Mean distance: {output_df['distance_to_interstate_mi'].mean():.2f} miles")
    print(f"Median distance: {output_df['distance_to_interstate_mi'].median():.2f} miles")


if __name__ == "__main__":
    print("Interstate Distance Calculator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nPre-loading highway data (first run downloads ~15MB from Census Bureau)...\n")

    preload_highway_data()

    calculate_site_distances()

    print("\n" + "=" * 60)
    print("Complete!")
