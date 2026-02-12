"""
Compute minimum distances from GSTV sites to Walmart and Target stores.

Uses the haversine formula to calculate great-circle distances between
each site and all store locations, then takes the minimum per site.
Outputs CSV files matching the existing distance file pattern:
  GTVID, Latitude, Longitude, min_distance_to_X_mi

Usage:
    python3 -m scripts.compute_retailer_distances
    python3 scripts/compute_retailer_distances.py
"""

import numpy as np
import polars as pl
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_PATH = PROJECT_ROOT / "data" / "input"
PLATINUM_PATH = INPUT_PATH / "platinum"

EARTH_RADIUS_MI = 3958.8  # Earth's mean radius in miles


def haversine_matrix(site_lats, site_lons, store_lats, store_lons):
    """
    Compute haversine distance matrix between all site-store pairs.

    Uses numpy broadcasting: sites (N,1) vs stores (1,M) → (N,M) matrix.
    Returns distances in miles.
    """
    # Convert to radians
    lat1 = np.radians(site_lats[:, np.newaxis])   # (N, 1)
    lon1 = np.radians(site_lons[:, np.newaxis])   # (N, 1)
    lat2 = np.radians(store_lats[np.newaxis, :])  # (1, M)
    lon2 = np.radians(store_lons[np.newaxis, :])  # (1, M)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MI * c  # (N, M) distance matrix in miles


def compute_min_distances(sites_df, stores_df, retailer_name, col_name):
    """
    Compute minimum distance from each site to the nearest store.

    For large store counts, processes in chunks to limit memory usage.
    """
    site_lats = sites_df["Latitude"].to_numpy().astype(np.float64)
    site_lons = sites_df["Longitude"].to_numpy().astype(np.float64)
    store_lats = stores_df["latitude"].to_numpy().astype(np.float64)
    store_lons = stores_df["longitude"].to_numpy().astype(np.float64)

    n_sites = len(site_lats)
    n_stores = len(store_lats)
    print(f"  Computing {n_sites:,} sites × {n_stores:,} {retailer_name} stores = {n_sites * n_stores:,} pairs")

    # Process in chunks if matrix would be very large (>500M elements)
    CHUNK_SIZE = 10000
    min_distances = np.full(n_sites, np.inf)

    for start in range(0, n_sites, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_sites)
        chunk_lats = site_lats[start:end]
        chunk_lons = site_lons[start:end]

        dist_matrix = haversine_matrix(chunk_lats, chunk_lons, store_lats, store_lons)
        min_distances[start:end] = dist_matrix.min(axis=1)

    result = pl.DataFrame({
        "GTVID": sites_df["GTVID"],
        "Latitude": sites_df["Latitude"],
        "Longitude": sites_df["Longitude"],
        col_name: min_distances,
    })

    print(f"  {retailer_name} distances — min: {min_distances.min():.3f} mi, "
          f"mean: {min_distances.mean():.1f} mi, max: {min_distances.max():.1f} mi")

    return result


def main():
    print("=" * 60)
    print("Computing retailer distances for GSTV sites")
    print("=" * 60)

    # Load site coordinates from existing distance file (canonical source)
    sites = pl.read_csv(INPUT_PATH / "site_kroger_distances.csv").select(
        ["GTVID", "Latitude", "Longitude"]
    )
    print(f"\nLoaded {len(sites):,} site locations")

    # --- Walmart ---
    t0 = time.time()
    walmart_raw = pl.read_csv(PLATINUM_PATH / "walmart_geodata.csv")
    walmart = walmart_raw.filter(pl.col("country_code") == "US")
    print(f"\nWalmart: {len(walmart):,} US stores (filtered from {len(walmart_raw):,} total)")

    walmart_dist = compute_min_distances(sites, walmart, "Walmart", "min_distance_to_walmart_mi")
    out_walmart = INPUT_PATH / "site_walmart_distances.csv"
    walmart_dist.write_csv(out_walmart)
    print(f"  Saved → {out_walmart} ({time.time() - t0:.1f}s)")

    # --- Target ---
    t0 = time.time()
    target_raw = pl.read_csv(PLATINUM_PATH / "target_geo_data.csv")
    target = target_raw.filter(pl.col("country_code") == "US")
    print(f"\nTarget: {len(target):,} US stores (filtered from {len(target_raw):,} total)")

    target_dist = compute_min_distances(sites, target, "Target", "min_distance_to_target_mi")
    out_target = INPUT_PATH / "site_target_distances.csv"
    target_dist.write_csv(out_target)
    print(f"  Saved → {out_target} ({time.time() - t0:.1f}s)")

    print(f"\nDone! Generated distance files for {len(sites):,} sites.")


if __name__ == "__main__":
    main()
