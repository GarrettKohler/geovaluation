"""
Compute minimum distances from sites to chain store locations using Haversine formula.

This module handles raw lat/lon geodata CSVs for McDonald's, Walmart, and Target,
auto-detecting column names and computing the minimum distance from each site
to the nearest chain store location.

Chunked processing avoids N x M memory blowup for large datasets
(e.g. 57K sites x 14K stores).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

EARTH_RADIUS_MI = 3958.8

# Candidate column names for latitude/longitude in chain geodata files.
# Each tuple is (lat_candidates, lon_candidates) tried in order.
_LAT_CANDIDATES = ["latitude", "lat", "Latitude", "LAT", "LATITUDE"]
_LON_CANDIDATES = ["longitude", "lng", "lon", "Longitude", "LNG", "LON", "LONGITUDE"]


def _detect_latlon_columns(
    columns: List[str],
) -> Tuple[str, str]:
    """
    Auto-detect latitude and longitude column names from a list of column names.

    Tries common naming conventions: latitude/longitude, lat/lng, Latitude/Longitude.

    Args:
        columns: List of column names from the DataFrame.

    Returns:
        Tuple of (lat_col, lon_col).

    Raises:
        ValueError: If no matching lat/lon columns are found.
    """
    lat_col = None
    lon_col = None

    for candidate in _LAT_CANDIDATES:
        if candidate in columns:
            lat_col = candidate
            break

    for candidate in _LON_CANDIDATES:
        if candidate in columns:
            lon_col = candidate
            break

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Could not detect lat/lon columns. "
            f"Available columns: {columns}. "
            f"Expected one of {_LAT_CANDIDATES} for latitude "
            f"and one of {_LON_CANDIDATES} for longitude."
        )

    return lat_col, lon_col


def haversine_distances(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Haversine distance in miles.

    Handles N x M distance matrix via broadcasting. Inputs should be
    broadcastable shapes, e.g. (N, 1) vs (1, M) for an N x M result.

    Args:
        lat1: Latitudes of first point set (radians after conversion).
        lon1: Longitudes of first point set.
        lat2: Latitudes of second point set.
        lon2: Longitudes of second point set.

    Returns:
        Distance array in miles, shape determined by broadcasting.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def compute_chain_distances(
    sites_df: pl.DataFrame,
    chain_df: pl.DataFrame,
    chain_name: str,
    site_lat_col: str = "latitude",
    site_lon_col: str = "longitude",
    site_id_col: str = "gtvid",
    chain_lat_col: Optional[str] = None,
    chain_lon_col: Optional[str] = None,
    chunk_size: int = 2000,
) -> pl.DataFrame:
    """
    Compute minimum distance from each site to nearest chain store location.

    Uses chunked processing to avoid N x M memory blowup.
    For 57K sites x 14K stores, processes in chunks of 2000 sites.

    If chain_lat_col / chain_lon_col are not provided, they are auto-detected
    from the chain DataFrame columns.

    Args:
        sites_df: DataFrame of site locations (must contain id, lat, lon cols).
        chain_df: DataFrame of chain store locations (must contain lat, lon cols).
        chain_name: Name used in the output column, e.g. "mcdonalds".
        site_lat_col: Column name for site latitude.
        site_lon_col: Column name for site longitude.
        site_id_col: Column name for site identifier.
        chain_lat_col: Column name for chain latitude (auto-detected if None).
        chain_lon_col: Column name for chain longitude (auto-detected if None).
        chunk_size: Number of sites per processing chunk.

    Returns:
        DataFrame with columns [GTVID, min_distance_to_{chain_name}_mi].
    """
    # Auto-detect chain lat/lon columns if not provided
    if chain_lat_col is None or chain_lon_col is None:
        detected_lat, detected_lon = _detect_latlon_columns(chain_df.columns)
        chain_lat_col = chain_lat_col or detected_lat
        chain_lon_col = chain_lon_col or detected_lon
        logger.info(
            "Auto-detected chain columns: lat=%s, lon=%s",
            chain_lat_col,
            chain_lon_col,
        )

    # Extract arrays
    site_ids = sites_df[site_id_col].to_numpy()
    site_lats = sites_df[site_lat_col].cast(pl.Float64).to_numpy()
    site_lons = sites_df[site_lon_col].cast(pl.Float64).to_numpy()

    chain_lats = chain_df[chain_lat_col].cast(pl.Float64).to_numpy()
    chain_lons = chain_df[chain_lon_col].cast(pl.Float64).to_numpy()

    # Drop chain rows with null/nan coordinates
    valid_chain = ~(np.isnan(chain_lats) | np.isnan(chain_lons))
    chain_lats = chain_lats[valid_chain]
    chain_lons = chain_lons[valid_chain]
    logger.info(
        "Computing distances: %d sites x %d %s locations (chunk_size=%d)",
        len(site_ids),
        len(chain_lats),
        chain_name,
        chunk_size,
    )

    n_sites = len(site_ids)
    min_distances = np.full(n_sites, np.inf)

    # Process in chunks to manage memory
    for start in range(0, n_sites, chunk_size):
        end = min(start + chunk_size, n_sites)
        chunk_lats = site_lats[start:end, np.newaxis]  # (chunk, 1)
        chunk_lons = site_lons[start:end, np.newaxis]

        # Broadcast against all chain locations: (chunk, n_chains)
        distances = haversine_distances(
            chunk_lats,
            chunk_lons,
            chain_lats[np.newaxis, :],
            chain_lons[np.newaxis, :],
        )
        min_distances[start:end] = distances.min(axis=1)

    col_name = f"min_distance_to_{chain_name}_mi"
    return pl.DataFrame(
        {
            "GTVID": site_ids,
            col_name: min_distances,
        }
    )


def load_chain_geodata(csv_path: Path) -> pl.DataFrame:
    """
    Load a chain geodata CSV, filtering to valid US locations.

    Reads the CSV, auto-detects lat/lon columns, and drops rows
    with missing coordinates.

    Args:
        csv_path: Path to the chain geodata CSV file.

    Returns:
        Polars DataFrame with at least lat/lon columns populated.
    """
    df = pl.read_csv(csv_path, infer_schema_length=5000, null_values=["", "NA", "null"])
    lat_col, lon_col = _detect_latlon_columns(df.columns)

    # Filter to rows with valid coordinates
    df = df.filter(
        pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
    )

    logger.info("Loaded %s: %d rows with valid coordinates", csv_path.name, len(df))
    return df


def compute_all_chain_distances(
    sites_df: pl.DataFrame,
    site_lat_col: str = "latitude",
    site_lon_col: str = "longitude",
    site_id_col: str = "gtvid",
    platinum_dir: Optional[Path] = None,
    chunk_size: int = 2000,
) -> pl.DataFrame:
    """
    Compute minimum distances from sites to all known chains (McDonald's, Walmart, Target).

    Looks for geodata CSVs in the platinum data directory and joins all
    distance columns onto the sites DataFrame by GTVID.

    Args:
        sites_df: DataFrame with site locations.
        site_lat_col: Column name for site latitude.
        site_lon_col: Column name for site longitude.
        site_id_col: Column name for site identifier.
        platinum_dir: Directory containing chain geodata CSVs.
                      Defaults to Settings.DATA_PLATINUM_DIR.
        chunk_size: Chunk size for distance computation.

    Returns:
        DataFrame with site ID and distance columns for each chain found.
    """
    if platinum_dir is None:
        settings = get_settings()
        platinum_dir = settings.DATA_PLATINUM_DIR

    # Map of chain name -> CSV filename
    chain_files: Dict[str, str] = {
        "mcdonalds": "mcdonalds_geodata.csv",
        "walmart": "walmart_geodata.csv",
        "target": "target_geo_data.csv",
    }

    result_df = sites_df.select([site_id_col]).rename({site_id_col: "GTVID"})

    for chain_name, filename in chain_files.items():
        csv_path = platinum_dir / filename
        if not csv_path.exists():
            logger.warning("Chain geodata not found: %s", csv_path)
            continue

        chain_df = load_chain_geodata(csv_path)
        distances_df = compute_chain_distances(
            sites_df=sites_df,
            chain_df=chain_df,
            chain_name=chain_name,
            site_lat_col=site_lat_col,
            site_lon_col=site_lon_col,
            site_id_col=site_id_col,
            chunk_size=chunk_size,
        )
        result_df = result_df.join(distances_df, on="GTVID", how="left")

    return result_df
