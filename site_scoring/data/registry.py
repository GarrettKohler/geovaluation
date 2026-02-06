"""
DataRegistry: Single source of truth for all data access.

This module provides a singleton registry that ensures consistent data loading
and filtering across both web visualization and ML training pipelines.

Key benefits:
- Consistent filtering rules (active sites, active_months threshold)
- Caching to prevent redundant disk I/O
- Single place to modify data loading behavior
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading


@dataclass
class DataPaths:
    """Central configuration for all data file paths."""
    base_dir: Path = Path(__file__).parent.parent.parent / "data"

    @property
    def input_dir(self) -> Path:
        return self.base_dir / "input"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "processed"

    @property
    def site_scores_csv(self) -> Path:
        return self.input_dir / "site_scores_revenue_and_diagnostics.csv"

    @property
    def training_parquet(self) -> Path:
        return self.processed_dir / "site_training_data.parquet"

    @property
    def precleaned_parquet(self) -> Path:
        return self.processed_dir / "site_aggregated_precleaned.parquet"

    @property
    def kroger_distances(self) -> Path:
        return self.input_dir / "site_kroger_distances.csv"

    @property
    def mcdonalds_distances(self) -> Path:
        return self.input_dir / "site_mcdonalds_distances.csv"

    @property
    def nearest_site_distances(self) -> Path:
        return self.input_dir / "nearest_site_distances.csv"

    @property
    def interstate_distances(self) -> Path:
        return self.input_dir / "site_interstate_distances.csv"


@dataclass
class FilterConfig:
    """Filtering configuration for consistent data access."""
    # Status filtering
    active_only: bool = False
    valid_statuses: Tuple[str, ...] = ("Active",)

    # Temporal filtering
    min_active_months: int = 0  # 0 = no filter, 12 = require 1 year of data

    # Revenue filtering
    exclude_negative_revenue: bool = False

    # Coordinate filtering
    require_coordinates: bool = False


class DataRegistry:
    """
    Singleton registry for all data access.

    Ensures consistency between web visualization and ML training by providing
    a single source of truth for data loading and filtering.

    Usage:
        registry = DataRegistry()

        # For web visualization (all sites)
        sites = registry.get_sites()

        # For ML training (filtered)
        training_data = registry.get_training_data()

        # Custom filtering
        active_sites = registry.get_sites(FilterConfig(active_only=True))
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._cache: Dict[str, Any] = {}
        self._paths = DataPaths()

        # Standard filter configurations
        self.FILTER_NONE = FilterConfig()
        self.FILTER_WEB = FilterConfig(require_coordinates=True)
        self.FILTER_ML = FilterConfig(
            active_only=True,
            min_active_months=12,
            exclude_negative_revenue=True
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def _cache_key(self, name: str, config: FilterConfig) -> str:
        """Generate cache key from name and filter config."""
        return f"{name}_{config.active_only}_{config.min_active_months}_{config.exclude_negative_revenue}_{config.require_coordinates}"

    # =========================================================================
    # Core Data Loaders
    # =========================================================================

    def get_raw_site_scores(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load raw site scores CSV (monthly data, ~1.4M rows).

        This is the base data - use get_sites() or get_training_data() for
        filtered/aggregated versions.
        """
        cache_key = "raw_site_scores"
        if cache_key not in self._cache or force_reload:
            path = self._paths.site_scores_csv
            if not path.exists():
                raise FileNotFoundError(f"Site scores file not found: {path}")

            print(f"Loading raw site scores from {path}...")
            self._cache[cache_key] = pl.read_csv(
                path,
                null_values=["", "NA", "null", "Unknown"],
                infer_schema_length=10000
            )
            print(f"  Loaded {len(self._cache[cache_key]):,} rows")

        return self._cache[cache_key]

    def get_training_parquet(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load pre-processed training parquet (one row per site, ML-ready).

        This is the fastest option for ML training if the parquet exists.
        """
        cache_key = "training_parquet"
        if cache_key not in self._cache or force_reload:
            path = self._paths.training_parquet
            if not path.exists():
                raise FileNotFoundError(
                    f"Training parquet not found: {path}\n"
                    "Run data_transform.py to generate it."
                )

            print(f"Loading training data from {path}...")
            self._cache[cache_key] = pl.read_parquet(path)
            print(f"  Loaded {len(self._cache[cache_key]):,} sites")

        return self._cache[cache_key]

    # =========================================================================
    # Geospatial Data Loaders
    # =========================================================================

    def get_kroger_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """Load Kroger distance data (one row per site)."""
        cache_key = "kroger_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.kroger_distances
            if path.exists():
                self._cache[cache_key] = pl.read_csv(path)
            else:
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    def get_mcdonalds_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """Load McDonald's distance data (one row per site)."""
        cache_key = "mcdonalds_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.mcdonalds_distances
            if path.exists():
                self._cache[cache_key] = pl.read_csv(path)
            else:
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    def get_nearest_site_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """Load nearest site distance data."""
        cache_key = "nearest_site_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.nearest_site_distances
            if path.exists():
                self._cache[cache_key] = pl.read_csv(path)
            else:
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    def get_interstate_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """Load interstate distance data (aggregated to min per site)."""
        cache_key = "interstate_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.interstate_distances
            if path.exists():
                df = pl.read_csv(path)
                # Aggregate to minimum distance per site
                self._cache[cache_key] = df.group_by("GTVID").agg(
                    pl.col("min_distance_to_interstate_mi").min()
                )
            else:
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    # =========================================================================
    # Filtered Data Access (Main API)
    # =========================================================================

    def get_sites(
        self,
        config: Optional[FilterConfig] = None,
        force_reload: bool = False
    ) -> pl.DataFrame:
        """
        Get site-level data with configurable filtering.

        Args:
            config: Filtering configuration. Defaults to FILTER_NONE (all sites).
            force_reload: Force reload from disk.

        Returns:
            DataFrame with one row per site, filtered according to config.
        """
        if config is None:
            config = self.FILTER_NONE

        cache_key = self._cache_key("sites", config)

        if cache_key not in self._cache or force_reload:
            # Try parquet first (faster), fall back to raw CSV
            try:
                df = self.get_training_parquet(force_reload)
            except FileNotFoundError:
                # Aggregate from raw CSV
                df = self._aggregate_sites_from_raw(force_reload)

            # Apply filters
            df = self._apply_filters(df, config)
            self._cache[cache_key] = df

        return self._cache[cache_key]

    def get_training_data(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Get ML training data with standard filtering.

        Applies:
        - active_only=True (status == 'Active')
        - min_active_months=12 (require 1 year of data)
        - exclude_negative_revenue=True

        Returns:
            DataFrame ready for ML training.
        """
        return self.get_sites(config=self.FILTER_ML, force_reload=force_reload)

    def get_web_sites(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Get sites for web visualization.

        Applies:
        - require_coordinates=True

        Returns:
            DataFrame with all sites that have valid coordinates.
        """
        return self.get_sites(config=self.FILTER_WEB, force_reload=force_reload)

    # =========================================================================
    # Revenue Metrics (for web visualization)
    # =========================================================================

    def get_revenue_metrics(
        self,
        use_active_for_percentiles: bool = True,
        force_reload: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate revenue metrics with percentile normalization.

        Args:
            use_active_for_percentiles: If True, calculate percentiles using
                only active sites (prevents inactive sites from skewing distribution).
            force_reload: Force recalculation.

        Returns:
            Dict mapping site_id to {score, avg_monthly, total, months}
        """
        cache_key = f"revenue_metrics_{use_active_for_percentiles}"

        if cache_key not in self._cache or force_reload:
            raw_df = self.get_raw_site_scores(force_reload)

            # Aggregate revenue by site
            site_metrics = raw_df.filter(
                pl.col("revenue").is_not_null() & pl.col("gtvid").is_not_null()
            ).group_by("gtvid").agg([
                pl.col("revenue").sum().alias("total_revenue"),
                pl.col("date").count().alias("active_months"),
                pl.col("statuis").first().alias("status")  # Note: typo in source data
            ])

            # Calculate derived metrics
            site_metrics = site_metrics.with_columns([
                (pl.col("total_revenue") / pl.col("active_months").clip(lower_bound=1))
                    .alias("avg_monthly_revenue"),
                (pl.col("total_revenue") / (pl.col("active_months") * 30).clip(lower_bound=1))
                    .alias("revenue_per_day")
            ])

            # Calculate percentiles (using active sites only if specified)
            if use_active_for_percentiles:
                active_metrics = site_metrics.filter(pl.col("status") == "Active")
                percentile_source = active_metrics["revenue_per_day"].to_numpy()
            else:
                percentile_source = site_metrics["revenue_per_day"].to_numpy()

            p20 = float(np.percentile(percentile_source, 20))
            p95 = float(np.percentile(percentile_source, 95))

            print(f"Revenue percentiles (p20-p95): ${p20:.2f} - ${p95:.2f}/day")

            # Build metrics dict
            metrics = {}
            for row in site_metrics.iter_rows(named=True):
                raw = row["revenue_per_day"]
                if p95 > p20:
                    normalized = (raw - p20) / (p95 - p20)
                else:
                    normalized = 0

                metrics[row["gtvid"]] = {
                    "score": max(0, min(1, normalized)),
                    "avg_monthly": row["avg_monthly_revenue"],
                    "total": row["total_revenue"],
                    "months": row["active_months"]
                }

            self._cache[cache_key] = metrics

        return self._cache[cache_key]

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _aggregate_sites_from_raw(self, force_reload: bool = False) -> pl.DataFrame:
        """Aggregate raw monthly data to site level."""
        raw_df = self.get_raw_site_scores(force_reload)

        # Group by site and aggregate
        # This is a simplified version - full aggregation is in data_transform.py
        site_df = raw_df.group_by("gtvid").agg([
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("date").count().alias("active_months"),
            pl.col("latitude").first(),
            pl.col("longitude").first(),
            pl.col("statuis").last().alias("status"),
            # Add more aggregations as needed
        ])

        return site_df

    def _apply_filters(self, df: pl.DataFrame, config: FilterConfig) -> pl.DataFrame:
        """Apply filter configuration to DataFrame."""
        result = df

        # Status filter
        if config.active_only:
            # Handle both 'status' and 'statuis' column names
            status_col = "status" if "status" in df.columns else "statuis"
            if status_col in df.columns:
                result = result.filter(pl.col(status_col).is_in(config.valid_statuses))

        # Active months filter
        if config.min_active_months > 0 and "active_months" in df.columns:
            result = result.filter(pl.col("active_months") > config.min_active_months)

        # Revenue filter
        if config.exclude_negative_revenue and "total_revenue" in df.columns:
            result = result.filter(pl.col("total_revenue") >= 0)

        # Coordinate filter
        if config.require_coordinates:
            if "latitude" in df.columns and "longitude" in df.columns:
                result = result.filter(
                    pl.col("latitude").is_not_null() &
                    pl.col("longitude").is_not_null()
                )

        return result


# Convenience function for quick access
def get_registry() -> DataRegistry:
    """Get the singleton DataRegistry instance."""
    return DataRegistry()
