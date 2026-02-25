"""
DataRegistry: Single source of truth for all data access in platinum1_1.

Provides a thread-safe singleton that ensures consistent data loading and
filtering across the web API and ML training pipeline.

Key design decisions:
    - Polars throughout (no pandas dependency in this module).
    - No Kroger data in this version.
    - Handles the ``statuis`` column typo transparently.
    - Raw geodata (McDonald's, Walmart, Target lat/lon) is NOT loaded here;
      distance computation is handled by the processing layer.
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl

from .paths import DataPaths


# =============================================================================
# Filter configuration
# =============================================================================

@dataclass
class FilterConfig:
    """Filtering rules applied consistently across web and ML pipelines."""

    # Status
    active_only: bool = False
    valid_statuses: Tuple[str, ...] = ("Active",)

    # Temporal
    min_active_months: int = 0      # 0 = no filter, 12 = require 1 year

    # Revenue
    exclude_negative_revenue: bool = False

    # Coordinates
    require_coordinates: bool = False


# Pre-built filter presets
FILTER_NONE = FilterConfig()
FILTER_WEB = FilterConfig(require_coordinates=True)
FILTER_ML = FilterConfig(
    active_only=True,
    min_active_months=12,
    exclude_negative_revenue=True,
)


# =============================================================================
# DataRegistry singleton
# =============================================================================

class DataRegistry:
    """
    Thread-safe singleton for all data access.

    Usage::

        registry = DataRegistry()

        # Web visualization (all sites with coordinates)
        sites = registry.get_web_sites()

        # ML training (active, 12+ months, positive revenue)
        training = registry.get_training_data()

        # Custom filter
        custom = registry.get_sites(FilterConfig(active_only=True))
    """

    _instance: Optional["DataRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DataRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._cache: Dict[str, Any] = {}
        self._paths = DataPaths()

    # -- public filter presets (instance-level aliases) -----------------------

    @property
    def FILTER_NONE(self) -> FilterConfig:
        return FILTER_NONE

    @property
    def FILTER_WEB(self) -> FilterConfig:
        return FILTER_WEB

    @property
    def FILTER_ML(self) -> FilterConfig:
        return FILTER_ML

    # =========================================================================
    # Cache management
    # =========================================================================

    def clear_cache(self) -> None:
        """Drop all cached DataFrames to free memory."""
        self._cache.clear()

    def _cache_key(self, name: str, config: FilterConfig) -> str:
        return (
            f"{name}"
            f"_{config.active_only}"
            f"_{config.min_active_months}"
            f"_{config.exclude_negative_revenue}"
            f"_{config.require_coordinates}"
        )

    # =========================================================================
    # Core data loaders
    # =========================================================================

    def get_raw_site_scores(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load the raw monthly site-scores CSV (~1.4 M rows, 927 MB).

        Prefer ``get_sites()`` or ``get_training_data()`` for filtered /
        aggregated access.
        """
        cache_key = "raw_site_scores"
        if cache_key not in self._cache or force_reload:
            path = self._paths.site_scores_csv
            if not path.exists():
                raise FileNotFoundError(f"Site scores file not found: {path}")

            print(f"Loading raw site scores from {path} ...")
            self._cache[cache_key] = pl.read_csv(
                path,
                null_values=["", "NA", "null", "Unknown"],
                infer_schema_length=10_000,
            )
            print(f"  Loaded {len(self._cache[cache_key]):,} rows")

        return self._cache[cache_key]

    def get_training_parquet(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load the pre-processed training parquet (one row per site).

        This is the fastest path for ML training when the parquet exists.
        """
        cache_key = "training_parquet"
        if cache_key not in self._cache or force_reload:
            path = self._paths.training_parquet
            if not path.exists():
                raise FileNotFoundError(
                    f"Training parquet not found: {path}\n"
                    "Run the data transform pipeline to generate it."
                )

            print(f"Loading training data from {path} ...")
            self._cache[cache_key] = pl.read_parquet(path)
            print(f"  Loaded {len(self._cache[cache_key]):,} sites")

        return self._cache[cache_key]

    # =========================================================================
    # Geospatial distance loaders
    # =========================================================================

    def get_nearest_site_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """Load pre-computed nearest-site distances (one row per site)."""
        cache_key = "nearest_site_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.nearest_site_distances
            if path.exists():
                self._cache[cache_key] = pl.read_csv(path)
            else:
                print(f"Warning: nearest site distances not found at {path}")
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    def get_interstate_distances(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Load interstate distances, aggregated to minimum per site.

        The source file may contain multiple interstates per site; we
        return the minimum distance for each GTVID.
        """
        cache_key = "interstate_distances"
        if cache_key not in self._cache or force_reload:
            path = self._paths.interstate_distances
            if path.exists():
                df = pl.read_csv(path)
                # Column name varies by source file
                dist_col = (
                    "min_distance_to_interstate_mi"
                    if "min_distance_to_interstate_mi" in df.columns
                    else "distance_to_interstate_mi"
                )
                # Aggregate to minimum distance per site
                self._cache[cache_key] = df.group_by("GTVID").agg(
                    pl.col(dist_col).min().alias("min_distance_to_interstate_mi")
                )
            else:
                print(f"Warning: interstate distances not found at {path}")
                self._cache[cache_key] = pl.DataFrame()
        return self._cache[cache_key]

    # =========================================================================
    # Filtered data access (primary API)
    # =========================================================================

    def get_sites(
        self,
        config: Optional[FilterConfig] = None,
        force_reload: bool = False,
    ) -> pl.DataFrame:
        """
        Site-level data with configurable filtering.

        Falls back to aggregating from the raw CSV when the training
        parquet is not available.

        Args:
            config:       Filtering rules (defaults to FILTER_NONE).
            force_reload: Bypass cache.

        Returns:
            Polars DataFrame with one row per site.
        """
        if config is None:
            config = FILTER_NONE

        cache_key = self._cache_key("sites", config)

        if cache_key not in self._cache or force_reload:
            # Prefer parquet (fast); fall back to raw CSV aggregation
            try:
                df = self.get_training_parquet(force_reload)
            except FileNotFoundError:
                df = self._aggregate_sites_from_raw(force_reload)

            df = self._apply_filters(df, config)
            self._cache[cache_key] = df

        return self._cache[cache_key]

    def get_training_data(self, force_reload: bool = False) -> pl.DataFrame:
        """
        ML-ready training data.

        Applies FILTER_ML: active only, 12+ months, non-negative revenue.
        """
        return self.get_sites(config=FILTER_ML, force_reload=force_reload)

    def get_web_sites(self, force_reload: bool = False) -> pl.DataFrame:
        """
        Sites for web visualization (requires valid coordinates).
        """
        return self.get_sites(config=FILTER_WEB, force_reload=force_reload)

    # =========================================================================
    # Revenue metrics (web visualization)
    # =========================================================================

    def get_revenue_metrics(
        self,
        use_active_for_percentiles: bool = True,
        force_reload: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-site revenue metrics with percentile normalization.

        Args:
            use_active_for_percentiles: Compute percentiles from active
                sites only to prevent inactive sites from skewing the
                distribution.
            force_reload: Bypass cache.

        Returns:
            ``{site_id: {score, avg_monthly, total, months}}``
        """
        cache_key = f"revenue_metrics_{use_active_for_percentiles}"

        if cache_key not in self._cache or force_reload:
            raw_df = self.get_raw_site_scores(force_reload)

            # Aggregate revenue by site
            # Note: source data uses ``statuis`` (not ``status``)
            status_col = (
                "status" if "status" in raw_df.columns else "statuis"
            )

            site_metrics = (
                raw_df
                .filter(
                    pl.col("revenue").is_not_null()
                    & pl.col("gtvid").is_not_null()
                )
                .group_by("gtvid")
                .agg([
                    pl.col("revenue").sum().alias("total_revenue"),
                    pl.col("date").count().alias("active_months"),
                    pl.col(status_col).first().alias("status"),
                ])
            )

            # Derived metrics
            site_metrics = site_metrics.with_columns([
                (
                    pl.col("total_revenue")
                    / pl.col("active_months").clip(lower_bound=1)
                ).alias("avg_monthly_revenue"),
                (
                    pl.col("total_revenue")
                    / (pl.col("active_months") * 30).clip(lower_bound=1)
                ).alias("revenue_per_day"),
            ])

            # Percentile source
            if use_active_for_percentiles:
                perc_source = (
                    site_metrics
                    .filter(pl.col("status") == "Active")["revenue_per_day"]
                    .to_numpy()
                )
            else:
                perc_source = site_metrics["revenue_per_day"].to_numpy()

            p20 = float(np.percentile(perc_source, 20))
            p95 = float(np.percentile(perc_source, 95))
            denom = p95 - p20 if p95 > p20 else 1.0

            print(f"Revenue percentiles (p20-p95): ${p20:.2f} - ${p95:.2f}/day")

            metrics: Dict[str, Dict[str, Any]] = {}
            for row in site_metrics.iter_rows(named=True):
                raw_val = row["revenue_per_day"]
                normalized = (raw_val - p20) / denom
                metrics[row["gtvid"]] = {
                    "score": max(0.0, min(1.0, normalized)),
                    "avg_monthly": row["avg_monthly_revenue"],
                    "total": row["total_revenue"],
                    "months": row["active_months"],
                }

            self._cache[cache_key] = metrics

        return self._cache[cache_key]

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _aggregate_sites_from_raw(
        self, force_reload: bool = False
    ) -> pl.DataFrame:
        """
        Aggregate raw monthly data to one row per site.

        This is a lightweight fallback when the training parquet does not
        exist.  The full ETL pipeline (data_transform) produces a richer
        aggregation.
        """
        raw_df = self.get_raw_site_scores(force_reload)

        # Determine status column name (handle ``statuis`` typo)
        status_col = (
            "status" if "status" in raw_df.columns else "statuis"
        )

        site_df = raw_df.group_by("gtvid").agg([
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("date").count().alias("active_months"),
            pl.col("latitude").first(),
            pl.col("longitude").first(),
            pl.col(status_col).last().alias("status"),
        ])

        return site_df

    def _apply_filters(
        self, df: pl.DataFrame, config: FilterConfig
    ) -> pl.DataFrame:
        """Apply FilterConfig rules to a site-level DataFrame."""
        result = df

        # Status filter -- handle both ``status`` and ``statuis``
        if config.active_only:
            status_col = (
                "status" if "status" in df.columns else "statuis"
            )
            if status_col in df.columns:
                result = result.filter(
                    pl.col(status_col).is_in(list(config.valid_statuses))
                )

        # Minimum active months
        if config.min_active_months > 0 and "active_months" in df.columns:
            result = result.filter(
                pl.col("active_months") > config.min_active_months
            )

        # Revenue floor
        if config.exclude_negative_revenue and "total_revenue" in df.columns:
            result = result.filter(pl.col("total_revenue") >= 0)

        # Coordinate requirement
        if config.require_coordinates:
            if "latitude" in df.columns and "longitude" in df.columns:
                result = result.filter(
                    pl.col("latitude").is_not_null()
                    & pl.col("longitude").is_not_null()
                )

        return result


# =============================================================================
# Convenience accessor
# =============================================================================

def get_registry() -> DataRegistry:
    """Return the singleton DataRegistry instance."""
    return DataRegistry()
