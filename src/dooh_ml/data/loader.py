"""Data loading from PostgreSQL database."""

from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
import geopandas as gpd

from ..config import Config, config as default_config


class DataLoader:
    """Load site and feature data from PostgreSQL."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._engine = None

    @property
    def engine(self):
        """Lazy-load SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(self.config.database.connection_string)
        return self._engine

    def load_sites(self, active_only: Optional[bool] = None) -> gpd.GeoDataFrame:
        """Load sites with geometry.

        Args:
            active_only: If True, only active sites. If False, only inactive.
                        If None, all sites.
        """
        where_clause = ""
        if active_only is True:
            where_clause = "WHERE is_active = TRUE"
        elif active_only is False:
            where_clause = "WHERE is_active = FALSE"

        query = f"""
        SELECT
            site_id,
            external_id,
            name,
            location,
            city,
            state,
            zip_code,
            market_region,
            site_type,
            traffic_volume,
            distance_to_highway_km,
            poi_density,
            competitor_count,
            is_active,
            activation_date
        FROM dooh.sites
        {where_clause}
        """

        gdf = gpd.read_postgis(
            query, self.engine, geom_col="location", crs="EPSG:4326"
        )
        return gdf

    def load_feature_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        site_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """Load feature snapshots for training.

        Args:
            start_date: Filter snapshots >= this date
            end_date: Filter snapshots <= this date
            site_ids: Filter to specific sites
        """
        conditions = []
        params = {}

        if start_date:
            conditions.append("snapshot_date >= :start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("snapshot_date <= :end_date")
            params["end_date"] = end_date

        if site_ids:
            conditions.append("site_id = ANY(:site_ids)")
            params["site_ids"] = site_ids

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT *
        FROM dooh.feature_snapshots
        {where_clause}
        ORDER BY site_id, snapshot_date
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        return df

    def load_revenue(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load daily revenue data."""
        conditions = []
        params = {}

        if start_date:
            conditions.append("revenue_date >= :start_date")
            params["start_date"] = start_date

        if end_date:
            conditions.append("revenue_date <= :end_date")
            params["end_date"] = end_date

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        SELECT
            site_id,
            revenue_date,
            gross_revenue,
            impressions,
            fill_rate
        FROM dooh.revenue
        {where_clause}
        ORDER BY site_id, revenue_date
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        return df

    def load_current_hardware(self) -> pd.DataFrame:
        """Load current hardware for all sites."""
        query = """
        SELECT
            site_id,
            display_technology,
            screen_size_inches,
            resolution,
            brightness_nits,
            installed_date
        FROM dooh.hardware
        WHERE is_current = TRUE
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df

    def load_training_dataset(
        self,
        snapshot_date: str,
    ) -> pd.DataFrame:
        """Load complete training dataset for a given snapshot date.

        Joins sites, features, and computes target variable.
        """
        query = """
        SELECT
            f.*,
            s.is_active,
            s.activation_date,
            CASE
                WHEN f.revenue_30d >= :threshold THEN 1
                ELSE 0
            END as reached_threshold
        FROM dooh.feature_snapshots f
        JOIN dooh.sites s ON f.site_id = s.site_id
        WHERE f.snapshot_date = :snapshot_date
        """

        # First, compute the threshold
        threshold_query = """
        SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY revenue_30d)
        FROM dooh.feature_snapshots
        WHERE snapshot_date = :snapshot_date
        AND revenue_30d IS NOT NULL
        """

        with self.engine.connect() as conn:
            threshold = conn.execute(
                text(threshold_query), {"snapshot_date": snapshot_date}
            ).scalar()

            df = pd.read_sql(
                text(query),
                conn,
                params={"snapshot_date": snapshot_date, "threshold": threshold or 0},
            )

        return df

    def load_sites_with_features(self) -> pd.DataFrame:
        """Load sites joined with latest features and hardware."""
        query = """
        SELECT
            s.site_id,
            s.name,
            s.market_region,
            s.is_active,
            s.traffic_volume,
            s.distance_to_highway_km,
            s.poi_density,
            s.competitor_count,
            h.display_technology,
            h.screen_size_inches,
            f.revenue_30d,
            f.revenue_90d,
            f.nearby_avg_revenue_30d,
            f.nearby_site_count,
            f.primary_content_type,
            f.primary_content_category,
            f.avg_loop_length_seconds,
            f.avg_cpm_floor
        FROM dooh.sites s
        LEFT JOIN dooh.hardware h ON s.site_id = h.site_id AND h.is_current = TRUE
        LEFT JOIN LATERAL (
            SELECT *
            FROM dooh.feature_snapshots fs
            WHERE fs.site_id = s.site_id
            ORDER BY fs.snapshot_date DESC
            LIMIT 1
        ) f ON TRUE
        """

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        return df
