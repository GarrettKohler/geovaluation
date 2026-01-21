"""Feature engineering and data splitting."""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..config import Config, config as default_config


@dataclass
class SplitResult:
    """Container for train/val/test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_sites: List[str]
    validation_sites: List[str]
    test_sites: List[str]


class TemporalSplitter:
    """Time-based splitting with gap periods and group awareness."""

    def __init__(
        self,
        date_column: str = "snapshot_date",
        group_column: str = "site_id",
        gap_days: int = 14,
    ):
        self.date_column = date_column
        self.group_column = group_column
        self.gap_days = gap_days

    def split(
        self,
        df: pd.DataFrame,
        train_end: str,
        validation_end: str,
        test_start: Optional[str] = None,
    ) -> SplitResult:
        """Split data temporally with gap periods.

        Args:
            df: DataFrame with date_column
            train_end: Last date for training data
            validation_end: Last date for validation data
            test_start: First date for test data (defaults to validation_end + gap)

        Returns:
            SplitResult with train, validation, test DataFrames
        """
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        train_end_dt = pd.to_datetime(train_end)
        validation_start_dt = train_end_dt + pd.Timedelta(days=self.gap_days)
        validation_end_dt = pd.to_datetime(validation_end)

        if test_start:
            test_start_dt = pd.to_datetime(test_start)
        else:
            test_start_dt = validation_end_dt + pd.Timedelta(days=self.gap_days)

        # Split by date
        train = df[df[self.date_column] <= train_end_dt]
        validation = df[
            (df[self.date_column] >= validation_start_dt)
            & (df[self.date_column] <= validation_end_dt)
        ]
        test = df[df[self.date_column] >= test_start_dt]

        return SplitResult(
            train=train,
            validation=validation,
            test=test,
            train_sites=train[self.group_column].unique().tolist(),
            validation_sites=validation[self.group_column].unique().tolist(),
            test_sites=test[self.group_column].unique().tolist(),
        )

    def split_by_site_groups(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        target_column: str = "reached_threshold",
    ) -> StratifiedGroupKFold:
        """Get cross-validation splitter that keeps sites together.

        All observations for a site stay in the same fold.
        """
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)


class FeatureEngineer:
    """Feature engineering for DOOH data."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_column: str = "gross_revenue",
        group_column: str = "site_id",
        date_column: str = "revenue_date",
        windows: List[int] = [7, 30, 90],
    ) -> pd.DataFrame:
        """Create rolling window features.

        IMPORTANT: Uses shift(1) to prevent data leakage.
        """
        df = df.copy()
        df = df.sort_values([group_column, date_column])

        for window in windows:
            # Shifted to avoid leakage
            df[f"{value_column}_rolling_{window}d"] = df.groupby(group_column)[
                value_column
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

            # Trend (% change over window)
            df[f"{value_column}_trend_{window}d"] = df.groupby(group_column)[
                value_column
            ].transform(
                lambda x: x.shift(1).pct_change(periods=window).fillna(0)
            )

        return df

    def create_market_proxies(
        self,
        sites_df: pd.DataFrame,
        revenue_df: pd.DataFrame,
        radius_km: float = 10.0,
    ) -> pd.DataFrame:
        """Create proxy features from nearby active sites.

        For inactive sites, compute features based on nearby active sites.
        """
        # This would use PostGIS spatial queries in practice
        # Simplified version for the structure
        proxies = []

        for _, site in sites_df[~sites_df["is_active"]].iterrows():
            # In production, use PostGIS ST_DWithin
            nearby_active = sites_df[sites_df["is_active"]]

            if len(nearby_active) > 0:
                nearby_revenue = revenue_df[
                    revenue_df["site_id"].isin(nearby_active["site_id"])
                ]

                if len(nearby_revenue) > 0:
                    avg_revenue = nearby_revenue["gross_revenue"].mean()
                    threshold = nearby_revenue["gross_revenue"].quantile(0.75)
                    high_pct = (nearby_revenue["gross_revenue"] > threshold).mean()
                else:
                    avg_revenue = 0
                    high_pct = 0

                proxies.append(
                    {
                        "site_id": site["site_id"],
                        "nearby_avg_revenue_30d": avg_revenue,
                        "nearby_high_revenue_pct": high_pct,
                        "nearby_site_count": len(nearby_active),
                    }
                )

        return pd.DataFrame(proxies)

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """Encode categorical variables.

        Returns DataFrame and mapping dict for inverse transform.
        """
        columns = columns or self.config.features.categorical_features
        df = df.copy()
        mappings = {}

        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")
                categories = df[col].unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                mappings[col] = mapping
                df[f"{col}_encoded"] = df[col].map(mapping)

        return df, mappings

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], List[int]]:
        """Prepare data for model training.

        Returns:
            X: Feature DataFrame
            y: Target Series
            feature_names: List of feature column names
            cat_indices: Indices of categorical columns (for CatBoost)
        """
        target_column = target_column or self.config.features.target_column

        continuous = self.config.features.continuous_features
        categorical = self.config.features.categorical_features

        # Select features that exist in the data
        available_continuous = [c for c in continuous if c in df.columns]
        available_categorical = [c for c in categorical if c in df.columns]

        feature_columns = available_continuous + available_categorical
        X = df[feature_columns].copy()

        # Fill missing values
        for col in available_continuous:
            X[col] = X[col].fillna(X[col].median())

        for col in available_categorical:
            X[col] = X[col].fillna("unknown").astype(str)

        y = df[target_column] if target_column in df.columns else None

        # Get categorical indices for CatBoost
        cat_indices = [
            feature_columns.index(c) for c in available_categorical
        ]

        return X, y, feature_columns, cat_indices
