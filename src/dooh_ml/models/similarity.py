"""Similarity model using Gower distance for lookalike analysis."""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import gower

from ..config import Config, config as default_config


class SimilarityModel:
    """Identify inactive sites resembling high-revenue performers.

    Uses Gower distance for mixed continuous/categorical features,
    which handles your feature types natively without encoding.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        high_performer_quantile: float = 0.75,
        config: Optional[Config] = None,
    ):
        """Initialize similarity model.

        Args:
            k_neighbors: Number of nearest high performers to consider
            high_performer_quantile: Revenue quantile defining "high performer"
            config: Configuration object
        """
        self.k_neighbors = k_neighbors
        self.high_performer_quantile = high_performer_quantile
        self.config = config or default_config

        self._high_performers: Optional[pd.DataFrame] = None
        self._feature_columns: Optional[List[str]] = None
        self._threshold: Optional[float] = None

    def fit(
        self,
        active_sites: pd.DataFrame,
        revenue_column: str = "revenue_30d",
        feature_columns: Optional[List[str]] = None,
    ) -> "SimilarityModel":
        """Fit model by identifying high-performing sites.

        Args:
            active_sites: DataFrame of active sites with features and revenue
            revenue_column: Column containing revenue for threshold calculation
            feature_columns: Columns to use for similarity (default: all features from config)
        """
        self._feature_columns = feature_columns or self.config.features.all_features
        self._feature_columns = [
            c for c in self._feature_columns if c in active_sites.columns
        ]

        # Calculate revenue threshold
        self._threshold = active_sites[revenue_column].quantile(
            self.high_performer_quantile
        )

        # Identify high performers
        self._high_performers = active_sites[
            active_sites[revenue_column] >= self._threshold
        ][self._feature_columns + ["site_id"]].copy()

        return self

    def score(self, sites: pd.DataFrame) -> np.ndarray:
        """Score sites by similarity to high performers.

        Args:
            sites: DataFrame of sites to score

        Returns:
            Array of lookalike scores (higher = more similar to high performers)
        """
        if self._high_performers is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare feature matrices
        query_features = sites[self._feature_columns].copy()
        reference_features = self._high_performers[self._feature_columns].copy()

        # Handle missing values
        for col in self._feature_columns:
            if col in self.config.features.continuous_features:
                median_val = pd.concat([query_features[col], reference_features[col]]).median()
                query_features[col] = query_features[col].fillna(median_val)
                reference_features[col] = reference_features[col].fillna(median_val)
            else:
                query_features[col] = query_features[col].fillna("unknown")
                reference_features[col] = reference_features[col].fillna("unknown")

        # Compute Gower distance matrix
        distances = gower.gower_matrix(
            query_features.values,
            reference_features.values,
        )

        # Score by average distance to k nearest high-performers
        k = min(self.k_neighbors, distances.shape[1])
        k_nearest_dist = np.partition(distances, k, axis=1)[:, :k].mean(axis=1)

        # Convert distance to similarity score (0-1, higher is better)
        scores = 1 / (1 + k_nearest_dist)

        return scores

    def find_lookalikes(
        self,
        inactive_sites: pd.DataFrame,
        top_n: int = 100,
    ) -> pd.DataFrame:
        """Find top lookalike sites from inactive pool.

        Args:
            inactive_sites: DataFrame of inactive sites
            top_n: Number of top sites to return

        Returns:
            DataFrame with site_id and lookalike_score, sorted descending
        """
        scores = self.score(inactive_sites)

        result = inactive_sites[["site_id"]].copy()
        result["lookalike_score"] = scores

        return (
            result.sort_values("lookalike_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def get_nearest_performers(
        self,
        site: pd.Series,
        n: int = 5,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get the nearest high-performing sites to a query site.

        Useful for explaining why a site scored high.

        Args:
            site: Single site Series with features
            n: Number of nearest performers to return

        Returns:
            Tuple of (nearest performers DataFrame, distances)
        """
        if self._high_performers is None:
            raise ValueError("Model not fitted. Call fit() first.")

        query_features = site[self._feature_columns].to_frame().T

        # Handle missing values
        for col in self._feature_columns:
            if col in self.config.features.continuous_features:
                query_features[col] = query_features[col].fillna(
                    self._high_performers[col].median()
                )
            else:
                query_features[col] = query_features[col].fillna("unknown")

        reference_features = self._high_performers[self._feature_columns].copy()
        for col in self._feature_columns:
            if col in self.config.features.continuous_features:
                reference_features[col] = reference_features[col].fillna(
                    reference_features[col].median()
                )
            else:
                reference_features[col] = reference_features[col].fillna("unknown")

        distances = gower.gower_matrix(
            query_features.values,
            reference_features.values,
        )[0]

        # Get indices of n nearest
        nearest_idx = np.argsort(distances)[:n]
        nearest_distances = distances[nearest_idx]

        return self._high_performers.iloc[nearest_idx], nearest_distances

    @property
    def threshold(self) -> Optional[float]:
        """Revenue threshold for high performer classification."""
        return self._threshold

    @property
    def n_high_performers(self) -> int:
        """Number of high-performing reference sites."""
        if self._high_performers is None:
            return 0
        return len(self._high_performers)
