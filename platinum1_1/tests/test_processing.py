"""Tests for processing/feature_processor.py and processing/geospatial.py."""

import numpy as np
import polars as pl
import pytest
import tempfile
from pathlib import Path

from platinum1_1.processing.feature_processor import FeatureProcessor, TensorBundle
from platinum1_1.processing.geospatial import haversine_distances, _detect_latlon_columns


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_zero_distance(self):
        """Distance from a point to itself should be ~0."""
        dist = haversine_distances(
            np.array([40.0]), np.array([-74.0]),
            np.array([40.0]), np.array([-74.0]),
        )
        assert abs(dist[0]) < 0.01

    def test_known_distance(self):
        """NYC to LA is approximately 2,451 miles."""
        nyc_lat, nyc_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437

        dist = haversine_distances(
            np.array([nyc_lat]), np.array([nyc_lon]),
            np.array([la_lat]), np.array([la_lon]),
        )
        assert 2400 < dist[0] < 2500

    def test_broadcasting(self):
        """N x M distance matrix via broadcasting."""
        lats1 = np.array([40.0, 41.0])[:, np.newaxis]
        lons1 = np.array([-74.0, -73.0])[:, np.newaxis]
        lats2 = np.array([34.0, 35.0, 36.0])[np.newaxis, :]
        lons2 = np.array([-118.0, -117.0, -116.0])[np.newaxis, :]

        dist = haversine_distances(lats1, lons1, lats2, lons2)
        assert dist.shape == (2, 3)
        assert np.all(dist > 0)


class TestDetectLatLon:
    def test_standard_columns(self):
        lat, lon = _detect_latlon_columns(["latitude", "longitude", "name"])
        assert lat == "latitude"
        assert lon == "longitude"

    def test_short_names(self):
        lat, lon = _detect_latlon_columns(["lat", "lng", "id"])
        assert lat == "lat"
        assert lon == "lng"

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="Could not detect"):
            _detect_latlon_columns(["name", "id", "value"])


# ---------------------------------------------------------------------------
# FeatureProcessor
# ---------------------------------------------------------------------------

class TestFeatureProcessor:
    @pytest.fixture
    def sample_df(self):
        """Create a minimal Polars DataFrame for testing."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            "rs_NVIs_95_185": np.random.randn(n),
            "rs_Revenue_95_185": np.random.randn(n),
            "log_min_distance_to_mcdonalds_mi": np.random.randn(n),
            "median_age": np.random.uniform(25, 65, n),
            "pct_female": np.random.uniform(0.4, 0.6, n),
            "network": np.random.choice(["A", "B", "C"], n),
            "program": np.random.choice(["X", "Y"], n),
            "c_emv_enabled_encoded": np.random.choice([0, 1], n),
            "c_nfc_enabled_encoded": np.random.choice([0, 1], n),
            "avg_monthly_revenue": np.random.uniform(100, 10000, n),
        })

    @pytest.fixture
    def config(self):
        """Minimal config for FeatureProcessor."""
        class Config:
            numeric_features = [
                "rs_NVIs_95_185", "rs_Revenue_95_185",
                "log_min_distance_to_mcdonalds_mi", "median_age", "pct_female",
            ]
            categorical_features = ["network", "program"]
            boolean_features = ["c_emv_enabled_encoded", "c_nfc_enabled_encoded"]
            target = "avg_monthly_revenue"
            task_type = "regression"
            lookalike_lower_percentile = 90
            lookalike_upper_percentile = 100
        return Config()

    def test_fit_transform_returns_bundle(self, config, sample_df):
        processor = FeatureProcessor(config)
        bundle = processor.fit_transform(sample_df)

        assert isinstance(bundle, TensorBundle)
        assert bundle.numeric.shape[0] == len(sample_df)
        assert bundle.categorical.shape[0] == len(sample_df)
        assert bundle.boolean.shape[0] == len(sample_df)
        assert bundle.target is not None
        assert bundle.target.shape[0] == len(sample_df)

    def test_clip_thresholds_stored(self, config, sample_df):
        """Clip thresholds should be stored during fit (train/serve skew fix)."""
        processor = FeatureProcessor(config)
        processor.fit_transform(sample_df)

        assert len(processor._clip_thresholds) > 0
        for col_idx, (p1, p99) in processor._clip_thresholds.items():
            assert p1 <= p99

    def test_transform_applies_clip_thresholds(self, config, sample_df):
        """Transform should apply the same clip thresholds as fit_transform."""
        processor = FeatureProcessor(config)
        processor.fit_transform(sample_df)

        # Create inference data with extreme values
        extreme_df = sample_df.clone()
        extreme_df = extreme_df.with_columns(
            pl.lit(1e6).alias("rs_NVIs_95_185")
        )

        bundle = processor.transform(extreme_df)
        # The extreme value should be clipped, so max scaled value should be bounded
        assert bundle.numeric[:, 0].max().item() <= 10.0

    def test_save_load_roundtrip(self, config, sample_df):
        """Processor should be saveable and loadable with identical behavior."""
        processor = FeatureProcessor(config)
        bundle1 = processor.fit_transform(sample_df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)

        try:
            processor.save(path)
            loaded = FeatureProcessor.load(path)

            bundle2 = loaded.transform(sample_df)
            assert bundle2.numeric.shape == bundle1.numeric.shape
            assert bundle2.categorical.shape == bundle1.categorical.shape
            # Clip thresholds should survive roundtrip
            assert len(loaded._clip_thresholds) == len(processor._clip_thresholds)
        finally:
            path.unlink(missing_ok=True)

    def test_inverse_transform_target(self, config, sample_df):
        """inverse_transform_target should return original scale values."""
        processor = FeatureProcessor(config)
        bundle = processor.fit_transform(sample_df)

        # Inverse transform the scaled targets
        original = processor.inverse_transform_target(bundle.target.numpy().flatten())
        # Should be in a reasonable revenue range
        assert original.min() > -1e6
        assert original.max() < 1e6

    def test_lookalike_binarization(self, sample_df):
        """Lookalike task should produce binary targets."""
        class LookalikeConfig:
            numeric_features = ["rs_NVIs_95_185", "median_age"]
            categorical_features = ["network"]
            boolean_features = ["c_emv_enabled_encoded"]
            target = "avg_monthly_revenue"
            task_type = "lookalike"
            lookalike_lower_percentile = 90
            lookalike_upper_percentile = 100

        processor = FeatureProcessor(LookalikeConfig())
        bundle = processor.fit_transform(sample_df)

        unique_values = bundle.target.unique().tolist()
        assert set(unique_values).issubset({0.0, 1.0})

    def test_unknown_categorical_handling(self, config, sample_df):
        """Unknown categories during transform should map to __UNKNOWN__."""
        processor = FeatureProcessor(config)
        processor.fit_transform(sample_df)

        new_df = sample_df.with_columns(
            pl.lit("NEVER_SEEN_BEFORE").alias("network")
        )
        bundle = processor.transform(new_df)
        # Should not raise and should have valid integer codes
        assert bundle.categorical.min().item() >= 0
