"""
Tests for /api/training/site-counts endpoint.

Verifies that site count computation works correctly for both
percentile and standard deviation threshold modes.
"""

import pytest
from pathlib import Path


class TestSiteCountsAPI:
    """Tests for the /api/training/site-counts endpoint."""

    def test_returns_200(self, client):
        """Endpoint returns 200 OK with default params."""
        response = client.get("/api/training/site-counts")
        assert response.status_code == 200

    def test_default_percentile_mode(self, client):
        """Default params use percentile mode p90-p100."""
        response = client.get("/api/training/site-counts")
        data = response.get_json()

        assert data["threshold_mode"] == "percentile"
        assert data["total_sites"] > 0
        assert data["positive_count"] > 0
        assert data["positive_count"] <= data["total_sites"]
        assert 0 <= data["positive_pct"] <= 100

    def test_percentile_returns_thresholds(self, client):
        """Percentile mode returns dollar thresholds."""
        response = client.get("/api/training/site-counts?mode=percentile&lower=80&upper=100")
        data = response.get_json()

        assert "lower_threshold" in data
        assert data["lower_threshold"] > 0  # revenue threshold should be positive

    def test_percentile_wider_range_more_sites(self, client):
        """Wider percentile range captures more sites."""
        narrow = client.get("/api/training/site-counts?mode=percentile&lower=90&upper=100").get_json()
        wide = client.get("/api/training/site-counts?mode=percentile&lower=50&upper=100").get_json()

        assert wide["positive_count"] >= narrow["positive_count"]
        assert wide["positive_pct"] >= narrow["positive_pct"]

    def test_stddev_mode(self, client):
        """Stddev mode returns correct structure."""
        response = client.get("/api/training/site-counts?mode=stddev&lower=1.0")
        data = response.get_json()

        assert data["threshold_mode"] == "stddev"
        assert data["total_sites"] > 0
        assert data["positive_count"] > 0
        assert "mean" in data
        assert "std" in data
        assert data["std"] > 0  # std dev should be positive

    def test_stddev_lower_sigma_captures_more(self, client):
        """Lower sigma threshold captures more sites (lower bar)."""
        high_bar = client.get("/api/training/site-counts?mode=stddev&lower=2.0").get_json()
        low_bar = client.get("/api/training/site-counts?mode=stddev&lower=0.5").get_json()

        assert low_bar["positive_count"] >= high_bar["positive_count"]

    def test_stddev_with_upper_bound(self, client):
        """Stddev mode with upper bound returns fewer sites than unbounded."""
        unbounded = client.get("/api/training/site-counts?mode=stddev&lower=0.5").get_json()
        bounded = client.get("/api/training/site-counts?mode=stddev&lower=0.5&upper=1.5").get_json()

        assert bounded["positive_count"] <= unbounded["positive_count"]

    def test_network_filter(self, client):
        """Network filter reduces total site count."""
        all_sites = client.get("/api/training/site-counts").get_json()
        filtered = client.get("/api/training/site-counts?network=Gilbarco").get_json()

        # Filtered should have fewer or equal total sites
        assert filtered["total_sites"] <= all_sites["total_sites"]

    def test_percentile_p100_upper_null_threshold(self, client):
        """Upper percentile of 100 means no upper bound (upper_threshold is null)."""
        response = client.get("/api/training/site-counts?mode=percentile&lower=90&upper=100")
        data = response.get_json()

        assert data["upper_threshold"] is None

    def test_no_nan_or_inf_in_response(self, client):
        """Response never contains NaN or Infinity (JSON-safe)."""
        import json

        response = client.get("/api/training/site-counts?mode=stddev&lower=-1.0")
        raw = response.data.decode("utf-8")

        # Valid JSON (no Infinity/NaN)
        data = json.loads(raw)
        assert data is not None

        # Check no string "Infinity" or "NaN" values
        for key, val in data.items():
            if isinstance(val, str):
                assert val not in ("Infinity", "-Infinity", "NaN"), f"{key} contains invalid value: {val}"


class TestComputeSiteCountsFunction:
    """Unit tests for the compute_site_counts service function."""

    def test_import_and_call(self):
        """Function is importable and callable."""
        from src.services.training_service import compute_site_counts
        result = compute_site_counts()
        assert isinstance(result, dict)
        assert "total_sites" in result

    def test_percentile_bounds(self):
        """Percentile mode respects bounds."""
        from src.services.training_service import compute_site_counts

        result = compute_site_counts(
            threshold_mode="percentile",
            lower_percentile=50,
            upper_percentile=100,
        )
        # Should be roughly 50% of sites
        assert 30 <= result["positive_pct"] <= 70

    def test_full_range_captures_all(self):
        """Percentile 0-100 captures all sites."""
        from src.services.training_service import compute_site_counts

        result = compute_site_counts(
            threshold_mode="percentile",
            lower_percentile=1,
            upper_percentile=100,
        )
        # p1-p100 should capture nearly all sites (minus any below p1)
        assert result["positive_pct"] >= 98

    def test_stddev_negative_sigma_captures_most(self):
        """Negative sigma threshold captures most sites."""
        from src.services.training_service import compute_site_counts

        result = compute_site_counts(
            threshold_mode="stddev",
            lower_sigma=-2.0,
        )
        # Mean - 2σ should capture ~97.7% of sites
        assert result["positive_pct"] >= 90
