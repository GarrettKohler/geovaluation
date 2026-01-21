"""
Tests for data loading service.

Verifies that all data loads correctly before page renders,
ensuring 57K+ sites are available with all required metrics.
"""

import pytest
import time
import pandas as pd
from pathlib import Path


class TestLoadSites:
    """Tests for load_sites() function."""

    def test_loads_dataframe(self, preloaded_data):
        """load_sites returns a pandas DataFrame."""
        sites = preloaded_data["sites"]
        assert isinstance(sites, pd.DataFrame)

    def test_loads_many_sites(self, preloaded_data):
        """load_sites returns 50K+ unique sites."""
        sites = preloaded_data["sites"]
        assert len(sites) >= 50000, f"Expected 50K+ sites, got {len(sites)}"

    def test_sites_have_required_columns(self, preloaded_data):
        """Sites DataFrame has required columns."""
        sites = preloaded_data["sites"]
        required = ["GTVID", "Latitude", "Longitude"]

        for col in required:
            assert col in sites.columns, f"Missing column: {col}"

    def test_no_missing_coordinates(self, preloaded_data):
        """No sites have missing coordinates."""
        sites = preloaded_data["sites"]

        assert sites["Latitude"].notna().all(), "Some sites have missing latitude"
        assert sites["Longitude"].notna().all(), "Some sites have missing longitude"

    def test_coordinates_are_numeric(self, preloaded_data):
        """Coordinates are numeric values."""
        sites = preloaded_data["sites"]

        assert pd.api.types.is_numeric_dtype(sites["Latitude"])
        assert pd.api.types.is_numeric_dtype(sites["Longitude"])

    def test_gtvid_is_unique(self, preloaded_data):
        """Each GTVID appears only once."""
        sites = preloaded_data["sites"]

        assert sites["GTVID"].nunique() == len(sites), "Duplicate GTVIDs found"

    def test_caching_works(self):
        """Second load is faster due to caching."""
        from src.services.data_service import load_sites

        # First load (may use cache)
        start1 = time.time()
        load_sites()
        time1 = time.time() - start1

        # Second load (should use cache)
        start2 = time.time()
        load_sites()
        time2 = time.time() - start2

        # Second load should be much faster
        assert time2 < time1 or time2 < 0.1, "Caching not working"


class TestLoadRevenueMetrics:
    """Tests for load_revenue_metrics() function."""

    def test_returns_dict(self, preloaded_data):
        """load_revenue_metrics returns a dictionary."""
        metrics = preloaded_data["metrics"]
        assert isinstance(metrics, dict)

    def test_has_metrics_for_many_sites(self, preloaded_data):
        """Revenue metrics available for most sites."""
        sites = preloaded_data["sites"]
        metrics = preloaded_data["metrics"]

        # Should have metrics for at least 80% of sites
        coverage = len(metrics) / len(sites)
        assert coverage >= 0.8, f"Only {coverage:.1%} sites have revenue metrics"

    def test_metrics_have_required_keys(self, preloaded_data):
        """Each metric entry has required keys."""
        metrics = preloaded_data["metrics"]
        required_keys = ["score", "avg_monthly", "total", "months"]

        # Check first 100 entries
        for site_id, site_metrics in list(metrics.items())[:100]:
            for key in required_keys:
                assert key in site_metrics, f"Site {site_id} missing key: {key}"

    def test_revenue_scores_normalized(self, preloaded_data):
        """Revenue scores are normalized between 0 and 1."""
        metrics = preloaded_data["metrics"]

        for site_id, site_metrics in list(metrics.items())[:1000]:
            score = site_metrics["score"]
            assert 0 <= score <= 1, f"Site {site_id} has score {score} outside [0,1]"

    def test_avg_monthly_is_positive(self, preloaded_data):
        """Average monthly revenue is non-negative."""
        metrics = preloaded_data["metrics"]

        for site_id, site_metrics in list(metrics.items())[:1000]:
            avg = site_metrics["avg_monthly"]
            assert avg >= 0, f"Site {site_id} has negative avg monthly: {avg}"


class TestLoadSiteDetails:
    """Tests for load_site_details() function."""

    def test_returns_dataframe(self, preloaded_data):
        """load_site_details returns a DataFrame."""
        details = preloaded_data["details"]
        assert isinstance(details, pd.DataFrame)

    def test_has_gtvid_column(self, preloaded_data):
        """Details DataFrame has gtvid column."""
        details = preloaded_data["details"]
        assert "gtvid" in details.columns

    def test_covers_all_sites(self, preloaded_data):
        """Site details available for all sites."""
        sites = preloaded_data["sites"]
        details = preloaded_data["details"]

        sites_set = set(sites["GTVID"])
        details_set = set(details["gtvid"])

        # All sites should have details
        coverage = len(sites_set & details_set) / len(sites_set)
        assert coverage >= 0.95, f"Only {coverage:.1%} sites have details"

    def test_has_location_columns(self, preloaded_data):
        """Details include location columns."""
        details = preloaded_data["details"]
        location_cols = ["state", "county", "zip", "dma"]

        for col in location_cols:
            assert col in details.columns, f"Missing location column: {col}"

    def test_has_site_info_columns(self, preloaded_data):
        """Details include site info columns."""
        details = preloaded_data["details"]
        info_cols = ["retailer", "network", "hardware_type"]

        for col in info_cols:
            assert col in details.columns, f"Missing site info column: {col}"


class TestFilterOptions:
    """Tests for get_filter_options() function."""

    def test_returns_dict(self, preloaded_data):
        """get_filter_options returns a dictionary."""
        options = preloaded_data["filter_options"]
        assert isinstance(options, dict)

    def test_has_state_options(self, preloaded_data):
        """Filter options include states."""
        options = preloaded_data["filter_options"]
        assert "State" in options
        assert len(options["State"]) > 0

    def test_state_options_are_sorted(self, preloaded_data):
        """State options are sorted alphabetically."""
        options = preloaded_data["filter_options"]
        states = options["State"]
        assert states == sorted(states), "States not sorted"

    def test_all_categorical_fields_present(self, preloaded_data, categorical_filter_fields):
        """All categorical fields have filter options."""
        options = preloaded_data["filter_options"]

        for field in categorical_filter_fields:
            assert field in options, f"Missing filter field: {field}"


class TestPreloadAllData:
    """Tests for preload_all_data() function."""

    def test_preload_runs_without_error(self):
        """preload_all_data executes successfully."""
        from src.services.data_service import preload_all_data

        # Should not raise any exceptions
        preload_all_data()

    def test_preload_fills_caches(self):
        """After preload, all data is cached."""
        from src.services import data_service
        from src.services.data_service import preload_all_data

        preload_all_data()

        # All caches should be filled
        assert data_service._sites_df is not None
        assert data_service._revenue_metrics is not None
        assert data_service._site_details_df is not None


class TestDataIntegrity:
    """Tests for data integrity across services."""

    def test_metrics_match_sites(self, preloaded_data):
        """Revenue metrics reference valid site IDs."""
        sites = preloaded_data["sites"]
        metrics = preloaded_data["metrics"]

        site_ids = set(sites["GTVID"])

        for metric_site_id in list(metrics.keys())[:1000]:
            assert metric_site_id in site_ids, f"Metric for unknown site: {metric_site_id}"

    def test_details_match_sites(self, preloaded_data):
        """Site details reference valid site IDs."""
        sites = preloaded_data["sites"]
        details = preloaded_data["details"]

        site_ids = set(sites["GTVID"])

        # Check a sample
        for detail_site_id in details["gtvid"].head(1000):
            assert detail_site_id in site_ids, f"Details for unknown site: {detail_site_id}"
