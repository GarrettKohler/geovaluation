"""
Unit tests for src/services/data_service.py.

Tests the data service functions with mocked data to validate
logic, edge cases, and error handling without requiring real CSV files.
"""

import pytest
import math
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Tests for _clean_nan_values()
# ---------------------------------------------------------------------------


class TestCleanNanValues:
    """Tests for _clean_nan_values() JSON sanitization function."""

    def setup_method(self):
        from src.services.data_service import _clean_nan_values
        self.clean = _clean_nan_values

    # --- Basic types ---

    def test_regular_float_unchanged(self):
        """Normal float values pass through unchanged."""
        assert self.clean(3.14) == 3.14

    def test_regular_int_unchanged(self):
        """Integer values pass through unchanged."""
        assert self.clean(42) == 42

    def test_string_unchanged(self):
        """String values pass through unchanged."""
        assert self.clean("hello") == "hello"

    def test_none_unchanged(self):
        """None values pass through unchanged."""
        assert self.clean(None) is None

    def test_bool_unchanged(self):
        """Boolean values pass through unchanged."""
        assert self.clean(True) is True
        assert self.clean(False) is False

    # --- NaN/Inf handling ---

    def test_nan_becomes_none(self):
        """Python float NaN is converted to None."""
        assert self.clean(float('nan')) is None

    def test_inf_becomes_none(self):
        """Python float Inf is converted to None."""
        assert self.clean(float('inf')) is None

    def test_negative_inf_becomes_none(self):
        """Python float -Inf is converted to None."""
        assert self.clean(float('-inf')) is None

    def test_math_nan_becomes_none(self):
        """math.nan is converted to None."""
        assert self.clean(math.nan) is None

    def test_math_inf_becomes_none(self):
        """math.inf is converted to None."""
        assert self.clean(math.inf) is None

    # --- NumPy types ---

    def test_numpy_nan_becomes_none(self):
        """np.nan is converted to None."""
        assert self.clean(np.nan) is None

    def test_numpy_inf_becomes_none(self):
        """np.inf is converted to None."""
        assert self.clean(np.inf) is None

    def test_numpy_float64_converted(self):
        """np.float64 is converted to Python float."""
        result = self.clean(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_float64_nan_becomes_none(self):
        """np.float64 NaN is converted to None."""
        assert self.clean(np.float64('nan')) is None

    def test_numpy_int64_converted(self):
        """np.int64 is converted to Python int."""
        result = self.clean(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_int32_converted(self):
        """np.int32 is converted to Python int."""
        result = self.clean(np.int32(7))
        assert result == 7
        assert isinstance(result, int)

    def test_numpy_array_converted(self):
        """np.ndarray is converted to list recursively."""
        arr = np.array([1.0, np.nan, 3.0])
        result = self.clean(arr)
        assert result == [1.0, None, 3.0]

    def test_numpy_2d_array_converted(self):
        """2D np.ndarray is converted to nested list."""
        arr = np.array([[1.0, np.nan], [np.inf, 4.0]])
        result = self.clean(arr)
        assert result == [[1.0, None], [None, 4.0]]

    # --- Nested structures ---

    def test_dict_with_nan_values(self):
        """NaN values in dict are cleaned."""
        data = {"a": 1.0, "b": float('nan'), "c": "text"}
        result = self.clean(data)
        assert result == {"a": 1.0, "b": None, "c": "text"}

    def test_list_with_nan_values(self):
        """NaN values in list are cleaned."""
        data = [1.0, float('nan'), float('inf'), 2.0]
        result = self.clean(data)
        assert result == [1.0, None, None, 2.0]

    def test_deeply_nested_structure(self):
        """Deeply nested structures are cleaned recursively."""
        data = {
            "level1": {
                "level2": [
                    {"value": float('nan')},
                    {"value": 42.0}
                ]
            }
        }
        result = self.clean(data)
        assert result["level1"]["level2"][0]["value"] is None
        assert result["level1"]["level2"][1]["value"] == 42.0

    def test_empty_dict(self):
        """Empty dict passes through."""
        assert self.clean({}) == {}

    def test_empty_list(self):
        """Empty list passes through."""
        assert self.clean([]) == []

    def test_mixed_numpy_and_python_types(self):
        """Mixed numpy and Python types in a dict are all handled."""
        data = {
            "py_float": 1.5,
            "np_float": np.float64(2.5),
            "np_int": np.int64(10),
            "np_nan": np.float64('nan'),
            "py_nan": float('nan'),
            "string": "test",
            "none": None,
        }
        result = self.clean(data)
        assert result == {
            "py_float": 1.5,
            "np_float": 2.5,
            "np_int": 10,
            "np_nan": None,
            "py_nan": None,
            "string": "test",
            "none": None,
        }

    def test_zero_float_not_cleaned(self):
        """Zero float is valid and not cleaned."""
        assert self.clean(0.0) == 0.0

    def test_negative_float_not_cleaned(self):
        """Negative float is valid and not cleaned."""
        assert self.clean(-5.5) == -5.5


# ---------------------------------------------------------------------------
# Tests for get_filtered_site_ids()
# ---------------------------------------------------------------------------


class TestGetFilteredSiteIds:
    """Tests for get_filtered_site_ids() with mocked data."""

    @pytest.fixture
    def mock_details_df(self):
        """Create a mock site details DataFrame."""
        return pd.DataFrame({
            'gtvid': ['SITE001', 'SITE002', 'SITE003', 'SITE004', 'SITE005'],
            'state': ['CA', 'CA', 'NY', 'TX', 'TX'],
            'network': ['NetA', 'NetB', 'NetA', 'NetA', 'NetC'],
            'retailer': ['RetX', 'RetX', 'RetY', 'RetZ', 'RetX'],
            'hardware_type': ['TypeA', 'TypeA', 'TypeB', 'TypeA', 'TypeB'],
        })

    def test_empty_filters_returns_empty(self, mock_details_df):
        """Empty filter dict returns empty list."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({})
            assert result == []

    def test_single_filter_matches(self, mock_details_df):
        """Single filter returns matching sites."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"State": "CA"})
            assert set(result) == {'SITE001', 'SITE002'}

    def test_multiple_filters_intersection(self, mock_details_df):
        """Multiple filters return intersection (AND logic)."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"State": "TX", "Network": "NetA"})
            assert result == ['SITE004']

    def test_no_matches_returns_empty(self, mock_details_df):
        """Filters with no matches return empty list."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"State": "FL"})
            assert result == []

    def test_invalid_field_name_ignored(self, mock_details_df):
        """Unknown field names are ignored (no error)."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"NonexistentField": "value"})
            # Should return all sites since no valid filter was applied...
            # Actually looking at the code, invalid display names not in CATEGORICAL_FIELDS
            # are skipped, so mask stays all-True, returning all sites
            assert len(result) == 5

    def test_none_value_filter_skipped(self, mock_details_df):
        """Filter with None value is skipped."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"State": None})
            # None value means the filter condition `if value` is False, so skipped
            assert len(result) == 5

    def test_empty_string_filter_skipped(self, mock_details_df):
        """Filter with empty string value is skipped."""
        from src.services.data_service import get_filtered_site_ids

        with patch('src.services.data_service.load_site_details', return_value=mock_details_df):
            result = get_filtered_site_ids({"State": ""})
            assert len(result) == 5


# ---------------------------------------------------------------------------
# Tests for get_site_details_for_display()
# ---------------------------------------------------------------------------


class TestGetSiteDetailsForDisplay:
    """Tests for get_site_details_for_display() with mocked data."""

    @pytest.fixture
    def mock_services(self):
        """Mock all three data loading functions."""
        details_df = pd.DataFrame({
            'gtvid': ['SITE001', 'SITE002'],
            'state': ['CA', 'NY'],
            'county': ['Los Angeles', 'Manhattan'],
            'zip': ['90001', '10001'],
            'dma': ['Los Angeles', 'New York'],
            'dma_rank': [2, 1],
            'retailer': ['Shell', 'BP'],
            'network': ['NetA', 'NetB'],
            'hardware_type': ['TypeA', 'TypeB'],
            'screen_count': [2, 1],
        })

        sites_df = pd.DataFrame({
            'GTVID': ['SITE001', 'SITE002', 'SITE003'],
            'Latitude': [34.05, 40.71, 37.77],
            'Longitude': [-118.24, -74.00, -122.42],
        })

        metrics = {
            'SITE001': {'score': 0.8, 'avg_monthly': 5000, 'total': 60000, 'months': 12},
            'SITE002': {'score': 0.5, 'avg_monthly': 2000, 'total': 24000, 'months': 12},
        }

        return details_df, sites_df, metrics

    def test_returns_full_details_for_known_site(self, mock_services):
        """Returns complete categorized details for a known site."""
        details_df, sites_df, metrics = mock_services
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('SITE001')

            assert result is not None
            assert result['site_id'] == 'SITE001'
            assert 'categories' in result
            assert 'partial' not in result  # Full data available

    def test_returns_none_for_unknown_site(self, mock_services):
        """Returns None for a site that doesn't exist anywhere."""
        details_df, sites_df, metrics = mock_services
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('NONEXISTENT')
            assert result is None

    def test_partial_data_for_coords_only_site(self, mock_services):
        """Returns partial data when site has coords but no details."""
        details_df, sites_df, metrics = mock_services
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            # SITE003 has coordinates but no details or revenue
            result = get_site_details_for_display('SITE003')

            assert result is not None
            assert result['partial'] is True
            assert 'coordinates' in result['available_data']

    def test_revenue_data_included(self, mock_services):
        """Revenue metrics are included in the Revenue category."""
        details_df, sites_df, metrics = mock_services
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('SITE001')
            revenue = result['categories']['Revenue']

            assert revenue['Avg Monthly Revenue'] == 5000
            assert revenue['Total Revenue'] == 60000
            assert revenue['Active Months'] == 12

    def test_categories_structure(self, mock_services):
        """All expected categories are present in result."""
        details_df, sites_df, metrics = mock_services
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('SITE001')
            expected_categories = ['Location', 'Site Info', 'Brands', 'Revenue',
                                   'Demographics', 'Performance', 'Capabilities', 'Sales']

            for cat in expected_categories:
                assert cat in result['categories'], f"Missing category: {cat}"

    def test_nan_values_cleaned_in_output(self, mock_services):
        """NaN values in site data are cleaned for JSON."""
        details_df, sites_df, metrics = mock_services
        # Add NaN to a detail field
        details_df.loc[0, 'dma_rank'] = float('nan')
        from src.services.data_service import get_site_details_for_display

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('SITE001')
            # NaN should be converted to None, not remain as float('nan')
            dma_rank = result['categories']['Location']['DMA Rank']
            assert dma_rank is None or not (isinstance(dma_rank, float) and math.isnan(dma_rank))

    def test_site_with_revenue_but_no_coords_or_details(self):
        """Site that only exists in revenue metrics returns partial."""
        from src.services.data_service import get_site_details_for_display

        details_df = pd.DataFrame({'gtvid': ['OTHER'], 'state': ['CA']})
        sites_df = pd.DataFrame({'GTVID': ['OTHER'], 'Latitude': [34.0], 'Longitude': [-118.0]})
        metrics = {'REVENUE_ONLY': {'score': 0.3, 'avg_monthly': 1000, 'total': 12000, 'months': 12}}

        with patch('src.services.data_service.load_site_details', return_value=details_df), \
             patch('src.services.data_service.load_revenue_metrics', return_value=metrics), \
             patch('src.services.data_service.load_sites', return_value=sites_df):

            result = get_site_details_for_display('REVENUE_ONLY')
            assert result is not None
            assert result['partial'] is True
            assert 'revenue' in result['available_data']


# ---------------------------------------------------------------------------
# Tests for load_sites() edge cases
# ---------------------------------------------------------------------------


class TestLoadSitesEdgeCases:
    """Edge case tests for load_sites() with mocked CSV."""

    def test_drops_rows_with_missing_lat(self):
        """Sites with missing latitude are excluded."""
        from src.services.data_service import load_sites
        import src.services.data_service as ds

        mock_df = pd.DataFrame({
            'gtvid': ['A', 'B', 'C'],
            'latitude': [34.0, None, 40.0],
            'longitude': [-118.0, -74.0, -122.0],
        })

        old_cache = ds._sites_df
        ds._sites_df = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_sites(force_reload=True)
                assert len(result) == 2
                assert 'B' not in result['GTVID'].values
        finally:
            ds._sites_df = old_cache

    def test_drops_rows_with_missing_lon(self):
        """Sites with missing longitude are excluded."""
        from src.services.data_service import load_sites
        import src.services.data_service as ds

        mock_df = pd.DataFrame({
            'gtvid': ['A', 'B', 'C'],
            'latitude': [34.0, 40.0, 37.0],
            'longitude': [-118.0, None, -122.0],
        })

        old_cache = ds._sites_df
        ds._sites_df = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_sites(force_reload=True)
                assert len(result) == 2
        finally:
            ds._sites_df = old_cache

    def test_deduplicates_by_gtvid(self):
        """Duplicate GTVIDs are deduplicated (first occurrence kept)."""
        from src.services.data_service import load_sites
        import src.services.data_service as ds

        mock_df = pd.DataFrame({
            'gtvid': ['A', 'A', 'B'],
            'latitude': [34.0, 35.0, 40.0],
            'longitude': [-118.0, -119.0, -74.0],
        })

        old_cache = ds._sites_df
        ds._sites_df = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_sites(force_reload=True)
                assert len(result) == 2
                # First occurrence of 'A' should be kept
                site_a = result[result['GTVID'] == 'A'].iloc[0]
                assert site_a['Latitude'] == 34.0
        finally:
            ds._sites_df = old_cache

    def test_columns_renamed_correctly(self):
        """Output columns are renamed to GTVID, Latitude, Longitude."""
        from src.services.data_service import load_sites
        import src.services.data_service as ds

        mock_df = pd.DataFrame({
            'gtvid': ['A'],
            'latitude': [34.0],
            'longitude': [-118.0],
        })

        old_cache = ds._sites_df
        ds._sites_df = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_sites(force_reload=True)
                assert 'GTVID' in result.columns
                assert 'Latitude' in result.columns
                assert 'Longitude' in result.columns
                assert 'gtvid' not in result.columns
        finally:
            ds._sites_df = old_cache


# ---------------------------------------------------------------------------
# Tests for load_revenue_metrics() edge cases
# ---------------------------------------------------------------------------


class TestLoadRevenueMetricsEdgeCases:
    """Edge case tests for load_revenue_metrics()."""

    def test_score_clamped_to_zero_one(self):
        """Revenue scores are clamped between 0 and 1."""
        from src.services.data_service import load_revenue_metrics
        import src.services.data_service as ds

        # Create data where one site has very high revenue (score > 1 before clamping)
        # and one has very low revenue (score < 0 before clamping)
        mock_df = pd.DataFrame({
            'gtvid': ['HIGH'] * 12 + ['LOW'] * 12 + ['MED'] * 12,
            'revenue': [100000] * 12 + [0.01] * 12 + [5000] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='ME').tolist() * 3,
            'site_activated_date': ['2024-01-01'] * 36,
        })

        old_cache = ds._revenue_metrics
        ds._revenue_metrics = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_revenue_metrics(force_reload=True)
                for site_id, metrics in result.items():
                    assert 0 <= metrics['score'] <= 1, \
                        f"Site {site_id} score {metrics['score']} not in [0,1]"
        finally:
            ds._revenue_metrics = old_cache

    def test_handles_zero_active_months(self):
        """Division by zero is prevented when active_months is 0."""
        from src.services.data_service import load_revenue_metrics
        import src.services.data_service as ds

        # clip(lower=1) prevents division by zero
        mock_df = pd.DataFrame({
            'gtvid': ['A', 'B'],
            'revenue': [1000.0, 2000.0],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'site_activated_date': pd.to_datetime(['2024-01-01', '2024-01-01']),
        })

        old_cache = ds._revenue_metrics
        ds._revenue_metrics = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_revenue_metrics(force_reload=True)
                # Should not raise ZeroDivisionError
                for metrics in result.values():
                    assert metrics['avg_monthly'] >= 0
        finally:
            ds._revenue_metrics = old_cache

    def test_drops_null_revenue_rows(self):
        """Rows with null revenue are excluded from calculations."""
        from src.services.data_service import load_revenue_metrics
        import src.services.data_service as ds

        mock_df = pd.DataFrame({
            'gtvid': ['A', 'A', 'B'],
            'revenue': [1000.0, None, 2000.0],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-01-01']),
            'site_activated_date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01']),
        })

        old_cache = ds._revenue_metrics
        ds._revenue_metrics = None
        try:
            with patch('pandas.read_csv', return_value=mock_df):
                result = load_revenue_metrics(force_reload=True)
                # Site A should only count 1 month (the non-null one)
                assert result['A']['months'] == 1
                assert result['A']['total'] == 1000.0
        finally:
            ds._revenue_metrics = old_cache


# ---------------------------------------------------------------------------
# Tests for get_filter_options() edge cases
# ---------------------------------------------------------------------------


class TestGetFilterOptionsEdgeCases:
    """Edge case tests for get_filter_options()."""

    def test_excludes_empty_strings(self):
        """Empty string values are excluded from filter options."""
        from src.services.data_service import get_filter_options
        import src.services.data_service as ds

        mock_details = pd.DataFrame({
            'gtvid': ['A', 'B', 'C'],
            'state': ['CA', '', 'NY'],
            'network': ['NetA', '  ', 'NetB'],
        })

        old_cache = ds._unique_values_cache
        ds._unique_values_cache = None
        try:
            with patch('src.services.data_service.load_site_details', return_value=mock_details):
                result = get_filter_options(force_reload=True)
                if 'State' in result:
                    assert '' not in result['State']
                    assert '  ' not in result.get('Network', [])
        finally:
            ds._unique_values_cache = old_cache

    def test_excludes_nan_values(self):
        """NaN values are excluded from filter options."""
        from src.services.data_service import get_filter_options
        import src.services.data_service as ds

        mock_details = pd.DataFrame({
            'gtvid': ['A', 'B', 'C'],
            'state': ['CA', None, 'NY'],
        })

        old_cache = ds._unique_values_cache
        ds._unique_values_cache = None
        try:
            with patch('src.services.data_service.load_site_details', return_value=mock_details):
                result = get_filter_options(force_reload=True)
                if 'State' in result:
                    assert None not in result['State']
        finally:
            ds._unique_values_cache = old_cache

    def test_options_are_sorted(self):
        """Filter options are sorted alphabetically."""
        from src.services.data_service import get_filter_options
        import src.services.data_service as ds

        mock_details = pd.DataFrame({
            'gtvid': ['A', 'B', 'C'],
            'state': ['NY', 'CA', 'TX'],
        })

        old_cache = ds._unique_values_cache
        ds._unique_values_cache = None
        try:
            with patch('src.services.data_service.load_site_details', return_value=mock_details):
                result = get_filter_options(force_reload=True)
                if 'State' in result:
                    assert result['State'] == sorted(result['State'])
        finally:
            ds._unique_values_cache = old_cache
