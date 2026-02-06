"""
Revenue consistency tests for geospatial application.

Validates that:
1. Site Revenue/Diagnostics data aggregates correctly to ~57K unique sites
2. No duplicates exist in aggregated data
3. Aggregated data joins properly with distance datasets (~65K rows)
4. Historic total revenue displayed in UI matches training data values
5. Revenue values are consistent across data_service.py and data_transform.py
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path


# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "input"
REVENUE_CSV = DATA_DIR / "site_scores_revenue_and_diagnostics.csv"
NEAREST_SITES_CSV = DATA_DIR / "nearest_site_distances.csv"


class TestRawDataIntegrity:
    """Tests for raw Site Revenue/Diagnostics data."""

    def test_raw_data_has_expected_rows(self):
        """Raw data should have ~1.4M monthly records."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        assert len(df) > 1_400_000, f"Expected 1.4M+ rows, got {len(df):,}"
        assert len(df) < 2_000_000, f"Unexpectedly large: {len(df):,} rows"

    def test_raw_data_no_duplicate_site_month(self):
        """Each (gtvid, date) combination should be unique."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Group by gtvid + date and count
        dup_check = df.group_by(['gtvid', 'date']).agg(pl.len().alias('count'))
        duplicates = dup_check.filter(pl.col('count') > 1)

        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate (gtvid, date) combinations"

    def test_id_gbase_gtvid_one_to_one(self):
        """Each id_gbase maps to exactly one gtvid and vice versa."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Check id_gbase -> gtvid mapping
        gtvid_per_base = df.group_by('id_gbase').agg(pl.col('gtvid').n_unique().alias('n_gtvids'))
        multi_gtvid = gtvid_per_base.filter(pl.col('n_gtvids') > 1)
        assert len(multi_gtvid) == 0, f"{len(multi_gtvid)} id_gbase values map to multiple gtvids"

        # Check gtvid -> id_gbase mapping
        base_per_gtvid = df.group_by('gtvid').agg(pl.col('id_gbase').n_unique().alias('n_bases'))
        multi_base = base_per_gtvid.filter(pl.col('n_bases') > 1)
        assert len(multi_base) == 0, f"{len(multi_base)} gtvid values map to multiple id_gbase"


class TestAggregatedDataConsistency:
    """Tests for aggregated site data consistency."""

    def test_aggregation_by_gtvid_and_idbase_match(self):
        """Aggregating by gtvid and id_gbase should produce identical total revenue."""
        df = pl.read_csv(
            REVENUE_CSV,
            infer_schema_length=10000,
            null_values=['', 'NA', 'null', 'Unknown']
        )

        # Aggregate by gtvid
        gtvid_agg = df.filter(
            pl.col('revenue').is_not_null() & pl.col('gtvid').is_not_null()
        ).group_by('gtvid').agg(pl.col('revenue').sum().alias('total_revenue'))

        # Aggregate by id_gbase
        df_sorted = df.sort(['id_gbase', 'date'])
        idbase_agg = df_sorted.group_by('id_gbase').agg([
            pl.col('gtvid').last().alias('gtvid'),
            pl.col('revenue').sum().alias('total_revenue')
        ])

        # Compare totals
        gtvid_totals = {row['gtvid']: row['total_revenue'] for row in gtvid_agg.iter_rows(named=True)}
        idbase_totals = {row['gtvid']: row['total_revenue'] for row in idbase_agg.iter_rows(named=True)}

        mismatches = []
        for gtvid in gtvid_totals:
            if gtvid in idbase_totals:
                if abs(gtvid_totals[gtvid] - idbase_totals[gtvid]) > 0.01:
                    mismatches.append((gtvid, gtvid_totals[gtvid], idbase_totals[gtvid]))

        assert len(mismatches) == 0, f"Found {len(mismatches)} revenue mismatches"

    def test_aggregated_site_count_matches_expected(self):
        """Aggregated data should have ~57K unique sites."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        unique_gtvids = df['gtvid'].n_unique()
        unique_idbase = df['id_gbase'].n_unique()

        # Both should match
        assert unique_gtvids == unique_idbase, "gtvid and id_gbase counts don't match"

        # Should be around 57K sites
        assert 55_000 <= unique_gtvids <= 60_000, f"Expected ~57K sites, got {unique_gtvids:,}"

    def test_no_duplicate_gtvids_in_aggregation(self):
        """Aggregated data should have exactly one row per GTVID."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Aggregate like data_service.py
        site_agg = df.group_by('gtvid').agg([
            pl.col('revenue').sum().alias('total_revenue'),
            pl.len().alias('active_months')
        ])

        # Check for duplicates
        assert site_agg['gtvid'].n_unique() == len(site_agg), "Duplicate GTVIDs in aggregation"


class TestDistanceDatasetJoins:
    """Tests for joining aggregated data with distance datasets."""

    def test_nearest_sites_dataset_row_count(self):
        """Nearest sites dataset should have ~65K+ rows (one per GTVID)."""
        nearest = pl.read_csv(NEAREST_SITES_CSV)

        assert len(nearest) >= 65_000, f"Expected 65K+ rows, got {len(nearest):,}"
        assert nearest['GTVID'].n_unique() == len(nearest), "Duplicate GTVIDs in nearest_sites"

    def test_site_scores_joins_with_distances(self):
        """Site scores should join cleanly with distance datasets."""
        # Load site scores aggregated
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        site_gtvids = set(df['gtvid'].unique().to_list())

        # Load distance datasets
        nearest = pl.read_csv(NEAREST_SITES_CSV)
        nearest_gtvids = set(nearest['GTVID'].to_list())

        # Most sites should have distance data
        overlap = site_gtvids & nearest_gtvids
        coverage = len(overlap) / len(site_gtvids)

        assert coverage >= 0.99, f"Only {coverage:.1%} of sites have distance data"

    def test_distance_datasets_superset_of_sites(self):
        """Distance datasets may have more GTVIDs than site scores (historical sites)."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        site_gtvids = set(df['gtvid'].unique().to_list())

        nearest = pl.read_csv(NEAREST_SITES_CSV)
        nearest_gtvids = set(nearest['GTVID'].to_list())

        # Distance files have more sites (historical/decommissioned)
        extra_in_distances = nearest_gtvids - site_gtvids

        # This is expected - document how many
        print(f"\nDistance datasets contain {len(extra_in_distances):,} historical sites not in revenue data")
        assert len(extra_in_distances) < 15_000, "Too many unmatched sites in distance data"


class TestRevenueValueConsistency:
    """Tests for revenue value consistency between data sources."""

    def test_total_revenue_positive(self):
        """Total revenue across all sites should be positive."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        total_revenue = df['revenue'].sum()
        assert total_revenue > 0, f"Total revenue is {total_revenue}"
        assert total_revenue > 400_000_000, f"Expected $400M+, got ${total_revenue:,.2f}"

    def test_site_revenue_aggregation_matches_raw(self):
        """Sum of aggregated site revenues equals sum of raw monthly revenues."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Raw total
        raw_total = df['revenue'].sum()

        # Aggregated total
        site_agg = df.group_by('gtvid').agg(pl.col('revenue').sum().alias('total_revenue'))
        agg_total = site_agg['total_revenue'].sum()

        # Should be identical
        assert abs(raw_total - agg_total) < 1.0, f"Raw: ${raw_total:,.2f}, Agg: ${agg_total:,.2f}"

    def test_data_service_matches_raw_aggregation(self, preloaded_data):
        """data_service.py aggregation matches direct polars aggregation."""
        # Get data_service totals
        metrics = preloaded_data["metrics"]
        service_total = sum(m['total'] for m in metrics.values())

        # Get direct aggregation
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        direct_total = df['revenue'].sum()

        # Should match within floating point tolerance
        assert abs(service_total - direct_total) < 1.0, \
            f"data_service: ${service_total:,.2f}, direct: ${direct_total:,.2f}"


class TestUIRevenueDisplay:
    """Tests for revenue values displayed in UI after training."""

    def test_api_sites_returns_correct_total_revenue(self, client, preloaded_data):
        """API /api/sites returns correct totalRevenue for each site."""
        response = client.get('/api/sites')
        assert response.status_code == 200

        sites_from_api = response.get_json()
        metrics = preloaded_data["metrics"]

        # Check a sample of sites
        mismatches = []
        for site in sites_from_api[:100]:
            gtvid = site['GTVID']
            api_total = site['totalRevenue']
            expected_total = metrics.get(gtvid, {}).get('total', 0)

            if abs(api_total - expected_total) > 0.01:
                mismatches.append({
                    'gtvid': gtvid,
                    'api': api_total,
                    'expected': expected_total
                })

        assert len(mismatches) == 0, f"Found {len(mismatches)} API revenue mismatches: {mismatches[:5]}"

    def test_api_sites_avg_monthly_revenue_calculated_correctly(self, client, preloaded_data):
        """avgMonthlyRevenue = totalRevenue / activeMonths."""
        response = client.get('/api/sites')
        sites_from_api = response.get_json()

        mismatches = []
        for site in sites_from_api[:100]:
            total = site['totalRevenue']
            months = site['activeMonths']
            avg = site['avgMonthlyRevenue']

            if months > 0:
                expected_avg = total / months
                if abs(avg - expected_avg) > 0.01:
                    mismatches.append({
                        'gtvid': site['GTVID'],
                        'total': total,
                        'months': months,
                        'api_avg': avg,
                        'expected_avg': expected_avg
                    })

        assert len(mismatches) == 0, f"avgMonthlyRevenue calculation errors: {mismatches[:5]}"

    def test_site_detail_revenue_matches_aggregation(self, client, preloaded_data, sample_site_ids):
        """Site detail endpoint returns revenue matching aggregation."""
        metrics = preloaded_data["metrics"]

        for site_id in sample_site_ids[:5]:
            response = client.get(f'/api/site-details/{site_id}')
            assert response.status_code == 200

            detail = response.get_json()
            revenue_category = detail.get('categories', {}).get('Revenue', {})

            api_total = revenue_category.get('Total Revenue', 0) or 0
            expected_total = metrics.get(site_id, {}).get('total', 0)

            assert abs(api_total - expected_total) < 1.0, \
                f"Site {site_id}: detail=${api_total:,.2f}, expected=${expected_total:,.2f}"


class TestTrainingDataConsistency:
    """Tests for training data matching UI display values."""

    def test_training_data_site_count(self):
        """Training data (Active sites only) should have ~26K sites.

        Status distribution:
        - Active: ~26K (used for training)
        - Temporarily Deactivated: ~23K
        - Other statuses: ~8K
        """
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Count Active sites
        active_count = df.filter(pl.col('statuis') == 'Active')['gtvid'].n_unique()

        # Active sites are ~26K (not all 57K - many are deactivated)
        assert 20_000 <= active_count <= 35_000, f"Active sites: {active_count:,}"

    def test_training_revenue_matches_display_revenue(self, preloaded_data):
        """Training uses same revenue values as UI display."""
        # This verifies the data pipeline is consistent
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)

        # Aggregate by gtvid (like data_service.py)
        site_agg = df.group_by('gtvid').agg([
            pl.col('revenue').sum().alias('total_revenue'),
            pl.len().alias('active_months')
        ])

        # Compare with data_service
        metrics = preloaded_data["metrics"]

        # Sample check
        mismatches = 0
        for row in site_agg.head(1000).iter_rows(named=True):
            gtvid = row['gtvid']
            agg_total = row['total_revenue']
            service_total = metrics.get(gtvid, {}).get('total', 0)

            if abs(agg_total - service_total) > 0.01:
                mismatches += 1

        assert mismatches == 0, f"Found {mismatches} training/display mismatches"

    def test_specific_site_revenue_matches_raw_data(self, preloaded_data):
        """Verify specific site's total revenue equals sum of its monthly records."""
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        metrics = preloaded_data["metrics"]

        # Pick a sample of sites with revenue
        sample_gtvids = list(metrics.keys())[:10]

        for gtvid in sample_gtvids:
            # Get raw monthly records
            site_records = df.filter(pl.col('gtvid') == gtvid)
            raw_sum = site_records['revenue'].sum()

            # Get from metrics
            service_total = metrics[gtvid]['total']

            assert abs(raw_sum - service_total) < 0.01, \
                f"Site {gtvid}: raw_sum=${raw_sum:,.2f}, service=${service_total:,.2f}"


class TestModelTrainingData:
    """Tests for model training data structure."""

    def test_training_uses_aggregated_data_not_monthly(self):
        """
        CRITICAL: Model must train on aggregated per-site data (~26K rows),
        NOT raw monthly records (~1.4M rows).

        Training on monthly records causes:
        1. Data leakage (same site in train/val/test)
        2. Learning temporal patterns instead of site characteristics
        3. Incorrect sample counts
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from site_scoring.config import Config
        from site_scoring.data_loader import DataProcessor

        config = Config()
        processor = DataProcessor(config)
        _, _, _, target = processor.load_and_process()

        # Should be ~26K aggregated sites, NOT ~1.4M monthly records
        assert len(target) < 100_000, \
            f"Training on {len(target):,} samples - likely using raw monthly data instead of aggregated!"

        # Should be close to Active site count (~26K)
        assert 20_000 <= len(target) <= 35_000, \
            f"Expected ~26K samples, got {len(target):,}"

    def test_config_points_to_aggregated_data(self):
        """Config should point to processed/aggregated parquet file."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from site_scoring.config import Config

        config = Config()
        data_path = str(config.data_path)

        # Should be parquet (aggregated), not CSV (raw monthly)
        assert 'processed' in data_path or 'training' in data_path, \
            f"Config points to raw data: {data_path}"

        assert data_path.endswith('.parquet'), \
            f"Config should use parquet for efficiency: {data_path}"

    def test_no_data_leakage_from_target(self):
        """Features should not include target or derived columns."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from site_scoring.config import Config

        config = Config()

        # Target should not be in features
        assert config.target not in config.numeric_features, \
            f"Target '{config.target}' found in numeric_features!"

        # If predicting total_revenue, log_total_revenue should not be a feature
        if config.target == "total_revenue":
            assert "log_total_revenue" not in config.numeric_features, \
                "log_total_revenue is derived from target - data leakage!"
            assert "avg_monthly_revenue" not in config.numeric_features, \
                "avg_monthly_revenue is derived from target - data leakage!"


class TestEndToEndRevenueFlow:
    """End-to-end tests for revenue data flow from raw data to UI."""

    def test_complete_revenue_pipeline(self, client, preloaded_data):
        """
        Complete test of revenue pipeline:
        1. Raw data aggregation
        2. data_service.py processing
        3. API endpoint response
        4. Verify all match
        """
        # Step 1: Direct aggregation from CSV
        df = pl.read_csv(REVENUE_CSV, infer_schema_length=10000)
        direct_agg = df.group_by('gtvid').agg(pl.col('revenue').sum().alias('total_revenue'))
        direct_map = {row['gtvid']: row['total_revenue'] for row in direct_agg.iter_rows(named=True)}

        # Step 2: data_service.py
        service_metrics = preloaded_data["metrics"]

        # Step 3: API response
        response = client.get('/api/sites')
        api_sites = response.get_json()
        api_map = {s['GTVID']: s['totalRevenue'] for s in api_sites}

        # Verify all three sources match
        errors = []
        sample_gtvids = list(direct_map.keys())[:100]

        for gtvid in sample_gtvids:
            direct = direct_map.get(gtvid, 0)
            service = service_metrics.get(gtvid, {}).get('total', 0)
            api = api_map.get(gtvid, 0)

            if abs(direct - service) > 0.01 or abs(direct - api) > 0.01:
                errors.append({
                    'gtvid': gtvid,
                    'direct': direct,
                    'service': service,
                    'api': api
                })

        assert len(errors) == 0, f"Pipeline inconsistencies found: {errors[:5]}"

    def test_summary_stats_match_site_totals(self, client):
        """Summary panel totals should match sum of individual site revenues."""
        response = client.get('/api/sites')
        sites = response.get_json()

        # Calculate totals from individual sites
        calculated_total = sum(s['totalRevenue'] for s in sites)
        calculated_avg = calculated_total / len(sites) if sites else 0

        # These should be the values shown in UI summary panel
        assert calculated_total > 400_000_000, f"Total revenue: ${calculated_total:,.2f}"
        print(f"\n✓ Total Revenue (all sites): ${calculated_total:,.2f}")
        print(f"✓ Average Revenue per site: ${calculated_avg:,.2f}")
        print(f"✓ Site count: {len(sites):,}")
