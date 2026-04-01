"""Tests for daily transactions/status merge → monthly site activity."""

import polars as pl
import pytest
from pathlib import Path

from site_scoring.data_transform import (
    load_daily_transactions,
    load_daily_status,
    build_monthly_site_activity,
)


@pytest.fixture
def sample_data(tmp_path):
    """Create minimal CSVs mimicking the real data format."""
    # Transactions: site A has 3 days (2 with transactions), site B has 1 day
    txn_csv = tmp_path / "site_transactions_daily.csv"
    txn_csv.write_text(
        "ID - Gbase,Date,Daily Transactions,GTVID\n"
        "site_a,2022-01-01,110,GTV001\n"
        "site_a,2022-01-02,0,GTV001\n"
        "site_a,2022-01-03,50,GTV001\n"
        "site_b,2022-01-01,200,GTV002\n"
    )

    # Status: site A active on day 1-2, deactivated day 3; site C is status-only
    status_csv = tmp_path / "site_status_daily.csv"
    status_csv.write_text(
        "Date,Status,GTVID,ID - Gbase\n"
        "2022-01-01,Active,GTV001,site_a\n"
        "2022-01-02,Active,GTV001,site_a\n"
        "2022-01-03,Deactivated,GTV001,site_a\n"
        "2022-01-01,Active,GTV003,site_c\n"
    )

    return tmp_path


def test_load_daily_transactions(sample_data):
    df = load_daily_transactions(sample_data)
    assert len(df) == 4
    assert set(df.columns) == {"id_gbase", "date", "daily_transactions", "gtvid"}
    assert df["id_gbase"].n_unique() == 2


def test_load_daily_status(sample_data):
    df = load_daily_status(sample_data)
    assert len(df) == 4
    assert set(df.columns) == {"date", "status", "gtvid", "id_gbase"}
    assert df["id_gbase"].n_unique() == 2


def test_build_monthly_activity_schema(sample_data):
    """Output has expected columns and types."""
    out = tmp_path_output(sample_data)
    monthly = build_monthly_site_activity(sample_data, out)

    assert set(monthly.columns) == {
        "id_gbase", "gtvid", "year_month",
        "active_days", "transaction_days", "total_days_in_data",
    }
    assert monthly["active_days"].dtype == pl.UInt16
    assert monthly["transaction_days"].dtype == pl.UInt16
    assert monthly["total_days_in_data"].dtype == pl.UInt16


def test_build_monthly_activity_values(sample_data):
    """Verify aggregated counts for site_a in 2022-01."""
    out = tmp_path_output(sample_data)
    monthly = build_monthly_site_activity(sample_data, out)

    site_a = monthly.filter(pl.col("id_gbase") == "site_a")
    assert len(site_a) == 1  # one month
    row = site_a.row(0, named=True)

    assert row["year_month"] == "2022-01"
    assert row["active_days"] == 2      # day 1 & 2 Active
    assert row["transaction_days"] == 2  # day 1 (110) & day 3 (50); day 2 is 0
    assert row["total_days_in_data"] == 3


def test_status_only_site(sample_data):
    """Site C exists only in status data — transaction_days should be 0."""
    out = tmp_path_output(sample_data)
    monthly = build_monthly_site_activity(sample_data, out)

    site_c = monthly.filter(pl.col("id_gbase") == "site_c")
    assert len(site_c) == 1
    row = site_c.row(0, named=True)

    assert row["active_days"] == 1
    assert row["transaction_days"] == 0
    assert row["total_days_in_data"] == 1


def test_transaction_only_site(sample_data):
    """Site B exists only in transactions data — active_days should be 0."""
    out = tmp_path_output(sample_data)
    monthly = build_monthly_site_activity(sample_data, out)

    site_b = monthly.filter(pl.col("id_gbase") == "site_b")
    assert len(site_b) == 1
    row = site_b.row(0, named=True)

    assert row["active_days"] == 0      # no status data → unknown ≠ Active
    assert row["transaction_days"] == 1
    assert row["total_days_in_data"] == 1


def test_parquet_written(sample_data):
    """Output parquet file is created."""
    out = tmp_path_output(sample_data)
    build_monthly_site_activity(sample_data, out)

    parquet = out / "site_monthly_activity.parquet"
    assert parquet.exists()

    # Round-trip: read it back and verify
    df = pl.read_parquet(parquet)
    assert len(df) == 3  # site_a, site_b, site_c — all in 2022-01


def test_gtvid_coalesced(sample_data):
    """GTVID is populated for all rows, coalesced from txn or status."""
    out = tmp_path_output(sample_data)
    monthly = build_monthly_site_activity(sample_data, out)

    assert monthly["gtvid"].null_count() == 0
    # site_b only in txn → gtvid comes from txn
    site_b = monthly.filter(pl.col("id_gbase") == "site_b")
    assert site_b["gtvid"][0] == "GTV002"
    # site_c only in status → gtvid comes from status
    site_c = monthly.filter(pl.col("id_gbase") == "site_c")
    assert site_c["gtvid"][0] == "GTV003"


def test_transactions_deduplication(tmp_path):
    """Duplicate (id_gbase, date) rows in transactions are collapsed to one."""
    txn_csv = tmp_path / "site_transactions_daily.csv"
    txn_csv.write_text(
        "ID - Gbase,Date,Daily Transactions,GTVID\n"
        "site_a,2022-01-01,67,GTV001\n"
        "site_a,2022-01-01,67,GTV001\n"
        "site_a,2022-01-01,67,GTV001\n"
        "site_a,2022-01-02,50,GTV001\n"
    )
    status_csv = tmp_path / "site_status_daily.csv"
    status_csv.write_text(
        "Date,Status,GTVID,ID - Gbase\n"
        "2022-01-01,Active,GTV001,site_a\n"
        "2022-01-02,Active,GTV001,site_a\n"
    )

    out = tmp_path / "output"
    monthly = build_monthly_site_activity(tmp_path, out)

    row = monthly.filter(pl.col("id_gbase") == "site_a").row(0, named=True)
    assert row["transaction_days"] == 2  # 2 unique days, not 4 raw rows
    assert row["total_days_in_data"] == 2


def test_multi_month(tmp_path):
    """Sites spanning two months produce two rows."""
    txn_csv = tmp_path / "site_transactions_daily.csv"
    txn_csv.write_text(
        "ID - Gbase,Date,Daily Transactions,GTVID\n"
        "site_x,2022-01-31,10,GTX\n"
        "site_x,2022-02-01,20,GTX\n"
    )
    status_csv = tmp_path / "site_status_daily.csv"
    status_csv.write_text(
        "Date,Status,GTVID,ID - Gbase\n"
        "2022-01-31,Active,GTX,site_x\n"
        "2022-02-01,Deactivated,GTX,site_x\n"
    )

    out = tmp_path / "output"
    monthly = build_monthly_site_activity(tmp_path, out)

    assert len(monthly) == 2
    months = monthly["year_month"].to_list()
    assert "2022-01" in months
    assert "2022-02" in months

    jan = monthly.filter(pl.col("year_month") == "2022-01").row(0, named=True)
    assert jan["active_days"] == 1
    assert jan["transaction_days"] == 1

    feb = monthly.filter(pl.col("year_month") == "2022-02").row(0, named=True)
    assert feb["active_days"] == 0  # Deactivated
    assert feb["transaction_days"] == 1


# Helper to create output dir inside tmp_path
def tmp_path_output(tmp_path: Path) -> Path:
    out = tmp_path / "processed"
    out.mkdir(exist_ok=True)
    return out
