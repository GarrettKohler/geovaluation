"""
Data transformation script for site scoring ML pipeline.

Transforms the monthly Site Scores data into a single-row-per-site dataset with:
1. Total revenue and auction metrics across all months
2. Average metrics per active month (number of months the site was active/present in dataset)
3. Joined geospatial features from auxiliary files
4. Log transformations of numeric metrics
5. One-hot encoded capability and restriction flags

This creates the "pre-cleaned" dataset ready for ML pipeline cleaning.
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Tuple

# Project root - dynamically resolved from this file's location
# site_scoring/data_transform.py -> go up one level to get project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default paths (relative to project root)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"

_GLOSSARY_STAGES = {
    "collection": {
        "title": "1. Data Collection",
        "question": "Where does our data come from and how do we gather it?",
        "intro": "Data collection is where everything begins. Our pipeline pulls structured data from <strong>multiple CSV sources</strong> to build a complete picture of every site in our network. Two primary sources come from Salesforce (site attributes and revenue), supplemented by geographic proximity data computed from 3rd-party retailer locations.",
        "analogy": "Imagine evaluating 67,000 gas stations for advertising potential. Salesforce hands you two spreadsheets: one describing each station (does it sell beer? what's the local median income?) and one showing ad revenue. Separately, a geography team measures how far each station is from the nearest highway, Walmart, McDonald's, and Target. Together these paint the full picture.",
        "why": "We keep collection simple and auditable \u2014 CSV files loaded via Polars (10\u201320x faster than pandas). Raw files are preserved in <code>data/input/</code> so any team member can open them in Excel. Proximity features are reproducible via scripts \u2014 if a retailer dataset updates, we rerun the distance calculations.",
    },
    "cleaning": {
        "title": "2. Data Cleaning",
        "question": "How do we fix errors, standardize formats, and prepare data for modeling?",
        "intro": "Before any modeling can happen, raw data needs to be <strong>cleaned, standardized, and transformed</strong>. Our ETL pipeline in <code>data_transform.py</code> takes ~1.4 million monthly records and transforms them into a clean, ML-ready dataset of active sites.",
        "analogy": "Imagine you received a massive stack of monthly reports from 67,000 gas stations. Some reports have typos, some leave fields blank, some use \"Yes\" where others use \"true.\" Before you can analyze anything, you need to: standardize every form, consolidate monthly reports into one summary per station, toss out stations that are no longer active, and convert everything to numbers a computer can process.",
        "why": "The full cleaning pipeline runs via <code>python3 -m site_scoring.data_transform</code> and outputs two files: <strong>site_aggregated_precleaned.parquet</strong> (all sites) as a checkpoint, and <strong>site_training_data.parquet</strong> (active sites only) as the final ML-ready dataset. Both are saved as Parquet for fast loading and as CSV for manual inspection.",
    },
    "combining": {
        "title": "3. Data Combining",
        "question": "How do we join data from different sources into one dataset?",
        "intro": "Data combining is where isolated sources become a <strong>single, unified dataset</strong>. Our pipeline merges temporal data (monthly records into site summaries), momentum features (relative strength trends), and spatial data (6 geographic proximity files) \u2014 all keyed on unique site identifiers. The result: one rich row per site with ~102 columns.",
        "analogy": "Imagine you're building a profile for each of 57,675 gas stations. You start with a giant ledger of monthly reports (1.4M rows) and condense it into one summary card per station. Then you staple on distance measurements \u2014 how far is this station from the nearest highway? Walmart? McDonald's? Finally, you add trend arrows showing whether each station is doing better or worse recently. The result is one comprehensive card per station.",
        "why": "The combining stage uses <strong>two different join keys</strong>: <code>id_gbase</code> for temporal aggregation (collapsing monthly \u2192 site-level) and <code>gtvid</code> for geospatial joins (attaching distance files). Both uniquely identify a site, but different source systems use different ID formats. All joins are left joins \u2014 preserving every site even when distance data is missing \u2014 to avoid silently dropping sites from the training set.",
    },
}

_GLOSSARY_SOURCES = [
    {
        "id": "sites_base",
        "icon": "\u26fd",
        "name": "Sites \u2013 Base Data Set",
        "desc": "Site attributes, capabilities, ad eligibility, demographics",
        "source": "Salesforce",
        "format": "CSV",
        "rows": "67,650",
        "cols": 43,
        "color": "accent",
        "fields": ["ID - Gbase", "Avg Daily Impressions", "Avg Household Income", "Median Age", "C - Sells Beer", "C - NFC Enabled", "C - Open 24 Hours", "R - Restaurants - QSR"],
        "sample": [["5fa964f8...", "121", "$87,841", "36.8", "Yes", "Unknown", "No", "false"], ["5fa965a5...", "83", "$58,509", "32.1", "No", "Yes", "No", "false"]],
        "notes": "Exported from Salesforce. Each row is a gas station/convenience store identified by Gbase ID. 43 columns: site capabilities (C- prefix: sells beer, diesel, NFC, EMV, 24hrs), ad category eligibility (R- prefix: lottery, automotive, restaurant, CPG ads), and local demographics (household income, median age). Also includes Average Daily Impressions \u2014 estimated foot traffic.",
    },
    {
        "id": "site_revenue",
        "icon": "\ud83d\udcb0",
        "name": "Site Revenue",
        "desc": "Revenue, program, network, DMA rank per site",
        "source": "Salesforce",
        "format": "CSV",
        "rows": "67,604",
        "cols": 7,
        "color": "cyan",
        "fields": ["ID - Gbase", "Sellable", "Schedulable", "Program", "Network", "Average Revenue", "Average DMA Rank"],
        "sample": [["57745759...", "true", "true", "IOTV2", "Wayne", "$136.13", "8"], ["57745759...", "true", "true", "Dover - IOTV2", "Dover", "$531.55", "8"]],
        "notes": "From Salesforce. Revenue per site-program combination. A single site can appear on multiple rows for different programs (IOTV2, Dover - IOTV2, etc.). Average Revenue is the target variable our model predicts. Sellable/Schedulable flags indicate whether the site is active for ad placement.",
    },
    {
        "id": "geo_prox",
        "icon": "\ud83d\udccd",
        "name": "Geographic Proximity Files",
        "desc": "Distances to interstates, nearest sites, and major retailers",
        "source": "Computed from geodata",
        "format": "5 CSVs",
        "rows": "~68K each",
        "cols": 8,
        "color": "green",
        "fields": ["GTVID", "Latitude", "Longitude", "nearest_site_distance_mi", "distance_to_interstate_mi"],
        "sample": [["5fa964f8...", "33.4484", "-112.074", "0.82 mi", "1.34 mi"], ["5fa965a5...", "40.7128", "-74.006", "0.15 mi", "3.21 mi"]],
        "notes": "Five CSV files: <strong>nearest_site_distances.csv</strong>, <strong>site_interstate_distances.csv</strong>, <strong>site_walmart_distances.csv</strong>, <strong>site_mcdonalds_distances.csv</strong>, <strong>site_target_distances.csv</strong>. Each measures distance from every site to the nearest instance of that feature. Computed via haversine formula from raw lat/lon coordinates.",
    },
    {
        "id": "retailer_geo",
        "icon": "\ud83c\udfea",
        "name": "Retailer Location Data",
        "desc": "McDonald\u2019s, Walmart, Target store locations (reference data)",
        "source": "3rd-party geodata",
        "format": "CSV",
        "rows": "13.5K / 9.7K / 1.9K",
        "cols": 27,
        "color": "orange",
        "fields": ["dba", "store_number", "address", "city", "state", "zip_code", "latitude", "longitude"],
        "sample": [["McDonald's", "12345", "123 Main St", "Phoenix", "AZ", "85001", "33.448", "-112.074"], ["Walmart", "4521", "456 Oak Ave", "Houston", "TX", "77001", "29.760", "-95.370"]],
        "notes": "Three CSVs: <strong>mcdonalds_geodata.csv</strong> (13,589), <strong>walmart_geodata.csv</strong> (9,784), <strong>target_geo_data.csv</strong> (1,982). Not used directly as model features \u2014 they\u2019re reference points for computing the proximity distances above. Proximity to high-traffic retailers is a strong signal for advertising revenue potential.",
    },
]


def load_site_scores(data_path: Path) -> pl.DataFrame:
    """Load the main Site Scores CSV file.

    @glossary: cleaning/null-handling
    @title: Null Value Handling
    @step: 0
    @color: accent
    @sub: Empty strings, "NA", "null", and "Unknown" treated as missing on load
    @analogy: Before analyzing any data, we declare what "missing" looks like. The source
        data uses four different representations of emptiness. Standardizing them upfront
        prevents the model from treating the word "Unknown" as meaningful data.
    @detail[Schema inference]: Schema inference uses the first 10,000 rows to determine
        column types. This catches most type issues early, but edge cases (like a column
        that is numeric for 10K rows then has a string) are handled by Polars strict typing.
    """
    print("Loading Site Scores data...")
    df = pl.read_csv(
        data_path / "site_scores_revenue_and_diagnostics.csv",
        infer_schema_length=10000,
        null_values=["", "NA", "null", "Unknown"]
    )
    print(f"  Loaded {len(df):,} monthly records for {df['id_gbase'].n_unique():,} unique sites")
    return df


def load_auxiliary_data(data_path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load nearest site, interstate, Kroger, McDonald's, Walmart, and Target distance files."""
    print("Loading auxiliary geospatial data...")

    # Nearest site distances
    nearest = pl.read_csv(data_path / "nearest_site_distances.csv")
    print(f"  Nearest sites: {len(nearest):,} records")

    # Interstate distances (some sites have multiple - take minimum distance)
    interstate = pl.read_csv(data_path / "site_interstate_distances.csv")
    interstate_agg = interstate.group_by('GTVID').agg([
        pl.col('distance_to_interstate_mi').min().alias('min_distance_to_interstate_mi'),
        pl.col('nearest_interstate').first().alias('nearest_interstate')
    ])
    print(f"  Interstate distances: {len(interstate_agg):,} unique sites")

    # Kroger distances (pre-aggregated - one row per site)
    kroger = pl.read_csv(data_path / "site_kroger_distances.csv")
    print(f"  Kroger distances: {len(kroger):,} records")

    # McDonald's distances (pre-aggregated - one row per site)
    mcdonalds = pl.read_csv(data_path / "site_mcdonalds_distances.csv")
    print(f"  McDonald's distances: {len(mcdonalds):,} records")

    # Walmart distances (computed via haversine from raw geodata)
    walmart = pl.read_csv(data_path / "site_walmart_distances.csv")
    print(f"  Walmart distances: {len(walmart):,} records")

    # Target distances (computed via haversine from raw geodata)
    target = pl.read_csv(data_path / "site_target_distances.csv")
    print(f"  Target distances: {len(target):,} records")

    return nearest, interstate_agg, kroger, mcdonalds, walmart, target


def load_daily_transactions(data_path: Path) -> pl.DataFrame:
    """Load daily transaction counts per site."""
    print("Loading daily transactions data...")
    df = pl.read_csv(
        data_path / "site_transactions_daily.csv",
        infer_schema_length=10000,
    )
    df = df.rename({
        "ID - Gbase": "id_gbase",
        "Date": "date",
        "Daily Transactions": "daily_transactions",
        "GTVID": "gtvid",
    })
    raw_count = len(df)
    # Deduplicate: some sites have exact duplicate rows per day
    df = df.group_by(["id_gbase", "date"]).agg(
        pl.col("daily_transactions").first(),
        pl.col("gtvid").first(),
    )
    deduped = raw_count - len(df)
    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"  Loaded {raw_count:,} rows, deduped {deduped:,} → {len(df):,} rows, {df['id_gbase'].n_unique():,} sites, dates {date_min} to {date_max}")
    return df


def load_daily_status(data_path: Path) -> pl.DataFrame:
    """Load daily status snapshots per site."""
    print("Loading daily status data...")
    df = pl.read_csv(
        data_path / "site_status_daily.csv",
        infer_schema_length=10000,
    )
    df = df.rename({
        "Date": "date",
        "Status": "status",
        "GTVID": "gtvid",
        "ID - Gbase": "id_gbase",
    })
    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"  Loaded {len(df):,} rows, {df['id_gbase'].n_unique():,} sites, dates {date_min} to {date_max}")
    return df


def load_active_days(data_path: Path) -> pl.DataFrame:
    """Load active days per month per site for daily revenue calculation."""
    print("Loading active days per month...")
    df = pl.read_csv(data_path / "active_days_per_month.csv", infer_schema_length=10000)
    df = df.rename({
        "ID - Gbase": "id_gbase",
        "Month": "date",
        "Active Days": "active_days",
    })
    df = df.select(["id_gbase", "date", "active_days"])
    # Deduplicate: source has per-GTVID rows alongside site-level rows for
    # the same (id_gbase, date). Take max to get the site-level active days.
    before = len(df)
    df = df.group_by(["id_gbase", "date"]).agg(pl.col("active_days").max())
    print(f"  Loaded {before:,} rows, deduped to {len(df):,}, {df['id_gbase'].n_unique():,} sites")
    return df


def build_monthly_site_activity(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pl.DataFrame:
    """
    Merge daily transactions and status data, then aggregate to monthly site activity.

    Produces one row per site per month with:
    - active_days: days with Status == "Active"
    - transaction_days: days with Daily Transactions > 0
    - total_days_in_data: total day-rows (coverage indicator)

    Output: data/processed/site_monthly_activity.parquet
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Load both daily sources
    txn = load_daily_transactions(input_path)
    status = load_daily_status(input_path)

    # Parse dates and derive year_month
    txn = txn.with_columns(pl.col("date").str.to_date().alias("date_parsed"))
    status = status.with_columns(pl.col("date").str.to_date().alias("date_parsed"))

    # Rename gtvid to avoid collision on join
    txn = txn.rename({"gtvid": "gtvid_txn"})
    status = status.rename({"gtvid": "gtvid_status"})

    # Full outer join on (id_gbase, date_parsed) — keeps all rows from both sources
    merged = txn.join(
        status,
        on=["id_gbase", "date_parsed"],
        how="full",
        coalesce=True,
    )

    # Coalesce gtvid from txn side first, then status
    merged = merged.with_columns(
        pl.coalesce(["gtvid_txn", "gtvid_status"]).alias("gtvid"),
        pl.col("date_parsed").dt.strftime("%Y-%m").alias("year_month"),
    )

    # Compute boolean flags (null status ≠ Active, null transactions ≠ >0)
    merged = merged.with_columns(
        (pl.col("status") == "Active").fill_null(False).alias("is_active"),
        (pl.col("daily_transactions") > 0).fill_null(False).alias("has_transactions"),
    )

    print(f"  Merged: {len(merged):,} day-rows across {merged['id_gbase'].n_unique():,} sites")

    # Group by (id_gbase, year_month) → monthly summary
    monthly = (
        merged
        .group_by(["id_gbase", "year_month"])
        .agg(
            pl.col("gtvid").first().alias("gtvid"),
            pl.col("is_active").sum().cast(pl.UInt16).alias("active_days"),
            pl.col("has_transactions").sum().cast(pl.UInt16).alias("transaction_days"),
            pl.len().cast(pl.UInt16).alias("total_days_in_data"),
        )
        .sort(["id_gbase", "year_month"])
    )

    print(f"  Monthly activity: {len(monthly):,} site-months")
    print(f"  Columns: {monthly.columns}")

    # Save
    out_file = output_path / "site_monthly_activity.parquet"
    monthly.write_parquet(out_file)
    print(f"  Saved to: {out_file}")

    return monthly


def calculate_relative_strength(
    df: pl.DataFrame,
    metric_col: str,
    output_col: str,
    short_days: int = 30,
    long_days: int = 90,
    as_of_date: Optional[date] = None,
    min_observations: int = 3,
) -> pl.DataFrame:
    """
    Calculate relative strength for any metric.

    This is a momentum indicator comparing recent performance to historical average.
    - Value > 1.0: Recent performance above historical (trending up)
    - Value < 1.0: Recent performance below historical (trending down)
    - Value = 1.0: Stable performance

    Args:
        df: DataFrame with time-series data (must have 'date', 'id_gbase', and metric_col)
        metric_col: Column name for the metric to calculate RS for
        output_col: Name for the output relative strength column
        short_days: Short-term window in days (default 30)
        long_days: Long-term window in days (default 90)
        as_of_date: Reference date for calculations (default: max date in data).
                    Use this in production to prevent look-ahead bias.
        min_observations: Minimum observations required in each window to produce
                          a valid RS value (default 3). Sites with fewer get RS=1.0.

    Returns:
        DataFrame with output_col added per site
    """
    if metric_col not in df.columns:
        print(f"  Warning: {metric_col} not found, skipping {output_col}")
        return pl.DataFrame({'id_gbase': [], output_col: []})

    # Ensure date is parsed properly
    if 'date_parsed' not in df.columns:
        df = df.with_columns(pl.col('date').str.to_date().alias('date_parsed'))

    # Use provided as_of_date or fall back to max date in dataset
    if as_of_date is not None:
        # Filter to only data available at as_of_date (prevents look-ahead bias)
        df = df.filter(pl.col('date_parsed') <= as_of_date)
        max_date = as_of_date
    else:
        max_date = df['date_parsed'].max()

    # Calculate cutoff dates
    short_cutoff = max_date - pl.duration(days=short_days)
    long_cutoff = max_date - pl.duration(days=long_days)

    # Calculate short-term average per site with observation count
    short_term = (
        df.filter(pl.col('date_parsed') >= short_cutoff)
        .group_by('id_gbase')
        .agg([
            pl.col(metric_col).mean().alias('short_avg'),
            pl.col(metric_col).count().alias('short_count'),
        ])
    )

    # Calculate long-term average per site with observation count
    long_term = (
        df.filter(pl.col('date_parsed') >= long_cutoff)
        .group_by('id_gbase')
        .agg([
            pl.col(metric_col).mean().alias('long_avg'),
            pl.col(metric_col).count().alias('long_count'),
        ])
    )

    # Join and calculate ratio
    # Use small epsilon (not 1.0 which biases small values significantly)
    epsilon = 1e-6
    rs_df = short_term.join(long_term, on='id_gbase', how='left').with_columns([
        # Only calculate RS if both windows have enough observations
        pl.when(
            (pl.col('short_count') >= min_observations) &
            (pl.col('long_count') >= min_observations)
        )
        .then((pl.col('short_avg') + epsilon) / (pl.col('long_avg') + epsilon))
        .otherwise(None)
        .alias(output_col)
    ]).select(['id_gbase', output_col])

    # Fill missing values with 1.0 (neutral - no trend data available)
    rs_df = rs_df.with_columns(
        pl.col(output_col).fill_null(1.0).fill_nan(1.0)
    )

    return rs_df


def calculate_all_relative_strength_features(
    df: pl.DataFrame,
    horizons: Optional[List[Tuple[int, int]]] = None,
    as_of_date: Optional[date] = None,
    min_observations: int = 3,
) -> pl.DataFrame:
    """
    Calculate all relative strength features for site metrics.

    Creates momentum indicators for multiple metrics across multiple time horizons.
    Multi-horizon RS helps capture both short-term momentum and longer-term trends,
    reducing overfitting compared to a single horizon.

    Default horizons (when horizons=None):
    - 30/90 days: Medium-term momentum (monthly vs quarterly) - backward compatible

    Recommended multi-horizon configuration:
    - 7/21 days: Short-term momentum (weekly vs 3-week)
    - 30/90 days: Medium-term momentum (monthly vs quarterly)
    - 90/180 days: Long-term momentum (quarterly vs semi-annual)

    Metrics tracked:
    - rs_Impressions: Impression trend (monthly_impressions)
    - rs_NVIs: Network video impression trend (monthly_nvis)
    - rs_Revenue: Revenue trend
    - rs_RevenuePerScreen: Per-screen revenue trend

    Args:
        df: DataFrame with monthly data (must have 'date', 'id_gbase', and metric columns)
        horizons: List of (short_days, long_days) tuples. If None, uses [(30, 90)].
                  When multiple horizons specified, column names include suffix
                  like rs_Revenue_7_21, rs_Revenue_30_90, etc.
        as_of_date: Reference date for calculations (prevents look-ahead bias in production)
        min_observations: Minimum observations required per window (default 3)

    Returns:
        DataFrame with all RS features per site (id_gbase as key)

    @glossary: combining/rs-features
    @title: Relative Strength Feature Join
    @step: 2
    @color: pink
    @sub: 12 momentum features computed from monthly time-series, joined back onto site rows
    @analogy: Think of momentum indicators in stock trading \u2014 is this site trending up
        or down? We compare recent performance (3 months) against longer windows (6, 12,
        24 months) to capture short, medium, and long-term trends.
    @why: RS > 1.0 means trending up (recent average beats historical). RS < 1.0 means
        trending down. RS = 1.0 is neutral (or insufficient data). Sites need at least
        2 months in each window to get a real RS value.
    @detail[Three time horizons]: Short-term: 95/185 days (3/6 months). Medium-term:
        185/370 days (6/12 months). Long-term: 370/740 days (12/24 months). Each horizon
        produces 4 features (Impressions, NVIs, Revenue, RevenuePerScreen) = 12 RS
        columns total.
    """
    # Default to single horizon for backward compatibility
    if horizons is None:
        horizons = [(30, 90)]

    multi_horizon = len(horizons) > 1
    print(f"Calculating relative strength features for {len(horizons)} horizon(s)...")

    # Parse date once for all calculations
    df = df.with_columns(pl.col('date').str.to_date().alias('date_parsed'))

    # Define metrics to calculate RS for
    rs_metrics = [
        ('monthly_impressions', 'Impressions'),
        ('monthly_nvis', 'NVIs'),
        ('revenue', 'Revenue'),
        ('monthly_revenue_per_screen', 'RevenuePerScreen'),
    ]

    # Calculate RS for each metric and horizon combination
    all_rs_dfs = []

    for short_days, long_days in horizons:
        # Add suffix for multi-horizon mode (e.g., _7_21, _30_90)
        horizon_suffix = f"_{short_days}_{long_days}" if multi_horizon else ""
        print(f"  Horizon: {short_days}/{long_days} days")

        for metric_col, metric_name in rs_metrics:
            output_col = f"rs_{metric_name}{horizon_suffix}"
            rs_df = calculate_relative_strength(
                df, metric_col, output_col,
                short_days=short_days,
                long_days=long_days,
                as_of_date=as_of_date,
                min_observations=min_observations,
            )
            if len(rs_df) > 0:
                all_rs_dfs.append(rs_df)
                mean_val = rs_df[output_col].mean()
                print(f"    {output_col}: {len(rs_df):,} sites (mean={mean_val:.3f})")

    # Join all RS features together
    if not all_rs_dfs:
        return pl.DataFrame({'id_gbase': []})

    result = all_rs_dfs[0]
    for rs_df in all_rs_dfs[1:]:
        # Use coalesce join to handle outer join properly
        result = result.join(rs_df, on='id_gbase', how='full', coalesce=True)

    # Fill any nulls with 1.0 (neutral)
    rs_cols = [col for col in result.columns if col.startswith('rs_')]
    for col in rs_cols:
        result = result.with_columns(pl.col(col).fill_null(1.0))

    print(f"  Total: {len(result):,} sites with {len(rs_cols)} RS features")
    return result


def aggregate_site_metrics(df: pl.DataFrame, active_days_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    Aggregate monthly records to one row per site with:
    - Total revenue and auction metrics
    - Average per active month
    - Site metadata (most recent values)
    - rs_Impressions (relative strength indicator)
    - avg_daily_revenue from actual active days (when active_days_df provided)

    @glossary: cleaning/aggregation
    @title: Monthly to Site-Level Aggregation
    @step: 1
    @color: cyan
    @sub: 1.4M monthly records consolidated into one row per unique site
    @analogy: You have 47 monthly report cards per station. This step staples them into
        one summary card with totals, averages, and the most recent status.
    @why: Different column types require different aggregation strategies: revenue metrics
        are summed then averaged, site metadata takes the most recent value, and temporal
        columns count distinct months.
    @detail[Relative strength features]: Also computes relative strength features \u2014
        momentum indicators that compare recent performance (3-month) to longer windows
        (6, 12, 24 months). These detect whether a site is trending up or down. Missing
        values default to 1.0 (neutral trend).

    @glossary: combining/temporal-aggregation
    @title: Temporal Aggregation (Monthly to Site-Level)
    @step: 1
    @color: cyan
    @sub: 1.4M monthly records collapsed into one row per site via group_by on id_gbase
    @analogy: This is really a vertical combine \u2014 collapsing multiple time-series rows
        into one summary row. We do not lose temporal signal because relative strength
        features capture momentum trends from the time-series data.
    @detail[Aggregation strategies]: Revenue and traffic metrics use sum totals plus
        average per month. Site metadata uses the last value (most recent month). Temporal
        columns count distinct months and track date ranges.
    """
    print("Aggregating site metrics...")

    # Define aggregation columns
    revenue_metrics = ['revenue', 'monthly_impressions', 'monthly_nvis',
                       'monthly_impressions_per_screen', 'monthly_nvis_per_screen',
                       'monthly_revenue_per_screen']

    # Build aggregation expressions
    agg_exprs = [
        # Count of active months (rows present in dataset)
        pl.len().alias('active_months'),

        # Date range
        pl.col('date').min().alias('first_month'),
        pl.col('date').max().alias('last_month'),

        # Most recent site metadata (use last value when sorted by date)
        pl.col('gtvid').last().alias('gtvid'),
        pl.col('site_activated_date').last().alias('site_activated_date'),
        pl.col('network').last().alias('network'),
        pl.col('state').last().alias('state'),
        pl.col('county').last().alias('county'),
        pl.col('latitude').last().alias('latitude'),
        pl.col('longitude').last().alias('longitude'),
        pl.col('zip').last().alias('zip'),
        pl.col('dma').last().alias('dma'),
        pl.col('dma_rank').last().alias('dma_rank'),
        pl.col('statuis').last().alias('status'),
        pl.col('program').last().alias('program'),
        pl.col('experience_type').last().alias('experience_type'),
        pl.col('hardware_type').last().alias('hardware_type'),
        pl.col('retailer').last().alias('retailer'),
        pl.col('screen_count').last().alias('screen_count'),

        # Demographics (last available)
        pl.col('avg_household_income').last().alias('avg_household_income'),
        pl.col('median_age').last().alias('median_age'),
        pl.col('pct_african_american').last().alias('pct_african_american'),
        pl.col('pct_asian').last().alias('pct_asian'),
        pl.col('pct_female').last().alias('pct_female'),
        pl.col('pct_male').last().alias('pct_male'),
        pl.col('pct_hispanic').last().alias('pct_hispanic'),

        # Brands
        pl.col('brand_fuel').last().alias('brand_fuel'),
        pl.col('brand_restaurant').last().alias('brand_restaurant'),
        pl.col('brand_c_store').last().alias('brand_c_store'),

        # Capability flags (last values)
        pl.col('schedule_site').last().alias('schedule_site'),
        pl.col('sellable_site').last().alias('sellable_site'),
        pl.col('c_emv_enabled').last().alias('c_emv_enabled'),
        pl.col('c_nfc_enabled').last().alias('c_nfc_enabled'),
        pl.col('c_open_24_hours').last().alias('c_open_24_hours'),
        pl.col('c_sells_beer').last().alias('c_sells_beer'),
        pl.col('c_sells_diesel_fuel').last().alias('c_sells_diesel_fuel'),
        pl.col('c_sells_lottery').last().alias('c_sells_lottery'),
        pl.col('c_vistar_programmatic_enabled').last().alias('c_vistar_programmatic_enabled'),
        pl.col('c_walk_up_enabled').last().alias('c_walk_up_enabled'),
        pl.col('c_sells_wine').last().alias('c_sells_wine'),

        # Restriction flags (last values)
        pl.col('r_lottery').last().alias('r_lottery'),
        pl.col('r_government').last().alias('r_government'),
        pl.col('r_travel_and_tourism').last().alias('r_travel_and_tourism'),
        pl.col('r_retail_car_wash').last().alias('r_retail_car_wash'),
        pl.col('r_cpg_beverage_beer_oof').last().alias('r_cpg_beverage_beer_oof'),
        pl.col('r_cpg_beverage_beer_vide').last().alias('r_cpg_beverage_beer_vide'),
        pl.col('r_cpg_beverage_wine_oof').last().alias('r_cpg_beverage_wine_oof'),
        pl.col('r_cpg_beverage_wine_video').last().alias('r_cpg_beverage_wine_video'),
        pl.col('r_finance_credit_cards').last().alias('r_finance_credit_cards'),
        pl.col('r_cpg_cbd_hemp_ingestibles_non_thc').last().alias('r_cpg_cbd_hemp_ingestibles_non_thc'),
        pl.col('r_cpg_non_food_beverage_cannabis_medical').last().alias('r_cpg_non_food_beverage_cannabis_medical'),
        pl.col('r_cpg_non_food_beverage_cannabis_recreational').last().alias('r_cpg_non_food_beverage_cannabis_recreational'),
        pl.col('r_cpg_non_food_beverage_cbd_hemp_non_thc').last().alias('r_cpg_non_food_beverage_cbd_hemp_non_thc'),
        pl.col('r_alcohol_drink_responsibly_message').last().alias('r_alcohol_drink_responsibly_message'),
        pl.col('r_alternative_transportation').last().alias('r_alternative_transportation'),
        pl.col('r_associations_and_npo_anti_smoking').last().alias('r_associations_and_npo_anti_smoking'),
        pl.col('r_automotive_after_market_oil').last().alias('r_automotive_after_market_oil'),
        pl.col('r_cpg_beverage_spirits_ooh').last().alias('r_cpg_beverage_spirits_ooh'),
        pl.col('r_cpg_beverage_spirits_video').last().alias('r_cpg_beverage_spirits_video'),
        pl.col('r_cpg_non_food_beverage_e_cigarette').last().alias('r_cpg_non_food_beverage_e_cigarette'),
        pl.col('r_entertainment_casinos_and_gambling').last().alias('r_entertainment_casinos_and_gambling'),
        pl.col('r_government_political').last().alias('r_government_political'),
        pl.col('r_automotive_electric').last().alias('r_automotive_electric'),
        pl.col('r_recruitment').last().alias('r_recruitment'),
        pl.col('r_restaurants_cdr').last().alias('r_restaurants_cdr'),
        pl.col('r_restaurants_qsr').last().alias('r_restaurants_qsr'),
        pl.col('r_retail_automotive_service').last().alias('r_retail_automotive_service'),
        pl.col('r_retail_grocery').last().alias('r_retail_grocery'),
        pl.col('r_retail_grocerty_with_fuel').last().alias('r_retail_grocerty_with_fuel'),
    ]

    # Add revenue/auction metric totals
    for col in revenue_metrics:
        if col in df.columns:
            agg_exprs.append(pl.col(col).sum().alias(f'total_{col}'))

    # Join active days for per-month daily revenue calculation
    if active_days_df is not None:
        df = df.join(
            active_days_df,
            on=['id_gbase', 'date'],
            how='left'
        )
        # Per-month daily revenue: 0 when active_days is 0, null only when unmatched
        df = df.with_columns(
            pl.when(pl.col('active_days').is_null())
            .then(None)
            .when(pl.col('active_days') == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col('revenue') / pl.col('active_days'))
            .alias('daily_revenue')
        )
    else:
        df = df.with_columns(
            pl.lit(None).cast(pl.Float64).alias('daily_revenue'),
            pl.lit(None).cast(pl.Int64).alias('active_days'),
        )

    # Daily revenue aggregation (from active_days join)
    agg_exprs.append(pl.col('daily_revenue').mean().alias('avg_daily_revenue'))
    agg_exprs.append(pl.col('active_days').sum().alias('total_active_days'))

    # Aggregate by site
    df_sorted = df.sort(['id_gbase', 'date'])
    site_agg = df_sorted.group_by('id_gbase').agg(agg_exprs)

    # Calculate averages per active month
    for col in revenue_metrics:
        total_col = f'total_{col}'
        if total_col in site_agg.columns:
            site_agg = site_agg.with_columns(
                (pl.col(total_col) / pl.col('active_months')).alias(f'avg_monthly_{col}')
            )

    # Fallback for sites without active_days data: use crude estimate
    site_agg = site_agg.with_columns(
        pl.when(pl.col('avg_daily_revenue').is_null())
        .then(pl.col('total_revenue') / (pl.col('active_months') * 30))
        .otherwise(pl.col('avg_daily_revenue'))
        .alias('avg_daily_revenue')
    )

    # Calculate and join all relative strength features (momentum indicators)
    # Multi-horizon RS: captures short, medium, and long-term momentum
    # NOTE: Data is MONTHLY (1 record per site per month on the 1st), so windows must be
    # calibrated for monthly data points. Using days that span month boundaries:
    # - 3/6 months (95/185 days): Recent quarter vs half-year (short-term)
    # - 6/12 months (185/370 days): Half-year vs year (medium-term)
    # - 12/24 months (370/740 days): Year vs 2-year (long-term trend)
    if 'date' in df.columns:
        rs_df = calculate_all_relative_strength_features(
            df,
            horizons=[(95, 185), (185, 370), (370, 740)],
            min_observations=2,  # Require at least 2 months of data in each window
        )
        if len(rs_df) > 0:
            site_agg = site_agg.join(rs_df, on='id_gbase', how='left')
            # Fill any missing RS values with 1.0 (neutral)
            rs_cols = [col for col in site_agg.columns if col.startswith('rs_')]
            for col in rs_cols:
                site_agg = site_agg.with_columns(pl.col(col).fill_null(1.0))

    print(f"  Aggregated to {len(site_agg):,} sites")
    return site_agg


def join_geospatial_features(
    site_df: pl.DataFrame,
    nearest_df: pl.DataFrame,
    interstate_df: pl.DataFrame,
    kroger_df: pl.DataFrame,
    mcdonalds_df: pl.DataFrame,
    walmart_df: pl.DataFrame,
    target_df: pl.DataFrame
) -> pl.DataFrame:
    """Join auxiliary geospatial data to site dataset.

    @glossary: cleaning/geo-joins
    @title: Geospatial Feature Joining
    @step: 2
    @color: green
    @sub: 6 distance files left-joined onto site data via GTVID
    @why: All 6 are left joins \u2014 the site table drives the result, so no sites are
        lost even if a distance file is missing that GTVID. The join key is gtvid (left)
        to GTVID (right) with case difference between the aggregated data and the
        distance files.
    @detail[Pre-aggregation]: Interstate distances are pre-aggregated before joining:
        some sites are near multiple interstates, so the pipeline takes the minimum
        distance per site via group_by(GTVID).agg(min(distance)). Walmart and Target
        distances are computed from raw geodata using the haversine formula in 10K-site
        chunks to control memory.

    @glossary: combining/geo-joins
    @title: Geospatial Feature Joins (6 Left Joins)
    @step: 3
    @color: green
    @sub: 6 distance CSVs joined sequentially via GTVID \u2014 nearest site, interstate,
        Kroger, McDonald's, Walmart, Target
    @why: Only selected columns are brought in to avoid polluting the dataset with
        duplicate lat/lon columns. The join key uses gtvid (left) to GTVID (right) to
        handle case differences between systems.
    """
    print("Joining geospatial features...")

    # Join nearest site distances (rename to consistent naming convention)
    site_df = site_df.join(
        nearest_df.select([
            'GTVID',
            'nearest_site',
            pl.col('nearest_site_distance_mi').alias('min_distance_to_nearest_site_mi')
        ]),
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    # Join interstate distances
    site_df = site_df.join(
        interstate_df,
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    # Join Kroger distances
    site_df = site_df.join(
        kroger_df.select(['GTVID', 'min_distance_to_kroger_mi']),
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    # Join McDonald's distances
    site_df = site_df.join(
        mcdonalds_df.select(['GTVID', 'min_distance_to_mcdonalds_mi']),
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    # Join Walmart distances
    site_df = site_df.join(
        walmart_df.select(['GTVID', 'min_distance_to_walmart_mi']),
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    # Join Target distances
    site_df = site_df.join(
        target_df.select(['GTVID', 'min_distance_to_target_mi']),
        left_on='gtvid',
        right_on='GTVID',
        how='left'
    )

    matched = site_df.filter(pl.col('nearest_site').is_not_null()).shape[0]
    kroger_matched = site_df.filter(pl.col('min_distance_to_kroger_mi').is_not_null()).shape[0]
    mcdonalds_matched = site_df.filter(pl.col('min_distance_to_mcdonalds_mi').is_not_null()).shape[0]
    walmart_matched = site_df.filter(pl.col('min_distance_to_walmart_mi').is_not_null()).shape[0]
    target_matched = site_df.filter(pl.col('min_distance_to_target_mi').is_not_null()).shape[0]
    print(f"  Matched {matched:,} sites with nearest site features")
    print(f"  Matched {kroger_matched:,} sites with Kroger distances")
    print(f"  Matched {mcdonalds_matched:,} sites with McDonald's distances")
    print(f"  Matched {walmart_matched:,} sites with Walmart distances")
    print(f"  Matched {target_matched:,} sites with Target distances")

    return site_df


def add_log_transformations(df: pl.DataFrame) -> pl.DataFrame:
    """Add log transformations for numeric metrics.

    @glossary: cleaning/log-transforms
    @title: Log Transformations
    @step: 3
    @color: orange
    @sub: Right-skewed revenue and distance features transformed to normalize distributions
    @analogy: Revenue and distances are like earthquake magnitudes \u2014 the difference
        between $100 and $1,000 matters more than between $100,000 and $101,000. Log
        compression makes the model treat differences proportionally.
    @detail[Why sign-preserving?]: The formula uses sign(x) \u00d7 log(1+|x|) for signed
        types and log(x+1) for unsigned integers. This handles rare negative revenue
        records without losing sign information. The +1 ensures log(0) does not produce
        negative infinity.
    """
    print("Adding log transformations...")

    numeric_cols = [
        # Total metrics only
        'total_revenue', 'total_monthly_impressions', 'total_monthly_nvis',
        'total_monthly_impressions_per_screen', 'total_monthly_nvis_per_screen',
        'total_monthly_revenue_per_screen',
        # Daily averages
        'avg_daily_revenue',
        # Geospatial distances (all use min_distance_to_X_mi naming convention)
        'min_distance_to_nearest_site_mi', 'min_distance_to_interstate_mi',
        'min_distance_to_kroger_mi', 'min_distance_to_mcdonalds_mi',
        'min_distance_to_walmart_mi', 'min_distance_to_target_mi',
        # Demographics
        'avg_household_income',
    ]

    for col in numeric_cols:
        if col in df.columns:
            col_dtype = df[col].dtype

            # For unsigned integers or columns that can't be negative, use simple log1p
            if col_dtype in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                df = df.with_columns(
                    (pl.col(col).cast(pl.Float64) + 1).log().alias(f'log_{col}')
                )
            else:
                # For signed types, use sign-preserving log: sign(x) * log(1 + |x|)
                # Cast to float first to handle all numeric types
                df = df.with_columns(
                    pl.when(pl.col(col).cast(pl.Float64) >= 0)
                    .then((pl.col(col).cast(pl.Float64) + 1).log())
                    .otherwise(-(-pl.col(col).cast(pl.Float64) + 1).log())
                    .alias(f'log_{col}')
                )

    log_cols = [c for c in df.columns if c.startswith('log_')]
    print(f"  Added {len(log_cols)} log-transformed columns")

    return df


def one_hot_encode_flags(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot encode capability (c_*) and restriction (r_*) flags.

    @glossary: cleaning/flag-encoding
    @title: Flag Encoding
    @step: 4
    @color: pink
    @sub: Capability and restriction flags converted from mixed strings to numeric 0/1
    @analogy: The source data is inconsistent \u2014 some flags say "Yes"/"No", others say
        "true"/"false", others use actual booleans. This step normalizes all 51 flags so
        the model sees clean 0/1 values.
    @why: Original string columns are dropped after encoding. ~40 encoded flag columns
        remain in the final dataset, with suffix _encoded.
    @detail[Encoding strategy]: Boolean values cast to Int8 (0/1). Yes/No/Unknown strings
        map to 1/0/null. Ad eligibility true/false strings are lowercased and cast.
        Multi-category flags get one-hot per category.
    """
    print("One-hot encoding capability and restriction flags...")

    # Identify flag columns
    capability_cols = [c for c in df.columns if c.startswith('c_')]
    restriction_cols = [c for c in df.columns if c.startswith('r_')]
    other_flags = ['schedule_site', 'sellable_site']

    all_flag_cols = capability_cols + restriction_cols + other_flags
    encoded_count = 0

    for col in all_flag_cols:
        if col not in df.columns:
            continue

        col_dtype = df[col].dtype

        if col_dtype == pl.Boolean:
            # Already boolean, convert to int
            df = df.with_columns(
                pl.col(col).cast(pl.Int8).alias(f'{col}_encoded')
            )
            encoded_count += 1
        elif col_dtype == pl.Utf8:
            # String column - check unique values
            unique_vals = df[col].unique().to_list()

            # Handle Yes/No/Unknown type columns
            if set(unique_vals) <= {'Yes', 'No', 'Unknown', None}:
                df = df.with_columns(
                    pl.when(pl.col(col) == 'Yes').then(1)
                    .when(pl.col(col) == 'No').then(0)
                    .otherwise(None)  # Unknown -> null
                    .cast(pl.Int8)
                    .alias(f'{col}_encoded')
                )
                encoded_count += 1
            # Handle true/false string columns
            elif set(unique_vals) <= {'true', 'false', 'True', 'False', None}:
                df = df.with_columns(
                    pl.col(col).str.to_lowercase().eq('true').cast(pl.Int8).alias(f'{col}_encoded')
                )
                encoded_count += 1
            else:
                # Multi-category - do full one-hot encoding
                for val in unique_vals:
                    if val is not None:
                        safe_val = str(val).replace(' ', '_').replace('-', '_').lower()
                        df = df.with_columns(
                            (pl.col(col) == val).cast(pl.Int8).alias(f'{col}_{safe_val}')
                        )
                        encoded_count += 1

    # Drop original flag columns (keep encoded versions)
    cols_to_drop = [c for c in all_flag_cols if c in df.columns]
    df = df.drop(cols_to_drop)

    print(f"  Created {encoded_count} encoded flag columns, dropped {len(cols_to_drop)} original columns")

    return df


def bin_high_cardinality(
    df: pl.DataFrame,
    column: str,
    top_n: int = 30,
) -> pl.DataFrame:
    """
    Bin a high-cardinality categorical column by keeping the top N most
    frequent values and replacing all others with 'Other'.

    @glossary: cleaning/binning
    @title: High-Cardinality Binning
    @step: 5
    @color: yellow
    @sub: Retailer and brand names binned to top N + "Other" to prevent overfitting
    @analogy: If your model learns that "Joe's Gas #47" means high revenue, it is
        memorizing noise. By grouping rare retailer names into "Other", we force the
        model to learn patterns from major chains while treating one-off names as generic.
    @detail[Top N selection]: Top N is determined by frequency \u2014 the most common
        30 retailer names are preserved, and all rare retailers are grouped into a single
        "Other" category. brand_fuel uses top 10 since it has fewer unique values.
    """
    if column not in df.columns:
        return df

    value_counts = df[column].value_counts().sort('count', descending=True)
    top_values = value_counts[column].head(top_n).to_list()
    original_unique = df[column].n_unique()

    df = df.with_columns(
        pl.when(pl.col(column).is_in(top_values))
        .then(pl.col(column))
        .otherwise(pl.lit('Other'))
        .alias(column)
    )

    new_unique = df[column].n_unique()
    print(f"  Binned '{column}': {original_unique:,} → {new_unique} categories (top {top_n} + Other)")

    return df


def prepare_training_dataset(
    df: pl.DataFrame,
    active_only: bool = True,
    drop_geo_ids: bool = True
) -> pl.DataFrame:
    """
    Prepare the final training dataset.

    Args:
        df: Pre-cleaned dataset
        active_only: If True, filter to only Active status sites
        drop_geo_ids: If True, drop state, county, dma, zip columns

    @glossary: cleaning/training-filter
    @title: Training Set Filtering
    @step: 6
    @color: red
    @sub: All sites filtered down to active-only, ML-ready rows
    @analogy: Not every gas station belongs in the training data. Deactivated sites would
        teach the model about failure, not success. Negative revenue records are data
        errors. Geographic IDs would let the model cheat by memorizing states instead of
        learning real patterns.
    @why: Geographic columns (state, county, DMA, zip) are dropped rather than filtered.
        If the model learned that "Texas" means high revenue, it would fail on new states.
        Instead, geographic signal comes through the proximity features (distance to
        Walmart, etc.) which generalize better.
    @detail[Three filters]: Active status only (removes ~31,500 deactivated sites).
        Remove negative revenue (removes ~2 data errors). Drop geographic identifiers
        (prevents model from memorizing state/county/zip).
    """
    print("Preparing training dataset...")

    # Filter to active sites only
    if active_only:
        original_count = len(df)
        df = df.filter(pl.col('status') == 'Active')
        print(f"  Filtered to Active sites: {len(df):,} (from {original_count:,})")

    # Remove negative revenue records (total AND daily)
    # total_revenue < 0: lifetime chargebacks exceed earnings
    # avg_daily_revenue < 0: extreme negative months amplified by few active days
    before_count = len(df)
    df = df.filter(
        (pl.col('total_revenue') >= 0) & (pl.col('avg_daily_revenue') >= 0)
    )
    removed = before_count - len(df)
    print(f"  Removed {removed:,} negative revenue records: {len(df):,} remaining")

    # Drop geographic identifier columns
    if drop_geo_ids:
        geo_cols_to_drop = ['state', 'county', 'dma', 'zip', 'zip_4']
        existing_geo_cols = [c for c in geo_cols_to_drop if c in df.columns]
        df = df.drop(existing_geo_cols)
        print(f"  Dropped geographic identifiers: {existing_geo_cols}")

    # Bin high-cardinality categorical features
    df = bin_high_cardinality(df, column='retailer', top_n=30)
    df = bin_high_cardinality(df, column='brand_c_store', top_n=30)
    df = bin_high_cardinality(df, column='brand_fuel', top_n=10)

    # Add log transformations
    df = add_log_transformations(df)

    # One-hot encode flags
    df = one_hot_encode_flags(df)

    return df


def calculate_dataset_totals(site_df: pl.DataFrame) -> dict:
    """Calculate total metrics across all sites for the entire dataset."""
    print("Calculating dataset totals...")

    total_cols = [c for c in site_df.columns if c.startswith('total_')]
    totals = {}

    for col in total_cols:
        metric_name = col.replace('total_', '')
        totals[f'dataset_total_{metric_name}'] = site_df[col].sum()

    # Also calculate dataset-wide averages
    num_sites = len(site_df)
    total_active_months = site_df['active_months'].sum()

    totals['total_sites'] = num_sites
    totals['total_active_site_months'] = total_active_months

    for col in total_cols:
        metric_name = col.replace('total_', '')
        totals[f'avg_per_active_month_{metric_name}'] = totals[f'dataset_total_{metric_name}'] / total_active_months

    return totals


def generate_summary_report(site_df: pl.DataFrame, totals: dict) -> str:
    """Generate a summary report of the pre-cleaned dataset."""

    report = []
    report.append("=" * 80)
    report.append("PRE-CLEANED DATASET SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append("\n" + "-" * 40)
    report.append("DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total unique sites: {len(site_df):,}")
    report.append(f"Total columns: {len(site_df.columns)}")
    report.append(f"Total active site-months: {totals['total_active_site_months']:,}")

    # Status distribution
    report.append("\n" + "-" * 40)
    report.append("SITE STATUS DISTRIBUTION")
    report.append("-" * 40)
    status_dist = site_df.group_by('status').agg(pl.len().alias('count')).sort('count', descending=True)
    for row in status_dist.iter_rows(named=True):
        pct = row['count'] / len(site_df) * 100
        report.append(f"  {row['status']}: {row['count']:,} ({pct:.1f}%)")

    # Active months distribution
    report.append("\n" + "-" * 40)
    report.append("ACTIVE MONTHS PER SITE")
    report.append("-" * 40)
    months_stats = site_df['active_months'].describe()
    report.append(f"  Min: {site_df['active_months'].min()}")
    report.append(f"  Max: {site_df['active_months'].max()}")
    report.append(f"  Mean: {site_df['active_months'].mean():.1f}")
    report.append(f"  Median: {site_df['active_months'].median():.1f}")

    # Revenue and auction totals
    report.append("\n" + "-" * 40)
    report.append("DATASET TOTALS (ALL SITES, ALL MONTHS)")
    report.append("-" * 40)
    for key, val in totals.items():
        if key.startswith('dataset_total_'):
            metric = key.replace('dataset_total_', '')
            if val is not None:
                report.append(f"  Total {metric}: {val:,.2f}")

    report.append("\n" + "-" * 40)
    report.append("AVERAGE PER ACTIVE SITE-MONTH")
    report.append("-" * 40)
    for key, val in totals.items():
        if key.startswith('avg_per_active_month_'):
            metric = key.replace('avg_per_active_month_', '')
            if val is not None:
                report.append(f"  Avg {metric}: {val:,.4f}")

    # Column categories
    report.append("\n" + "-" * 40)
    report.append("COLUMN CATEGORIES")
    report.append("-" * 40)

    total_cols = [c for c in site_df.columns if c.startswith('total_')]
    avg_monthly_cols = [c for c in site_df.columns if c.startswith('avg_monthly_')]
    metadata_cols = [c for c in site_df.columns if c not in total_cols + avg_monthly_cols]

    report.append(f"  Site metadata columns: {len(metadata_cols)}")
    report.append(f"  Total metric columns: {len(total_cols)}")
    report.append(f"  Avg monthly metric columns: {len(avg_monthly_cols)}")

    report.append("\n" + "-" * 40)
    report.append("GEOSPATIAL FEATURES")
    report.append("-" * 40)
    nearest_matched = site_df.filter(pl.col('nearest_site').is_not_null()).shape[0]
    interstate_matched = site_df.filter(pl.col('min_distance_to_interstate_mi').is_not_null()).shape[0]
    kroger_matched = site_df.filter(pl.col('min_distance_to_kroger_mi').is_not_null()).shape[0] if 'min_distance_to_kroger_mi' in site_df.columns else 0
    mcdonalds_matched = site_df.filter(pl.col('min_distance_to_mcdonalds_mi').is_not_null()).shape[0] if 'min_distance_to_mcdonalds_mi' in site_df.columns else 0
    report.append(f"  Sites with nearest site distance: {nearest_matched:,} ({nearest_matched/len(site_df)*100:.1f}%)")
    report.append(f"  Sites with interstate distance: {interstate_matched:,} ({interstate_matched/len(site_df)*100:.1f}%)")
    report.append(f"  Sites with Kroger distance: {kroger_matched:,} ({kroger_matched/len(site_df)*100:.1f}%)")
    report.append(f"  Sites with McDonald's distance: {mcdonalds_matched:,} ({mcdonalds_matched/len(site_df)*100:.1f}%)")

    # Sample rows
    report.append("\n" + "-" * 40)
    report.append("SAMPLE DATA (5 rows)")
    report.append("-" * 40)
    sample_cols = ['id_gbase', 'gtvid', 'status', 'active_months',
                   'total_revenue', 'avg_monthly_revenue', 'avg_daily_revenue',
                   'total_monthly_impressions', 'avg_monthly_monthly_impressions']
    available_cols = [c for c in sample_cols if c in site_df.columns]
    report.append(str(site_df.select(available_cols).head(5)))

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def generate_training_report(df: pl.DataFrame) -> str:
    """Generate a summary report of the training dataset."""

    report = []
    report.append("=" * 80)
    report.append("TRAINING DATASET SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append("\n" + "-" * 40)
    report.append("DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total sites (Active only): {len(df):,}")
    report.append(f"Total columns: {len(df.columns)}")

    # Column categories
    log_cols = [c for c in df.columns if c.startswith('log_')]
    encoded_cols = [c for c in df.columns if '_encoded' in c]
    total_cols = [c for c in df.columns if c.startswith('total_')]
    avg_cols = [c for c in df.columns if c.startswith('avg_')]

    report.append("\n" + "-" * 40)
    report.append("COLUMN BREAKDOWN")
    report.append("-" * 40)
    report.append(f"  Log-transformed columns: {len(log_cols)}")
    report.append(f"  One-hot encoded flags: {len(encoded_cols)}")
    report.append(f"  Total metric columns: {len(total_cols)}")
    report.append(f"  Average metric columns: {len(avg_cols)}")

    # Log columns detail
    report.append("\n" + "-" * 40)
    report.append("LOG-TRANSFORMED COLUMNS")
    report.append("-" * 40)
    for col in sorted(log_cols):
        stats = df[col]
        report.append(f"  {col}: min={stats.min():.4f}, max={stats.max():.4f}, mean={stats.mean():.4f}")

    # Encoded flags detail
    report.append("\n" + "-" * 40)
    report.append("ONE-HOT ENCODED FLAGS (% True)")
    report.append("-" * 40)
    for col in sorted(encoded_cols):
        if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            pct_true = df[col].sum() / len(df) * 100
            null_count = df[col].null_count()
            report.append(f"  {col}: {pct_true:.1f}% (nulls: {null_count})")

    # Sample data
    report.append("\n" + "-" * 40)
    report.append("SAMPLE DATA (3 rows, key columns)")
    report.append("-" * 40)
    key_cols = ['id_gbase', 'gtvid', 'active_months',
                'total_revenue', 'log_total_revenue',
                'avg_monthly_revenue', 'avg_daily_revenue', 'log_avg_daily_revenue']
    available = [c for c in key_cols if c in df.columns]
    report.append(str(df.select(available).head(3)))

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def transform_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH
) -> pl.DataFrame:
    """
    Main transformation function.

    Returns the pre-cleaned dataset for review before ML cleaning.

    @glossary: combining/output
    @title: Combined Output: Pre-Cleaned Dataset
    @step: 4
    @color: orange
    @sub: All sites with all features saved as site_aggregated_precleaned.parquet
    @why: Column count grows from 94 (raw CSV) to ~102 (precleaned) because aggregation
        adds computed columns (totals, averages, RS features) and joins add distance
        columns. It then jumps to ~111 in the training set because log transforms and
        encoded flags add more columns while geo IDs are dropped.
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all data
    site_scores = load_site_scores(input_path)
    nearest, interstate, kroger, mcdonalds, walmart, target = load_auxiliary_data(input_path)
    active_days = load_active_days(input_path)

    # Aggregate to one row per site
    site_agg = aggregate_site_metrics(site_scores, active_days_df=active_days)

    # Join geospatial features
    site_final = join_geospatial_features(site_agg, nearest, interstate, kroger, mcdonalds, walmart, target)

    # Calculate dataset totals
    totals = calculate_dataset_totals(site_final)

    # Generate summary report
    report = generate_summary_report(site_final, totals)
    print("\n" + report)

    # Save outputs
    output_file = output_path / "site_aggregated_precleaned.parquet"
    site_final.write_parquet(output_file)
    print(f"\nSaved pre-cleaned dataset to: {output_file}")

    # Save report
    report_file = output_path / "precleaned_summary.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Saved summary report to: {report_file}")

    # Also save as CSV for easy inspection
    csv_file = output_path / "site_aggregated_precleaned.csv"
    site_final.write_csv(csv_file)
    print(f"Saved CSV version to: {csv_file}")

    return site_final


def create_training_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH
) -> pl.DataFrame:
    """
    Create the final training dataset with:
    - Active sites only
    - Log transformations
    - One-hot encoded flags
    - Geographic identifiers dropped
    """
    # First run full transformation
    site_df = transform_data(input_path, output_path)

    print("\n" + "=" * 80)
    print("CREATING TRAINING DATASET")
    print("=" * 80)

    # Prepare training dataset
    train_df = prepare_training_dataset(
        site_df,
        active_only=True,
        drop_geo_ids=True
    )

    # Generate training report
    train_report = generate_training_report(train_df)
    print("\n" + train_report)

    # Save training dataset
    train_file = output_path / "site_training_data.parquet"
    train_df.write_parquet(train_file)
    print(f"\nSaved training dataset to: {train_file}")

    # Save training report
    train_report_file = output_path / "training_data_summary.txt"
    with open(train_report_file, 'w') as f:
        f.write(train_report)
    print(f"Saved training report to: {train_report_file}")

    # Save CSV version
    train_csv = output_path / "site_training_data.csv"
    train_df.write_csv(train_csv)
    print(f"Saved CSV version to: {train_csv}")

    return train_df


# Module-level cache for prediction data (~25MB)
_prediction_data_cache: Optional[pl.DataFrame] = None


def get_all_sites_for_prediction() -> pl.DataFrame:
    """
    Load all sites (all statuses) with derived features for batch prediction.

    Loads site_aggregated_precleaned.parquet (57K sites) and applies the same
    transforms as training (log transforms, one-hot encoding, binning) but
    without the active-only filter.

    Results are cached at module level — computed once, reused for all requests.

    @glossary: productionizing/inference-data-prep
    @title: Inference Data Preparation
    @step: 2
    @color: yellow
    @sub: Load precleaned parquet (all statuses) and apply the same training
        transforms without the active-only filter
    @analogy: Training learned the recipe from a curated subset of sites
        (Active, 12+ months of history). Inference needs to score every
        site in the network — including Deactivated, Awaiting Installation,
        and Cancelled ones — so we re-open the full precleaned parquet
        and run it through the exact same prep steps the model expects.
    @why: Calling prepare_training_dataset(active_only=False, drop_geo_ids=True)
        guarantees train/inference parity: the same log transforms, one-hot
        encoding, and binning that shaped the training feature space are
        applied to every site we score. The fitted preprocessor (scalers
        and label encoders, loaded by BatchPredictor) then takes over for
        the numeric/categorical/boolean transforms downstream. Skipping
        any of these steps would silently shift the input distribution
        and degrade predictions.
    @detail[Module-level cache]: A module-global _prediction_data_cache holds
        the prepared DataFrame after the first call. Subsequent requests
        return the cached frame in O(1) instead of re-reading the ~25MB
        parquet and recomputing derived features. The cache lives for the
        process lifetime — restarting the Flask app rebuilds it.
    """
    global _prediction_data_cache
    if _prediction_data_cache is not None:
        return _prediction_data_cache

    parquet_path = DEFAULT_OUTPUT_PATH / "site_aggregated_precleaned.parquet"
    print(f"Loading all sites for prediction from {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} sites (all statuses)")

    # Apply same transforms as training but keep all sites
    df = prepare_training_dataset(df, active_only=False, drop_geo_ids=True)
    print(f"  Prepared {len(df):,} sites with derived features")

    _prediction_data_cache = df
    return df


if __name__ == "__main__":
    # Create both pre-cleaned and training datasets
    train_df = create_training_dataset()
