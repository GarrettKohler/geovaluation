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
from datetime import datetime

# Project root - dynamically resolved from this file's location
# site_scoring/data_transform.py -> go up one level to get project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default paths (relative to project root)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"


def load_site_scores(data_path: Path) -> pl.DataFrame:
    """Load the main Site Scores CSV file."""
    print("Loading Site Scores data...")
    df = pl.read_csv(
        data_path / "Site Scores - Site Revenue, Impressions, and Diagnostics.csv",
        infer_schema_length=10000,
        null_values=["", "NA", "null", "Unknown"]
    )
    print(f"  Loaded {len(df):,} monthly records for {df['id_gbase'].n_unique():,} unique sites")
    return df


def load_auxiliary_data(data_path: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load nearest site, interstate, Kroger, and McDonald's distance files."""
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

    return nearest, interstate_agg, kroger, mcdonalds


def calculate_relative_strength(
    df: pl.DataFrame,
    metric_col: str,
    output_col: str,
    short_days: int = 30,
    long_days: int = 90
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

    Returns:
        DataFrame with output_col added per site
    """
    if metric_col not in df.columns:
        print(f"  Warning: {metric_col} not found, skipping {output_col}")
        return pl.DataFrame({'id_gbase': [], output_col: []})

    # Ensure date is parsed properly
    if 'date_parsed' not in df.columns:
        df = df.with_columns(pl.col('date').str.to_date().alias('date_parsed'))

    # Get the most recent date in the dataset
    max_date = df['date_parsed'].max()

    # Calculate cutoff dates
    short_cutoff = max_date - pl.duration(days=short_days)
    long_cutoff = max_date - pl.duration(days=long_days)

    # Calculate short-term average per site (most recent short_days)
    short_term = (
        df.filter(pl.col('date_parsed') >= short_cutoff)
        .group_by('id_gbase')
        .agg(pl.col(metric_col).mean().alias('short_avg'))
    )

    # Calculate long-term average per site (most recent long_days)
    long_term = (
        df.filter(pl.col('date_parsed') >= long_cutoff)
        .group_by('id_gbase')
        .agg(pl.col(metric_col).mean().alias('long_avg'))
    )

    # Join and calculate ratio
    # Use smoothing (add small epsilon) to handle cases where long_avg is zero
    epsilon = 1.0  # Small constant to avoid division by zero
    rs_df = short_term.join(long_term, on='id_gbase', how='left').with_columns(
        ((pl.col('short_avg') + epsilon) / (pl.col('long_avg') + epsilon))
        .alias(output_col)
    ).select(['id_gbase', output_col])

    # Fill missing values with 1.0 (neutral - no trend data available)
    rs_df = rs_df.with_columns(
        pl.col(output_col).fill_null(1.0).fill_nan(1.0)
    )

    return rs_df


def calculate_all_relative_strength_features(df: pl.DataFrame, short_days: int = 30, long_days: int = 90) -> pl.DataFrame:
    """
    Calculate all relative strength features for site metrics.

    Creates momentum indicators for:
    - rs_Impressions: Impression trend (monthly_impressions)
    - rs_NVIs: Network video impression trend (monthly_nvis)
    - rs_Revenue: Revenue trend (revenue)
    - rs_RevenuePerScreen: Per-screen revenue trend (monthly_revenue_per_screen)

    Args:
        df: DataFrame with monthly data
        short_days: Short-term window (default 30)
        long_days: Long-term window (default 90)

    Returns:
        DataFrame with all RS features per site (id_gbase as key)
    """
    print(f"Calculating relative strength features (short={short_days}d, long={long_days}d)...")

    # Parse date once for all calculations
    df = df.with_columns(pl.col('date').str.to_date().alias('date_parsed'))

    # Define metrics to calculate RS for
    rs_metrics = [
        ('monthly_impressions', 'rs_Impressions'),
        ('monthly_nvis', 'rs_NVIs'),
        ('revenue', 'rs_Revenue'),
        ('monthly_revenue_per_screen', 'rs_RevenuePerScreen'),
    ]

    # Calculate each RS feature
    rs_dfs = []
    for metric_col, output_col in rs_metrics:
        rs_df = calculate_relative_strength(df, metric_col, output_col, short_days, long_days)
        if len(rs_df) > 0:
            rs_dfs.append(rs_df)
            mean_val = rs_df[output_col].mean()
            print(f"  {output_col}: {len(rs_df):,} sites (mean={mean_val:.3f})")

    # Join all RS features together
    if not rs_dfs:
        return pl.DataFrame({'id_gbase': []})

    result = rs_dfs[0]
    for rs_df in rs_dfs[1:]:
        # Use coalesce join to handle outer join properly
        result = result.join(rs_df, on='id_gbase', how='full', coalesce=True)

    # Fill any nulls with 1.0 (neutral)
    rs_cols = [col for col in result.columns if col.startswith('rs_')]
    for col in rs_cols:
        result = result.with_columns(pl.col(col).fill_null(1.0))

    print(f"  Total: {len(result):,} sites with RS features")
    return result


def aggregate_site_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate monthly records to one row per site with:
    - Total revenue and auction metrics
    - Average per active month
    - Site metadata (most recent values)
    - rs_Impressions (relative strength indicator)
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

    # Calculate and join all relative strength features (momentum indicators)
    # These compare recent performance (30 days) to longer-term average (90 days)
    if 'date' in df.columns:
        rs_df = calculate_all_relative_strength_features(df, short_days=30, long_days=90)
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
    mcdonalds_df: pl.DataFrame
) -> pl.DataFrame:
    """Join auxiliary geospatial data to site dataset."""
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

    matched = site_df.filter(pl.col('nearest_site').is_not_null()).shape[0]
    kroger_matched = site_df.filter(pl.col('min_distance_to_kroger_mi').is_not_null()).shape[0]
    mcdonalds_matched = site_df.filter(pl.col('min_distance_to_mcdonalds_mi').is_not_null()).shape[0]
    print(f"  Matched {matched:,} sites with nearest site features")
    print(f"  Matched {kroger_matched:,} sites with Kroger distances")
    print(f"  Matched {mcdonalds_matched:,} sites with McDonald's distances")

    return site_df


def add_log_transformations(df: pl.DataFrame) -> pl.DataFrame:
    """Add log transformations for numeric metrics."""
    print("Adding log transformations...")

    numeric_cols = [
        # Total metrics only
        'total_revenue', 'total_monthly_impressions', 'total_monthly_nvis',
        'total_monthly_impressions_per_screen', 'total_monthly_nvis_per_screen',
        'total_monthly_revenue_per_screen',
        # Geospatial distances (all use min_distance_to_X_mi naming convention)
        'min_distance_to_nearest_site_mi', 'min_distance_to_interstate_mi',
        'min_distance_to_kroger_mi', 'min_distance_to_mcdonalds_mi',
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
    """One-hot encode capability (c_*) and restriction (r_*) flags."""
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
    """
    print("Preparing training dataset...")

    # Filter to active sites only
    if active_only:
        original_count = len(df)
        df = df.filter(pl.col('status') == 'Active')
        print(f"  Filtered to Active sites: {len(df):,} (from {original_count:,})")

    # Remove negative revenue records
    before_count = len(df)
    df = df.filter(pl.col('total_revenue') >= 0)
    removed = before_count - len(df)
    print(f"  Removed {removed:,} negative revenue records: {len(df):,} remaining")

    # Drop geographic identifier columns
    if drop_geo_ids:
        geo_cols_to_drop = ['state', 'county', 'dma', 'zip', 'zip_4']
        existing_geo_cols = [c for c in geo_cols_to_drop if c in df.columns]
        df = df.drop(existing_geo_cols)
        print(f"  Dropped geographic identifiers: {existing_geo_cols}")

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
                   'total_revenue', 'avg_monthly_revenue',
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
                'avg_monthly_revenue', 'log_avg_monthly_revenue']
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
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all data
    site_scores = load_site_scores(input_path)
    nearest, interstate, kroger, mcdonalds = load_auxiliary_data(input_path)

    # Aggregate to one row per site
    site_agg = aggregate_site_metrics(site_scores)

    # Join geospatial features
    site_final = join_geospatial_features(site_agg, nearest, interstate, kroger, mcdonalds)

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


if __name__ == "__main__":
    # Create both pre-cleaned and training datasets
    train_df = create_training_dataset()
