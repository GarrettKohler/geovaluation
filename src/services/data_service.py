"""
Data loading and caching service for site visualization.

Provides centralized data access for sites, revenue metrics, and site details.
All data is cached in memory after first load for fast subsequent access.
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Optional, Any


def _clean_nan_values(obj: Any) -> Any:
    """
    Recursively convert NaN/Inf values to None for JSON serialization.

    JSON doesn't support NaN or Inf, so we convert them to null.
    """
    if isinstance(obj, dict):
        return {k: _clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _clean_nan_values(obj.tolist())
    return obj


# Data file paths (go up: services -> src -> project root)
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "input"
REVENUE_CSV = DATA_DIR / "Site Scores - Site Revenue, Impressions, and Diagnostics.csv"

# Cached data (module-level singletons)
_sites_df: Optional[pd.DataFrame] = None
_revenue_metrics: Optional[Dict[str, Dict[str, Any]]] = None
_site_details_df: Optional[pd.DataFrame] = None
_unique_values_cache: Optional[Dict[str, List[str]]] = None

# Columns to load for site details (organized by category)
DETAIL_COLUMNS = [
    'gtvid',
    # Location
    'state', 'county', 'zip', 'dma', 'dma_rank',
    # Site info
    'retailer', 'network', 'hardware_type', 'experience_type', 'program', 'statuis',
    'screen_count', 'site_activated_date',
    # Brands
    'brand_fuel', 'brand_restaurant', 'brand_c_store',
    # Demographics
    'avg_household_income', 'median_age',
    'pct_african_american', 'pct_asian', 'pct_hispanic', 'pct_female', 'pct_male',
    # Performance
    'avg_daily_impressions', 'avg_daily_nvis', 'avg_latency',
    # Capabilities
    'c_emv_enabled', 'c_nfc_enabled', 'c_open_24_hours',
    'c_sells_beer', 'c_sells_wine', 'c_sells_diesel_fuel', 'c_sells_lottery',
    'c_vistar_programmatic_enabled', 'c_walk_up_enabled',
    # Sales info
    'sellable_site', 'schedule_site'
]

# Categorical fields that can be used as filters (display_name -> column_name)
CATEGORICAL_FIELDS = {
    'State': 'state',
    'County': 'county',
    'DMA': 'dma',
    'Retailer': 'retailer',
    'Network': 'network',
    'Hardware': 'hardware_type',
    'Experience': 'experience_type',
    'Program': 'program',
    'Status': 'statuis',
    'Fuel Brand': 'brand_fuel',
    'Restaurant': 'brand_restaurant',
    'C-Store': 'brand_c_store',
}

# Category organization for site details display
SITE_DETAIL_CATEGORIES = {
    'Location': ['State', 'County', 'ZIP', 'DMA', 'DMA Rank', 'Latitude', 'Longitude'],
    'Site Info': ['Retailer', 'Network', 'Hardware', 'Experience', 'Program', 'Status', 'Screen Count', 'Activated'],
    'Brands': ['Fuel Brand', 'Restaurant', 'C-Store'],
    'Revenue': ['Avg Monthly Revenue', 'Total Revenue', 'Active Months', 'Revenue Score'],
    'Demographics': ['Avg Household Income', 'Median Age', '% African American', '% Asian', '% Hispanic', '% Female', '% Male'],
    'Performance': ['Avg Daily Impressions', 'Avg Daily Visits', 'Avg Latency'],
    'Capabilities': ['EMV Enabled', 'NFC Enabled', 'Open 24 Hours', 'Sells Beer', 'Sells Wine', 'Sells Diesel', 'Sells Lottery', 'Programmatic Enabled', 'Walk-up Enabled'],
    'Sales': ['Sellable Site', 'Schedule Site'],
}


def load_sites(force_reload: bool = False) -> pd.DataFrame:
    """
    Load and cache sites data from the Site Scores CSV.

    Extracts unique sites with their coordinates from the revenue data file.

    Args:
        force_reload: If True, reload from disk even if cached.

    Returns:
        DataFrame with site locations (GTVID, Latitude, Longitude).
    """
    global _sites_df
    if _sites_df is None or force_reload:
        print("Loading sites data from Site Scores...")
        # Load only the columns we need for sites
        df = pd.read_csv(
            REVENUE_CSV,
            usecols=['gtvid', 'latitude', 'longitude'],
            dtype={'gtvid': str}
        )
        # Get unique sites (first occurrence of each)
        _sites_df = df.groupby('gtvid').first().reset_index()
        _sites_df = _sites_df.dropna(subset=['latitude', 'longitude'])
        # Rename columns to match expected format
        _sites_df = _sites_df.rename(columns={
            'gtvid': 'GTVID',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        })
        print(f"Loaded {len(_sites_df):,} unique sites")
    return _sites_df


def load_revenue_metrics(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Calculate and cache revenue metrics for each site.

    Computes:
    - Total revenue
    - Average monthly revenue
    - Revenue per day
    - Normalized revenue score (0-1, using p10-p90 percentiles)

    Args:
        force_reload: If True, recalculate even if cached.

    Returns:
        Dict mapping site ID to metrics dict with keys:
        - score: Normalized revenue score (0-1)
        - avg_monthly: Average monthly revenue
        - total: Total revenue
        - months: Number of active months
    """
    global _revenue_metrics
    if _revenue_metrics is not None and not force_reload:
        return _revenue_metrics

    print("Calculating revenue metrics...")

    # Load revenue data (only needed columns for speed)
    rev_df = pd.read_csv(
        REVENUE_CSV,
        usecols=['gtvid', 'revenue', 'date', 'site_activated_date'],
        dtype={'gtvid': str, 'revenue': float},
        parse_dates=['date', 'site_activated_date']
    )

    # Filter to rows with revenue data
    rev_df = rev_df.dropna(subset=['revenue', 'gtvid'])

    # Group by site and calculate metrics
    site_metrics = rev_df.groupby('gtvid').agg({
        'revenue': 'sum',
        'date': ['min', 'max', 'count'],
        'site_activated_date': 'first'
    }).reset_index()

    # Flatten column names
    site_metrics.columns = ['gtvid', 'total_revenue', 'first_month', 'last_month', 'active_months', 'activated_date']

    # Calculate average monthly revenue
    site_metrics['avg_monthly_revenue'] = site_metrics['total_revenue'] / site_metrics['active_months'].clip(lower=1)

    # Calculate active days for revenue per day
    site_metrics['active_days'] = site_metrics['active_months'] * 30
    site_metrics['revenue_per_day'] = site_metrics['total_revenue'] / site_metrics['active_days'].clip(lower=1)

    # Calculate percentiles for normalization and logging
    rev_per_day_p10 = np.percentile(site_metrics['revenue_per_day'].values, 10)
    rev_per_day_p90 = np.percentile(site_metrics['revenue_per_day'].values, 90)

    avg_monthly_p10 = np.percentile(site_metrics['avg_monthly_revenue'].values, 10)
    avg_monthly_p90 = np.percentile(site_metrics['avg_monthly_revenue'].values, 90)

    total_rev_p10 = np.percentile(site_metrics['total_revenue'].values, 10)
    total_rev_p90 = np.percentile(site_metrics['total_revenue'].values, 90)

    months_p10 = np.percentile(site_metrics['active_months'].values, 10)
    months_p90 = np.percentile(site_metrics['active_months'].values, 90)

    # Create lookup dict
    _revenue_metrics = {}
    for _, row in site_metrics.iterrows():
        raw = row['revenue_per_day']
        normalized = (raw - rev_per_day_p10) / (rev_per_day_p90 - rev_per_day_p10) if rev_per_day_p90 > rev_per_day_p10 else 0
        _revenue_metrics[row['gtvid']] = {
            'score': max(0, min(1, normalized)),
            'avg_monthly': row['avg_monthly_revenue'],
            'total': row['total_revenue'],
            'months': row['active_months']
        }

    print(f"Calculated revenue metrics for {len(_revenue_metrics):,} sites")
    print(f"  Revenue/day:    ${rev_per_day_p10:.2f} (p10) to ${rev_per_day_p90:.2f} (p90)")
    print(f"  Avg Monthly:    ${avg_monthly_p10:,.0f} (p10) to ${avg_monthly_p90:,.0f} (p90)")
    print(f"  Total Revenue:  ${total_rev_p10:,.0f} (p10) to ${total_rev_p90:,.0f} (p90)")
    print(f"  Active Months:  {months_p10:.0f} (p10) to {months_p90:.0f} (p90)")

    return _revenue_metrics


def load_site_details(force_reload: bool = False) -> pd.DataFrame:
    """
    Load full site details, keeping first non-null value per site for each column.

    Args:
        force_reload: If True, reload from disk even if cached.

    Returns:
        DataFrame with one row per site containing all detail columns.
    """
    global _site_details_df
    if _site_details_df is not None and not force_reload:
        return _site_details_df

    print("Loading site details...")

    # Load full CSV
    df = pd.read_csv(REVENUE_CSV, low_memory=False)
    available_cols = [c for c in DETAIL_COLUMNS if c in df.columns]

    # For each site, get first non-null value for each column
    _site_details_df = df[available_cols].groupby('gtvid').first().reset_index()

    # Replace NaN with None for JSON serialization
    _site_details_df = _site_details_df.where(pd.notnull(_site_details_df), None)

    print(f"Loaded details for {len(_site_details_df):,} sites with {len(available_cols)} columns")

    # Log most common values for categorical columns
    print("  Most common values by category:")
    for display_name, col_name in CATEGORICAL_FIELDS.items():
        if col_name in _site_details_df.columns:
            mode_series = _site_details_df[col_name].dropna()
            if len(mode_series) > 0:
                value_counts = mode_series.value_counts()
                if len(value_counts) > 0:
                    top_value = value_counts.index[0]
                    top_count = value_counts.iloc[0]
                    pct = (top_count / len(mode_series)) * 100
                    print(f"    {display_name}: {top_value} ({top_count:,} sites, {pct:.1f}%)")

    return _site_details_df


def get_filter_options(force_reload: bool = False) -> Dict[str, List[str]]:
    """
    Get unique values for all categorical fields that can be used as filters.

    Args:
        force_reload: If True, recalculate even if cached.

    Returns:
        Dict mapping field display name to sorted list of unique values.
    """
    global _unique_values_cache
    if _unique_values_cache is not None and not force_reload:
        return _unique_values_cache

    details_df = load_site_details()

    options = {}
    for display_name, col_name in CATEGORICAL_FIELDS.items():
        if col_name in details_df.columns:
            unique_vals = details_df[col_name].dropna().unique().tolist()
            unique_vals = sorted([v for v in unique_vals if v and str(v).strip()])
            options[display_name] = unique_vals

    _unique_values_cache = options
    return options


def get_filtered_site_ids(filters: Dict[str, str]) -> List[str]:
    """
    Get site IDs matching the specified filters.

    Args:
        filters: Dict mapping field display name to filter value.

    Returns:
        List of matching site IDs.
    """
    if not filters:
        return []

    details_df = load_site_details()

    # Start with all sites
    mask = pd.Series([True] * len(details_df), index=details_df.index)

    # Apply each filter
    for display_name, value in filters.items():
        if display_name in CATEGORICAL_FIELDS and value:
            col_name = CATEGORICAL_FIELDS[display_name]
            if col_name in details_df.columns:
                mask = mask & (details_df[col_name] == value)

    return details_df.loc[mask, 'gtvid'].tolist()


def get_site_details_for_display(site_id: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive site details organized by category for display.

    Returns partial data if full details aren't available but the site exists
    in the sites or revenue data.

    Args:
        site_id: The site GTVID.

    Returns:
        Dict with site_id, categories, and partial flag if data is incomplete.
        Returns None only if the site doesn't exist anywhere in the data.
    """
    details_df = load_site_details()
    metrics = load_revenue_metrics()
    sites = load_sites()

    # Get site details from details DataFrame
    site_row = details_df[details_df['gtvid'] == site_id]
    has_full_details = not site_row.empty

    # Check if site exists in sites DataFrame (has coordinates)
    site_coords = sites[sites['GTVID'] == site_id]
    has_coordinates = not site_coords.empty

    # Check if site has revenue metrics
    has_revenue = site_id in metrics

    # If site doesn't exist anywhere, return None
    if not has_full_details and not has_coordinates and not has_revenue:
        return None

    # Build site_data from available sources
    if has_full_details:
        site_data = site_row.iloc[0].to_dict()
    else:
        # Initialize with empty dict for partial data
        site_data = {'gtvid': site_id}

    # Get coordinates
    site_coords = sites[sites['GTVID'] == site_id]
    if not site_coords.empty:
        site_data['latitude'] = site_coords.iloc[0]['Latitude']
        site_data['longitude'] = site_coords.iloc[0]['Longitude']

    # Add revenue metrics
    site_metrics = metrics.get(site_id, {})
    site_data['revenue_score'] = site_metrics.get('score', 0)
    site_data['avg_monthly_revenue'] = site_metrics.get('avg_monthly', 0)
    site_data['total_revenue'] = site_metrics.get('total', 0)
    site_data['active_months'] = site_metrics.get('months', 0)

    # Column name mapping (internal -> display)
    col_to_display = {
        'state': 'State', 'county': 'County', 'zip': 'ZIP', 'dma': 'DMA', 'dma_rank': 'DMA Rank',
        'latitude': 'Latitude', 'longitude': 'Longitude',
        'retailer': 'Retailer', 'network': 'Network', 'hardware_type': 'Hardware',
        'experience_type': 'Experience', 'program': 'Program', 'statuis': 'Status',
        'screen_count': 'Screen Count', 'site_activated_date': 'Activated',
        'brand_fuel': 'Fuel Brand', 'brand_restaurant': 'Restaurant', 'brand_c_store': 'C-Store',
        'avg_monthly_revenue': 'Avg Monthly Revenue', 'total_revenue': 'Total Revenue',
        'active_months': 'Active Months', 'revenue_score': 'Revenue Score',
        'avg_household_income': 'Avg Household Income', 'median_age': 'Median Age',
        'pct_african_american': '% African American', 'pct_asian': '% Asian',
        'pct_hispanic': '% Hispanic', 'pct_female': '% Female', 'pct_male': '% Male',
        'avg_daily_impressions': 'Avg Daily Impressions', 'avg_daily_nvis': 'Avg Daily Visits',
        'avg_latency': 'Avg Latency',
        'c_emv_enabled': 'EMV Enabled', 'c_nfc_enabled': 'NFC Enabled',
        'c_open_24_hours': 'Open 24 Hours', 'c_sells_beer': 'Sells Beer',
        'c_sells_wine': 'Sells Wine', 'c_sells_diesel_fuel': 'Sells Diesel',
        'c_sells_lottery': 'Sells Lottery', 'c_vistar_programmatic_enabled': 'Programmatic Enabled',
        'c_walk_up_enabled': 'Walk-up Enabled',
        'sellable_site': 'Sellable Site', 'schedule_site': 'Schedule Site',
    }

    # Build categories
    categories = {}
    for cat_name, fields in SITE_DETAIL_CATEGORIES.items():
        cat_data = {}
        for field in fields:
            # Find the internal column name
            col_name = next((k for k, v in col_to_display.items() if v == field), field.lower())
            cat_data[field] = site_data.get(col_name)
        categories[cat_name] = cat_data

    result = {
        'site_id': site_id,
        'categories': categories
    }

    # Add partial flag if site details were incomplete
    if not has_full_details:
        result['partial'] = True
        result['available_data'] = []
        if has_coordinates:
            result['available_data'].append('coordinates')
        if has_revenue:
            result['available_data'].append('revenue')

    # Clean NaN/Inf values for JSON serialization
    return _clean_nan_values(result)


def preload_all_data() -> None:
    """Pre-load all data into memory for faster API responses."""
    load_sites()
    load_revenue_metrics()
    load_site_details()
    get_filter_options()
    print("All data pre-loaded.")
