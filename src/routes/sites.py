"""Site data and filtering API routes."""

from flask import Blueprint, jsonify, request

from src.services.data_service import (
    load_sites,
    load_revenue_metrics,
    load_site_details,
    get_filter_options,
    get_filtered_site_ids,
    get_site_details_for_display,
    _clean_nan_values,
)

sites_bp = Blueprint('sites', __name__, url_prefix='/api')


@sites_bp.route('/sites')
def get_sites():
    """
    Get all sites with their coordinates, revenue metrics, and status.

    Returns:
        JSON array of sites with:
        - GTVID: Site identifier
        - Latitude, Longitude: Coordinates
        - revenueScore: Normalized revenue score (0-1)
        - avgMonthlyRevenue: Average monthly revenue in dollars
        - status: Site status (Active, Inactive, etc.)
    """
    df = load_sites()
    metrics = load_revenue_metrics()
    details_df = load_site_details()

    # Create status lookup from site details
    status_lookup = {}
    if 'statuis' in details_df.columns:
        status_lookup = dict(zip(details_df['gtvid'], details_df['statuis']))

    sites = []
    for _, row in df.iterrows():
        site_id = row['GTVID']
        site_metrics = metrics.get(site_id, {'score': 0, 'avg_monthly': 0, 'total': 0, 'months': 0})
        site_status = status_lookup.get(site_id, 'Unknown')

        sites.append({
            'GTVID': site_id,
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'revenueScore': site_metrics.get('score', 0) if isinstance(site_metrics, dict) else 0,
            'avgMonthlyRevenue': site_metrics.get('avg_monthly', 0) if isinstance(site_metrics, dict) else 0,
            'totalRevenue': site_metrics.get('total', 0) if isinstance(site_metrics, dict) else 0,
            'activeMonths': site_metrics.get('months', 0) if isinstance(site_metrics, dict) else 0,
            'status': site_status if site_status else 'Unknown'
        })

    return jsonify(sites)


@sites_bp.route('/site-details/<site_id>')
def get_full_site_details(site_id):
    """
    Get comprehensive site details for the side panel display.

    Args:
        site_id: The site GTVID.

    Returns:
        JSON with site_id and categories dict containing all site fields
        organized by category (Location, Site Info, Brands, Revenue, etc.).
    """
    result = get_site_details_for_display(site_id)
    if result is None:
        return jsonify({'error': 'Site not found'}), 404
    return jsonify(result)


@sites_bp.route('/bulk-site-details', methods=['POST'])
def get_bulk_site_details():
    """
    Get detailed information for multiple sites at once.

    Request Body:
        {"site_ids": ["SFR001", "GHR001", ...]}

    Returns:
        JSON dict mapping site_id to all available fields for that site.
    """
    data = request.get_json()
    site_ids = data.get('site_ids', [])

    if not site_ids:
        return jsonify({})

    details_df = load_site_details()

    result = {}
    for site_id in site_ids:
        site_row = details_df[details_df['gtvid'] == site_id]
        if not site_row.empty:
            # Convert row to dict and clean NaN/Inf values for JSON
            site_data = site_row.iloc[0].to_dict()
            result[site_id] = _clean_nan_values(site_data)

    return jsonify(result)


@sites_bp.route('/filter-options')
def api_get_filter_options():
    """
    Get unique values for all categorical fields that can be used as filters.

    Returns:
        JSON dict mapping field display name to sorted list of unique values.
        Fields include: State, County, DMA, Retailer, Network, Hardware, etc.
    """
    return jsonify(get_filter_options())


@sites_bp.route('/filtered-sites', methods=['POST'])
def api_get_filtered_sites():
    """
    Get sites matching the specified filters.

    Request Body:
        {"filters": {"State": "TX", "Network": "Gilbarco", ...}}

    Returns:
        JSON with:
        - site_ids: List of matching site GTVIDs
        - count: Number of matching sites
    """
    data = request.get_json()
    filters = data.get('filters', {})

    matching_sites = get_filtered_site_ids(filters)

    return jsonify({
        'site_ids': matching_sites,
        'count': len(matching_sites)
    })
