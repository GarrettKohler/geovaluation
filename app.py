"""
Flask application for visualizing site-to-highway connections.

Features:
- Interactive map with all sites displayed
- Lasso selection and click-to-select sites
- Highway connection visualization with distance calculations
- Side panel with comprehensive site details
- Filtering by categorical fields (State, Network, Retailer, etc.)

Run with: python app.py
Then open http://localhost:5000 in your browser.
"""

from flask import Flask, render_template, jsonify, request, Response
from src.services.interstate_distance import distance_to_nearest_interstate, preload_highway_data
from src.services.data_service import (
    load_sites,
    load_revenue_metrics,
    load_site_details,
    get_filter_options,
    get_filtered_site_ids,
    get_site_details_for_display,
    CATEGORICAL_FIELDS,
    preload_all_data,
    _clean_nan_values,
)
from src.services.training_service import (
    get_system_info,
    start_training,
    stop_training,
    get_training_status,
    stream_training_progress,
)

app = Flask(__name__)


# =============================================================================
# Page Routes
# =============================================================================

@app.route('/')
def index():
    """Render the main map visualization page."""
    return render_template('index.html')


# =============================================================================
# API Routes - Sites Data
# =============================================================================

@app.route('/api/sites')
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


@app.route('/api/site/<site_id>')
def get_site_detail(site_id):
    """
    Get basic site information including highway distance.

    Args:
        site_id: The site GTVID.

    Returns:
        JSON with site coordinates, nearest highway, and distance.
    """
    df = load_sites()
    site_row = df[df['GTVID'] == site_id]

    if site_row.empty:
        return jsonify({'error': 'Site not found'}), 404

    site = site_row.iloc[0]
    lat, lon = site['Latitude'], site['Longitude']

    result = distance_to_nearest_interstate(lat, lon, include_nearest_point=True)

    return jsonify({
        'site_id': site_id,
        'latitude': lat,
        'longitude': lon,
        'nearest_highway': result['nearest_highway'],
        'distance_miles': round(result['distance_miles'], 2),
        'highway_point': {
            'lat': result['nearest_point_lat'],
            'lon': result['nearest_point_lon']
        }
    })


@app.route('/api/site-details/<site_id>')
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


@app.route('/api/bulk-site-details', methods=['POST'])
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


# =============================================================================
# API Routes - Highway Connections
# =============================================================================

@app.route('/api/highway-connections', methods=['POST'])
def get_highway_connections():
    """
    Calculate highway connections for selected sites.

    Request Body:
        {"site_ids": ["SFR001", "GHR001", ...]}

    Returns:
        JSON with connections array, each containing:
        - site_id, site_lat, site_lon: Site information
        - highway_lat, highway_lon, highway_name: Nearest highway point
        - highway_segment: Coordinates for highlighting the highway section
        - distance_miles: Distance from site to highway
    """
    data = request.get_json()
    site_ids = data.get('site_ids', [])

    if not site_ids:
        return jsonify({'connections': []})

    df = load_sites()
    connections = []

    for site_id in site_ids:
        site_row = df[df['GTVID'] == site_id]
        if site_row.empty:
            continue

        site = site_row.iloc[0]
        lat, lon = site['Latitude'], site['Longitude']

        # Get nearest interstate with highway segment
        result = distance_to_nearest_interstate(
            lat, lon,
            include_nearest_point=True,
            include_highway_segment=True,
            segment_length_meters=800
        )

        connections.append({
            'site_id': site_id,
            'site_lat': lat,
            'site_lon': lon,
            'highway_lat': result['nearest_point_lat'],
            'highway_lon': result['nearest_point_lon'],
            'highway_name': result['nearest_highway'],
            'highway_segment': result.get('highway_segment', []),
            'distance_miles': round(result['distance_miles'], 2)
        })

    return jsonify({'connections': connections})


# =============================================================================
# API Routes - Filtering
# =============================================================================

@app.route('/api/filter-options')
def api_get_filter_options():
    """
    Get unique values for all categorical fields that can be used as filters.

    Returns:
        JSON dict mapping field display name to sorted list of unique values.
        Fields include: State, County, DMA, Retailer, Network, Hardware, etc.
    """
    return jsonify(get_filter_options())


@app.route('/api/filtered-sites', methods=['POST'])
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


# =============================================================================
# API Routes - Model Training
# =============================================================================

@app.route('/api/training/system-info')
def api_get_system_info():
    """
    Get system information for training (GPU availability, etc.).

    Returns:
        JSON with PyTorch version, CUDA/MPS availability, recommended device.
    """
    return jsonify(get_system_info())


@app.route('/api/training/start', methods=['POST'])
def api_start_training():
    """
    Start a new model training job.

    Request Body:
        {
            "model_type": "neural_network",
            "target": "revenue",
            "epochs": 50,
            "batch_size": 4096,
            "learning_rate": 0.0001,
            "dropout": 0.2,
            "hidden_layers": [512, 256, 128, 64],
            "device": "mps"
        }

    Returns:
        JSON with success status and job_id or error message.
    """
    config = request.get_json() or {}
    success, result = start_training(config)

    if success:
        return jsonify({'success': True, 'job_id': result})
    else:
        return jsonify({'success': False, 'error': result}), 400


@app.route('/api/training/stop', methods=['POST'])
def api_stop_training():
    """
    Stop the current training job.

    Returns:
        JSON with success status and message.
    """
    success, message = stop_training()
    return jsonify({'success': success, 'message': message})


@app.route('/api/training/status')
def api_get_training_status():
    """
    Get current training job status.

    Returns:
        JSON with job status, progress metrics, and final results if complete.
    """
    status = get_training_status()
    if status is None:
        return jsonify({'status': 'no_job', 'message': 'No training job exists'})
    return jsonify(status)


@app.route('/api/training/stream')
def api_stream_training():
    """
    Server-Sent Events stream for real-time training progress.

    Returns:
        SSE stream with training progress updates.
    """
    return Response(
        stream_training_progress(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Site Visualization Server")
    print("=" * 60)

    # Pre-load all data
    print("\nLoading data...")
    preload_all_data()

    print("\nPre-loading highway data (this may take a moment on first run)...")
    preload_highway_data()

    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("Open http://localhost:8080 in your browser")
    print("=" * 60)

    app.run(debug=True, port=8080)
