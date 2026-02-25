"""
Flask application for visualizing geospatial site data with ML scoring.

Features:
- Interactive map with all sites displayed
- Lasso selection and click-to-select sites
- Side panel with comprehensive site details
- Filtering by categorical fields (State, Network, Retailer, etc.)
- ML model training (regression and lookalike)
- Explainability features (tier classification, counterfactuals)

Run with: python app.py
Then open http://localhost:8080 in your browser.
"""

from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response
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
    load_explainability_components,
    explain_prediction,
    scan_experiment_folders,
)
from site_scoring.config import get_all_model_presets, get_model_preset, get_all_available_features, DEFAULT_OUTPUT_DIR
from src.services.fleet_analysis_service import (
    start_fleet_analysis,
    get_fleet_analysis_status,
    export_fleet_analysis_to_excel,
)
from src.services.shap_service import ShapCache, generate_shap_plots

app = Flask(__name__)


# =============================================================================
# Page Routes
# =============================================================================

@app.route('/')
def home():
    """Render the experiment hub home page."""
    return render_template('home.html')


@app.route('/map')
@app.route('/map/<job_id>')
def map_view(job_id=None):
    """Render the map visualization page."""
    return render_template('index.html', job_id=job_id)


@app.route('/training-details')
def training_details():
    """Render the training details page with site records."""
    return render_template('training_details.html')


@app.route('/glossary')
def glossary():
    """Render the ML/statistics glossary page."""
    return render_template('glossary.html')


@app.route('/shap-values')
def shap_values():
    """Render the SHAP feature importance visualization page."""
    return render_template('shap_values.html')


@app.route('/datasets')
def datasets_index():
    """Render the dataset index page showing pipeline-active source datasets."""
    from src.services.lineage_service import load_ontology, get_pipeline_datasets
    ontology = load_ontology(Path(__file__).parent / "docs" / "data_ontology.yaml")
    datasets = get_pipeline_datasets(ontology)
    return render_template('datasets_index.html', datasets=datasets)


@app.route('/datasets/<dataset_id>')
def dataset_lineage(dataset_id):
    """Render the column lineage detail page for a specific dataset."""
    from src.services.lineage_service import load_ontology, get_dataset_info, get_lineage_for_dataset
    ontology = load_ontology(Path(__file__).parent / "docs" / "data_ontology.yaml")
    dataset = get_dataset_info(ontology, dataset_id)
    if not dataset:
        return "Dataset not found", 404
    lineages = get_lineage_for_dataset(dataset_id)
    kept = [l for l in lineages if not l.dropped]
    dropped = [l for l in lineages if l.dropped]
    return render_template('dataset_lineage.html', dataset=dataset, kept=kept, dropped=dropped)


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


@app.route('/api/training/presets')
def api_get_training_presets():
    """
    Get available model presets with feature counts.

    Returns:
        JSON dict mapping preset key to {name, description, counts}.
    """
    return jsonify(get_all_model_presets())


@app.route('/api/training/features')
def api_get_training_features():
    """
    Get the feature lists for a model preset (or the default config).

    Query Params:
        preset: Optional model preset name (model_a, model_b)

    Returns:
        JSON with numeric, categorical, and boolean feature lists plus counts.
    """
    from site_scoring.config import Config
    cfg = Config()

    preset_name = request.args.get('preset')
    if preset_name:
        try:
            cfg.apply_model_preset(preset_name)
        except ValueError:
            pass  # Fall back to default if invalid preset

    return jsonify({
        'target': cfg.target,
        'task_type': cfg.task_type,
        'numeric': cfg.numeric_features,
        'categorical': cfg.categorical_features,
        'boolean': cfg.boolean_features,
        'preset': preset_name or 'default',
        'counts': {
            'numeric': len(cfg.numeric_features),
            'categorical': len(cfg.categorical_features),
            'boolean': len(cfg.boolean_features),
            'total': len(cfg.numeric_features) + len(cfg.categorical_features) + len(cfg.boolean_features),
        },
    })


@app.route('/api/training/all-features')
def api_get_all_features():
    """
    Get all available features across all presets.
    Used by the feature selection UI to show what can be selected.

    Returns:
        JSON with numeric, categorical, and boolean feature lists (union of all presets).
    """
    all_features = get_all_available_features()
    return jsonify({
        'numeric': all_features['numeric'],
        'categorical': all_features['categorical'],
        'boolean': all_features['boolean'],
        'counts': {
            'numeric': len(all_features['numeric']),
            'categorical': len(all_features['categorical']),
            'boolean': len(all_features['boolean']),
            'total': (len(all_features['numeric']) +
                      len(all_features['categorical']) +
                      len(all_features['boolean'])),
        },
    })


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
    Stop a training job.
    
    Request Body:
        {"job_id": "job_123..."}

    Returns:
        JSON with success status and message.
    """
    data = request.get_json() or {}
    job_id = data.get('job_id')
    
    success, message = stop_training(job_id)
    return jsonify({'success': success, 'message': message})


@app.route('/api/training/status')
def api_get_training_status():
    """
    Get training status.
    
    Query Params:
        job_id: Optional job ID to get specific status.

    Returns:
        JSON with job status, progress metrics, and final results if complete.
    """
    job_id = request.args.get('job_id')
    status = get_training_status(job_id)
    return jsonify(status)


@app.route('/api/experiments')
def api_get_experiments():
    """
    Get a list of all experiments (active and history).
    """
    # This maps to get_training_status() with no args which returns all jobs
    return jsonify(get_training_status())


@app.route('/api/experiments/catalog')
def api_get_catalog():
    """Persistent experiment catalog — survives Flask restarts.

    Scans all experiment folders on disk, parses config and metadata files,
    and returns a sorted list of experiments with their metrics, features,
    and artifact inventory.
    """
    catalog = scan_experiment_folders()
    return jsonify({"experiments": catalog})


@app.route('/api/experiments/compare', methods=['POST'])
def api_compare_experiments():
    """Compare two or more experiments across metrics and features."""
    import json as json_mod

    data = request.get_json() or {}
    experiment_ids = data.get('experiment_ids', [])

    if len(experiment_ids) < 2:
        return jsonify({'error': 'Need at least 2 experiment IDs to compare'}), 400

    catalog = scan_experiment_folders()
    catalog_lookup = {exp['job_id']: exp for exp in catalog}

    metrics_comparison = {}
    feature_comparison = {}

    for exp_id in experiment_ids:
        exp = catalog_lookup.get(exp_id)
        if exp is None:
            return jsonify({'error': f'Experiment not found: {exp_id}'}), 404

        metrics_comparison[exp_id] = {
            'model_type': exp.get('model_type'),
            'task_type': exp.get('task_type'),
            'test_metrics': exp.get('test_metrics', {}),
            'created_at': exp.get('created_at'),
        }

        # Feature comparison
        training_features = exp.get('training_features', {})
        all_features = []
        for feat_type in ['numeric', 'categorical', 'boolean']:
            all_features.extend(training_features.get(feat_type, []))

        # Load SHAP importance if available
        shap_path = DEFAULT_OUTPUT_DIR / "experiments" / exp_id / "shap_importance.json"
        top_features = []
        if shap_path.exists():
            try:
                with open(shap_path) as f:
                    shap_data = json_mod.load(f)
                top_features = shap_data if isinstance(shap_data, list) else shap_data.get('features', [])
            except Exception:
                pass

        feature_comparison[exp_id] = {
            'feature_count': exp.get('feature_count', {}),
            'all_features': all_features,
            'top_features': top_features[:15],
        }

    # Compute feature overlap between experiments
    all_feature_sets = {eid: set(fc['all_features']) for eid, fc in feature_comparison.items()}
    if len(all_feature_sets) == 2:
        sets = list(all_feature_sets.values())
        overlap = sets[0] & sets[1]
        only_first = sets[0] - sets[1]
        only_second = sets[1] - sets[0]
        feature_overlap = {
            'shared': sorted(overlap),
            'shared_count': len(overlap),
            'unique_to_first': sorted(only_first),
            'unique_to_second': sorted(only_second),
        }
    else:
        common = set.intersection(*all_feature_sets.values()) if all_feature_sets else set()
        feature_overlap = {
            'shared': sorted(common),
            'shared_count': len(common),
        }

    # Score comparison (optional — expensive, requires two inference passes)
    score_comparison = None
    include_scores = data.get('include_scores', False)

    if include_scores and len(experiment_ids) == 2:
        try:
            import numpy as np
            from site_scoring.data_transform import get_all_sites_for_prediction
            from site_scoring.predict import BatchPredictor

            all_sites_df = get_all_sites_for_prediction()

            exp_id_a, exp_id_b = experiment_ids[0], experiment_ids[1]
            dir_a = DEFAULT_OUTPUT_DIR / "experiments" / exp_id_a
            dir_b = DEFAULT_OUTPUT_DIR / "experiments" / exp_id_b

            if dir_a.exists() and dir_b.exists():
                predictor_a = BatchPredictor(dir_a)
                predictor_b = BatchPredictor(dir_b)

                scores_a = predictor_a.predict(all_sites_df)
                scores_b = predictor_b.predict(all_sites_df)

                # Align scores on common gtvids
                common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
                arr_a = np.array([scores_a[gid] for gid in common_ids])
                arr_b = np.array([scores_b[gid] for gid in common_ids])

                # Pearson correlation
                correlation = float(np.corrcoef(arr_a, arr_b)[0, 1]) if len(arr_a) > 1 else None

                # Mean absolute difference
                mean_diff = float(np.mean(np.abs(arr_a - arr_b)))

                # Quintile agreement: % of sites in same quintile by both models
                def quintile_labels(arr):
                    percentiles = np.percentile(arr, [20, 40, 60, 80])
                    return np.digitize(arr, percentiles)

                q_a = quintile_labels(arr_a)
                q_b = quintile_labels(arr_b)
                agreement_pct = float(np.mean(q_a == q_b) * 100)

                # Top 20 sites with largest score disagreement
                diffs = np.abs(arr_a - arr_b)
                top_diff_indices = np.argsort(diffs)[-20:][::-1]
                sites_with_large_diff = []
                for idx in top_diff_indices:
                    sites_with_large_diff.append({
                        'gtvid': common_ids[idx],
                        'score_a': float(arr_a[idx]),
                        'score_b': float(arr_b[idx]),
                        'diff': float(diffs[idx]),
                    })

                score_comparison = {
                    'correlation': correlation,
                    'mean_diff': mean_diff,
                    'agreement_pct': round(agreement_pct, 1),
                    'n_common_sites': len(common_ids),
                    'sites_with_large_diff': sites_with_large_diff,
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            score_comparison = {'error': str(e)}

    result = {
        'metrics_comparison': _clean_nan_values(metrics_comparison),
        'feature_comparison': feature_comparison,
        'feature_overlap': feature_overlap,
    }
    if score_comparison is not None:
        result['score_comparison'] = _clean_nan_values(score_comparison)

    return jsonify(result)


@app.route('/api/training/stream')
def api_stream_training():
    """
    Server-Sent Events stream for real-time training progress for all jobs.

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
# API Routes - Batch Prediction
# =============================================================================

# Module-level cache for BatchPredictor (reloads only when experiment changes)
_cached_predictor = None
_cached_predictor_experiment = None


def _find_latest_experiment() -> Path:
    """Find the most recent experiment directory with a complete model."""
    experiments_dir = DEFAULT_OUTPUT_DIR / "experiments"
    if not experiments_dir.exists():
        return None

    experiment_dirs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for exp_dir in experiment_dirs:
        config_path = exp_dir / "config.json"
        preprocessor_path = exp_dir / "preprocessor.pkl"
        has_model = (
            (exp_dir / "best_model.pt").exists()
            or (exp_dir / "model_wrapper.pkl").exists()
        )
        if config_path.exists() and preprocessor_path.exists() and has_model:
            return exp_dir

    return None


def _get_cached_predictor(experiment_dir: Path):
    """Get or create a cached BatchPredictor for the given experiment."""
    global _cached_predictor, _cached_predictor_experiment

    if _cached_predictor is not None and _cached_predictor_experiment == str(experiment_dir):
        return _cached_predictor

    from site_scoring.predict import BatchPredictor
    _cached_predictor = BatchPredictor(experiment_dir)
    _cached_predictor_experiment = str(experiment_dir)
    return _cached_predictor


@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    """
    Run batch prediction using a trained model on all (or filtered) sites.

    Request Body:
        {
            "experiment_dir": "job_xxx",   // optional, defaults to latest
            "filter": {"network": ["Wayne"], "state": ["TX"]}  // optional
        }

    Returns:
        JSON with:
        - predictions: {gtvid: score} mapping
        - model_type: str
        - task_type: str
        - count: int
        - summary: {mean, median, std, min, max, p10, p25, p75, p90}
    """
    import numpy as np
    import polars as pl

    data = request.get_json() or {}
    experiment_name = data.get('experiment_dir')
    filters = data.get('filter')

    # Find experiment directory
    if experiment_name:
        experiment_dir = DEFAULT_OUTPUT_DIR / "experiments" / experiment_name
        if not experiment_dir.exists():
            return jsonify({'error': f'Experiment not found: {experiment_name}'}), 404
    else:
        experiment_dir = _find_latest_experiment()
        if experiment_dir is None:
            return jsonify({'error': 'No trained model found. Train a model first.'}), 404

    try:
        # Load predictor (cached)
        predictor = _get_cached_predictor(experiment_dir)

        # Load all sites for prediction
        from site_scoring.data_transform import get_all_sites_for_prediction
        all_sites_df = get_all_sites_for_prediction()

        # Apply filters if provided
        if filters:
            matching_ids = get_filtered_site_ids(filters)
            all_sites_df = all_sites_df.filter(pl.col("gtvid").is_in(matching_ids))

        # Run prediction
        predictions = predictor.predict(all_sites_df)

        # Compute summary statistics
        scores = np.array(list(predictions.values()))
        summary = {}
        if len(scores) > 0:
            summary = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'p10': float(np.percentile(scores, 10)),
                'p25': float(np.percentile(scores, 25)),
                'p75': float(np.percentile(scores, 75)),
                'p90': float(np.percentile(scores, 90)),
            }

        # Clean NaN/Inf for JSON safety
        clean_predictions = _clean_nan_values(predictions)

        return jsonify(_clean_nan_values({
            'predictions': clean_predictions,
            'model_type': predictor.model_type,
            'task_type': predictor.task_type,
            'count': len(clean_predictions),
            'experiment_dir': experiment_dir.name,
            'summary': summary,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict/export', methods=['POST'])
def api_predict_export():
    """Export scored site predictions as CSV or Excel download.

    For regression tasks: scores all sites with predicted revenue + site metadata.
    For classification tasks: combines training sites (with actual_label) and
    non-active sites (with predicted_probability) into a single export with
    a 'category' column (TRAINING / NON_ACTIVE).
    """
    from flask import send_file
    from io import BytesIO
    from datetime import datetime
    import polars as pl
    import csv as csv_mod

    data = request.get_json() or {}
    experiment_name = data.get('experiment_dir')
    export_format = data.get('format', 'csv')
    filters = data.get('filter')

    # Find experiment
    if experiment_name:
        experiment_dir = DEFAULT_OUTPUT_DIR / "experiments" / experiment_name
        if not experiment_dir.exists():
            return jsonify({'error': f'Experiment not found: {experiment_name}'}), 404
    else:
        experiment_dir = _find_latest_experiment()
        if experiment_dir is None:
            return jsonify({'error': 'No trained model found.'}), 404

    try:
        predictor = _get_cached_predictor(experiment_dir)

        from site_scoring.data_transform import get_all_sites_for_prediction
        all_sites_df = get_all_sites_for_prediction()

        # Apply filters if provided
        if filters:
            matching_ids = get_filtered_site_ids(filters)
            all_sites_df = all_sites_df.filter(pl.col("gtvid").is_in(matching_ids))

        # For classification: load training labels from experiment artifacts
        training_labels = None
        if predictor.task_type == "lookalike":
            training_labels = _load_training_labels(experiment_dir)

        # Get predictions with metadata (+ labels for classification)
        result_df = predictor.predict_with_metadata(
            all_sites_df, training_labels=training_labels
        )

        # Add export metadata columns
        result_df = result_df.with_columns([
            pl.lit(predictor.model_type).alias("model_type"),
            pl.lit(experiment_dir.name).alias("experiment_id"),
            pl.lit(datetime.now().isoformat()).alias("scored_at"),
        ])

        # Generate file
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"site_scores_{predictor.task_type}_{predictor.model_type}_{timestamp}"

        buf = BytesIO()
        if export_format == 'xlsx':
            result_df.write_excel(buf)
            buf.seek(0)
            return send_file(
                buf,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'{filename}.xlsx',
            )
        else:
            csv_bytes = result_df.write_csv().encode('utf-8')
            buf.write(csv_bytes)
            buf.seek(0)
            return send_file(
                buf,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{filename}.csv',
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


def _load_training_labels(experiment_dir: Path) -> dict:
    """Load training site labels from an experiment's training_sites.csv.

    Returns dict mapping gtvid -> actual_label (0 or 1).
    Falls back to empty dict if the file doesn't exist.
    """
    import csv as csv_mod

    labels = {}
    training_csv = experiment_dir / "training_sites.csv"
    if training_csv.exists():
        with open(training_csv) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                gtvid = row.get("gtvid", "")
                label = int(row.get("actual_label", 0))
                if gtvid:
                    labels[gtvid] = label
    return labels


@app.route('/api/predict/filtered', methods=['POST'])
def api_predict_filtered():
    """Score only sites matching filter criteria.

    Request Body:
        {
            "experiment_dir": "job_xxx",  (optional, defaults to latest)
            "filters": {
                "network": ["Wayne", "Dover"],
                "state": ["TX", "CA"],
                "status": ["Active"]
            }
        }

    Returns same format as /api/predict/batch with summary stats.
    """
    import numpy as np
    import polars as pl

    data = request.get_json() or {}
    filters = data.get('filters', {})
    experiment_name = data.get('experiment_dir')

    if not filters:
        return jsonify({'error': 'No filters provided. Use /api/predict/batch for all sites.'}), 400

    # Find experiment directory
    if experiment_name:
        experiment_dir = DEFAULT_OUTPUT_DIR / "experiments" / experiment_name
        if not experiment_dir.exists():
            return jsonify({'error': f'Experiment not found: {experiment_name}'}), 404
    else:
        experiment_dir = _find_latest_experiment()
        if experiment_dir is None:
            return jsonify({'error': 'No trained model found. Train a model first.'}), 404

    try:
        predictor = _get_cached_predictor(experiment_dir)

        from site_scoring.data_transform import get_all_sites_for_prediction
        all_sites_df = get_all_sites_for_prediction()

        # Apply filters
        matching_ids = get_filtered_site_ids(filters)
        filtered_df = all_sites_df.filter(pl.col("gtvid").is_in(matching_ids))

        if len(filtered_df) == 0:
            return jsonify({
                'predictions': {},
                'model_type': predictor.model_type,
                'task_type': predictor.task_type,
                'count': 0,
                'experiment_dir': experiment_dir.name,
                'filter_applied': filters,
                'summary': {},
            })

        predictions = predictor.predict(filtered_df)
        clean_predictions = _clean_nan_values(predictions)

        # Summary stats
        scores = np.array(list(predictions.values()))
        summary = {}
        if len(scores) > 0:
            summary = {
                'mean': float(np.mean(scores)),
                'median': float(np.median(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'p10': float(np.percentile(scores, 10)),
                'p25': float(np.percentile(scores, 25)),
                'p75': float(np.percentile(scores, 75)),
                'p90': float(np.percentile(scores, 90)),
            }

        return jsonify(_clean_nan_values({
            'predictions': clean_predictions,
            'model_type': predictor.model_type,
            'task_type': predictor.task_type,
            'count': len(clean_predictions),
            'experiment_dir': experiment_dir.name,
            'filter_applied': filters,
            'summary': summary,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# =============================================================================
# API Routes - SHAP Feature Importance
# =============================================================================

SHAP_OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def get_latest_shap_directory() -> Path:
    """
    Find the most recent experiment directory containing SHAP data.

    First checks the experiments subdirectory for job-specific SHAP caches,
    then falls back to the default output directory for backward compatibility.

    Returns:
        Path to the directory containing the most recent shap_cache.npz
    """
    experiments_dir = DEFAULT_OUTPUT_DIR / "experiments"

    # Check experiment directories (newest first by modification time)
    if experiments_dir.exists():
        experiment_dirs = sorted(
            [d for d in experiments_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        for exp_dir in experiment_dirs:
            shap_cache_path = exp_dir / "shap_cache.npz"
            if shap_cache_path.exists():
                return exp_dir

    # Fallback to default output directory (legacy location)
    return DEFAULT_OUTPUT_DIR


@app.route('/api/shap/available')
def api_shap_available():
    """
    Check if SHAP data is available from the most recent training run.
    Searches experiment directories first, then falls back to default location.

    Returns:
        JSON with available flag and basic info.
    """
    shap_dir = get_latest_shap_directory()
    cache = ShapCache(shap_dir)
    if cache.exists():
        info = cache.get_feature_importance(top_n=1)
        return jsonify({
            'available': True,
            'n_samples': info['n_samples'] if info else 0,
            'n_features': info['n_features'] if info else 0,
            'experiment_dir': str(shap_dir.name) if shap_dir != DEFAULT_OUTPUT_DIR else None,
        })
    return jsonify({'available': False})


@app.route('/api/shap/summary')
def api_shap_summary():
    """
    Get SHAP feature importance summary data.

    Query Params:
        top_n: Number of top features to return (default: 30)

    Returns:
        JSON with ranked feature importance list, base value, and sample counts.
    """
    top_n = request.args.get('top_n', 30, type=int)
    shap_dir = get_latest_shap_directory()
    cache = ShapCache(shap_dir)
    result = cache.get_feature_importance(top_n=top_n)

    if result is None:
        return jsonify({'error': 'No SHAP data available. Train a model first.'}), 404

    return jsonify(result)


@app.route('/api/shap/plots')
def api_shap_plots():
    """
    Get SHAP visualization plots as base64-encoded PNG images.

    Returns:
        JSON with bar_plot and summary_plot as base64 strings,
        or error if SHAP data/matplotlib unavailable.
    """
    shap_dir = get_latest_shap_directory()
    plots = generate_shap_plots(shap_dir)
    if plots is None:
        return jsonify({'error': 'No SHAP plots available. Train a model first.'}), 404
    return jsonify(plots)


# =============================================================================
# API Routes - Explainability (Conformal Prediction & Tier Classification)
# =============================================================================

@app.route('/api/explainability/available')
def api_explainability_available():
    """
    Check if the explainability pipeline is available.

    The pipeline is fitted automatically after training lookalike models
    and provides:
    - Probability calibration (isotonic regression)
    - Conformal prediction (prediction sets with coverage guarantees)
    - Tier classification (executive-friendly labels)

    Returns:
        JSON with:
        - available: bool
        - calibration_method: str (if available)
        - conformal_alpha: float (if available)
    """
    components = load_explainability_components(SHAP_OUTPUT_DIR)

    if components is None:
        return jsonify({
            'available': False,
            'message': 'No explainability pipeline found. Train a lookalike model first.'
        })

    metadata = components.get('metadata', {})
    return jsonify({
        'available': True,
        'calibration_method': 'isotonic',
        'conformal_alpha': metadata.get('conformal_alpha', 0.10),
        'n_features': metadata.get('n_numeric', 0) + metadata.get('n_categorical', 0) + metadata.get('n_boolean', 0),
        'n_calibration_samples': metadata.get('n_calibration_samples', 0),
    })


@app.route('/api/explainability/explain', methods=['POST'])
def api_explain_prediction():
    """
    Explain a prediction with calibration and tier classification.

    This lightweight endpoint takes a raw model probability and returns:
    - Calibrated probability
    - Tier classification with confidence statement
    - Historical accuracy for the tier

    Request Body:
        {"probability": 0.75}  # Raw sigmoid output from model

    Returns:
        JSON with:
        - raw_probability: float
        - calibrated_probability: float
        - tier: int (1-4)
        - tier_label: str (Recommended, Promising, Review Required, Not Recommended)
        - tier_action: str (recommended action)
        - confidence_statement: str (e.g., "8 out of 10 similar sites succeeded")
        - historical_accuracy: float or null
        - color: str (hex color for UI)
    """
    data = request.get_json() or {}
    probability = data.get('probability')

    if probability is None:
        return jsonify({'error': 'Missing probability parameter'}), 400

    try:
        probability = float(probability)
        if not 0 <= probability <= 1:
            return jsonify({'error': 'Probability must be between 0 and 1'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid probability value'}), 400

    result = explain_prediction(probability, SHAP_OUTPUT_DIR)

    if 'error' in result:
        return jsonify(result), 404

    return jsonify(result)


@app.route('/api/explainability/explain-batch', methods=['POST'])
def api_explain_batch():
    """
    Explain multiple predictions at once.

    Request Body:
        {"probabilities": {"site1": 0.75, "site2": 0.42, ...}}

    Returns:
        JSON dict mapping site_id to explanation result.
    """
    data = request.get_json() or {}
    probabilities = data.get('probabilities', {})

    if not probabilities:
        return jsonify({})

    components = load_explainability_components(SHAP_OUTPUT_DIR)
    if components is None:
        return jsonify({'error': 'Explainability pipeline not available'}), 404

    calibrator = components.get('calibrator')
    tier_classifier = components.get('tier_classifier')

    if calibrator is None or tier_classifier is None:
        return jsonify({'error': 'Missing calibrator or tier classifier'}), 404

    import numpy as np
    results = {}

    for site_id, raw_prob in probabilities.items():
        try:
            raw_prob = float(raw_prob)
            calibrated_prob = calibrator.calibrate(np.array([raw_prob]))[0]
            tier_result = tier_classifier.classify(calibrated_prob)

            results[site_id] = {
                'raw_probability': raw_prob,
                'calibrated_probability': float(calibrated_prob),
                'tier': tier_result.tier,
                'tier_label': tier_result.label,
                'tier_action': tier_result.action,
                'confidence_statement': tier_result.confidence_statement,
                'historical_accuracy': tier_result.historical_accuracy,
                'color': tier_result.color,
            }
        except (TypeError, ValueError) as e:
            results[site_id] = {'error': str(e)}

    return jsonify(results)


@app.route('/api/explainability/tier-summary', methods=['POST'])
def api_tier_summary():
    """
    Get tier distribution summary for a set of predictions.

    Useful for fleet-wide analysis to understand how sites are distributed
    across tiers.

    Request Body:
        {"probabilities": [0.85, 0.72, 0.45, ...]}

    Returns:
        JSON with:
        - tier_distribution: dict mapping tier to {count, percentage, label, color}
        - total_sites: int
        - calibration_applied: bool
    """
    data = request.get_json() or {}
    probabilities = data.get('probabilities', [])

    if not probabilities:
        return jsonify({
            'tier_distribution': {},
            'total_sites': 0,
            'calibration_applied': False,
        })

    components = load_explainability_components(SHAP_OUTPUT_DIR)
    if components is None:
        return jsonify({'error': 'Explainability pipeline not available'}), 404

    calibrator = components.get('calibrator')
    tier_classifier = components.get('tier_classifier')

    if calibrator is None or tier_classifier is None:
        return jsonify({'error': 'Missing calibrator or tier classifier'}), 404

    import numpy as np

    # Calibrate all probabilities
    raw_probs = np.array([float(p) for p in probabilities])
    calibrated_probs = calibrator.calibrate(raw_probs)

    # Get tier distribution
    from site_scoring.explainability.tiers import TIER_LABELS, TIER_COLORS
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for prob in calibrated_probs:
        tier_result = tier_classifier.classify(prob)
        tier_counts[tier_result.tier] += 1

    total = len(probabilities)
    tier_distribution = {}
    for tier, count in tier_counts.items():
        tier_distribution[tier] = {
            'count': count,
            'percentage': count / total if total > 0 else 0,
            'label': TIER_LABELS[tier],
            'color': TIER_COLORS[tier],
        }

    return jsonify({
        'tier_distribution': tier_distribution,
        'total_sites': total,
        'calibration_applied': True,
    })


# =============================================================================
# API Routes - Fleet-Wide Intervention Analysis (Phase 5)
# =============================================================================

@app.route('/api/explainability/fleet-analysis', methods=['POST'])
def api_start_fleet_analysis():
    """
    Start fleet-wide intervention analysis.

    This endpoint analyzes all low-performing sites (Tier 3-4) to identify
    strategic interventions that could upgrade the entire portfolio.

    Request Body:
        {
            "site_ids": ["SFR001", "GHR001", ...],
            "probabilities": [0.35, 0.42, ...],
            "n_counterfactuals": 3,  # optional, default 3
            "n_clusters": 5          # optional, default 5
        }

    Returns:
        JSON with:
        - success: bool
        - job_id: str (for tracking progress)
        - message: str
    """
    data = request.get_json() or {}
    site_ids = data.get('site_ids', [])
    probabilities = data.get('probabilities', [])
    n_counterfactuals = data.get('n_counterfactuals', 3)
    n_clusters = data.get('n_clusters', 5)

    if not site_ids or not probabilities:
        return jsonify({
            'success': False,
            'error': 'Missing site_ids or probabilities'
        }), 400

    if len(site_ids) != len(probabilities):
        return jsonify({
            'success': False,
            'error': f'Length mismatch: {len(site_ids)} site_ids vs {len(probabilities)} probabilities'
        }), 400

    # Load explainability components
    components = load_explainability_components(SHAP_OUTPUT_DIR)
    if components is None:
        return jsonify({
            'success': False,
            'error': 'Explainability pipeline not available. Train a lookalike model first.'
        }), 404

    calibrator = components.get('calibrator')
    tier_classifier = components.get('tier_classifier')
    metadata = components.get('metadata', {})

    if calibrator is None or tier_classifier is None:
        return jsonify({
            'success': False,
            'error': 'Missing calibrator or tier classifier'
        }), 404

    # Load site feature data
    try:
        import pandas as pd
        details_df = load_site_details()

        # Filter to requested sites
        site_data = details_df[details_df['gtvid'].isin(site_ids)].copy()

        if site_data.empty:
            return jsonify({
                'success': False,
                'error': 'No matching sites found in site details'
            }), 404

        # Get feature names from metadata
        feature_names = metadata.get('feature_names', [])
        # Derive continuous features: first n_numeric features in the ordered list
        n_numeric = metadata.get('n_numeric', 0)
        continuous_features = metadata.get('continuous_features', feature_names[:n_numeric])

        if not feature_names:
            return jsonify({
                'success': False,
                'error': 'Feature names not found in metadata'
            }), 500

        # Load the trained model for counterfactual generation
        model_path = SHAP_OUTPUT_DIR / 'best_model.pt'
        if not model_path.exists():
            return jsonify({
                'success': False,
                'error': 'Trained model not found. Train a model first.'
            }), 404

        # Load sklearn wrapper for counterfactuals
        from site_scoring.explainability.conformal import SklearnModelWrapper
        import torch

        # Load the actual PyTorch model from checkpoint (contains config + weights)
        device = 'cpu'
        from site_scoring.model import SiteScoringModel
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        nn_config = checkpoint['config']
        # Load preprocessor to get categorical vocab sizes
        import pickle
        preprocessor_path = SHAP_OUTPUT_DIR / 'preprocessor.pkl'
        with open(preprocessor_path, 'rb') as pf:
            preprocessor_data = pickle.load(pf)
        model = SiteScoringModel.from_config(nn_config, preprocessor_data['categorical_vocab_sizes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        n_numeric = metadata.get('n_numeric', 0)
        n_categorical = metadata.get('n_categorical', 0)
        n_boolean = metadata.get('n_boolean', 0)

        # Create sklearn wrapper
        sklearn_model = SklearnModelWrapper(
            model=model,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            device=device,
        )

        # Prepare training data for counterfactual generator
        # (Uses feature ranges from training data)
        train_data = site_data[feature_names].copy() if all(f in site_data.columns for f in feature_names) else site_data.copy()

        # Start fleet analysis job
        job_id = start_fleet_analysis(
            model=sklearn_model,
            train_data=train_data,
            feature_names=feature_names,
            continuous_features=continuous_features,
            calibrator=calibrator,
            tier_classifier=tier_classifier,
            site_data=site_data,
            site_ids=site_ids,
            raw_probabilities=probabilities,
            n_counterfactuals=n_counterfactuals,
            n_clusters=n_clusters,
            output_dir=SHAP_OUTPUT_DIR / 'fleet_analysis',
        )

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Fleet analysis started. Analyzing {len(site_ids)} sites.',
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to start fleet analysis: {str(e)}'
        }), 500


@app.route('/api/explainability/fleet-analysis/status/<job_id>')
def api_fleet_analysis_status(job_id: str):
    """
    Get status of a fleet analysis job.

    Returns:
        JSON with:
        - job_id: str
        - status: 'pending' | 'running' | 'completed' | 'failed'
        - progress_pct: float
        - progress_message: str
        - results: dict (when completed)
    """
    status = get_fleet_analysis_status(job_id)

    if status is None:
        return jsonify({
            'error': f'Job {job_id} not found'
        }), 404

    return jsonify(status)


@app.route('/api/explainability/export-report/<job_id>')
def api_export_fleet_report(job_id: str):
    """
    Export fleet analysis results to Excel.

    Returns:
        Excel file download, or JSON error
    """
    from flask import send_file

    excel_path = export_fleet_analysis_to_excel(job_id)

    if excel_path is None:
        return jsonify({
            'error': 'Export failed. Job may not be complete or openpyxl not installed.'
        }), 400

    return send_file(
        excel_path,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'fleet_intervention_analysis_{job_id}.xlsx',
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == '__main__':
    import os

    # Only print startup banner once (not in reloader child process)
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

    if not is_reloader:
        print("=" * 60)
        print("Site Visualization Server")
        print("=" * 60)

    # Pre-load all data (in both processes for data availability)
    if not is_reloader:
        print("\nLoading data...")
    preload_all_data()

    if not is_reloader:
        print("\n" + "=" * 60)
        print("Starting Flask server...")
        print("Open http://localhost:8080 in your browser")
        print("=" * 60)

    app.run(debug=True, port=8080)