"""Batch prediction, export, and filtered scoring API routes."""

from pathlib import Path

from flask import Blueprint, jsonify, request

from src.services.data_service import get_filtered_site_ids, _clean_nan_values
from site_scoring.config import DEFAULT_OUTPUT_DIR

prediction_bp = Blueprint('prediction', __name__, url_prefix='/api')

# Module-level cache for BatchPredictor (reloads only when experiment changes)
_cached_predictor = None
_cached_predictor_experiment = None


def _find_latest_experiment() -> Path:
    """Find the most recent experiment directory with a complete model.

    @glossary: productionizing/experiment-discovery
    @title: Experiment Discovery
    @step: 0
    @color: accent
    @sub: Find the most recent valid experiment by modification time and
        artifact validation
    @analogy: Before scoring sites, we need to pick which trained model
        to use. The system sorts experiment folders by modification time
        (newest first) and validates each one has the required artifacts:
        config.json, preprocessor.pkl, and either best_model.pt or
        model_wrapper.pkl.
    @why: An experiment is only valid if it has all three artifact types.
        Partially-saved experiments (e.g., training was interrupted) are
        skipped automatically.
    """
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


@prediction_bp.route('/predict/batch', methods=['POST'])
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


@prediction_bp.route('/predict/export', methods=['POST'])
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


@prediction_bp.route('/predict/filtered', methods=['POST'])
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

    @glossary: productionizing/filtered-scoring
    @title: Filtered Scoring
    @step: 4
    @color: orange
    @sub: Apply map filter state (network, state, status) to score a
        subset of sites
    @analogy: Instead of scoring all 57,000 sites, filtered scoring lets
        users focus on what they care about. "Show me just the Wayne
        network sites in Texas" scores only that subset, returning
        results with summary statistics in under 2 seconds.
    @why: Filters support multi-select on network, state, and status,
        plus min/max revenue bounds and explicit gtvid lists. The filter
        is applied to the prediction DataFrame before scoring, not
        after, for efficiency.
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
