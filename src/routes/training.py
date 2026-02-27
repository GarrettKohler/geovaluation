"""Training and experiment management API routes."""

from flask import Blueprint, Response, jsonify, request

from src.services.data_service import _clean_nan_values
from src.services.training_service import (
    get_system_info,
    start_training,
    stop_training,
    get_training_status,
    stream_training_progress,
    scan_experiment_folders,
)
from site_scoring.config import get_all_model_presets, get_all_available_features, DEFAULT_OUTPUT_DIR

training_bp = Blueprint('training', __name__, url_prefix='/api')


@training_bp.route('/training/system-info')
def api_get_system_info():
    """
    Get system information for training (GPU availability, etc.).

    Returns:
        JSON with PyTorch version, CUDA/MPS availability, recommended device.
    """
    return jsonify(get_system_info())


@training_bp.route('/training/presets')
def api_get_training_presets():
    """
    Get available model presets with feature counts.

    Returns:
        JSON dict mapping preset key to {name, description, counts}.
    """
    return jsonify(get_all_model_presets())


@training_bp.route('/training/features')
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


@training_bp.route('/training/all-features')
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


@training_bp.route('/training/start', methods=['POST'])
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


@training_bp.route('/training/stop', methods=['POST'])
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


@training_bp.route('/training/status')
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


@training_bp.route('/experiments')
def api_get_experiments():
    """
    Get a list of all experiments (active and history).
    """
    # This maps to get_training_status() with no args which returns all jobs
    return jsonify(get_training_status())


@training_bp.route('/experiments/catalog')
def api_get_catalog():
    """Persistent experiment catalog — survives Flask restarts.

    Scans all experiment folders on disk, parses config and metadata files,
    and returns a sorted list of experiments with their metrics, features,
    and artifact inventory.
    """
    catalog = scan_experiment_folders()
    return jsonify({"experiments": catalog})


@training_bp.route('/experiments/compare', methods=['POST'])
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


@training_bp.route('/training/stream')
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
