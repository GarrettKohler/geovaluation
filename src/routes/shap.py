"""SHAP feature importance API routes."""

from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request

from site_scoring.config import DEFAULT_OUTPUT_DIR
from src.services.shap_service import ShapCache, generate_shap_plots

shap_bp = Blueprint('shap', __name__, url_prefix='/api')

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


def _resolve_shap_directory(job_id: Optional[str]) -> Optional[Path]:
    """
    Resolve a job_id to its experiment directory, or fall back to latest.

    Returns None when a job_id is supplied but no matching experiment
    directory exists — callers should treat this as a 404 condition rather
    than silently serving the latest experiment's SHAP (which would defeat
    the per-experiment routing).
    """
    if job_id:
        candidate = DEFAULT_OUTPUT_DIR / "experiments" / job_id
        if candidate.is_dir():
            return candidate
        return None
    return get_latest_shap_directory()


@shap_bp.route('/shap/available')
def api_shap_available():
    """
    Check if SHAP data is available for a given experiment, or the most
    recent run if no job_id is supplied.

    Query Params:
        job_id: Experiment ID (e.g. job_1777670490_7d6d9c62). Optional.

    Returns:
        JSON with available flag and basic info.
    """
    job_id = request.args.get('job_id')
    shap_dir = _resolve_shap_directory(job_id)
    if shap_dir is None:
        return jsonify({'available': False, 'error': f'Experiment not found: {job_id}'}), 404
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


@shap_bp.route('/shap/summary')
def api_shap_summary():
    """
    Get SHAP feature importance summary data.

    Query Params:
        top_n: Number of top features to return (default: 30)
        job_id: Experiment ID. Optional — falls back to latest experiment.

    Returns:
        JSON with ranked feature importance list, base value, and sample counts.
    """
    top_n = request.args.get('top_n', 30, type=int)
    job_id = request.args.get('job_id')
    shap_dir = _resolve_shap_directory(job_id)
    if shap_dir is None:
        return jsonify({'error': f'Experiment not found: {job_id}'}), 404
    cache = ShapCache(shap_dir)
    result = cache.get_feature_importance(top_n=top_n)

    if result is None:
        return jsonify({'error': 'No SHAP data available for this experiment.'}), 404

    result['experiment_dir'] = shap_dir.name if shap_dir != DEFAULT_OUTPUT_DIR else None
    return jsonify(result)


@shap_bp.route('/shap/plots')
def api_shap_plots():
    """
    Get SHAP visualization plots as base64-encoded PNG images.

    Query Params:
        job_id: Experiment ID. Optional — falls back to latest experiment.

    Returns:
        JSON with bar_plot and summary_plot as base64 strings,
        or error if SHAP data/matplotlib unavailable.
    """
    job_id = request.args.get('job_id')
    shap_dir = _resolve_shap_directory(job_id)
    if shap_dir is None:
        return jsonify({'error': f'Experiment not found: {job_id}'}), 404
    plots = generate_shap_plots(shap_dir)
    if plots is None:
        return jsonify({'error': 'No SHAP plots available for this experiment.'}), 404
    return jsonify(plots)
