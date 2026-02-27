"""Explainability API routes — calibration, tiers, fleet analysis."""

from flask import Blueprint, jsonify, request

from src.services.data_service import load_site_details
from src.services.training_service import (
    load_explainability_components,
    explain_prediction,
)
from src.services.fleet_analysis_service import (
    start_fleet_analysis,
    get_fleet_analysis_status,
    export_fleet_analysis_to_excel,
)
from src.routes.shap import SHAP_OUTPUT_DIR

explainability_bp = Blueprint('explainability', __name__, url_prefix='/api')


@explainability_bp.route('/explainability/available')
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


@explainability_bp.route('/explainability/explain', methods=['POST'])
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


@explainability_bp.route('/explainability/explain-batch', methods=['POST'])
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

    calibrator = components.get('calibrator') if components else None
    tier_classifier = components.get('tier_classifier') if components else None

    # Always have a tier classifier — use default thresholds if none was fitted
    if tier_classifier is None:
        from site_scoring.explainability import TierClassifier
        tier_classifier = TierClassifier()

    import numpy as np
    is_calibrated = calibrator is not None
    results = {}

    for site_id, raw_prob in probabilities.items():
        try:
            raw_prob = float(raw_prob)
            calibrated_prob = float(calibrator.calibrate(np.array([raw_prob]))[0]) if is_calibrated else raw_prob
            tier_result = tier_classifier.classify(calibrated_prob)

            results[site_id] = {
                'raw_probability': raw_prob,
                'calibrated_probability': calibrated_prob,
                'is_calibrated': is_calibrated,
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


@explainability_bp.route('/explainability/tier-summary', methods=['POST'])
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

    calibrator = components.get('calibrator') if components else None
    tier_classifier = components.get('tier_classifier') if components else None

    # Always have a tier classifier — use default thresholds if none was fitted
    if tier_classifier is None:
        from site_scoring.explainability import TierClassifier
        tier_classifier = TierClassifier()

    import numpy as np

    # Calibrate all probabilities (or use raw if no calibrator)
    raw_probs = np.array([float(p) for p in probabilities])
    is_calibrated = calibrator is not None
    calibrated_probs = calibrator.calibrate(raw_probs) if is_calibrated else raw_probs

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
        'calibration_applied': is_calibrated,
    })


@explainability_bp.route('/explainability/fleet-analysis', methods=['POST'])
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


@explainability_bp.route('/explainability/fleet-analysis/status/<job_id>')
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


@explainability_bp.route('/explainability/export-report/<job_id>')
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
