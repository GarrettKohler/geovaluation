"""Page routes — template-rendering endpoints and health probes."""

from flask import Blueprint, jsonify, render_template

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/health')
def health():
    """Liveness probe — confirms the process is alive and Flask can serve."""
    return jsonify({"status": "ok"})


@pages_bp.route('/ready')
def ready():
    """Readiness probe — checks if data caches are populated."""
    from src.services import data_service
    data_loaded = data_service._sites_df is not None
    if not data_loaded:
        return jsonify({"status": "not_ready", "data_loaded": False}), 503
    return jsonify({"status": "ready", "data_loaded": True})


@pages_bp.route('/')
def home():
    """Render the experiment hub home page."""
    return render_template('home.html')


@pages_bp.route('/map')
@pages_bp.route('/map/<job_id>')
def map_view(job_id=None):
    """Render the map visualization page."""
    return render_template('index.html', job_id=job_id)


@pages_bp.route('/training-details')
def training_details():
    """Render the training details page with site records."""
    return render_template('training_details.html')


@pages_bp.route('/glossary')
def glossary():
    """Render the ML/statistics glossary page."""
    return render_template('glossary.html')


@pages_bp.route('/shap-values')
@pages_bp.route('/shap-values/<job_id>')
def shap_values(job_id=None):
    """Render the SHAP feature importance visualization page.

    With no job_id, the page loads SHAP for the most recent experiment that
    has a shap_cache.npz. With a job_id, it loads SHAP for that specific
    experiment (or shows an error if SHAP wasn't computed for it).
    """
    return render_template('shap_values.html', job_id=job_id)
