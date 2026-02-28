"""Page routes — template-rendering endpoints."""

from flask import Blueprint, render_template

pages_bp = Blueprint('pages', __name__)


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
def shap_values():
    """Render the SHAP feature importance visualization page."""
    return render_template('shap_values.html')
