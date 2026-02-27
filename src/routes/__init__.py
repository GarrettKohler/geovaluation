"""
Blueprint registration for the Flask application.

Organizes routes into domain-specific modules:
- pages: Template-rendering page routes
- sites: Site data and filtering API
- training: Model training and experiment management
- prediction: Batch prediction, export, filtered scoring
- shap: SHAP feature importance
- explainability: Calibration, tiers, fleet analysis
"""

from src.routes.pages import pages_bp
from src.routes.sites import sites_bp
from src.routes.training import training_bp
from src.routes.prediction import prediction_bp
from src.routes.shap import shap_bp
from src.routes.explainability import explainability_bp


def register_blueprints(app):
    """Register all route blueprints on the Flask app."""
    app.register_blueprint(pages_bp)
    app.register_blueprint(sites_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(shap_bp)
    app.register_blueprint(explainability_bp)
