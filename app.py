"""
Flask application for visualizing geospatial site data with ML scoring.

Routes are organized into Blueprints under src/routes/:
- pages: Template-rendering page routes
- sites: Site data and filtering API
- training: Model training and experiment management
- prediction: Batch prediction, export, filtered scoring
- shap: SHAP feature importance
- explainability: Calibration, tiers, fleet analysis

Run with: python app.py
Then open http://localhost:8080 in your browser.
"""

from flask import Flask
from src.routes import register_blueprints
from src.services.data_service import preload_all_data

app = Flask(__name__)
register_blueprints(app)

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
