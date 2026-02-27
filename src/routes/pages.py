"""Page routes — template-rendering endpoints."""

from pathlib import Path

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


@pages_bp.route('/datasets')
def datasets_index():
    """Render the dataset index page showing pipeline-active source datasets."""
    from src.services.lineage_service import load_ontology, get_pipeline_datasets
    ontology = load_ontology(Path(__file__).parents[2] / "docs" / "data_ontology.yaml")
    datasets = get_pipeline_datasets(ontology)
    return render_template('datasets_index.html', datasets=datasets)


@pages_bp.route('/datasets/<dataset_id>')
def dataset_lineage(dataset_id):
    """Render the column lineage detail page for a specific dataset."""
    from src.services.lineage_service import load_ontology, get_dataset_info, get_lineage_for_dataset
    ontology = load_ontology(Path(__file__).parents[2] / "docs" / "data_ontology.yaml")
    dataset = get_dataset_info(ontology, dataset_id)
    if not dataset:
        return "Dataset not found", 404
    lineages = get_lineage_for_dataset(dataset_id)
    kept = [l for l in lineages if not l.dropped]
    dropped = [l for l in lineages if l.dropped]
    return render_template('dataset_lineage.html', dataset=dataset, kept=kept, dropped=dropped)
