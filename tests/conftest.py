"""
Shared pytest fixtures for geospatial application tests.

Provides:
- Flask test client
- Sample data fixtures
- Mock training configurations
- Data service reset utilities
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def app():
    """Create Flask application for testing."""
    from app import app as flask_app
    flask_app.config.update({
        "TESTING": True,
    })
    return flask_app


@pytest.fixture(scope="session")
def client(app):
    """Flask test client for API testing."""
    return app.test_client()


@pytest.fixture(scope="session")
def preloaded_data():
    """
    Preload all data once for the test session.
    Returns tuple of (sites_df, revenue_metrics, site_details_df).
    """
    from src.services.data_service import (
        load_sites,
        load_revenue_metrics,
        load_site_details,
        get_filter_options,
    )

    sites = load_sites()
    metrics = load_revenue_metrics()
    details = load_site_details()
    filters = get_filter_options()

    return {
        "sites": sites,
        "metrics": metrics,
        "details": details,
        "filter_options": filters,
    }


@pytest.fixture
def sample_site_ids(preloaded_data):
    """Get a sample of valid site IDs for testing."""
    sites_df = preloaded_data["sites"]
    # Return first 10 site IDs
    return sites_df["GTVID"].head(10).tolist()


@pytest.fixture
def single_site_id(preloaded_data):
    """Get a single valid site ID for testing."""
    sites_df = preloaded_data["sites"]
    return sites_df["GTVID"].iloc[0]


@pytest.fixture
def training_config_minimal():
    """Minimal training configuration for quick tests."""
    return {
        "model_type": "neural_network",
        "target": "revenue",
        "epochs": 1,
        "batch_size": 512,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "hidden_layers": [64, 32],
        "device": "cpu",  # Use CPU for faster test setup
    }


@pytest.fixture
def model_output_dir():
    """Path to trained model outputs."""
    return Path(__file__).parent.parent / "site_scoring" / "outputs"


@pytest.fixture
def required_site_fields():
    """Fields required for frontend map rendering."""
    return [
        "GTVID",
        "Latitude",
        "Longitude",
        "revenueScore",
        "avgMonthlyRevenue",
        "totalRevenue",
        "activeMonths",
        "status",
    ]


@pytest.fixture
def categorical_filter_fields():
    """Categorical fields that should have filter options."""
    return [
        "State",
        "County",
        "DMA",
        "Retailer",
        "Network",
        "Hardware",
        "Experience",
        "Program",
        "Status",
        "Fuel Brand",
        "Restaurant",
        "C-Store",
    ]


def reset_data_caches():
    """Reset all data service caches for fresh loading."""
    from src.services import data_service
    data_service._sites_df = None
    data_service._revenue_metrics = None
    data_service._site_details_df = None
    data_service._unique_values_cache = None
