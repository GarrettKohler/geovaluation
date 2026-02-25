"""Tests for the FastAPI application (app.py) and schemas."""

import pytest
from fastapi.testclient import TestClient

from platinum1_1.app import app
from platinum1_1.api.schemas import (
    ModelType,
    TaskType,
    TrainingRequest,
    TrainingResponse,
    SystemInfoResponse,
    FeatureListResponse,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

class TestSystemInfo:
    def test_system_info_endpoint(self, client):
        response = client.get("/api/training/system-info")
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "mps_available" in data
        assert "cuda_available" in data


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeaturesEndpoint:
    def test_features_endpoint(self, client):
        response = client.get("/api/training/features")
        assert response.status_code == 200
        data = response.json()
        assert "numeric" in data
        assert "categorical" in data
        assert "boolean" in data
        assert data["total"] > 0
        assert len(data["numeric"]) > 0

    def test_no_kroger_in_features(self, client):
        response = client.get("/api/training/features")
        data = response.json()
        all_features = data["numeric"] + data["categorical"] + data["boolean"]
        for f in all_features:
            assert "kroger" not in f.lower()


# ---------------------------------------------------------------------------
# Experiments (empty state)
# ---------------------------------------------------------------------------

class TestExperimentsEndpoint:
    def test_list_experiments_empty(self, client):
        response = client.get("/api/experiments")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_training_request_defaults(self):
        req = TrainingRequest()
        assert req.model_type == ModelType.NEURAL_NETWORK
        assert req.task_type == TaskType.REGRESSION
        assert req.target == "avg_monthly_revenue"
        assert req.epochs == 50
        assert req.batch_size == 4096

    def test_training_request_validation(self):
        """Epochs must be >= 1."""
        with pytest.raises(Exception):
            TrainingRequest(epochs=0)

    def test_training_response(self):
        resp = TrainingResponse(job_id="abc", status="pending", message="ok")
        assert resp.job_id == "abc"

    def test_system_info_response(self):
        resp = SystemInfoResponse(
            device="cpu", mps_available=False, cuda_available=False
        )
        assert resp.device == "cpu"

    def test_feature_list_response(self):
        resp = FeatureListResponse(
            numeric=["a", "b"], categorical=["c"], boolean=["d"], total=4
        )
        assert resp.total == 4
