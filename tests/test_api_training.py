"""
Tests for training API endpoints.

Verifies training system info, job management, and SSE streaming.
"""

import pytest
import json
import time


class TestSystemInfoAPI:
    """Tests for /api/training/system-info endpoint."""

    def test_system_info_returns_200(self, client):
        """API returns 200 OK."""
        response = client.get("/api/training/system-info")
        assert response.status_code == 200

    def test_system_info_has_pytorch_version(self, client):
        """System info includes PyTorch version."""
        response = client.get("/api/training/system-info")
        data = response.get_json()

        assert "pytorch_version" in data
        assert data["pytorch_version"] is not None

    def test_system_info_has_device_availability(self, client):
        """System info includes device availability flags."""
        response = client.get("/api/training/system-info")
        data = response.get_json()

        assert "cuda_available" in data
        assert "mps_available" in data
        assert "recommended_device" in data

        # recommended_device should be one of cuda, mps, or cpu
        assert data["recommended_device"] in ["cuda", "mps", "cpu"]

    def test_system_info_includes_chip_detection(self, client):
        """On Apple Silicon, system info includes chip details."""
        response = client.get("/api/training/system-info")
        data = response.get_json()

        if data.get("mps_available"):
            # Should include Apple Silicon chip info
            assert "detected_chip" in data or "chip_name" in data


class TestTrainingStatusAPI:
    """Tests for /api/training/status endpoint."""

    def test_status_no_job_returns_no_job(self, client):
        """Status endpoint indicates when no job exists."""
        # First stop any running job
        client.post("/api/training/stop")
        time.sleep(0.5)

        response = client.get("/api/training/status")
        data = response.get_json()

        # Either no_job status or not running
        assert data.get("status") == "no_job" or data.get("is_running") is False


class TestTrainingStartAPI:
    """Tests for /api/training/start endpoint."""

    def test_start_training_returns_200(self, client, training_config_minimal):
        """API returns 200 OK when starting training."""
        # Ensure no job is running first
        client.post("/api/training/stop")
        time.sleep(1)

        response = client.post(
            "/api/training/start",
            json=training_config_minimal,
            content_type="application/json",
        )

        # Should return 200 (success) or 400 (already running)
        assert response.status_code in [200, 400]

    def test_start_training_returns_job_id(self, client, training_config_minimal):
        """Successful start returns a job ID."""
        client.post("/api/training/stop")
        time.sleep(1)

        response = client.post(
            "/api/training/start",
            json=training_config_minimal,
            content_type="application/json",
        )

        if response.status_code == 200:
            data = response.get_json()
            assert "success" in data
            if data["success"]:
                assert "job_id" in data

        # Clean up
        client.post("/api/training/stop")

    def test_cannot_start_duplicate_training(self, client, training_config_minimal):
        """Cannot start training when job already running."""
        client.post("/api/training/stop")
        time.sleep(1)

        # Start first job
        first_response = client.post(
            "/api/training/start",
            json=training_config_minimal,
            content_type="application/json",
        )

        if first_response.status_code == 200 and first_response.get_json().get("success"):
            # Try to start second job immediately
            second_response = client.post(
                "/api/training/start",
                json=training_config_minimal,
                content_type="application/json",
            )

            # Second should fail
            data = second_response.get_json()
            assert data.get("success") is False or second_response.status_code == 400

        # Clean up
        client.post("/api/training/stop")


class TestTrainingStopAPI:
    """Tests for /api/training/stop endpoint."""

    def test_stop_training_returns_200(self, client):
        """API returns 200 OK for stop request."""
        response = client.post("/api/training/stop")
        assert response.status_code == 200

    def test_stop_returns_message(self, client):
        """Stop endpoint returns success status and message."""
        response = client.post("/api/training/stop")
        data = response.get_json()

        assert "success" in data
        assert "message" in data


class TestTrainingStreamAPI:
    """Tests for /api/training/stream SSE endpoint."""

    def test_stream_endpoint_returns_sse(self, client):
        """Stream endpoint returns server-sent events content type."""
        response = client.get("/api/training/stream")

        # Should be event-stream content type
        assert "text/event-stream" in response.content_type

    def test_stream_headers_correct(self, client):
        """Stream response has correct headers for SSE."""
        response = client.get("/api/training/stream")

        # Check cache control
        assert response.headers.get("Cache-Control") == "no-cache"

    def test_stream_returns_data(self, client):
        """Stream returns SSE data format."""
        response = client.get("/api/training/stream")

        # Get raw response data
        data = response.data.decode("utf-8")

        # SSE format starts with "data: "
        assert "data:" in data


class TestTrainingConfigValidation:
    """Tests for training configuration validation."""

    def test_accepts_valid_model_type(self, client):
        """API accepts valid model type."""
        client.post("/api/training/stop")
        time.sleep(0.5)

        config = {
            "model_type": "neural_network",
            "epochs": 1,
            "device": "cpu",
        }

        response = client.post(
            "/api/training/start",
            json=config,
            content_type="application/json",
        )

        # Should not error on config parsing
        assert response.status_code in [200, 400]  # 400 only if job running

        client.post("/api/training/stop")

    def test_accepts_different_targets(self, client):
        """API accepts different target variables."""
        targets = ["revenue", "monthly_impressions"]

        for target in targets:
            client.post("/api/training/stop")
            time.sleep(0.5)

            config = {
                "target": target,
                "epochs": 1,
                "device": "cpu",
            }

            response = client.post(
                "/api/training/start",
                json=config,
                content_type="application/json",
            )

            # Should handle all valid targets
            assert response.status_code in [200, 400]

            client.post("/api/training/stop")
            time.sleep(0.5)
