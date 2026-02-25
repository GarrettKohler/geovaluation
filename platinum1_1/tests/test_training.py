"""Tests for training/progress.py and training/experiment.py."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from platinum1_1.training.progress import TrainingProgress, sanitize_for_json
from platinum1_1.training.experiment import ExperimentManager


# ---------------------------------------------------------------------------
# sanitize_for_json
# ---------------------------------------------------------------------------

class TestSanitizeForJson:
    def test_replaces_inf(self):
        assert sanitize_for_json(float("inf")) is None
        assert sanitize_for_json(float("-inf")) is None

    def test_replaces_nan(self):
        assert sanitize_for_json(float("nan")) is None

    def test_normal_float(self):
        assert sanitize_for_json(3.14) == 3.14

    def test_nested_dict(self):
        data = {"a": 1.0, "b": float("inf"), "c": {"d": float("nan")}}
        result = sanitize_for_json(data)
        assert result == {"a": 1.0, "b": None, "c": {"d": None}}

    def test_nested_list(self):
        data = [1.0, float("inf"), [float("nan"), 2.0]]
        result = sanitize_for_json(data)
        assert result == [1.0, None, [None, 2.0]]

    def test_numpy_scalar(self):
        val = np.float64(3.14)
        assert sanitize_for_json(val) == pytest.approx(3.14)

    def test_numpy_inf(self):
        assert sanitize_for_json(np.float64("inf")) is None

    def test_numpy_integer(self):
        assert sanitize_for_json(np.int64(42)) == 42

    def test_numpy_array(self):
        arr = np.array([1.0, float("inf"), 3.0])
        result = sanitize_for_json(arr)
        assert result == [1.0, None, 3.0]

    def test_result_is_json_serializable(self):
        """Final result must be serializable by json.dumps (no Infinity)."""
        data = {
            "loss": float("inf"),
            "metrics": {"r2": float("-inf"), "mae": float("nan")},
            "history": [float("inf"), 1.0, float("nan")],
        }
        sanitized = sanitize_for_json(data)
        # Should not raise
        json_str = json.dumps(sanitized)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str


# ---------------------------------------------------------------------------
# TrainingProgress
# ---------------------------------------------------------------------------

class TestTrainingProgress:
    def test_default_values(self):
        p = TrainingProgress()
        assert p.status == "running"
        assert p.epoch == 0

    def test_to_sse_event(self):
        p = TrainingProgress(epoch=5, total_epochs=50, train_loss=0.5)
        event = p.to_sse_event()
        assert event.startswith("event: progress\ndata: ")
        assert '"epoch": 5' in event

    def test_to_sse_event_sanitizes_inf(self):
        p = TrainingProgress(best_val_loss=float("inf"))
        event = p.to_sse_event()
        assert "Infinity" not in event

    def test_to_dict(self):
        p = TrainingProgress(epoch=1, train_loss=0.3, val_loss=0.4)
        d = p.to_dict()
        assert d["epoch"] == 1
        assert d["train_loss"] == 0.3


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------

class TestExperimentManager:
    @pytest.fixture
    def manager(self):
        tmp = Path(tempfile.mkdtemp())
        mgr = ExperimentManager(base_dir=tmp, max_experiments=3)
        yield mgr
        shutil.rmtree(tmp, ignore_errors=True)

    def test_create_experiment(self, manager):
        exp_dir = manager.create_experiment("test_001")
        assert exp_dir.exists()
        assert (exp_dir / "metadata.json").exists()

        meta = json.loads((exp_dir / "metadata.json").read_text())
        assert meta["job_id"] == "test_001"
        assert meta["status"] == "running"

    def test_complete_experiment(self, manager):
        manager.create_experiment("test_002")
        manager.complete_experiment("test_002", {"r2": 0.85, "mae": 100})

        exp = manager.get_experiment("test_002")
        assert exp["status"] == "completed"
        assert exp["metrics"]["r2"] == 0.85

    def test_fail_experiment(self, manager):
        manager.create_experiment("test_003")
        manager.fail_experiment("test_003", "OOM error")

        exp = manager.get_experiment("test_003")
        assert exp["status"] == "failed"
        assert "OOM" in exp["error"]

    def test_fifo_cleanup(self, manager):
        """FIFO: creating 4th experiment should delete oldest."""
        for i in range(1, 5):
            manager.create_experiment(f"exp_{i:03d}")

        experiments = manager.list_experiments()
        job_ids = [e["job_id"] for e in experiments]
        assert "exp_001" not in job_ids  # Oldest should be gone
        assert len(experiments) <= 3

    def test_list_experiments(self, manager):
        manager.create_experiment("a")
        manager.create_experiment("b")

        exps = manager.list_experiments()
        assert len(exps) == 2

    def test_get_nonexistent(self, manager):
        assert manager.get_experiment("nonexistent") is None
