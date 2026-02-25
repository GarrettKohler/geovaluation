"""Tests for the experiment catalog scanner.

Verifies that scan_experiment_folders() correctly reads experiment
directories, parses config.json/model_metadata.json, handles edge
cases (missing files, corrupt JSON, empty dirs), and caches results.
"""

import json
import os
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers to create mock experiment folders
# ---------------------------------------------------------------------------

def _make_experiment(tmp_path, job_id, config=None, metadata=None, artifacts=None):
    """Create a mock experiment folder with specified artifacts."""
    exp_dir = tmp_path / "experiments" / job_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    if config is not None:
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f)

    if metadata is not None:
        with open(exp_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f)

    for artifact in (artifacts or []):
        (exp_dir / artifact).write_text("")

    return exp_dir


def _minimal_config(**overrides):
    """Return a minimal config.json dict with sensible defaults."""
    cfg = {
        "model_type": "neural_network",
        "task_type": "regression",
        "target": "avg_monthly_revenue",
        "epochs": 50,
        "network_filter": None,
        "lookalike_lower_percentile": 90,
        "lookalike_upper_percentile": 100,
        "training_features": {
            "numeric": ["feat_a", "feat_b"],
            "categorical": ["cat_a"],
            "boolean": ["bool_a", "bool_b", "bool_c"],
        },
    }
    cfg.update(overrides)
    return cfg


def _minimal_metadata(**overrides):
    """Return a minimal model_metadata.json dict."""
    meta = {
        "model_type": "neural_network",
        "task_type": "regression",
        "test_metrics": {
            "test_r2": 0.85,
            "test_mae": 120.5,
            "test_rmse": 180.0,
            "test_smape": 25.0,
            "test_mape": 30.0,
            "test_f1": 0.0,
            "test_logloss": 0.0,
        },
    }
    meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseExperimentFolder:
    """Tests for _parse_experiment_folder() helper."""

    def test_missing_config_returns_none(self, tmp_path):
        """Folder without config.json should be skipped."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = tmp_path / "experiments" / "job_1234_abcd"
        exp_dir.mkdir(parents=True)
        (exp_dir / "best_model.pt").write_text("")

        assert _parse_experiment_folder(exp_dir) is None

    def test_corrupt_json_returns_none(self, tmp_path):
        """Corrupt config.json should be skipped gracefully."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = tmp_path / "experiments" / "job_1234_abcd"
        exp_dir.mkdir(parents=True)
        (exp_dir / "config.json").write_text("{invalid json!!")

        assert _parse_experiment_folder(exp_dir) is None

    def test_complete_nn_experiment(self, tmp_path):
        """Complete NN experiment with config + metadata + model + preprocessor."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771882528_ae3755b1",
            config=_minimal_config(),
            metadata=_minimal_metadata(),
            artifacts=["best_model.pt", "preprocessor.pkl", "shap_cache.npz"],
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry is not None
        assert entry["job_id"] == "job_1771882528_ae3755b1"
        assert entry["model_type"] == "neural_network"
        assert entry["task_type"] == "regression"
        assert entry["is_complete"] is True
        assert entry["has_shap"] is True
        assert entry["has_predictions"] is False
        assert entry["test_metrics"]["test_r2"] == 0.85
        assert "config.json" in entry["artifacts"]
        assert "best_model.pt" in entry["artifacts"]
        assert "preprocessor.pkl" in entry["artifacts"]

    def test_complete_xgboost_experiment(self, tmp_path):
        """Complete XGBoost experiment with model_wrapper.pkl instead of best_model.pt."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771881740_ed065fc5",
            config=_minimal_config(model_type="xgboost"),
            metadata=_minimal_metadata(model_type="xgboost"),
            artifacts=["model_wrapper.pkl", "preprocessor.pkl", "best_model.json"],
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry is not None
        assert entry["model_type"] == "xgboost"
        assert entry["is_complete"] is True
        assert "model_wrapper.pkl" in entry["artifacts"]

    def test_incomplete_experiment(self, tmp_path):
        """Experiment with only config.json should be marked incomplete."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771273203_1ba0517a",
            config=_minimal_config(),
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry is not None
        assert entry["is_complete"] is False
        assert entry["test_metrics"] == {}

    def test_created_at_parsed_from_job_id(self, tmp_path):
        """Unix timestamp in job_id is correctly converted to ISO datetime."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771882528_ae3755b1",
            config=_minimal_config(),
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry["created_at"] is not None
        # 1771882528 = 2026-02-24T11:35:28Z (approx, depends on timezone)
        assert "2026" in entry["created_at"]

    def test_feature_count_from_config(self, tmp_path):
        """Feature counts are correctly extracted from training_features."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771882528_ae3755b1",
            config=_minimal_config(),
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry["feature_count"]["numeric"] == 2
        assert entry["feature_count"]["categorical"] == 1
        assert entry["feature_count"]["boolean"] == 3

    def test_artifacts_list_matches_files(self, tmp_path):
        """Artifacts list should exactly match files present in folder."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771882528_ae3755b1",
            config=_minimal_config(),
            metadata=_minimal_metadata(),
            artifacts=["best_model.pt", "preprocessor.pkl", "test_predictions.csv",
                        "training_sites.csv", "shap_cache.npz", "shap_importance.json"],
        )

        entry = _parse_experiment_folder(exp_dir)
        # config.json and model_metadata.json are also created by _make_experiment
        expected = {
            "config.json", "model_metadata.json", "best_model.pt",
            "preprocessor.pkl", "test_predictions.csv", "training_sites.csv",
            "shap_cache.npz", "shap_importance.json",
        }
        assert set(entry["artifacts"]) == expected

    def test_lookalike_config_fields(self, tmp_path):
        """Lookalike percentile bounds are included in catalog entry."""
        from src.services.training_service import _parse_experiment_folder

        exp_dir = _make_experiment(
            tmp_path,
            "job_1771882528_ae3755b1",
            config=_minimal_config(
                task_type="lookalike",
                lookalike_lower_percentile=80,
                lookalike_upper_percentile=95,
            ),
        )

        entry = _parse_experiment_folder(exp_dir)
        assert entry["task_type"] == "lookalike"
        assert entry["lookalike_lower_percentile"] == 80
        assert entry["lookalike_upper_percentile"] == 95


class TestScanExperimentFolders:
    """Tests for scan_experiment_folders() function."""

    def test_scan_empty_dir(self, tmp_path, monkeypatch):
        """Empty experiments directory returns empty list."""
        from src.services import training_service

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result = training_service.scan_experiment_folders()
        assert result == []

    def test_scan_nonexistent_dir(self, tmp_path, monkeypatch):
        """Nonexistent experiments directory returns empty list."""
        from src.services import training_service

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result = training_service.scan_experiment_folders()
        assert result == []

    def test_scan_multiple_experiments(self, tmp_path, monkeypatch):
        """Multiple experiment folders are returned, sorted newest first."""
        from src.services import training_service

        _make_experiment(tmp_path, "job_1000000000_aaa", config=_minimal_config())
        _make_experiment(tmp_path, "job_1771882528_bbb", config=_minimal_config())
        _make_experiment(tmp_path, "job_1500000000_ccc", config=_minimal_config())

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result = training_service.scan_experiment_folders()
        assert len(result) == 3
        # Newest first
        assert result[0]["job_id"] == "job_1771882528_bbb"
        assert result[1]["job_id"] == "job_1500000000_ccc"
        assert result[2]["job_id"] == "job_1000000000_aaa"

    def test_skips_non_job_dirs(self, tmp_path, monkeypatch):
        """Directories not starting with 'job_' are skipped."""
        from src.services import training_service

        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        (experiments_dir / ".DS_Store").write_text("")
        (experiments_dir / "random_folder").mkdir()
        _make_experiment(tmp_path, "job_1771882528_aaa", config=_minimal_config())

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result = training_service.scan_experiment_folders()
        assert len(result) == 1

    def test_cache_returns_same_object(self, tmp_path, monkeypatch):
        """Scanning twice without changes returns cached result."""
        from src.services import training_service

        _make_experiment(tmp_path, "job_1771882528_aaa", config=_minimal_config())

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result1 = training_service.scan_experiment_folders()
        result2 = training_service.scan_experiment_folders()
        assert result1 is result2  # Same object (cached)

    def test_cache_invalidation_on_dir_change(self, tmp_path, monkeypatch):
        """Adding a new experiment invalidates the cache."""
        from src.services import training_service

        _make_experiment(tmp_path, "job_1771882528_aaa", config=_minimal_config())

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result1 = training_service.scan_experiment_folders()
        assert len(result1) == 1

        # Add a new experiment (changes dir mtime)
        time.sleep(0.05)  # Ensure mtime changes
        _make_experiment(tmp_path, "job_1771882529_bbb", config=_minimal_config())

        result2 = training_service.scan_experiment_folders()
        assert len(result2) == 2
        assert result1 is not result2  # New object (cache invalidated)

    def test_graceful_on_corrupt_json(self, tmp_path, monkeypatch):
        """Corrupt config.json is skipped; other experiments still returned."""
        from src.services import training_service

        # Good experiment
        _make_experiment(tmp_path, "job_1771882528_aaa", config=_minimal_config())

        # Corrupt experiment
        exp_dir = tmp_path / "experiments" / "job_1771882529_bbb"
        exp_dir.mkdir(parents=True)
        (exp_dir / "config.json").write_text("NOT JSON {{{")

        monkeypatch.setattr(training_service, "DEFAULT_OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(training_service, "_catalog_cache", None)
        monkeypatch.setattr(training_service, "_catalog_cache_mtime", 0.0)

        result = training_service.scan_experiment_folders()
        assert len(result) == 1
        assert result[0]["job_id"] == "job_1771882528_aaa"


class TestApiEndpoint:
    """Test the /api/experiments/catalog Flask route."""

    def test_catalog_endpoint_returns_200(self):
        """GET /api/experiments/catalog returns 200 with experiments list."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from app import app

        with app.test_client() as client:
            response = client.get('/api/experiments/catalog')
            assert response.status_code == 200
            data = response.get_json()
            assert "experiments" in data
            assert isinstance(data["experiments"], list)
