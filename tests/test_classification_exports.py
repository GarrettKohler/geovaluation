"""
Unit tests for classification (lookalike) training exports.

Tests the _export_classification_results() function that produces:
  1. training_sites.csv — all active sites with revenue and class labels
  2. test_predictions.csv — test split predictions with probabilities
  3. non_active_classification.csv — non-active sites scored by the model

These tests use mock data and do NOT require GPU hardware or trained models.
"""

import csv
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_dir(tmp_path):
    """Create a temporary experiment directory with required config files."""
    exp_dir = tmp_path / "test_experiment"
    exp_dir.mkdir()

    # Minimal config.json required by BatchPredictor
    config = {
        "model_type": "xgboost",
        "task_type": "lookalike",
        "target": "avg_monthly_revenue",
        "training_features": {
            "numeric": ["feature_a", "feature_b"],
            "categorical": ["cat_a"],
            "boolean": ["bool_a"],
        },
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f)

    return exp_dir


@pytest.fixture
def mock_processor():
    """Build a mock DataProcessor with known gtvids and revenues."""
    processor = MagicMock()

    # 20 sites total: some high revenue (label=1), some low (label=0)
    n_sites = 20
    processor.source_gtvids = [f"GTX{i:03d}" for i in range(n_sites)]
    processor.source_revenues = [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    ]
    processor.source_statuses = ["Active"] * n_sites

    # Revenue threshold at $1500 → sites 14-19 are top performers (label=1)
    processor.top_performer_threshold = 1500.0

    # Split indices: 14 train, 3 val, 3 test
    processor.train_indices = list(range(0, 14))
    processor.val_indices = list(range(14, 17))
    processor.test_indices = list(range(17, 20))

    return processor


@pytest.fixture
def mock_job(experiment_dir):
    """Build a mock TrainingJob."""
    job = MagicMock()
    job.job_id = "test_job_001"
    job.output_dir = experiment_dir
    job.config = MagicMock()
    job.config.target = "avg_monthly_revenue"
    return job


@pytest.fixture
def test_predictions():
    """Predicted probabilities for the 3 test sites (indices 17-19)."""
    return np.array([0.85, 0.92, 0.78])


@pytest.fixture
def test_targets():
    """Actual labels for the 3 test sites (indices 17-19 → revenues 1800, 1900, 2000 → all ≥ 1500)."""
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def mock_non_active_data():
    """Build mock non-active site data (Polars DataFrame mimic)."""
    import polars as pl

    return pl.DataFrame({
        "gtvid": ["NAX001", "NAX002", "NAX003", "NAX004"],
        "status": ["Temporarily Deactivated", "Removed", "Removed", "Temporarily Deactivated"],
        "avg_monthly_revenue": [200.0, 1800.0, 50.0, 1600.0],
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [5.0, 6.0, 7.0, 8.0],
        "cat_a": ["x", "y", "x", "y"],
        "bool_a": [True, False, True, False],
    })


@pytest.fixture
def run_export(mock_job, mock_processor, test_predictions, test_targets, mock_non_active_data):
    """Run the export function with mocked non-active site scoring."""
    from src.services.training_service import _export_classification_results

    mock_scores = {
        "NAX001": 0.15,
        "NAX002": 0.88,
        "NAX003": 0.05,
        "NAX004": 0.72,
    }

    with patch("site_scoring.data_transform.get_all_sites_for_prediction", return_value=mock_non_active_data), \
         patch("site_scoring.predict.BatchPredictor") as MockBP:

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = mock_scores
        MockBP.return_value = mock_predictor

        _export_classification_results(
            job=mock_job,
            processor=mock_processor,
            test_predictions=test_predictions,
            test_targets=test_targets,
            test_roc_auc=0.87,
        )

    return mock_job.output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingSitesCsv:
    """Tests for training_sites.csv export."""

    def test_training_sites_csv_columns(self, run_export):
        """Verify CSV has required columns: gtvid, avg_monthly_revenue, actual_label."""
        path = run_export / "training_sites.csv"
        assert path.exists(), "training_sites.csv was not created"

        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {"gtvid", "avg_monthly_revenue", "actual_label"}

    def test_training_sites_includes_all_active(self, run_export, mock_processor):
        """All source gtvids are present in the export."""
        path = run_export / "training_sites.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            exported_gtvids = {row["gtvid"] for row in reader}

        expected_gtvids = set(mock_processor.source_gtvids)
        assert exported_gtvids == expected_gtvids

    def test_actual_labels_are_binary_training(self, run_export):
        """actual_label is 0 or 1 in training_sites.csv."""
        path = run_export / "training_sites.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            labels = {row["actual_label"] for row in reader}

        assert labels.issubset({"0", "1"})


class TestTestPredictionsCsv:
    """Tests for test_predictions.csv export."""

    def test_test_predictions_csv_columns(self, run_export):
        """Verify CSV has required columns."""
        path = run_export / "test_predictions.csv"
        assert path.exists(), "test_predictions.csv was not created"

        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "gtvid", "predicted_probability", "actual_label", "model_roc_auc"
            }

    def test_roc_auc_in_test_predictions(self, run_export):
        """Each test prediction row has model_roc_auc = 0.87."""
        path = run_export / "test_predictions.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["model_roc_auc"]) == pytest.approx(0.87, abs=1e-6)

    def test_probabilities_in_range_test(self, run_export):
        """predicted_probability is between 0 and 1 in test_predictions.csv."""
        path = run_export / "test_predictions.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                prob = float(row["predicted_probability"])
                assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"

    def test_actual_labels_are_binary_test(self, run_export):
        """actual_label is 0 or 1 in test_predictions.csv."""
        path = run_export / "test_predictions.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            labels = {row["actual_label"] for row in reader}

        assert labels.issubset({"0", "1"})


class TestNonActiveCsv:
    """Tests for non_active_classification.csv export."""

    def test_non_active_csv_columns(self, run_export):
        """Verify non-active CSV has required columns including top_5000."""
        path = run_export / "non_active_classification.csv"
        assert path.exists(), "non_active_classification.csv was not created"

        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "gtvid", "status", "predicted_probability", "actual_label", "model_roc_auc", "top_5000"
            }

    def test_non_active_excludes_active_sites(self, run_export):
        """No Active-status sites in non-active export."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            statuses = {row["status"] for row in reader}

        assert "Active" not in statuses

    def test_probabilities_in_range_non_active(self, run_export):
        """predicted_probability is between 0 and 1 in non-active export."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                prob = float(row["predicted_probability"])
                assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"

    def test_actual_labels_are_binary_non_active(self, run_export):
        """actual_label is 0 or 1 in non_active_classification.csv."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            labels = {row["actual_label"] for row in reader}

        assert labels.issubset({"0", "1"})

    def test_top_5000_column_is_binary(self, run_export):
        """top_5000 is 0 or 1 in non_active_classification.csv."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            flags = {row["top_5000"] for row in reader}

        assert flags.issubset({"0", "1"})

    def test_top_5000_all_flagged_when_under_5000(self, run_export):
        """With only 4 non-active sites, all should be flagged as top_5000=1."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            flags = [row["top_5000"] for row in reader]

        assert all(f == "1" for f in flags), f"Expected all top_5000=1, got {flags}"

    def test_non_active_sorted_by_probability_descending(self, run_export):
        """non_active_classification.csv is sorted by predicted_probability descending."""
        path = run_export / "non_active_classification.csv"

        with open(path) as f:
            reader = csv.DictReader(f)
            probs = [float(row["predicted_probability"]) for row in reader]

        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], (
                f"Row {i} prob {probs[i]} < row {i+1} prob {probs[i+1]}"
            )
