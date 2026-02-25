"""Tests for the explainability module."""

import numpy as np
import pytest

from platinum1_1.explainability.calibration import ProbabilityCalibrator
from platinum1_1.explainability.tiers import TierClassifier, TierResult, TIER_THRESHOLDS
from platinum1_1.explainability.counterfactuals import IMMUTABLE_FEATURES, ACTIONABLE_FEATURES


# ---------------------------------------------------------------------------
# ProbabilityCalibrator
# ---------------------------------------------------------------------------

class TestProbabilityCalibrator:
    def test_isotonic_fit_and_calibrate(self):
        cal = ProbabilityCalibrator(method="isotonic")
        y_proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])

        cal.fit(y_proba, y_true)
        calibrated = cal.calibrate(y_proba)
        assert len(calibrated) == 5
        assert all(0 <= p <= 1 for p in calibrated)

    def test_platt_fit_and_calibrate(self):
        np.random.seed(42)
        n = 100
        y_proba = np.random.uniform(0, 1, n)
        y_true = (y_proba > 0.5).astype(int)

        cal = ProbabilityCalibrator(method="platt")
        cal.fit(y_proba, y_true)
        calibrated = cal.calibrate(y_proba)
        assert len(calibrated) == n
        assert all(0 <= p <= 1 for p in calibrated)

    def test_brier_score_tracked(self):
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])
        y_true = np.array([0, 0, 1, 1])

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_proba, y_true)

        assert cal.brier_score_before is not None
        assert cal.brier_score_after is not None

    def test_calibrate_before_fit_raises(self):
        cal = ProbabilityCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.calibrate(np.array([0.5]))

    def test_save_load_roundtrip(self, tmp_path):
        y_proba = np.random.uniform(0, 1, 50)
        y_true = (y_proba > 0.5).astype(int)

        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y_proba, y_true)

        path = tmp_path / "cal.pkl"
        cal.save(path)

        loaded = ProbabilityCalibrator.load(path)
        result = loaded.calibrate(y_proba)
        assert len(result) == 50

    def test_calibration_summary(self):
        y_proba = np.array([0.2, 0.5, 0.8])
        y_true = np.array([0, 1, 1])

        cal = ProbabilityCalibrator()
        assert cal.get_calibration_summary()["fitted"] is False

        cal.fit(y_proba, y_true)
        summary = cal.get_calibration_summary()
        assert summary["fitted"] is True
        assert summary["method"] == "isotonic"


# ---------------------------------------------------------------------------
# TierClassifier
# ---------------------------------------------------------------------------

class TestTierClassifier:
    def test_tier_1(self):
        tc = TierClassifier()
        result = tc.classify(0.90)
        assert result.tier == 1
        assert result.label == "Recommended"

    def test_tier_2(self):
        tc = TierClassifier()
        result = tc.classify(0.75)
        assert result.tier == 2
        assert result.label == "Promising"

    def test_tier_3(self):
        tc = TierClassifier()
        result = tc.classify(0.55)
        assert result.tier == 3
        assert result.label == "Review Required"

    def test_tier_4(self):
        tc = TierClassifier()
        result = tc.classify(0.30)
        assert result.tier == 4
        assert result.label == "Not Recommended"

    def test_boundary_values(self):
        """Test exact threshold boundaries."""
        tc = TierClassifier()
        assert tc.classify(0.85).tier == 1  # >= 0.85 is Tier 1
        assert tc.classify(0.65).tier == 2  # >= 0.65 is Tier 2
        assert tc.classify(0.50).tier == 3  # >= 0.50 is Tier 3
        assert tc.classify(0.49).tier == 4  # < 0.50 is Tier 4

    def test_confidence_statement(self):
        tc = TierClassifier()
        result = tc.classify(0.90)
        assert "out of 10" in result.confidence_statement

    def test_historical_accuracy(self):
        tc = TierClassifier()
        result = tc.classify(0.90)
        assert result.historical_accuracy == 0.88

    def test_tier_colors(self):
        tc = TierClassifier()
        result = tc.classify(0.90)
        assert result.color.startswith("#")

    def test_classify_batch(self):
        tc = TierClassifier()
        probs = np.array([0.90, 0.75, 0.55, 0.30])
        results = tc.classify_batch(probs)
        assert [r.tier for r in results] == [1, 2, 3, 4]

    def test_to_dict_roundtrip(self):
        tc = TierClassifier()
        d = tc.to_dict()
        loaded = TierClassifier.from_dict(d)
        assert loaded.thresholds == tc.thresholds
        assert loaded.historical_accuracy == tc.historical_accuracy

    def test_descending_thresholds_required(self):
        with pytest.raises(ValueError, match="descending"):
            TierClassifier(thresholds=[0.50, 0.65, 0.85])  # Wrong order


# ---------------------------------------------------------------------------
# Counterfactual constants
# ---------------------------------------------------------------------------

class TestCounterfactualConstants:
    def test_no_kroger_in_immutable(self):
        """Kroger should NOT be in immutable features for platinum1_1."""
        for f in IMMUTABLE_FEATURES:
            assert "kroger" not in f.lower(), f"Kroger found in IMMUTABLE: {f}"

    def test_actionable_features_exist(self):
        assert len(ACTIONABLE_FEATURES) > 0

    def test_no_overlap(self):
        """Actionable and immutable features should not overlap."""
        overlap = set(ACTIONABLE_FEATURES) & set(IMMUTABLE_FEATURES)
        assert len(overlap) == 0, f"Overlapping features: {overlap}"
