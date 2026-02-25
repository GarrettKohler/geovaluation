"""
Probability Calibration for Site Scoring Model.

Ensures predicted probabilities match observed frequencies. When the model
outputs 75% confidence, calibration ensures ~75% of similar predictions
were actually correct.

Methods:
- Isotonic Regression: Non-parametric, flexible calibration curve
- Platt Scaling: Logistic regression on model outputs (parametric)

Reference: Niculescu-Mizil & Caruana (2005) - "Predicting Good Probabilities"
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from typing import Literal, Optional, Tuple
import pickle
from pathlib import Path


class ProbabilityCalibrator:
    """
    Calibrates model probability outputs to match true frequencies.

    Args:
        method: 'isotonic' (default, non-parametric) or 'platt' (logistic)
    """

    def __init__(self, method: Literal['isotonic', 'platt'] = 'isotonic'):
        self.method = method
        self._fitted = False

        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            self.calibrator = LogisticRegression(solver='lbfgs')
        else:
            raise ValueError(f"Unknown method: {method}. Use 'isotonic' or 'platt'")

        self.brier_score_before: Optional[float] = None
        self.brier_score_after: Optional[float] = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on held-out calibration data.

        IMPORTANT: Use a separate calibration set not used for training.
        """
        y_proba = np.asarray(y_proba).ravel()
        y_true = np.asarray(y_true).ravel()

        if len(y_proba) != len(y_true):
            raise ValueError(f"Length mismatch: y_proba={len(y_proba)}, y_true={len(y_true)}")

        self.brier_score_before = brier_score_loss(y_true, y_proba)

        if self.method == 'isotonic':
            self.calibrator.fit(y_proba, y_true)
        else:
            self.calibrator.fit(y_proba.reshape(-1, 1), y_true)

        self._fitted = True
        y_calibrated = self.calibrate(y_proba)
        self.brier_score_after = brier_score_loss(y_true, y_calibrated)

        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_proba = np.asarray(y_proba).ravel()

        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(y_proba)
        else:
            calibrated = self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]

        return np.clip(calibrated, 0, 1)

    def get_reliability_data(
        self, y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute data for reliability diagram (calibration curve)."""
        y_proba = np.asarray(y_proba).ravel()
        y_true = np.asarray(y_true).ravel()

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins[1:-1])

        bin_means, observed_freqs, bin_counts = [], [], []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(y_proba[mask].mean())
                observed_freqs.append(y_true[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_means.append(np.nan)
                observed_freqs.append(np.nan)
                bin_counts.append(0)

        return np.array(bin_means), np.array(observed_freqs), np.array(bin_counts)

    def get_expected_calibration_error(
        self, y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE). Lower is better."""
        bin_means, observed_freqs, bin_counts = self.get_reliability_data(y_proba, y_true, n_bins)

        valid = ~np.isnan(bin_means) & ~np.isnan(observed_freqs)
        if not valid.any():
            return np.nan

        bin_means = bin_means[valid]
        observed_freqs = observed_freqs[valid]
        bin_counts = np.array(bin_counts)[valid]

        total_samples = bin_counts.sum()
        ece = np.sum(bin_counts * np.abs(observed_freqs - bin_means)) / total_samples
        return float(ece)

    def get_calibration_summary(self) -> dict:
        """Get summary of calibration performance."""
        if not self._fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "method": self.method,
            "brier_score_before": self.brier_score_before,
            "brier_score_after": self.brier_score_after,
            "improvement": (
                (self.brier_score_before - self.brier_score_after) / self.brier_score_before
                if self.brier_score_before and self.brier_score_before > 0
                else 0
            ),
        }

    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        with open(Path(path), 'wb') as f:
            pickle.dump({
                'method': self.method,
                'calibrator': self.calibrator,
                'fitted': self._fitted,
                'brier_score_before': self.brier_score_before,
                'brier_score_after': self.brier_score_after,
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'ProbabilityCalibrator':
        """Load calibrator from disk."""
        with open(Path(path), 'rb') as f:
            data = pickle.load(f)

        instance = cls(method=data['method'])
        instance.calibrator = data['calibrator']
        instance._fitted = data['fitted']
        instance.brier_score_before = data['brier_score_before']
        instance.brier_score_after = data['brier_score_after']
        return instance
