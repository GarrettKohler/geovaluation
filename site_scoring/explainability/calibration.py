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

    Why Calibration Matters:
    ------------------------
    Neural networks (especially with softmax) often produce overconfident
    predictions. Calibration maps raw scores to probabilities that reflect
    actual observed outcomes.

    Example: If calibrated probability is 0.70, then historically ~70% of
    sites with this score succeeded.

    Args:
        method: 'isotonic' (default, non-parametric) or 'platt' (logistic)

    Isotonic vs Platt:
    - Isotonic: More flexible, handles complex calibration curves, needs
      more data (1000+ samples recommended)
    - Platt: Parametric (2 params), works with less data but assumes
      sigmoid relationship between scores and true probabilities
    """

    def __init__(self, method: Literal['isotonic', 'platt'] = 'isotonic'):
        self.method = method
        self._fitted = False

        if method == 'isotonic':
            # out_of_bounds='clip' ensures predictions stay in [0,1]
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'platt':
            # Platt scaling uses logistic regression on raw scores
            self.calibrator = LogisticRegression(solver='lbfgs')
        else:
            raise ValueError(f"Unknown method: {method}. Use 'isotonic' or 'platt'")

        # Store calibration diagnostics
        self.brier_score_before: Optional[float] = None
        self.brier_score_after: Optional[float] = None

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on held-out calibration data.

        IMPORTANT: Use a separate calibration set that wasn't used for training.
        Typically 15-20% of validation data is reserved for calibration.

        Args:
            y_proba: Uncalibrated probability predictions from model (shape: n_samples,)
            y_true: True binary labels (0 or 1)

        Returns:
            self for method chaining
        """
        y_proba = np.asarray(y_proba).ravel()
        y_true = np.asarray(y_true).ravel()

        if len(y_proba) != len(y_true):
            raise ValueError(f"Length mismatch: y_proba={len(y_proba)}, y_true={len(y_true)}")

        # Store pre-calibration Brier score
        self.brier_score_before = brier_score_loss(y_true, y_proba)

        if self.method == 'isotonic':
            self.calibrator.fit(y_proba, y_true)
        else:
            # Platt scaling needs 2D input
            self.calibrator.fit(y_proba.reshape(-1, 1), y_true)

        # Mark as fitted before computing post-calibration Brier score
        self._fitted = True

        # Compute post-calibration Brier score
        y_calibrated = self.calibrate(y_proba)
        self.brier_score_after = brier_score_loss(y_true, y_calibrated)

        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities to calibrated probabilities.

        Args:
            y_proba: Uncalibrated probabilities (shape: n_samples,)

        Returns:
            Calibrated probabilities in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_proba = np.asarray(y_proba).ravel()

        if self.method == 'isotonic':
            calibrated = self.calibrator.predict(y_proba)
        else:
            # Platt scaling returns probability of class 1
            calibrated = self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]

        # Ensure output is in [0, 1] (isotonic should handle this, but be safe)
        return np.clip(calibrated, 0, 1)

    def get_reliability_data(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram (calibration curve).

        A well-calibrated model shows points close to the diagonal.

        Args:
            y_proba: Predicted probabilities (can be uncalibrated or calibrated)
            y_true: True binary labels
            n_bins: Number of bins for grouping predictions

        Returns:
            Tuple of (bin_means, observed_frequencies, bin_counts)
        """
        y_proba = np.asarray(y_proba).ravel()
        y_true = np.asarray(y_true).ravel()

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins[1:-1])

        bin_means = []
        observed_freqs = []
        bin_counts = []

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

        return (
            np.array(bin_means),
            np.array(observed_freqs),
            np.array(bin_counts),
        )

    def get_expected_calibration_error(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE is the weighted average of the difference between predicted
        probability and observed frequency across bins. Lower is better.

        A well-calibrated model has ECE < 0.05.

        Args:
            y_proba: Predicted probabilities
            y_true: True binary labels
            n_bins: Number of bins

        Returns:
            ECE value (0 = perfect calibration)
        """
        bin_means, observed_freqs, bin_counts = self.get_reliability_data(
            y_proba, y_true, n_bins
        )

        # Filter out empty bins
        valid = ~np.isnan(bin_means) & ~np.isnan(observed_freqs)
        if not valid.any():
            return np.nan

        bin_means = bin_means[valid]
        observed_freqs = observed_freqs[valid]
        bin_counts = np.array(bin_counts)[valid]

        # Weighted average of |accuracy - confidence|
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
        path = Path(path)
        with open(path, 'wb') as f:
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
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(method=data['method'])
        instance.calibrator = data['calibrator']
        instance._fitted = data['fitted']
        instance.brier_score_before = data['brier_score_before']
        instance.brier_score_after = data['brier_score_after']
        return instance
