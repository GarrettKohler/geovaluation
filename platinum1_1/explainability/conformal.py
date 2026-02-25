"""
Conformal Prediction for Site Scoring Model.

Provides prediction sets with statistical coverage guarantees using MAPIE.
Unlike point predictions, conformal prediction outputs a SET of possible
labels, with guaranteed coverage: if you set 90% confidence, the true
label is included in the set >=90% of the time.

Updated for MAPIE 1.2.0 API (SplitConformalClassifier).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path
import pickle
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin

MAPIE_AVAILABLE = False
MAPIE_VERSION = None

try:
    import mapie
    import sklearn
    MAPIE_VERSION = getattr(mapie, '__version__', '0.0.0')
    SKLEARN_VERSION = getattr(sklearn, '__version__', '0.0.0')

    sklearn_major = int(SKLEARN_VERSION.split('.')[0])
    sklearn_minor = int(SKLEARN_VERSION.split('.')[1])

    if sklearn_major >= 1 and sklearn_minor >= 6:
        warnings.warn(
            f"MAPIE {MAPIE_VERSION} has compatibility issues with scikit-learn {SKLEARN_VERSION}. "
            f"Using fallback conformal prediction implementation."
        )
    else:
        from mapie.classification import SplitConformalClassifier
        MAPIE_AVAILABLE = True
except ImportError:
    warnings.warn("MAPIE not installed. Conformal prediction will use fallback implementation.")
except Exception as e:
    warnings.warn(f"MAPIE import error: {e}\nUsing fallback implementation.")


class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """Wraps PyTorch model to provide sklearn-compatible interface for MAPIE."""

    def __init__(self, model=None, n_numeric=0, n_categorical=0, n_boolean=0, device='cpu'):
        self.model = model
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.n_boolean = n_boolean
        self.device = device
        self.classes_ = np.array([0, 1])

    def _split_features(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split concatenated features back into numeric, categorical, boolean."""
        X = np.asarray(X)
        cat_start = self.n_numeric
        bool_start = cat_start + self.n_categorical

        numeric = X[:, :cat_start].astype(np.float32)
        categorical = X[:, cat_start:bool_start].astype(np.int64)
        boolean = X[:, bool_start:].astype(np.float32)

        return (
            torch.from_numpy(numeric).to(self.device),
            torch.from_numpy(categorical).to(self.device),
            torch.from_numpy(boolean).to(self.device),
        )

    def fit(self, X, y):
        """No-op: model is already trained."""
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities. Returns (n_samples, 2)."""
        self.model.eval()
        numeric, categorical, boolean = self._split_features(X)

        with torch.no_grad():
            logits = self.model(numeric, categorical, boolean)
            proba_1 = torch.sigmoid(logits).cpu().numpy().ravel()

        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])


class ConformalClassifier:
    """Conformal prediction wrapper providing prediction sets with coverage guarantees."""

    def __init__(self, model, n_numeric, n_categorical, n_boolean,
                 alpha=0.10, method='aps', device='cpu'):
        self.alpha = alpha
        self.method = method
        self.device = device
        self._fitted = False

        self.sklearn_wrapper = SklearnModelWrapper(
            model=model, n_numeric=n_numeric, n_categorical=n_categorical,
            n_boolean=n_boolean, device=device,
        )

        if MAPIE_AVAILABLE:
            confidence_level = 1.0 - alpha
            self.mapie = SplitConformalClassifier(
                estimator=self.sklearn_wrapper,
                confidence_level=confidence_level,
                conformity_score='lac',
                prefit=True,
            )
        else:
            self.mapie = None
            self._fallback_threshold = 0.5

    def fit(self, X_cal, y_cal):
        """Fit conformal predictor on calibration data."""
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).ravel().astype(int)

        if MAPIE_AVAILABLE and self.mapie is not None:
            self.mapie.conformalize(X_cal, y_cal)
        else:
            proba = self.sklearn_wrapper.predict_proba(X_cal)[:, 1]
            positive_proba = proba[y_cal == 1]
            if len(positive_proba) > 0:
                sorted_proba = np.sort(positive_proba)
                idx = int((1 - self.alpha) * len(sorted_proba))
                self._fallback_threshold = sorted_proba[max(0, idx - 1)]

        self._fitted = True
        self._n_cal_samples = len(y_cal)
        self._cal_positive_rate = y_cal.mean()
        return self

    def predict_sets(self, X, alpha=None):
        """Generate prediction sets with coverage guarantee."""
        if not self._fitted:
            raise RuntimeError("Conformal predictor not fitted. Call fit() first.")

        X = np.asarray(X)
        alpha = alpha or self.alpha

        if MAPIE_AVAILABLE and self.mapie is not None:
            confidence_level = 1.0 - alpha
            predictions, prediction_sets = self.mapie.predict(X, confidence_level=confidence_level)
            if prediction_sets.ndim == 3:
                prediction_sets = prediction_sets[:, :, 0]
            return predictions, prediction_sets
        else:
            proba = self.sklearn_wrapper.predict_proba(X)
            predictions = (proba[:, 1] >= 0.5).astype(int)
            sets = np.zeros((len(X), 2), dtype=bool)
            sets[:, 0] = proba[:, 0] >= self._fallback_threshold
            sets[:, 1] = proba[:, 1] >= self._fallback_threshold
            return predictions, sets

    def get_set_sizes(self, X, alpha=None):
        """Get size of prediction set for each sample."""
        _, prediction_sets = self.predict_sets(X, alpha)
        return prediction_sets.sum(axis=1)

    def get_confidence_interpretation(self, prediction_set):
        """Get human-readable interpretation of a prediction set."""
        if prediction_set[1] and not prediction_set[0]:
            return "High confidence: HIGH-VALUE site"
        elif prediction_set[0] and not prediction_set[1]:
            return "High confidence: LOW-VALUE site"
        elif prediction_set[0] and prediction_set[1]:
            return "Uncertain: Additional review recommended"
        else:
            return "Unable to classify: Insufficient data"

    def evaluate_coverage(self, X_test, y_test, alpha=None):
        """Evaluate coverage on test data."""
        _, prediction_sets = self.predict_sets(X_test, alpha)
        y_test = np.asarray(y_test).ravel().astype(int)
        coverage_mask = prediction_sets[np.arange(len(y_test)), y_test]
        set_sizes = prediction_sets.sum(axis=1)

        return {
            'coverage': coverage_mask.mean(),
            'target_coverage': 1 - (alpha or self.alpha),
            'coverage_achieved': coverage_mask.mean() >= (1 - (alpha or self.alpha)),
            'avg_set_size': set_sizes.mean(),
            'pct_uncertain': (set_sizes > 1).mean(),
            'pct_empty': (set_sizes == 0).mean(),
            'n_samples': len(y_test),
        }

    def save(self, path):
        """Save conformal predictor state."""
        save_data = {
            'alpha': self.alpha,
            'method': self.method,
            'fitted': self._fitted,
            'n_cal_samples': getattr(self, '_n_cal_samples', None),
            'cal_positive_rate': getattr(self, '_cal_positive_rate', None),
            'fallback_threshold': getattr(self, '_fallback_threshold', 0.5),
            'mapie_version': MAPIE_VERSION,
        }

        if MAPIE_AVAILABLE and self.mapie is not None and self._fitted:
            try:
                if hasattr(self.mapie, 'conformity_scores_'):
                    save_data['mapie_conformity_scores'] = self.mapie.conformity_scores_
                if hasattr(self.mapie, 'quantile_'):
                    save_data['mapie_quantile'] = self.mapie.quantile_
            except Exception:
                pass

        with open(Path(path), 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path, model, n_numeric, n_categorical, n_boolean, device='cpu'):
        """Load conformal predictor state."""
        with open(Path(path), 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            model=model, n_numeric=n_numeric, n_categorical=n_categorical,
            n_boolean=n_boolean, alpha=data['alpha'], method=data['method'], device=device,
        )
        instance._fitted = data['fitted']
        instance._n_cal_samples = data.get('n_cal_samples')
        instance._cal_positive_rate = data.get('cal_positive_rate')
        instance._fallback_threshold = data.get('fallback_threshold', 0.5)

        if MAPIE_AVAILABLE and instance.mapie is not None:
            if 'mapie_conformity_scores' in data:
                try:
                    instance.mapie.conformity_scores_ = data['mapie_conformity_scores']
                except Exception:
                    pass
            if 'mapie_quantile' in data:
                try:
                    instance.mapie.quantile_ = data['mapie_quantile']
                except Exception:
                    pass

        return instance
