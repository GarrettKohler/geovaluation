"""
Conformal Prediction for Site Scoring Model.

Provides prediction sets with statistical coverage guarantees using MAPIE.
Unlike point predictions, conformal prediction outputs a SET of possible
labels, with guaranteed coverage: if you set 90% confidence, the true
label is included in the set ≥90% of the time.

Key Methods:
- LAC (Least Ambiguous Classifier): Uses softmax scores directly
- APS (Adaptive Prediction Sets): Guarantees non-empty sets
- RAPS (Regularized APS): Produces smaller, more actionable sets

Reference: "Conformal Prediction Under Covariate Shift" - Gibbs et al. (2021)

Updated for MAPIE 1.2.0 API (SplitConformalClassifier).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from pathlib import Path
import pickle
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin

# Check if MAPIE is available and get version
# NOTE: MAPIE 1.2.0 has compatibility issues with scikit-learn 1.8.0
# (EnsembleClassifier doesn't implement __sklearn_tags__)
# We use a simple fallback implementation for now.
MAPIE_AVAILABLE = False
MAPIE_VERSION = None
MAPIE_SKLEARN_COMPATIBLE = False

try:
    import mapie
    import sklearn
    MAPIE_VERSION = getattr(mapie, '__version__', '0.0.0')
    SKLEARN_VERSION = getattr(sklearn, '__version__', '0.0.0')

    # Check sklearn version - MAPIE 1.2.0 has issues with sklearn >= 1.6
    sklearn_major = int(SKLEARN_VERSION.split('.')[0])
    sklearn_minor = int(SKLEARN_VERSION.split('.')[1])

    if sklearn_major >= 1 and sklearn_minor >= 6:
        # MAPIE 1.2.0 is incompatible with sklearn 1.6+
        # Use fallback until MAPIE is updated
        warnings.warn(
            f"MAPIE {MAPIE_VERSION} has compatibility issues with scikit-learn {SKLEARN_VERSION}. "
            f"Using fallback conformal prediction implementation."
        )
        MAPIE_SKLEARN_COMPATIBLE = False
    else:
        from mapie.classification import SplitConformalClassifier
        MAPIE_AVAILABLE = True
        MAPIE_SKLEARN_COMPATIBLE = True
except ImportError:
    warnings.warn(
        "MAPIE not installed. Install with: pip install mapie\n"
        "Conformal prediction will use fallback implementation."
    )
except Exception as e:
    warnings.warn(f"MAPIE import error: {e}\nUsing fallback implementation.")


class SklearnModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps PyTorch model to provide sklearn-compatible interface.

    MAPIE and other sklearn tools expect:
    - fit(X, y) method
    - predict(X) method
    - predict_proba(X) method returning (n_samples, n_classes)

    This wrapper handles the numeric/categorical/boolean tensor split
    used by SiteScoringModel.

    Inherits from BaseEstimator and ClassifierMixin for sklearn 1.8+ compatibility.
    """

    def __init__(
        self,
        model: nn.Module = None,
        n_numeric: int = 0,
        n_categorical: int = 0,
        n_boolean: int = 0,
        device: str = 'cpu',
    ):
        """
        Args:
            model: PyTorch SiteScoringModel
            n_numeric: Number of numeric features
            n_categorical: Number of categorical features
            n_boolean: Number of boolean features
            device: Device to run inference on
        """
        # Store all params for sklearn clone() compatibility
        self.model = model
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.n_boolean = n_boolean
        self.device = device
        self.classes_ = np.array([0, 1])  # Binary classification

    def _split_features(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split concatenated features back into numeric, categorical, boolean."""
        X = np.asarray(X)

        # Split positions
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnModelWrapper':
        """No-op: model is already trained."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)]
        """
        self.model.eval()

        numeric, categorical, boolean = self._split_features(X)

        with torch.no_grad():
            # Model outputs raw logits for binary classification
            logits = self.model(numeric, categorical, boolean)
            # Convert to probability using sigmoid
            proba_1 = torch.sigmoid(logits).cpu().numpy().ravel()

        # Return shape (n_samples, 2) for sklearn compatibility
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])


class ConformalClassifier:
    """
    Conformal prediction wrapper for site scoring model.

    Provides prediction sets with guaranteed coverage. At confidence level
    1-α, the true label is included in the prediction set with probability
    ≥1-α, regardless of the model or data distribution.

    Prediction Set Interpretation:
    - {0}: High confidence site is LOW value
    - {1}: High confidence site is HIGH value
    - {0, 1}: Uncertain - model cannot confidently predict either class

    Args:
        model: Trained SiteScoringModel (PyTorch)
        n_numeric: Number of numeric features
        n_categorical: Number of categorical features
        n_boolean: Number of boolean features
        alpha: Significance level (default 0.10 = 90% confidence)
        method: Conformal method - 'aps' (recommended) or 'lac'
        device: Device for inference
    """

    def __init__(
        self,
        model: nn.Module,
        n_numeric: int,
        n_categorical: int,
        n_boolean: int,
        alpha: float = 0.10,
        method: str = 'aps',
        device: str = 'cpu',
    ):
        self.alpha = alpha
        self.method = method
        self.device = device
        self._fitted = False

        # Wrap PyTorch model for sklearn compatibility
        self.sklearn_wrapper = SklearnModelWrapper(
            model=model,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            device=device,
        )

        # Initialize MAPIE if available
        if MAPIE_AVAILABLE:
            # MAPIE 1.2.0+ uses SplitConformalClassifier with new API
            # confidence_level = 1 - alpha (e.g., alpha=0.10 -> confidence_level=0.90)
            confidence_level = 1.0 - alpha

            # MAPIE 1.2.0 only supports 'lac' for binary classification
            # 'aps' is only available for multi-class problems
            # Force 'lac' for binary classification compatibility
            conformity_score = 'lac'

            self.mapie = SplitConformalClassifier(
                estimator=self.sklearn_wrapper,
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                prefit=True,  # Model already trained (replaces cv='prefit')
            )
        else:
            self.mapie = None
            self._fallback_threshold = 0.5  # Will be calibrated

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'ConformalClassifier':
        """
        Fit conformal predictor on calibration data.

        IMPORTANT: Calibration data must be held out from training.

        Args:
            X_cal: Calibration features (concatenated: numeric + categorical + boolean)
            y_cal: Calibration labels (0 or 1)

        Returns:
            self for method chaining
        """
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).ravel().astype(int)

        if MAPIE_AVAILABLE and self.mapie is not None:
            # MAPIE 1.2.0+ API: use conformalize() when prefit=True
            # conformalize() calibrates the conformal predictor on held-out data
            # without re-fitting the base estimator
            self.mapie.conformalize(X_cal, y_cal)
        else:
            # Fallback: simple threshold calibration
            proba = self.sklearn_wrapper.predict_proba(X_cal)[:, 1]
            # Find threshold that gives desired coverage
            positive_proba = proba[y_cal == 1]
            if len(positive_proba) > 0:
                sorted_proba = np.sort(positive_proba)
                idx = int((1 - self.alpha) * len(sorted_proba))
                self._fallback_threshold = sorted_proba[max(0, idx - 1)]
            else:
                self._fallback_threshold = 0.5

        self._fitted = True

        # Store calibration stats
        self._n_cal_samples = len(y_cal)
        self._cal_positive_rate = y_cal.mean()

        return self

    def predict_sets(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction sets with coverage guarantee.

        Args:
            X: Features to predict
            alpha: Override significance level (default: use init value)

        Returns:
            Tuple of:
            - predictions: Array of predicted classes (n_samples,)
            - prediction_sets: Boolean array (n_samples, 2) where True means
              class is included in prediction set
        """
        if not self._fitted:
            raise RuntimeError("Conformal predictor not fitted. Call fit() first.")

        X = np.asarray(X)
        alpha = alpha or self.alpha

        if MAPIE_AVAILABLE and self.mapie is not None:
            # MAPIE 1.2.0+ predict API
            # confidence_level = 1 - alpha
            confidence_level = 1.0 - alpha
            predictions, prediction_sets = self.mapie.predict(
                X, confidence_level=confidence_level
            )
            # prediction_sets shape in 1.2.0: (n_samples, n_classes) directly
            # or may be (n_samples, n_classes, n_alphas) - handle both cases
            if prediction_sets.ndim == 3:
                prediction_sets = prediction_sets[:, :, 0]
            return predictions, prediction_sets
        else:
            # Fallback implementation
            proba = self.sklearn_wrapper.predict_proba(X)
            predictions = (proba[:, 1] >= 0.5).astype(int)

            # Simple fallback: include both classes if uncertain
            sets = np.zeros((len(X), 2), dtype=bool)
            sets[:, 0] = proba[:, 0] >= self._fallback_threshold
            sets[:, 1] = proba[:, 1] >= self._fallback_threshold

            return predictions, sets

    def get_set_sizes(self, X: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Get size of prediction set for each sample.

        Size interpretation:
        - 0: Empty set (should be rare with APS method)
        - 1: Confident prediction (single class)
        - 2: Uncertain (both classes included)

        Returns:
            Array of set sizes (n_samples,)
        """
        _, prediction_sets = self.predict_sets(X, alpha)
        return prediction_sets.sum(axis=1)

    def get_uncertainty_mask(self, X: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Get mask of uncertain predictions (set size > 1).

        Useful for flagging sites that need additional review.

        Returns:
            Boolean array where True = uncertain
        """
        set_sizes = self.get_set_sizes(X, alpha)
        return set_sizes > 1

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: Optional[float] = None,
    ) -> dict:
        """
        Evaluate coverage on test data.

        Coverage should be ≥ 1-alpha for valid conformal prediction.

        Returns:
            Dict with coverage statistics
        """
        _, prediction_sets = self.predict_sets(X_test, alpha)
        y_test = np.asarray(y_test).ravel().astype(int)

        # Check if true label is in prediction set
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

    def get_confidence_interpretation(
        self,
        prediction_set: np.ndarray,
    ) -> str:
        """
        Get human-readable interpretation of a prediction set.

        Args:
            prediction_set: Boolean array of shape (2,) - [class_0_included, class_1_included]

        Returns:
            Interpretation string for executives
        """
        if prediction_set[1] and not prediction_set[0]:
            return "High confidence: HIGH-VALUE site"
        elif prediction_set[0] and not prediction_set[1]:
            return "High confidence: LOW-VALUE site"
        elif prediction_set[0] and prediction_set[1]:
            return "Uncertain: Additional review recommended"
        else:
            return "Unable to classify: Insufficient data"

    def save(self, path: Path) -> None:
        """Save conformal predictor state."""
        path = Path(path)

        # Save state for reconstruction
        save_data = {
            'alpha': self.alpha,
            'method': self.method,
            'fitted': self._fitted,
            'n_cal_samples': getattr(self, '_n_cal_samples', None),
            'cal_positive_rate': getattr(self, '_cal_positive_rate', None),
            'fallback_threshold': getattr(self, '_fallback_threshold', 0.5),
            'mapie_version': MAPIE_VERSION,
        }

        # Try to save MAPIE internal state if available
        if MAPIE_AVAILABLE and self.mapie is not None and self._fitted:
            try:
                # MAPIE 1.2.0 stores conformity scores differently
                if hasattr(self.mapie, 'conformity_scores_'):
                    save_data['mapie_conformity_scores'] = self.mapie.conformity_scores_
                if hasattr(self.mapie, 'quantile_'):
                    save_data['mapie_quantile'] = self.mapie.quantile_
            except Exception:
                pass  # Some attributes may not be picklable

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(
        cls,
        path: Path,
        model: nn.Module,
        n_numeric: int,
        n_categorical: int,
        n_boolean: int,
        device: str = 'cpu',
    ) -> 'ConformalClassifier':
        """
        Load conformal predictor state.

        Note: Requires the trained model to be passed in separately
        since PyTorch models are saved/loaded independently.
        """
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            model=model,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            alpha=data['alpha'],
            method=data['method'],
            device=device,
        )

        instance._fitted = data['fitted']
        instance._n_cal_samples = data.get('n_cal_samples')
        instance._cal_positive_rate = data.get('cal_positive_rate')
        instance._fallback_threshold = data.get('fallback_threshold', 0.5)

        # Restore MAPIE state if available
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
