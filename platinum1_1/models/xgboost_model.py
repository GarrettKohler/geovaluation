"""
XGBoost wrapper with consistent interface for regression and classification.

Handles the XGBoost 2.0+ API change where callbacks were removed from
.fit() and must be set via .set_params() before training.

Note: XGBoost does not support MPS (Metal Performance Shaders).
      tree_method="hist" on CPU is used for fast training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel:
    """
    Wrapper for XGBoost with consistent interface.

    Provides a unified API for regression (XGBRegressor) and classification
    (XGBClassifier) with built-in early stopping, feature importance, and
    safe serialization (callbacks cleared before save).
    """

    def __init__(self, task_type: str = "regression", **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            )

        self.task_type = task_type
        self.model = None
        self.is_fitted = False
        self._build_model(**kwargs)

    def _build_model(self, **kwargs):
        """Construct the underlying XGBoost estimator."""
        common_params = {
            "n_estimators": kwargs.get("n_estimators", 1000),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "tree_method": "hist",
            "device": "cpu",  # XGBoost does not support MPS
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 50),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
            "verbosity": kwargs.get("verbosity", 0),
        }

        if self.task_type == "regression":
            self.model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **common_params,
            )
        else:
            self.model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common_params,
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callbacks: Optional[List] = None,
    ) -> "XGBoostModel":
        """
        Train with optional validation and callbacks.

        Args:
            X_train: Training features array (n_samples, n_features).
            y_train: Training target array (n_samples,).
            X_val: Validation features (enables early stopping).
            y_val: Validation target.
            callbacks: Optional list of xgboost.callback.TrainingCallback
                       instances. Set via set_params per XGBoost 2.0+ API.

        Returns:
            self (fitted model).
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # XGBoost 2.0+ removed callbacks from fit(); use set_params instead
        if callbacks:
            self.model.set_params(callbacks=callbacks)

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.is_fitted = True

        # Clear callbacks before any serialization to avoid pickling issues
        if callbacks:
            self.model.set_params(callbacks=None)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.predict(X)

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get feature importance scores as native Python floats.

        Args:
            feature_names: Optional list of feature names. If provided and
                           length matches, keys are feature names; otherwise
                           keys are integer indices.

        Returns:
            Dict mapping feature name (or index) to importance score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self.model.feature_importances_

        if feature_names and len(feature_names) == len(importance):
            return {name: float(val) for name, val in zip(feature_names, importance)}
        return {str(i): float(val) for i, val in enumerate(importance)}

    @property
    def best_iteration(self) -> int:
        """Get the best iteration (with early stopping)."""
        if not self.is_fitted:
            return 0
        return getattr(self.model, "best_iteration", self.model.n_estimators)

    def save(self, path: Path) -> None:
        """Save model to file (XGBoost native format)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Clear callbacks before serialization
        self.model.set_params(callbacks=None)
        self.model.save_model(str(path))
        logger.info("Saved XGBoost model to %s", path)

    def load(self, path: Path) -> None:
        """Load model from file (XGBoost native format)."""
        self.model.load_model(str(path))
        self.is_fitted = True
        logger.info("Loaded XGBoost model from %s", path)
