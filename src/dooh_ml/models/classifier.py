"""Classification model for activation success prediction."""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV

from ..config import Config, config as default_config


class ActivationClassifier:
    """Predict first-year activation success using CatBoost.

    CatBoost with class weights outperforms SMOTE-based resampling
    for imbalanced data with categorical features.

    Key advantages:
    - Native categorical handling (no encoding needed)
    - Ordered boosting prevents target leakage
    - GPU acceleration available
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.1,
        depth: int = 6,
        early_stopping_rounds: int = 50,
        calibrate: bool = True,
        config: Optional[Config] = None,
    ):
        """Initialize classifier.

        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            early_stopping_rounds: Early stopping patience
            calibrate: Whether to calibrate probabilities
            config: Configuration object
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.early_stopping_rounds = early_stopping_rounds
        self.calibrate = calibrate
        self.config = config or default_config

        self._model = None
        self._calibrated_model = None
        self._feature_names: List[str] = []
        self._cat_features: List[str] = []
        self._optimal_threshold: float = 0.5
        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
    ) -> "ActivationClassifier":
        """Fit classifier with automatic class weight balancing.

        Args:
            X_train: Training features
            y_train: Training labels (0/1)
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            cat_features: List of categorical column names
        """
        from catboost import CatBoostClassifier

        self._feature_names = list(X_train.columns)

        # Identify categorical features
        if cat_features is None:
            cat_features = [
                c for c in self.config.features.categorical_features
                if c in X_train.columns
            ]
        self._cat_features = cat_features

        # Calculate class weight for imbalance
        scale_weight = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)

        self._model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            scale_pos_weight=scale_weight,
            cat_features=cat_features,
            eval_metric="AUC",
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            random_seed=42,
            verbose=False,
        )

        # Prepare data
        X_train_processed = self._prepare_features(X_train)

        if X_val is not None and y_val is not None:
            X_val_processed = self._prepare_features(X_val)
            self._model.fit(
                X_train_processed,
                y_train,
                eval_set=[(X_val_processed, y_val)],
            )
        else:
            self._model.fit(X_train_processed, y_train)

        # Calibrate probabilities if requested
        if self.calibrate and X_val is not None and y_val is not None:
            self._calibrated_model = CalibratedClassifierCV(
                self._model, method="sigmoid", cv="prefit"
            )
            self._calibrated_model.fit(X_val_processed, y_val)

        # Find optimal threshold using validation data
        if X_val is not None and y_val is not None:
            y_proba = self.predict_proba(X_val)
            self._optimal_threshold = self._find_optimal_threshold(y_val, y_proba)

        self._is_fitted = True
        return self

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for CatBoost."""
        X = X.copy()

        # Ensure categorical columns are strings
        for col in self._cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("unknown").astype(str)

        # Fill numeric missing values
        numeric_cols = [c for c in X.columns if c not in self._cat_features]
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())

        return X

    def _find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
    ) -> float:
        """Find threshold that maximizes F1 score."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)

        if optimal_idx < len(thresholds):
            return thresholds[optimal_idx]
        return 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.

        Uses calibrated model if available.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_processed = self._prepare_features(X)

        if self._calibrated_model is not None:
            return self._calibrated_model.predict_proba(X_processed)[:, 1]
        else:
            return self._model.predict_proba(X_processed)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features
            threshold: Classification threshold (default: optimal from training)
        """
        threshold = threshold or self._optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Returns metrics suited for imbalanced data:
        - PR-AUC: Primary metric, focuses on positive class
        - ROC-AUC: Overall discrimination
        - MCC: Balanced single metric
        - Lift@10%: Business-relevant prioritization measure
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        # Calculate lift in top decile
        top_10_pct_idx = np.argsort(y_proba)[-int(len(y_proba) * 0.1):]
        baseline_rate = y.mean()
        top_10_rate = y.iloc[top_10_pct_idx].mean()
        lift_10 = top_10_rate / baseline_rate if baseline_rate > 0 else 0

        return {
            "pr_auc": average_precision_score(y, y_proba),
            "roc_auc": roc_auc_score(y, y_proba),
            "mcc": matthews_corrcoef(y, y_pred),
            "lift_top_10pct": lift_10,
            "optimal_threshold": self._optimal_threshold,
        }

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importance = self._model.feature_importances_

        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def explain_prediction(
        self,
        X: pd.DataFrame,
        site_idx: int = 0,
    ) -> Dict[str, Any]:
        """Explain a single prediction using SHAP.

        Args:
            X: Features DataFrame
            site_idx: Index of site to explain

        Returns:
            Dict with prediction and feature contributions
        """
        try:
            import shap

            X_processed = self._prepare_features(X)
            explainer = shap.TreeExplainer(self._model)
            shap_values = explainer(X_processed)

            site_shap = shap_values[site_idx]

            # Get top contributing features
            contributions = pd.DataFrame({
                "feature": self._feature_names,
                "value": X_processed.iloc[site_idx].values,
                "shap_value": site_shap.values,
            }).sort_values("shap_value", key=abs, ascending=False)

            return {
                "prediction": self.predict_proba(X.iloc[[site_idx]])[0],
                "base_value": explainer.expected_value,
                "top_features": contributions.head(10).to_dict("records"),
            }
        except ImportError:
            return {
                "error": "SHAP not installed. Run: pip install shap",
                "prediction": self.predict_proba(X.iloc[[site_idx]])[0],
            }

    @property
    def optimal_threshold(self) -> float:
        """Get the optimal classification threshold."""
        return self._optimal_threshold
