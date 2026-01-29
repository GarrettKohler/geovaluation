"""
SHAP-Select: Feature Selection using SHAP Values and Statistical Significance.

Implements the method from:
"shap-select: Lightweight Feature Selection Using SHAP Values and Regression"
https://arxiv.org/html/2410.06815v1 (October 2024)

Key idea: After training, compute SHAP values on validation data, then use
statistical significance testing via linear/logistic regression on the
SHAP values to iteratively eliminate the least significant features.

For regression: features should have positive, statistically significant
coefficients when SHAP values are regressed against the target.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Callable
import warnings


class ShapSelect:
    """
    SHAP-Select: Feature selection using SHAP values and statistical significance.

    This is a POST-TRAINING technique that:
    1. Computes SHAP values for all features on validation data
    2. Fits linear/logistic regression of target ~ SHAP values
    3. Iteratively removes features with lowest statistical significance
    4. Returns features with significant positive contributions

    Args:
        task_type: 'regression' or 'classification'
        significance_level: P-value threshold for feature significance (default: 0.05)
        min_l1_weight: Minimal L1 regularization for stability (default: 1e-6)
        max_iterations: Maximum number of elimination iterations
    """

    def __init__(
        self,
        task_type: str = 'regression',
        significance_level: float = 0.05,
        min_l1_weight: float = 1e-6,
        max_iterations: Optional[int] = None,
    ):
        self.task_type = task_type
        self.significance_level = significance_level
        self.min_l1_weight = min_l1_weight
        self.max_iterations = max_iterations

        # Results storage
        self.elimination_history: List[Tuple[str, float, float]] = []
        self.final_t_statistics: Dict[str, float] = {}

    def select_features(
        self,
        shap_values: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        verbose: bool = True,
    ) -> Dict:
        """
        Perform SHAP-Select feature selection.

        Args:
            shap_values: SHAP values array, shape (n_samples, n_features)
            y_val: Validation target values, shape (n_samples,) or (n_samples, 1)
            feature_names: List of feature names
            verbose: Print progress during elimination

        Returns:
            Dictionary with:
            - selected_features: List of selected feature names
            - eliminated_features: List of eliminated feature names
            - elimination_order: List of (name, t_stat, p_value) tuples
            - importance_scores: Dict mapping feature name to final importance
            - n_selected: Number of selected features
        """
        # Validate inputs
        if shap_values.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP values have {shap_values.shape[1]} features "
                f"but {len(feature_names)} feature names provided"
            )

        y_val = np.asarray(y_val).flatten()

        # Initialize
        remaining_indices = list(range(len(feature_names)))
        self.elimination_history = []
        max_iter = self.max_iterations or len(feature_names) - 1

        if verbose:
            print(f"SHAP-Select: Starting with {len(feature_names)} features")
            print(f"Significance level: {self.significance_level}")

        iteration = 0
        while len(remaining_indices) > 1 and iteration < max_iter:
            # Get SHAP values for remaining features
            shap_subset = shap_values[:, remaining_indices]

            # Fit regression on SHAP values
            coefs, t_stats, p_values = self._fit_regression(shap_subset, y_val)

            # Find feature to eliminate
            # Priority: negative coefficients first, then lowest t-statistic
            elimination_idx, should_stop = self._find_elimination_candidate(
                coefs, t_stats, p_values, remaining_indices, feature_names
            )

            if should_stop:
                if verbose:
                    print(f"Stopping: all remaining features are significant")
                break

            # Record elimination
            feat_idx = remaining_indices[elimination_idx]
            feat_name = feature_names[feat_idx]
            self.elimination_history.append((
                feat_name,
                float(t_stats[elimination_idx]),
                float(p_values[elimination_idx])
            ))

            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}: Eliminating '{feat_name}' "
                      f"(t={t_stats[elimination_idx]:.3f}, p={p_values[elimination_idx]:.4f})")

            # Remove feature
            remaining_indices.pop(elimination_idx)
            iteration += 1

        # Final statistics for remaining features
        if len(remaining_indices) > 0:
            shap_final = shap_values[:, remaining_indices]
            coefs, t_stats, p_values = self._fit_regression(shap_final, y_val)
            self.final_t_statistics = {
                feature_names[remaining_indices[i]]: float(t_stats[i])
                for i in range(len(remaining_indices))
            }

        # Build results
        selected = [feature_names[i] for i in remaining_indices]
        eliminated = [name for name, _, _ in self.elimination_history]

        # Compute importance scores (mean absolute SHAP value)
        importance_scores = {}
        for i in remaining_indices:
            importance_scores[feature_names[i]] = float(np.abs(shap_values[:, i]).mean())

        if verbose:
            print(f"SHAP-Select complete: {len(selected)}/{len(feature_names)} features selected")

        return {
            'selected_features': selected,
            'eliminated_features': eliminated,
            'elimination_order': self.elimination_history,
            'importance_scores': importance_scores,
            'final_t_statistics': self.final_t_statistics,
            'n_selected': len(selected),
            'n_eliminated': len(eliminated),
        }

    def _fit_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit regression and compute t-statistics.

        Returns:
            coefs: Regression coefficients
            t_stats: T-statistics for each coefficient
            p_values: P-values for each coefficient
        """
        n, p = X.shape

        if self.task_type == 'regression':
            # Linear regression with minimal L1
            try:
                from sklearn.linear_model import Lasso
                reg = Lasso(alpha=self.min_l1_weight, fit_intercept=True, max_iter=10000)
                reg.fit(X, y)
                coefs = reg.coef_
            except ImportError:
                # Fallback to OLS
                X_with_intercept = np.column_stack([np.ones(n), X])
                coefs_full = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                coefs = coefs_full[1:]  # Exclude intercept
        else:
            # Logistic regression for classification
            try:
                from sklearn.linear_model import LogisticRegression
                reg = LogisticRegression(
                    penalty='l1',
                    C=1/max(self.min_l1_weight, 1e-6),
                    solver='saga',
                    max_iter=10000
                )
                reg.fit(X, y)
                coefs = reg.coef_.flatten()
            except ImportError:
                raise ImportError("sklearn required for classification SHAP-Select")

        # Compute standard errors and t-statistics
        t_stats, p_values = self._compute_statistics(X, y, coefs)

        return coefs, t_stats, p_values

    def _compute_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute t-statistics and p-values for coefficients."""
        n, p = X.shape

        # Predictions and residuals
        y_pred = X @ coefs
        residuals = y - y_pred
        df = max(n - p - 1, 1)

        # Mean squared error
        mse = np.sum(residuals ** 2) / df

        # Standard errors: sqrt(MSE * diag((X'X)^{-1}))
        try:
            XtX = X.T @ X
            # Add small regularization for numerical stability
            XtX_reg = XtX + np.eye(p) * 1e-8
            XtX_inv_diag = np.diag(np.linalg.inv(XtX_reg))
        except np.linalg.LinAlgError:
            XtX_inv_diag = np.ones(p)

        se = np.sqrt(np.maximum(mse * XtX_inv_diag, 1e-10))

        # T-statistics and p-values
        t_stats = coefs / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

        return t_stats, p_values

    def _find_elimination_candidate(
        self,
        coefs: np.ndarray,
        t_stats: np.ndarray,
        p_values: np.ndarray,
        remaining_indices: List[int],
        feature_names: List[str],
    ) -> Tuple[int, bool]:
        """
        Find the feature to eliminate.

        Priority:
        1. Negative coefficients (model using feature incorrectly)
        2. Lowest t-statistic among non-significant features

        Returns:
            elimination_idx: Index in the remaining_indices list to eliminate
            should_stop: True if all remaining features are significant
        """
        n_features = len(coefs)

        # Check for negative coefficients first
        negative_mask = coefs < 0
        if negative_mask.any():
            # Eliminate most negative coefficient
            negative_indices = np.where(negative_mask)[0]
            most_negative = negative_indices[np.argmin(coefs[negative_indices])]
            return most_negative, False

        # All coefficients positive - check significance
        non_significant = p_values >= self.significance_level

        if not non_significant.any():
            # All features are significant
            return 0, True

        # Eliminate lowest t-statistic among non-significant
        non_sig_indices = np.where(non_significant)[0]
        lowest_t_idx = non_sig_indices[np.argmin(t_stats[non_sig_indices])]

        return lowest_t_idx, False


def apply_shap_select(
    model_predict_fn: Callable,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    n_background: int = 100,
    n_shap_samples: int = 100,
    task_type: str = 'regression',
    significance_level: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Convenience function to apply SHAP-Select to a trained model.

    This function handles SHAP value computation and then applies SHAP-Select.

    Args:
        model_predict_fn: Function that takes X and returns predictions
        X_val: Validation features, shape (n_samples, n_features)
        y_val: Validation targets
        feature_names: List of feature names
        background_data: Background data for SHAP (default: use X_val[:n_background])
        n_background: Number of background samples
        n_shap_samples: Number of samples for SHAP approximation
        task_type: 'regression' or 'classification'
        significance_level: P-value threshold
        verbose: Print progress

    Returns:
        SHAP-Select results dictionary
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library required. Install with: pip install shap")

    # Prepare background data
    if background_data is None:
        n_bg = min(n_background, len(X_val))
        background_data = X_val[:n_bg]

    if verbose:
        print(f"Computing SHAP values for {len(X_val)} samples...")

    # Create explainer and compute SHAP values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(model_predict_fn, background_data)
        shap_values = explainer.shap_values(X_val, nsamples=n_shap_samples)

    # Handle multi-output SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 2:
        shap_values = shap_values.squeeze()

    if verbose:
        print(f"SHAP values computed: shape {shap_values.shape}")

    # Apply SHAP-Select
    selector = ShapSelect(
        task_type=task_type,
        significance_level=significance_level,
    )

    return selector.select_features(
        shap_values=shap_values,
        y_val=y_val,
        feature_names=feature_names,
        verbose=verbose,
    )


def get_shap_feature_importance(
    model_predict_fn: Callable,
    X_val: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    n_background: int = 100,
    n_shap_samples: int = 100,
) -> Dict[str, float]:
    """
    Compute SHAP-based feature importance without selection.

    Returns dictionary mapping feature names to mean |SHAP| values.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library required. Install with: pip install shap")

    if background_data is None:
        n_bg = min(n_background, len(X_val))
        background_data = X_val[:n_bg]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(model_predict_fn, background_data)
        shap_values = explainer.shap_values(X_val, nsamples=n_shap_samples)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 2:
        shap_values = shap_values.squeeze()

    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    return {name: float(imp) for name, imp in zip(feature_names, importance)}
