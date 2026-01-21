"""Causal inference model using Double Machine Learning."""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from ..config import Config, config as default_config


@dataclass
class CausalEffect:
    """Container for causal effect estimates."""

    effect: float
    lower_bound: float
    upper_bound: float
    confident: bool  # True if lower_bound > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect": self.effect,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confident": self.confident,
        }


class CausalModel:
    """Causal inference for hardware treatment effects.

    Uses Double Machine Learning (DML) via EconML's CausalForestDML to:
    - Separate correlation from causation
    - Estimate heterogeneous treatment effects (CATE)
    - Provide confidence intervals for decision-making

    Key insight: DML isolates the "as-if-random" variation in treatment
    assignment after controlling for observable confounders.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 50,
        cv_folds: int = 5,
        config: Optional[Config] = None,
    ):
        """Initialize causal model.

        Args:
            n_estimators: Number of trees in causal forest
            min_samples_leaf: Minimum samples per leaf (controls variance)
            cv_folds: Cross-fitting folds (prevents overfitting bias)
            config: Configuration object
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.cv_folds = cv_folds
        self.config = config or default_config

        self._model = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        outcome_column: Optional[str] = None,
        treatment_column: Optional[str] = None,
        confounder_columns: Optional[list] = None,
        effect_modifier_columns: Optional[list] = None,
    ) -> "CausalModel":
        """Fit causal model using Double Machine Learning.

        Args:
            df: Training data with outcomes, treatments, and features
            outcome_column: Revenue/outcome column (Y)
            treatment_column: Treatment column, e.g., hardware type (T)
            confounder_columns: Variables affecting both T and Y (W)
            effect_modifier_columns: Variables where treatment effect varies (X)
        """
        # Import here to allow graceful failure if not installed
        from econml.dml import CausalForestDML

        outcome_column = outcome_column or self.config.features.outcome_column
        treatment_column = treatment_column or self.config.features.treatment_column
        confounder_columns = confounder_columns or self.config.features.confounder_columns
        effect_modifier_columns = (
            effect_modifier_columns or self.config.features.effect_modifier_columns
        )

        # Filter to available columns
        confounder_columns = [c for c in confounder_columns if c in df.columns]
        effect_modifier_columns = [c for c in effect_modifier_columns if c in df.columns]

        # Prepare data
        Y = df[outcome_column].values
        T = df[treatment_column].values

        # Encode treatment if categorical
        if df[treatment_column].dtype == "object":
            T = pd.Categorical(T).codes

        W = df[confounder_columns].copy()
        X = df[effect_modifier_columns].copy()

        # Handle missing/categorical in confounders
        for col in W.columns:
            if W[col].dtype == "object":
                W[col] = pd.Categorical(W[col]).codes
            W[col] = W[col].fillna(W[col].median())

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.Categorical(X[col]).codes
            X[col] = X[col].fillna(X[col].median())

        # Initialize CausalForestDML
        self._model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=200, max_depth=5),
            model_t=GradientBoostingClassifier(n_estimators=200, max_depth=5),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            cv=self.cv_folds,
            random_state=42,
        )

        # Fit model
        self._model.fit(Y=Y, T=T, X=X.values, W=W.values)

        self._effect_modifier_columns = effect_modifier_columns
        self._is_fitted = True

        return self

    def effect(
        self,
        sites: pd.DataFrame,
        effect_modifier_columns: Optional[list] = None,
    ) -> np.ndarray:
        """Estimate treatment effects for sites.

        Args:
            sites: DataFrame with effect modifier features
            effect_modifier_columns: Columns to use (default: from fit)

        Returns:
            Array of conditional average treatment effects (CATE)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        effect_modifier_columns = effect_modifier_columns or self._effect_modifier_columns

        X = sites[effect_modifier_columns].copy()
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.Categorical(X[col]).codes
            X[col] = X[col].fillna(X[col].median())

        return self._model.effect(X.values).flatten()

    def effect_interval(
        self,
        sites: pd.DataFrame,
        alpha: float = 0.05,
        effect_modifier_columns: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for treatment effects.

        Args:
            sites: DataFrame with effect modifier features
            alpha: Significance level (0.05 = 95% CI)
            effect_modifier_columns: Columns to use

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        effect_modifier_columns = effect_modifier_columns or self._effect_modifier_columns

        X = sites[effect_modifier_columns].copy()
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.Categorical(X[col]).codes
            X[col] = X[col].fillna(X[col].median())

        lower, upper = self._model.effect_interval(X.values, alpha=alpha)
        return lower.flatten(), upper.flatten()

    def estimate_site_effects(
        self,
        sites: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Get full effect estimates for sites.

        Args:
            sites: DataFrame with site_id and effect modifier features
            alpha: Significance level for confidence intervals

        Returns:
            DataFrame with site_id, effect, bounds, and confidence flag
        """
        effects = self.effect(sites)
        lower, upper = self.effect_interval(sites, alpha=alpha)

        result = sites[["site_id"]].copy()
        result["expected_uplift"] = effects
        result["uplift_lower_bound"] = lower
        result["uplift_upper_bound"] = upper
        result["uplift_confident"] = lower > 0

        return result

    def generate_recommendation(
        self,
        site: pd.Series,
        treatment_name: str = "LED display",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Generate human-readable recommendation for a site.

        Args:
            site: Single site Series
            treatment_name: Name of the treatment for output
            alpha: Confidence level

        Returns:
            Dict with recommendation details
        """
        site_df = site.to_frame().T
        effect = self.effect(site_df)[0]
        lower, upper = self.effect_interval(site_df, alpha=alpha)
        lower, upper = lower[0], upper[0]

        confident = lower > 0

        if confident:
            confidence_level = "High" if lower > effect * 0.3 else "Medium"
            return {
                "site_id": site.get("site_id", "unknown"),
                "action": f"Upgrade to {treatment_name}",
                "expected_uplift": f"${effect:,.0f}/month",
                "confidence_interval": f"${lower:,.0f} to ${upper:,.0f}",
                "confidence": confidence_level,
                "recommend": True,
            }
        else:
            return {
                "site_id": site.get("site_id", "unknown"),
                "action": "No upgrade recommended",
                "expected_uplift": f"${effect:,.0f}/month",
                "confidence_interval": f"${lower:,.0f} to ${upper:,.0f}",
                "confidence": "Low",
                "recommend": False,
            }

    def feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance for effect modifiers."""
        if not self._is_fitted:
            return None
        return self._model.feature_importances_
