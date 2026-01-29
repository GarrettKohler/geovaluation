"""
Unified Explainability Pipeline for Site Scoring.

Combines all explainability components into a single interface:
- Probability Calibration (ensures predicted probabilities are accurate)
- Conformal Prediction (provides prediction sets with coverage guarantees)
- SHAP Feature Importance (explains why a prediction was made)
- Counterfactual Generation (explains what would change the prediction)
- Tier Classification (executive-friendly business recommendations)

Usage:
    pipeline = ExplainabilityPipeline(model, train_data, ...)
    pipeline.fit_calibration(X_cal, y_cal)
    result = pipeline.explain_site(site_features)
    print(f"Tier: {result.tier_label} - {result.confidence_statement}")
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import pickle
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Feature importance will be limited.")

from .calibration import ProbabilityCalibrator
from .conformal import ConformalClassifier, SklearnModelWrapper
from .counterfactuals import (
    CounterfactualGenerator,
    UpgradePathClusterer,
    Counterfactual,
    ACTIONABLE_FEATURES,
    IMMUTABLE_FEATURES,
    DICE_AVAILABLE,
)
from .tiers import TierClassifier, TierResult


@dataclass
class ExplanationResult:
    """
    Complete explanation for a single site prediction.

    Combines all explanation types into a single result object
    suitable for both API responses and UI rendering.
    """
    # Core prediction
    site_id: Optional[Any] = None
    raw_prediction: float = 0.0
    calibrated_probability: float = 0.0

    # Conformal prediction
    prediction_set: List[int] = field(default_factory=list)
    is_uncertain: bool = False
    confidence_interpretation: str = ""

    # Tier classification
    tier: int = 0
    tier_label: str = ""
    tier_action: str = ""
    confidence_statement: str = ""
    historical_accuracy: Optional[float] = None

    # Feature importance (SHAP)
    top_positive_drivers: List[Tuple[str, float]] = field(default_factory=list)
    top_negative_drivers: List[Tuple[str, float]] = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None

    # Counterfactuals (if applicable)
    counterfactuals: List[Counterfactual] = field(default_factory=list)
    recommended_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'site_id': self.site_id,
            'prediction': {
                'raw': self.raw_prediction,
                'calibrated': self.calibrated_probability,
                'prediction_set': self.prediction_set,
                'is_uncertain': self.is_uncertain,
            },
            'tier': {
                'tier': self.tier,
                'label': self.tier_label,
                'action': self.tier_action,
                'confidence': self.confidence_statement,
                'historical_accuracy': self.historical_accuracy,
            },
            'drivers': {
                'positive': [
                    {'feature': f, 'impact': float(v)}
                    for f, v in self.top_positive_drivers
                ],
                'negative': [
                    {'feature': f, 'impact': float(v)}
                    for f, v in self.top_negative_drivers
                ],
            },
            'counterfactuals': [cf.to_dict() for cf in self.counterfactuals],
            'recommended_changes': self.recommended_changes,
        }

    def get_executive_summary(self) -> str:
        """Generate executive-friendly text summary."""
        lines = [
            f"Site Classification: {self.tier_label.upper()}",
            f"Recommendation: {self.tier_action}",
            f"Confidence: {self.confidence_statement}",
            "",
        ]

        if self.top_positive_drivers:
            lines.append("Key Strengths:")
            for feature, impact in self.top_positive_drivers[:3]:
                lines.append(f"  + {feature} (contributes +{impact:.2f})")
            lines.append("")

        if self.top_negative_drivers:
            lines.append("Areas for Improvement:")
            for feature, impact in self.top_negative_drivers[:3]:
                lines.append(f"  - {feature} (reduces by {abs(impact):.2f})")
            lines.append("")

        if self.recommended_changes:
            lines.append("Upgrade Path:")
            for change in self.recommended_changes[:3]:
                lines.append(f"  → {change}")

        return "\n".join(lines)


class ExplainabilityPipeline:
    """
    Unified interface for all explainability components.

    This is the main class users should interact with. It combines:
    1. Probability calibration (fit once, use for all predictions)
    2. Conformal prediction (prediction sets with coverage guarantee)
    3. SHAP analysis (feature importance)
    4. Counterfactual generation (upgrade paths)
    5. Tier classification (executive-friendly labels)

    Example:
        # Initialize
        pipeline = ExplainabilityPipeline(
            model=trained_model,
            train_data=train_df,
            feature_names=feature_list,
            ...
        )

        # Fit calibration on held-out data
        pipeline.fit_calibration(X_cal, y_cal)

        # Explain a single site
        result = pipeline.explain_site(site_features)
        print(result.get_executive_summary())

        # Fleet-wide analysis
        interventions = pipeline.get_fleet_interventions(low_value_sites)
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: pd.DataFrame,
        feature_names: List[str],
        continuous_features: List[str],
        n_numeric: int,
        n_categorical: int,
        n_boolean: int,
        outcome_name: str = 'target',
        actionable_features: Optional[List[str]] = None,
        device: str = 'cpu',
        alpha: float = 0.10,
    ):
        """
        Initialize the explainability pipeline.

        Args:
            model: Trained PyTorch SiteScoringModel
            train_data: DataFrame used for training
            feature_names: Ordered list of feature names
            continuous_features: Names of continuous (numeric) features
            n_numeric: Count of numeric features
            n_categorical: Count of categorical features
            n_boolean: Count of boolean features
            outcome_name: Name of target column
            actionable_features: Features that can be changed (for counterfactuals)
            device: Device for model inference
            alpha: Significance level for conformal prediction (default 0.10 = 90%)
        """
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.n_boolean = n_boolean
        self.device = device
        self.alpha = alpha

        # Set actionable features
        self.actionable_features = actionable_features or [
            f for f in ACTIONABLE_FEATURES if f in feature_names
        ]

        # Initialize components
        self._init_calibrator()
        self._init_conformal(model, n_numeric, n_categorical, n_boolean)
        self._init_tier_classifier()
        self._init_shap(train_data)
        self._init_counterfactuals(model, train_data, outcome_name)

        self._fitted = False

    def _init_calibrator(self):
        """Initialize probability calibrator."""
        self.calibrator = ProbabilityCalibrator(method='isotonic')

    def _init_conformal(self, model, n_numeric, n_categorical, n_boolean):
        """Initialize conformal predictor."""
        self.conformal = ConformalClassifier(
            model=model,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            alpha=self.alpha,
            method='aps',
            device=self.device,
        )
        # Store sklearn wrapper for other uses
        self.sklearn_wrapper = self.conformal.sklearn_wrapper

    def _init_tier_classifier(self):
        """Initialize tier classifier."""
        self.tier_classifier = TierClassifier()

    def _init_shap(self, train_data: pd.DataFrame):
        """Initialize SHAP explainer."""
        if SHAP_AVAILABLE:
            try:
                # Use KernelExplainer for model-agnostic SHAP
                # Sample background data for efficiency
                background = train_data[self.feature_names].sample(
                    min(100, len(train_data)),
                    random_state=42,
                ).values

                self.shap_explainer = shap.KernelExplainer(
                    self.sklearn_wrapper.predict_proba,
                    background,
                )
            except Exception as e:
                warnings.warn(f"SHAP initialization failed: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

    def _init_counterfactuals(self, model, train_data: pd.DataFrame, outcome_name: str):
        """Initialize counterfactual generator."""
        if DICE_AVAILABLE:
            try:
                self.cf_generator = CounterfactualGenerator(
                    model=self.sklearn_wrapper,
                    train_data=train_data,
                    feature_names=self.feature_names,
                    continuous_features=self.continuous_features,
                    outcome_name=outcome_name,
                    actionable_features=self.actionable_features,
                )
            except Exception as e:
                warnings.warn(f"Counterfactual initialization failed: {e}")
                self.cf_generator = None
        else:
            self.cf_generator = None

        self.path_clusterer = UpgradePathClusterer(n_clusters=5)

    def fit_calibration(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> 'ExplainabilityPipeline':
        """
        Fit calibration and conformal prediction on held-out data.

        IMPORTANT: Use data not used for model training.

        Args:
            X_cal: Calibration features (concatenated numeric + categorical + boolean)
            y_cal: Calibration labels (0 or 1)

        Returns:
            self for method chaining
        """
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).ravel()

        # Get raw probabilities for calibration
        raw_proba = self.sklearn_wrapper.predict_proba(X_cal)[:, 1]

        # Fit probability calibrator
        self.calibrator.fit(raw_proba, y_cal)

        # Fit conformal predictor
        self.conformal.fit(X_cal, y_cal)

        self._fitted = True
        return self

    def explain_site(
        self,
        site_features: pd.DataFrame,
        site_id: Optional[Any] = None,
        generate_counterfactuals: bool = True,
    ) -> ExplanationResult:
        """
        Generate complete explanation for a single site.

        Args:
            site_features: DataFrame with single row of site features
            site_id: Optional identifier for the site
            generate_counterfactuals: Whether to generate counterfactuals
                                     (slower, only useful for low-value sites)

        Returns:
            ExplanationResult with all explanation components
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_calibration() first.")

        if len(site_features) != 1:
            raise ValueError(f"Expected single site, got {len(site_features)}")

        # Convert to numpy for model
        X = site_features[self.feature_names].values

        # 1. Get raw prediction
        raw_proba = self.sklearn_wrapper.predict_proba(X)[:, 1][0]

        # 2. Calibrate probability
        calibrated_prob = self.calibrator.calibrate(np.array([raw_proba]))[0]

        # 3. Conformal prediction
        _, pred_sets = self.conformal.predict_sets(X)
        prediction_set = [i for i in range(2) if pred_sets[0, i]]
        is_uncertain = len(prediction_set) > 1
        confidence_interp = self.conformal.get_confidence_interpretation(pred_sets[0])

        # 4. Tier classification
        tier_result: TierResult = self.tier_classifier.classify(calibrated_prob)

        # 5. SHAP analysis
        top_positive, top_negative, shap_values = self._compute_shap(X)

        # 6. Counterfactuals (only for low-value predictions)
        counterfactuals = []
        recommended_changes = []

        if generate_counterfactuals and calibrated_prob < 0.5 and self.cf_generator:
            try:
                counterfactuals = self.cf_generator.generate(
                    site_features,
                    n_counterfactuals=3,
                    desired_class=1,
                )
                if counterfactuals:
                    # Extract unique recommended changes
                    all_changes = set()
                    for cf in counterfactuals:
                        for change in cf.changes.keys():
                            all_changes.add(change)
                    recommended_changes = list(all_changes)[:5]
            except Exception as e:
                warnings.warn(f"Counterfactual generation failed: {e}")

        return ExplanationResult(
            site_id=site_id,
            raw_prediction=float(raw_proba),
            calibrated_probability=float(calibrated_prob),
            prediction_set=prediction_set,
            is_uncertain=is_uncertain,
            confidence_interpretation=confidence_interp,
            tier=tier_result.tier,
            tier_label=tier_result.label,
            tier_action=tier_result.action,
            confidence_statement=tier_result.confidence_statement,
            historical_accuracy=tier_result.historical_accuracy,
            top_positive_drivers=top_positive,
            top_negative_drivers=top_negative,
            shap_values=shap_values,
            counterfactuals=counterfactuals,
            recommended_changes=recommended_changes,
        )

    def explain_batch(
        self,
        sites: pd.DataFrame,
        site_ids: Optional[List[Any]] = None,
        generate_counterfactuals: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> List[ExplanationResult]:
        """
        Explain multiple sites.

        Args:
            sites: DataFrame with multiple sites
            site_ids: Optional list of site identifiers
            generate_counterfactuals: Whether to generate counterfactuals
            progress_callback: Optional callback(current, total)

        Returns:
            List of ExplanationResult objects
        """
        results = []
        n_total = len(sites)
        site_ids = site_ids or [None] * n_total

        for i, (idx, row) in enumerate(sites.iterrows()):
            site_df = pd.DataFrame([row])
            result = self.explain_site(
                site_df,
                site_id=site_ids[i] if site_ids else idx,
                generate_counterfactuals=generate_counterfactuals,
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, n_total)

        return results

    def _compute_shap(
        self,
        X: np.ndarray,
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], Optional[np.ndarray]]:
        """Compute SHAP values and extract top drivers."""
        if self.shap_explainer is None:
            return [], [], None

        try:
            shap_values = self.shap_explainer.shap_values(X, nsamples=50)

            # For binary classification, get class 1 SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = shap_values.ravel()

            # Pair with feature names and sort
            feature_impacts = list(zip(self.feature_names, shap_values))
            positive = sorted(
                [(f, v) for f, v in feature_impacts if v > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            negative = sorted(
                [(f, v) for f, v in feature_impacts if v < 0],
                key=lambda x: x[1],
            )[:5]

            return positive, negative, shap_values

        except Exception as e:
            warnings.warn(f"SHAP computation failed: {e}")
            return [], [], None

    def get_fleet_interventions(
        self,
        low_value_sites: pd.DataFrame,
        progress_callback: Optional[callable] = None,
    ) -> List[dict]:
        """
        Identify fleet-wide upgrade interventions.

        Clusters counterfactual changes across multiple sites to identify
        strategic patterns like "42 sites would upgrade with 24/7 hours".

        Args:
            low_value_sites: DataFrame of sites with calibrated_prob < 0.5
            progress_callback: Optional progress callback

        Returns:
            List of intervention recommendations
        """
        if self.cf_generator is None:
            warnings.warn("Counterfactual generator not available")
            return []

        # Generate counterfactuals for batch
        cf_results = self.cf_generator.generate_batch(
            low_value_sites,
            n_counterfactuals=3,
            progress_callback=progress_callback,
        )

        # Cluster to find patterns
        self.path_clusterer.fit(cf_results, self.actionable_features)

        # Get upgrade paths
        paths = self.path_clusterer.get_upgrade_paths(len(low_value_sites))

        return [path.to_dict() for path in paths]

    def get_calibration_stats(self) -> dict:
        """Get calibration performance statistics."""
        return self.calibrator.get_calibration_summary()

    def get_coverage_stats(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Get conformal prediction coverage statistics."""
        return self.conformal.evaluate_coverage(X_test, y_test)

    def save(self, directory: Path) -> None:
        """
        Save pipeline components to directory.

        Note: Model must be saved separately using PyTorch save.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save calibrator
        self.calibrator.save(directory / 'calibrator.pkl')

        # Save conformal state
        self.conformal.save(directory / 'conformal.pkl')

        # Save tier classifier
        with open(directory / 'tier_classifier.pkl', 'wb') as f:
            pickle.dump(self.tier_classifier.to_dict(), f)

        # Save metadata
        with open(directory / 'pipeline_metadata.pkl', 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'continuous_features': self.continuous_features,
                'n_numeric': self.n_numeric,
                'n_categorical': self.n_categorical,
                'n_boolean': self.n_boolean,
                'actionable_features': self.actionable_features,
                'alpha': self.alpha,
                'fitted': self._fitted,
            }, f)

    @classmethod
    def load(
        cls,
        directory: Path,
        model: nn.Module,
        train_data: pd.DataFrame,
        device: str = 'cpu',
    ) -> 'ExplainabilityPipeline':
        """
        Load pipeline from directory.

        Args:
            directory: Directory containing saved pipeline
            model: Trained PyTorch model (loaded separately)
            train_data: Training data for re-initializing SHAP/DiCE
            device: Device for inference

        Returns:
            Loaded ExplainabilityPipeline
        """
        directory = Path(directory)

        # Load metadata
        with open(directory / 'pipeline_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Create pipeline
        pipeline = cls(
            model=model,
            train_data=train_data,
            feature_names=metadata['feature_names'],
            continuous_features=metadata['continuous_features'],
            n_numeric=metadata['n_numeric'],
            n_categorical=metadata['n_categorical'],
            n_boolean=metadata['n_boolean'],
            actionable_features=metadata['actionable_features'],
            device=device,
            alpha=metadata['alpha'],
        )

        # Load calibrator
        pipeline.calibrator = ProbabilityCalibrator.load(directory / 'calibrator.pkl')

        # Load conformal
        pipeline.conformal = ConformalClassifier.load(
            directory / 'conformal.pkl',
            model=model,
            n_numeric=metadata['n_numeric'],
            n_categorical=metadata['n_categorical'],
            n_boolean=metadata['n_boolean'],
            device=device,
        )
        pipeline.sklearn_wrapper = pipeline.conformal.sklearn_wrapper

        # Load tier classifier
        with open(directory / 'tier_classifier.pkl', 'rb') as f:
            tier_data = pickle.load(f)
            pipeline.tier_classifier = TierClassifier.from_dict(tier_data)

        pipeline._fitted = metadata['fitted']
        return pipeline
