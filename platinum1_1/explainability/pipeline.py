"""
Unified Explainability Pipeline for Site Scoring.

Combines all explainability components into a single interface:
- Probability Calibration
- Conformal Prediction
- SHAP Feature Importance
- Counterfactual Generation
- Tier Classification

Usage:
    pipeline = ExplainabilityPipeline(model, train_data, ...)
    pipeline.fit_calibration(X_cal, y_cal)
    result = pipeline.explain_site(site_features)
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

from .calibration import ProbabilityCalibrator
from .conformal import ConformalClassifier, SklearnModelWrapper
from .counterfactuals import (
    CounterfactualGenerator, UpgradePathClusterer, Counterfactual,
    ACTIONABLE_FEATURES, DICE_AVAILABLE,
)
from .tiers import TierClassifier, TierResult


@dataclass
class ExplanationResult:
    """Complete explanation for a single site prediction."""
    site_id: Optional[Any] = None
    raw_prediction: float = 0.0
    calibrated_probability: float = 0.0
    prediction_set: List[int] = field(default_factory=list)
    is_uncertain: bool = False
    confidence_interpretation: str = ""
    tier: int = 0
    tier_label: str = ""
    tier_action: str = ""
    confidence_statement: str = ""
    historical_accuracy: Optional[float] = None
    top_positive_drivers: List[Tuple[str, float]] = field(default_factory=list)
    top_negative_drivers: List[Tuple[str, float]] = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None
    counterfactuals: List[Counterfactual] = field(default_factory=list)
    recommended_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
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
                'positive': [{'feature': f, 'impact': float(v)} for f, v in self.top_positive_drivers],
                'negative': [{'feature': f, 'impact': float(v)} for f, v in self.top_negative_drivers],
            },
            'counterfactuals': [cf.to_dict() for cf in self.counterfactuals],
            'recommended_changes': self.recommended_changes,
        }

    def get_executive_summary(self) -> str:
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
                lines.append(f"  -> {change}")
        return "\n".join(lines)


class ExplainabilityPipeline:
    """Unified interface for all explainability components."""

    def __init__(self, model, train_data, feature_names, continuous_features,
                 n_numeric, n_categorical, n_boolean, outcome_name='target',
                 actionable_features=None, device='cpu', alpha=0.10):
        self.model = model
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.n_numeric = n_numeric
        self.n_categorical = n_categorical
        self.n_boolean = n_boolean
        self.device = device
        self.alpha = alpha

        self.actionable_features = actionable_features or [
            f for f in ACTIONABLE_FEATURES if f in feature_names
        ]

        self.calibrator = ProbabilityCalibrator(method='isotonic')

        self.conformal = ConformalClassifier(
            model=model, n_numeric=n_numeric, n_categorical=n_categorical,
            n_boolean=n_boolean, alpha=alpha, method='aps', device=device,
        )
        self.sklearn_wrapper = self.conformal.sklearn_wrapper
        self.tier_classifier = TierClassifier()

        # SHAP
        if SHAP_AVAILABLE:
            try:
                background = train_data[feature_names].sample(
                    min(100, len(train_data)), random_state=42
                ).values
                self.shap_explainer = shap.KernelExplainer(
                    self.sklearn_wrapper.predict_proba, background
                )
            except Exception as e:
                warnings.warn(f"SHAP initialization failed: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

        # Counterfactuals
        if DICE_AVAILABLE:
            try:
                self.cf_generator = CounterfactualGenerator(
                    model=self.sklearn_wrapper, train_data=train_data,
                    feature_names=feature_names, continuous_features=continuous_features,
                    outcome_name=outcome_name, actionable_features=self.actionable_features,
                )
            except Exception as e:
                warnings.warn(f"Counterfactual initialization failed: {e}")
                self.cf_generator = None
        else:
            self.cf_generator = None

        self.path_clusterer = UpgradePathClusterer(n_clusters=5)
        self._fitted = False

    def fit_calibration(self, X_cal, y_cal):
        """Fit calibration and conformal prediction on held-out data."""
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal).ravel()
        raw_proba = self.sklearn_wrapper.predict_proba(X_cal)[:, 1]
        self.calibrator.fit(raw_proba, y_cal)
        self.conformal.fit(X_cal, y_cal)
        self._fitted = True
        return self

    def explain_site(self, site_features, site_id=None, generate_counterfactuals=True):
        """Generate complete explanation for a single site."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_calibration() first.")
        if len(site_features) != 1:
            raise ValueError(f"Expected single site, got {len(site_features)}")

        X = site_features[self.feature_names].values
        raw_proba = self.sklearn_wrapper.predict_proba(X)[:, 1][0]
        calibrated_prob = self.calibrator.calibrate(np.array([raw_proba]))[0]

        _, pred_sets = self.conformal.predict_sets(X)
        prediction_set = [i for i in range(2) if pred_sets[0, i]]
        is_uncertain = len(prediction_set) > 1
        confidence_interp = self.conformal.get_confidence_interpretation(pred_sets[0])

        tier_result = self.tier_classifier.classify(calibrated_prob)
        top_positive, top_negative, shap_values = self._compute_shap(X)

        counterfactuals = []
        recommended_changes = []
        if generate_counterfactuals and calibrated_prob < 0.5 and self.cf_generator:
            try:
                counterfactuals = self.cf_generator.generate(site_features, n_counterfactuals=3)
                if counterfactuals:
                    all_changes = set()
                    for cf in counterfactuals:
                        all_changes.update(cf.changes.keys())
                    recommended_changes = list(all_changes)[:5]
            except Exception as e:
                warnings.warn(f"Counterfactual generation failed: {e}")

        return ExplanationResult(
            site_id=site_id, raw_prediction=float(raw_proba),
            calibrated_probability=float(calibrated_prob),
            prediction_set=prediction_set, is_uncertain=is_uncertain,
            confidence_interpretation=confidence_interp,
            tier=tier_result.tier, tier_label=tier_result.label,
            tier_action=tier_result.action, confidence_statement=tier_result.confidence_statement,
            historical_accuracy=tier_result.historical_accuracy,
            top_positive_drivers=top_positive, top_negative_drivers=top_negative,
            shap_values=shap_values, counterfactuals=counterfactuals,
            recommended_changes=recommended_changes,
        )

    def _compute_shap(self, X):
        if self.shap_explainer is None:
            return [], [], None
        try:
            shap_values = self.shap_explainer.shap_values(X, nsamples=50)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap_values = shap_values.ravel()
            feature_impacts = list(zip(self.feature_names, shap_values))
            positive = sorted([(f, v) for f, v in feature_impacts if v > 0], key=lambda x: x[1], reverse=True)[:5]
            negative = sorted([(f, v) for f, v in feature_impacts if v < 0], key=lambda x: x[1])[:5]
            return positive, negative, shap_values
        except Exception as e:
            warnings.warn(f"SHAP computation failed: {e}")
            return [], [], None

    def get_fleet_interventions(self, low_value_sites, progress_callback=None):
        """Identify fleet-wide upgrade interventions."""
        if self.cf_generator is None:
            return []
        cf_results = self.cf_generator.generate_batch(low_value_sites, 3, progress_callback)
        self.path_clusterer.fit(cf_results, self.actionable_features)
        paths = self.path_clusterer.get_upgrade_paths(len(low_value_sites))
        return [path.to_dict() for path in paths]

    def save(self, directory):
        """Save pipeline components."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.calibrator.save(directory / 'calibrator.pkl')
        self.conformal.save(directory / 'conformal.pkl')
        with open(directory / 'tier_classifier.pkl', 'wb') as f:
            pickle.dump(self.tier_classifier.to_dict(), f)
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
    def load(cls, directory, model, train_data, device='cpu'):
        """Load pipeline from directory."""
        directory = Path(directory)
        with open(directory / 'pipeline_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        pipeline = cls(
            model=model, train_data=train_data,
            feature_names=metadata['feature_names'],
            continuous_features=metadata['continuous_features'],
            n_numeric=metadata['n_numeric'],
            n_categorical=metadata['n_categorical'],
            n_boolean=metadata['n_boolean'],
            actionable_features=metadata['actionable_features'],
            device=device, alpha=metadata['alpha'],
        )
        pipeline.calibrator = ProbabilityCalibrator.load(directory / 'calibrator.pkl')
        pipeline.conformal = ConformalClassifier.load(
            directory / 'conformal.pkl', model=model,
            n_numeric=metadata['n_numeric'], n_categorical=metadata['n_categorical'],
            n_boolean=metadata['n_boolean'], device=device,
        )
        pipeline.sklearn_wrapper = pipeline.conformal.sklearn_wrapper
        with open(directory / 'tier_classifier.pkl', 'rb') as f:
            tier_data = pickle.load(f)
            pipeline.tier_classifier = TierClassifier.from_dict(tier_data)
        pipeline._fitted = metadata['fitted']
        return pipeline
