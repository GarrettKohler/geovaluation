"""Explainability module for site scoring predictions.

Components:
- ProbabilityCalibrator: Ensures predicted probabilities match observed frequencies
- TierClassifier: Maps probabilities to executive-friendly tiers
- ConformalClassifier: Prediction sets with coverage guarantees
- CounterfactualGenerator: "What-if" explanations for low-value sites
- ExplainabilityPipeline: Unified interface combining all components
"""

from .calibration import ProbabilityCalibrator
from .tiers import TierClassifier, TierResult
from .conformal import ConformalClassifier, SklearnModelWrapper
from .counterfactuals import CounterfactualGenerator, UpgradePathClusterer, Counterfactual
from .pipeline import ExplainabilityPipeline, ExplanationResult

__all__ = [
    "ProbabilityCalibrator",
    "TierClassifier",
    "TierResult",
    "ConformalClassifier",
    "SklearnModelWrapper",
    "CounterfactualGenerator",
    "UpgradePathClusterer",
    "Counterfactual",
    "ExplainabilityPipeline",
    "ExplanationResult",
]
