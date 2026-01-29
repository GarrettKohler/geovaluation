"""
Explainability Module for Site Scoring.

Provides uncertainty quantification, probability calibration, and counterfactual
explanations for the site scoring model.

Key Components:
- ConformalClassifier: MAPIE wrapper for prediction sets with coverage guarantees
- ProbabilityCalibrator: Isotonic/Platt calibration for probability alignment
- CounterfactualGenerator: DiCE integration for "what-if" explanations
- TierClassifier: Executive-friendly confidence tiers
- ExplainabilityPipeline: Unified interface combining all components
"""

from .calibration import ProbabilityCalibrator
from .conformal import ConformalClassifier
from .counterfactuals import CounterfactualGenerator, UpgradePathClusterer
from .tiers import TierClassifier, TIER_LABELS, TIER_ACTIONS
from .pipeline import ExplainabilityPipeline, ExplanationResult

__all__ = [
    "ProbabilityCalibrator",
    "ConformalClassifier",
    "CounterfactualGenerator",
    "UpgradePathClusterer",
    "TierClassifier",
    "TIER_LABELS",
    "TIER_ACTIONS",
    "ExplainabilityPipeline",
    "ExplanationResult",
]
