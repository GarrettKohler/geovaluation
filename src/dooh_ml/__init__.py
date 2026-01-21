"""DOOH Site Optimization ML Package.

Three-model architecture for site prioritization:
- Similarity: Gower distance lookalike modeling
- Causal: Double ML for hardware treatment effects
- Classifier: CatBoost for activation success prediction
"""

__version__ = "0.1.0"

from .models.similarity import SimilarityModel
from .models.causal import CausalModel
from .models.classifier import ActivationClassifier
from .inference.prioritizer import SitePrioritizer

__all__ = [
    "SimilarityModel",
    "CausalModel",
    "ActivationClassifier",
    "SitePrioritizer",
]
