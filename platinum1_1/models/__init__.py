"""
Models module for the platinum1_1 backend.

Provides neural network and XGBoost model implementations
along with a factory function for model creation.
"""

from .neural_network import SiteScoringModel, ResidualBlock, CategoricalEmbedding
from .xgboost_model import XGBoostModel
from .factory import create_model

__all__ = [
    "SiteScoringModel",
    "ResidualBlock",
    "CategoricalEmbedding",
    "XGBoostModel",
    "create_model",
]
