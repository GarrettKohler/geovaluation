"""
Configuration package for the platinum1_1 backend.

Public API:
    Settings, get_settings  -- environment-based application settings
    FeatureType, FeatureDefinition, FeatureRegistry -- extensible feature catalog
    ModelConfig             -- full training-run configuration
"""

from .settings import Settings, get_settings
from .features import (
    FeatureType,
    FeatureDefinition,
    FeatureRegistry,
    ModelConfig,
)

__all__ = [
    "Settings",
    "get_settings",
    "FeatureType",
    "FeatureDefinition",
    "FeatureRegistry",
    "ModelConfig",
]
