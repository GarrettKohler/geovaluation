"""
Feature Selection Module for Site Scoring.

Implements 5 techniques for dynamic feature selection during neural network training:
1. LassoNet - Hierarchical L1 constraints with skip connections
2. Stochastic Gates (STG) - Learnable Bernoulli gates for L0 regularization
3. SHAP-Select - Post-training iterative elimination using statistical significance
4. TabNet - Instance-wise sparsemax attention (external library integration)
5. Gradient Analysis - Weight/gradient profile tracking during training

Usage:
    from site_scoring.feature_selection import (
        FeatureSelectionConfig,
        FeatureSelectionMethod,
        FeatureSelectionTrainer,
        create_feature_selection_model,
        get_preset,
    )

    # Quick start with preset
    config = get_preset('stg_light')
    model, fs_trainer = create_feature_selection_model(config, base_model, ...)
"""

# Configuration
from .config import FeatureSelectionConfig, FeatureSelectionMethod, get_preset, PRESETS

# Integration
from .integration import (
    FeatureSelectionTrainer,
    create_feature_selection_model,
    STGAugmentedModel,
)

# Individual techniques
from .stochastic_gates import StochasticGates, STGWrapper
from .lassonet import LassoNetLayer, LassoNetModel, HierProx, LassoNetWrapper
from .shap_select import ShapSelect, apply_shap_select, get_shap_feature_importance
from .gradient_analyzer import GradientFeatureAnalyzer, EpochWiseFeatureEliminator
from .tabnet_wrapper import TabNetWrapper, create_tabnet_model

__all__ = [
    # Configuration
    'FeatureSelectionConfig',
    'FeatureSelectionMethod',
    'get_preset',
    'PRESETS',
    # Integration
    'FeatureSelectionTrainer',
    'create_feature_selection_model',
    'STGAugmentedModel',
    # Stochastic Gates
    'StochasticGates',
    'STGWrapper',
    # LassoNet
    'LassoNetLayer',
    'LassoNetModel',
    'HierProx',
    'LassoNetWrapper',
    # SHAP-Select
    'ShapSelect',
    'apply_shap_select',
    'get_shap_feature_importance',
    # Gradient Analysis
    'GradientFeatureAnalyzer',
    'EpochWiseFeatureEliminator',
    # TabNet
    'TabNetWrapper',
    'create_tabnet_model',
]
