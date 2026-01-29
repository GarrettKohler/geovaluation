"""
Feature Selection Configuration.

Defines configuration options for all feature selection techniques.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class FeatureSelectionMethod(str, Enum):
    """Available feature selection methods."""
    NONE = "none"                           # No feature selection
    STOCHASTIC_GATES = "stochastic_gates"   # STG: Learnable Bernoulli gates
    LASSONET = "lassonet"                   # Hierarchical L1 constraints
    SHAP_SELECT = "shap_select"             # Post-training SHAP-based elimination
    TABNET = "tabnet"                       # Sparsemax attention (replaces MLP)
    GRADIENT_ANALYSIS = "gradient_analysis" # Weight/gradient tracking


@dataclass
class FeatureSelectionConfig:
    """
    Configuration for feature selection during training.

    This config controls which technique(s) to use and their hyperparameters.
    Multiple techniques can be combined (e.g., STG during training + SHAP post-training).
    """

    # Primary method to use during training
    method: FeatureSelectionMethod = FeatureSelectionMethod.NONE

    # Whether to run SHAP-Select post-training validation (can combine with any method)
    run_shap_validation: bool = False

    # Whether to track gradient/weight profiles (can combine with any method)
    track_gradients: bool = False

    # ==========================================================================
    # Stochastic Gates (STG) Parameters
    # ==========================================================================
    stg_sigma: float = 0.5          # Gate distribution spread
    stg_lambda: float = 0.1         # L0 regularization weight (higher = fewer features)
    stg_init_mean: float = 0.5      # Initial gate activation (higher = more features start active)
    stg_threshold: float = 0.5      # Threshold for considering a feature selected

    # ==========================================================================
    # LassoNet Parameters
    # ==========================================================================
    lassonet_M: float = 10.0        # Hierarchy coefficient (higher = more nonlinearity allowed)
    lassonet_lambda: float = 0.01   # L1 regularization strength
    lassonet_lambda_path: bool = False  # Whether to train across lambda path
    lassonet_n_lambdas: int = 20    # Number of lambdas in path
    lassonet_lambda_max: float = 1.0    # Maximum lambda value

    # ==========================================================================
    # SHAP-Select Parameters
    # ==========================================================================
    shap_significance_level: float = 0.05  # P-value threshold for feature significance
    shap_n_background: int = 100           # Number of background samples for SHAP
    shap_n_samples: int = 100              # Number of samples for SHAP approximation

    # ==========================================================================
    # TabNet Parameters
    # ==========================================================================
    tabnet_n_d: int = 64            # Width of decision prediction layer
    tabnet_n_a: int = 64            # Width of attention embedding
    tabnet_n_steps: int = 5         # Number of decision steps
    tabnet_gamma: float = 1.5       # Feature reuse penalty
    tabnet_mask_type: str = 'sparsemax'  # 'sparsemax' or 'entmax'

    # ==========================================================================
    # Gradient Analysis Parameters
    # ==========================================================================
    gradient_analysis_interval: int = 5    # Record every N epochs
    gradient_elimination_interval: int = 10  # Check elimination every N epochs
    gradient_elimination_percentile: float = 5  # Bottom X% to consider for elimination
    gradient_min_features: int = 5         # Minimum features to keep

    # ==========================================================================
    # General Parameters
    # ==========================================================================
    # Minimum selection threshold (features below this are eliminated)
    selection_threshold: float = 0.01

    # Maximum features to eliminate per epoch (safety limit)
    max_elimination_per_epoch: int = 5

    # Warm-up epochs before starting elimination
    warmup_epochs: int = 5

    def get_method_display_name(self) -> str:
        """Get human-readable name for the selected method."""
        names = {
            FeatureSelectionMethod.NONE: "None",
            FeatureSelectionMethod.STOCHASTIC_GATES: "Stochastic Gates (STG)",
            FeatureSelectionMethod.LASSONET: "LassoNet",
            FeatureSelectionMethod.SHAP_SELECT: "SHAP-Select",
            FeatureSelectionMethod.TABNET: "TabNet (Sparsemax)",
            FeatureSelectionMethod.GRADIENT_ANALYSIS: "Gradient Analysis",
        }
        return names.get(self.method, str(self.method))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'run_shap_validation': self.run_shap_validation,
            'track_gradients': self.track_gradients,
            'stg_sigma': self.stg_sigma,
            'stg_lambda': self.stg_lambda,
            'stg_threshold': self.stg_threshold,
            'lassonet_M': self.lassonet_M,
            'lassonet_lambda': self.lassonet_lambda,
            'tabnet_n_steps': self.tabnet_n_steps,
            'tabnet_gamma': self.tabnet_gamma,
            'shap_significance_level': self.shap_significance_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureSelectionConfig':
        """Create from dictionary."""
        config = cls()

        if 'method' in data:
            config.method = FeatureSelectionMethod(data['method'])

        # Copy other fields
        for key, value in data.items():
            if key != 'method' and hasattr(config, key):
                setattr(config, key, value)

        return config


# Preset configurations for common use cases
PRESETS = {
    'none': FeatureSelectionConfig(
        method=FeatureSelectionMethod.NONE,
    ),

    'stg_light': FeatureSelectionConfig(
        method=FeatureSelectionMethod.STOCHASTIC_GATES,
        stg_lambda=0.05,
        stg_sigma=0.5,
        track_gradients=True,
    ),

    'stg_aggressive': FeatureSelectionConfig(
        method=FeatureSelectionMethod.STOCHASTIC_GATES,
        stg_lambda=0.3,
        stg_sigma=0.5,
        run_shap_validation=True,
    ),

    'lassonet_standard': FeatureSelectionConfig(
        method=FeatureSelectionMethod.LASSONET,
        lassonet_M=10.0,
        lassonet_lambda=0.01,
        track_gradients=True,
    ),

    'lassonet_path': FeatureSelectionConfig(
        method=FeatureSelectionMethod.LASSONET,
        lassonet_M=10.0,
        lassonet_lambda_path=True,
        lassonet_n_lambdas=30,
    ),

    'shap_only': FeatureSelectionConfig(
        method=FeatureSelectionMethod.NONE,
        run_shap_validation=True,
        shap_significance_level=0.05,
    ),

    'tabnet': FeatureSelectionConfig(
        method=FeatureSelectionMethod.TABNET,
        tabnet_n_steps=5,
        tabnet_gamma=1.5,
    ),

    'hybrid_stg_shap': FeatureSelectionConfig(
        method=FeatureSelectionMethod.STOCHASTIC_GATES,
        stg_lambda=0.1,
        run_shap_validation=True,
        track_gradients=True,
    ),
}


def get_preset(name: str) -> FeatureSelectionConfig:
    """
    Get a preset configuration by name.

    Available presets:
    - 'none': No feature selection
    - 'stg_light': Light STG regularization (keeps most features)
    - 'stg_aggressive': Aggressive STG with SHAP validation
    - 'lassonet_standard': Standard LassoNet configuration
    - 'lassonet_path': LassoNet with full lambda path training
    - 'shap_only': Only post-training SHAP-Select
    - 'tabnet': TabNet with sparsemax attention
    - 'hybrid_stg_shap': STG during training + SHAP post-training
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return PRESETS[name]
