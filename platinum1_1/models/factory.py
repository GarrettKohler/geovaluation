"""
Model factory for creating neural network or XGBoost models.

Provides a single entry point for model creation based on configuration,
abstracting away the import and construction details.
"""

from typing import Any, Dict


def create_model(
    config: Any,
    categorical_vocab_sizes: Dict[str, int],
    n_numeric: int,
    n_boolean: int,
) -> Any:
    """
    Create model based on config.model_type.

    Args:
        config: Configuration object with at least a `model_type` attribute
                ("neural_network" or "xgboost") and `task_type` attribute.
                For neural networks, also uses: embedding_dim, hidden_dims,
                dropout, use_batch_norm.
        categorical_vocab_sizes: Mapping of categorical feature name to
                                 vocabulary size (from FeatureProcessor).
        n_numeric: Number of numeric features (from FeatureProcessor).
        n_boolean: Number of boolean features (from FeatureProcessor).

    Returns:
        Model instance (SiteScoringModel or XGBoostModel).

    Raises:
        ValueError: If config.model_type is not recognized.
    """
    model_type = getattr(config, "model_type", "neural_network")

    if model_type == "neural_network":
        from .neural_network import SiteScoringModel

        return SiteScoringModel.from_config(
            config, categorical_vocab_sizes, n_numeric, n_boolean
        )

    elif model_type == "xgboost":
        from .xgboost_model import XGBoostModel

        task_type = getattr(config, "task_type", "regression")
        return XGBoostModel(
            task_type=task_type,
            n_estimators=getattr(config, "n_estimators", 1000),
            learning_rate=getattr(config, "learning_rate", 0.05),
            max_depth=getattr(config, "max_depth", 6),
            early_stopping_rounds=getattr(config, "early_stopping_rounds", 50),
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: neural_network, xgboost"
        )
