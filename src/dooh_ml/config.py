"""Configuration management for DOOH ML platform."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration."""

    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("DB_NAME", "dooh_sites"))
    username: str = field(default_factory=lambda: os.getenv("DB_USER", "pgadmin"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    schema: str = "dooh"

    @property
    def connection_string(self) -> str:
        """SQLAlchemy connection string."""
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def async_connection_string(self) -> str:
        """Async SQLAlchemy connection string."""
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", "azureml://eastus.api.azureml.ms/mlflow/v1.0"
        )
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "dooh-site-optimization")
    )
    registry_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_REGISTRY_URI", "")
    )


@dataclass
class ModelConfig:
    """Model hyperparameters and settings."""

    # Similarity model
    similarity_k_neighbors: int = 10
    similarity_high_performer_quantile: float = 0.75

    # Causal model
    causal_n_estimators: int = 500
    causal_min_samples_leaf: int = 50
    causal_cv_folds: int = 5

    # Classifier model
    classifier_iterations: int = 500
    classifier_learning_rate: float = 0.1
    classifier_depth: int = 6
    classifier_early_stopping_rounds: int = 50

    # Prioritization weights
    priority_weight_lookalike: float = 0.2
    priority_weight_success: float = 0.3
    priority_weight_uplift: float = 0.5


@dataclass
class FeatureConfig:
    """Feature column definitions."""

    # Continuous features
    continuous_features: list = field(
        default_factory=lambda: [
            "traffic_volume",
            "distance_to_highway_km",
            "poi_density",
            "competitor_count",
            "screen_size_inches",
            "avg_loop_length_seconds",
            "avg_cpm_floor",
            "revenue_30d",
            "revenue_90d",
            "nearby_avg_revenue_30d",
        ]
    )

    # Categorical features
    categorical_features: list = field(
        default_factory=lambda: [
            "market_region",
            "display_technology",
            "primary_content_type",
            "primary_content_category",
        ]
    )

    # Treatment variable (for causal model)
    treatment_column: str = "display_technology"
    treatment_control: str = "LCD"

    # Outcome variable
    outcome_column: str = "revenue_30d"

    # Target for classifier
    target_column: str = "reached_threshold"

    # Confounders (affect both treatment and outcome)
    confounder_columns: list = field(
        default_factory=lambda: [
            "traffic_volume",
            "distance_to_highway_km",
            "poi_density",
            "market_region",
            "nearby_avg_revenue_30d",
        ]
    )

    # Effect modifiers (where treatment effect varies)
    effect_modifier_columns: list = field(
        default_factory=lambda: [
            "market_region",
            "traffic_volume",
            "poi_density",
        ]
    )

    @property
    def all_features(self) -> list:
        """All feature columns."""
        return self.continuous_features + self.categorical_features


@dataclass
class Config:
    """Main configuration container."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Training settings
    train_end_date: Optional[str] = None
    validation_start_date: Optional[str] = None
    validation_end_date: Optional[str] = None
    test_start_date: Optional[str] = None
    gap_days: int = 14  # Gap between train and validation

    # Revenue threshold for "success"
    revenue_threshold_quantile: float = 0.75
    success_months: int = 12  # Months to reach threshold


# Global config instance
config = Config()


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    return Config(
        database=DatabaseConfig(),
        mlflow=MLflowConfig(),
        model=ModelConfig(),
        features=FeatureConfig(),
    )
