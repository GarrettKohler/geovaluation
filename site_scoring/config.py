"""
Configuration for site scoring model.
Optimized for Apple M4 with MPS backend.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch

# Import feature selection config
from .feature_selection.config import FeatureSelectionConfig, FeatureSelectionMethod, get_preset

# Project root - dynamically resolved from this file's location
# site_scoring/config.py -> go up one level to get project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default paths (relative to project root)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "site_training_data.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "site_scoring" / "outputs"


@dataclass
class Config:
    """Configuration for site scoring pipeline."""

    # Paths - USE AGGREGATED DATA (one row per site, ~26K Active sites)
    # NOT the raw monthly data (1.4M rows - that causes data leakage!)
    data_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    # Target variable - aggregated metrics
    # Options: avg_monthly_revenue (recommended), total_revenue, total_monthly_impressions
    target: str = "avg_monthly_revenue"

    # Task type: "regression" (predict revenue) or "lookalike" (classify top performers)
    task_type: str = "regression"

    # Lookalike classifier percentile bounds
    # Sites with revenue percentile >= lower_percentile and <= upper_percentile are labeled as "top performers" (1)
    # All other sites (below lower_percentile) are labeled as "non-performers" (0)
    lookalike_lower_percentile: int = 90  # Default: top 10% (90th percentile and above)
    lookalike_upper_percentile: int = 100  # Default: include all above lower bound

    # Clustering configuration (Deep Embedded Clustering)
    # Used to segment top performers identified by lookalike classifier
    n_clusters: int = 5  # Number of clusters to discover (2-10 recommended)
    latent_dim: int = 32  # Dimension of autoencoder latent space
    pretrain_epochs: int = 20  # Epochs for autoencoder pretraining
    clustering_epochs: int = 30  # Epochs for clustering refinement with KL-divergence
    cluster_probability_threshold: float = 0.5  # Min lookalike probability to include in clustering

    # Device - auto-detect M4 MPS
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")

    # Data loading - optimized for large files
    batch_size: int = 4096  # Large batches for M4 GPU
    num_workers: int = 4    # M4 has efficient cores
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Feature columns by type - AGGREGATED DATA COLUMNS
    # These match the columns in site_training_data.parquet (one row per site)
    # Note: NVI, per-screen metrics, and raw coordinates excluded
    numeric_features: List[str] = field(default_factory=lambda: [
        # Multi-horizon relative strength indicators (momentum features)
        # Data is MONTHLY (1 record per site per month), windows calibrated for monthly points:
        # Short-term (3/6 months = 95/185 days): Recent quarter vs half-year momentum
        "rs_Impressions_95_185", "rs_NVIs_95_185", "rs_Revenue_95_185", "rs_RevenuePerScreen_95_185",
        # Medium-term (6/12 months = 185/370 days): Half-year vs annual
        "rs_Impressions_185_370", "rs_NVIs_185_370", "rs_Revenue_185_370", "rs_RevenuePerScreen_185_370",
        # Long-term (12/24 months = 370/740 days): Annual vs 2-year trend
        "rs_Impressions_370_740", "rs_NVIs_370_740", "rs_Revenue_370_740", "rs_RevenuePerScreen_370_740",
        # Revenue metrics
        "avg_monthly_revenue",
        "log_total_revenue",
        # Geospatial distances (log-transformed for better distribution)
        "log_min_distance_to_nearest_site_mi", "log_min_distance_to_interstate_mi",
        "log_min_distance_to_kroger_mi", "log_min_distance_to_mcdonalds_mi",
        # Demographics
        "avg_household_income", "median_age",
        "pct_female", "pct_male",
    ])

    categorical_features: List[str] = field(default_factory=lambda: [
        # Site metadata (no month/year - data is aggregated!)
        "network", "program", "experience_type", "hardware_type", "retailer",
        "brand_fuel", "brand_restaurant", "brand_c_store",
        "nearest_interstate",  # Geospatial
    ])

    boolean_features: List[str] = field(default_factory=lambda: [
        # Pre-encoded boolean flags from aggregated data (already 0/1 integers)
        # Restriction flags (r_*_encoded)
        "r_lottery_encoded", "r_government_encoded", "r_travel_and_tourism_encoded",
        "r_retail_car_wash_encoded", "r_cpg_beverage_beer_oof_encoded",
        "r_cpg_beverage_beer_vide_encoded", "r_cpg_beverage_wine_oof_encoded",
        "r_cpg_beverage_wine_video_encoded", "r_finance_credit_cards_encoded",
        "r_cpg_cbd_hemp_ingestibles_non_thc_encoded",
        "r_cpg_non_food_beverage_cannabis_medical_encoded",
        "r_cpg_non_food_beverage_cannabis_recreational_encoded",
        "r_cpg_non_food_beverage_cbd_hemp_non_thc_encoded",
        "r_alcohol_drink_responsibly_message_encoded", "r_alternative_transportation_encoded",
        "r_associations_and_npo_anti_smoking_encoded", "r_automotive_after_market_oil_encoded",
        "r_cpg_beverage_spirits_ooh_encoded", "r_cpg_beverage_spirits_video_encoded",
        "r_cpg_non_food_beverage_e_cigarette_encoded",
        "r_entertainment_casinos_and_gambling_encoded",
        "r_government_political_encoded", "r_automotive_electric_encoded",
        "r_recruitment_encoded", "r_restaurants_cdr_encoded", "r_restaurants_qsr_encoded",
        "r_retail_automotive_service_encoded", "r_retail_grocery_encoded",
        "r_retail_grocerty_with_fuel_encoded",
        # Capability flags (c_*_encoded)
        "c_emv_enabled_encoded", "c_nfc_enabled_encoded", "c_open_24_hours_encoded",
        "c_sells_beer_encoded", "c_sells_diesel_fuel_encoded", "c_sells_lottery_encoded",
        "c_vistar_programmatic_enabled_encoded", "c_walk_up_enabled_encoded",
        "c_sells_wine_encoded",
        # Other booleans
        "schedule_site_encoded", "sellable_site_encoded",
    ])

    # Model architecture
    embedding_dim: int = 16          # Embedding dimension for categoricals
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4  # Lower LR for stability
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    early_stopping_patience: int = 10

    # Mixed precision for M4 optimization
    use_amp: bool = False  # MPS doesn't fully support AMP yet

    # Logging
    log_interval: int = 100

    # Feature Selection Configuration
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Remove target and derived columns from features to prevent data leakage
        leakage_columns = {self.target}
        if self.target == "total_revenue":
            leakage_columns.update(["log_total_revenue", "avg_monthly_revenue"])
        elif self.target == "avg_monthly_revenue":
            leakage_columns.update(["total_revenue", "log_total_revenue"])

        self.numeric_features = [f for f in self.numeric_features if f not in leakage_columns]

    def set_feature_selection_preset(self, preset_name: str):
        """
        Set feature selection configuration using a preset.

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
        self.feature_selection = get_preset(preset_name)

    def apply_model_preset(self, preset_name: str):
        """
        Apply a model preset that controls which features are included.

        Available presets:
        - 'model_a': All available features from the source dataset
        - 'model_b': Curated features (excludes retailer, pct_male, nearest_interstate)
        """
        preset = get_model_preset(preset_name)
        self.numeric_features = list(preset['numeric'])
        self.categorical_features = list(preset['categorical'])
        self.boolean_features = list(preset['boolean'])
        # Re-run leakage removal with new feature lists
        self.__post_init__()


# =============================================================================
# Model Presets: Define which features each model variant uses
# =============================================================================

# Model B: Curated feature set — removes retailer, pct_male, nearest_interstate
# Hypothesis: these features may be noise or proxies that don't generalize
_MODEL_B_NUMERIC = [
    # Multi-horizon RS (momentum) - reduces overfitting vs single horizon
    # Horizons: 3/6 months, 6/12 months, 12/24 months
    "rs_Impressions_95_185", "rs_NVIs_95_185", "rs_Revenue_95_185", "rs_RevenuePerScreen_95_185",
    "rs_Impressions_185_370", "rs_NVIs_185_370", "rs_Revenue_185_370", "rs_RevenuePerScreen_185_370",
    "rs_Impressions_370_740", "rs_NVIs_370_740", "rs_Revenue_370_740", "rs_RevenuePerScreen_370_740",
    "avg_monthly_revenue", "log_total_revenue",
    "log_min_distance_to_nearest_site_mi", "log_min_distance_to_interstate_mi",
    "log_min_distance_to_kroger_mi", "log_min_distance_to_mcdonalds_mi",
    "avg_household_income", "median_age",
    "pct_female",
    # pct_male removed (collinear with pct_female)
]

_MODEL_B_CATEGORICAL = [
    "network", "program", "experience_type", "hardware_type",
    # retailer removed
    "brand_fuel", "brand_restaurant", "brand_c_store",
    # nearest_interstate removed
]

# Model A: All available features from site_training_data.parquet
# Includes additional demographic, per-screen, and volume metrics
_MODEL_A_NUMERIC = [
    # Multi-horizon relative strength indicators (momentum)
    # Horizons: 3/6 months, 6/12 months, 12/24 months
    "rs_Impressions_95_185", "rs_NVIs_95_185", "rs_Revenue_95_185", "rs_RevenuePerScreen_95_185",
    "rs_Impressions_185_370", "rs_NVIs_185_370", "rs_Revenue_185_370", "rs_RevenuePerScreen_185_370",
    "rs_Impressions_370_740", "rs_NVIs_370_740", "rs_Revenue_370_740", "rs_RevenuePerScreen_370_740",
    # Revenue metrics
    "avg_monthly_revenue", "log_total_revenue",
    # Volume metrics (additional)
    "avg_monthly_monthly_impressions", "avg_monthly_monthly_nvis",
    "avg_monthly_monthly_impressions_per_screen",
    "avg_monthly_monthly_nvis_per_screen",
    "avg_monthly_monthly_revenue_per_screen",
    # Log-transformed totals (additional)
    "log_total_monthly_impressions", "log_total_monthly_nvis",
    "log_total_monthly_impressions_per_screen",
    "log_total_monthly_nvis_per_screen",
    "log_total_monthly_revenue_per_screen",
    # Geospatial distances
    "log_min_distance_to_nearest_site_mi", "log_min_distance_to_interstate_mi",
    "log_min_distance_to_kroger_mi", "log_min_distance_to_mcdonalds_mi",
    # Demographics (full set)
    "avg_household_income", "median_age",
    "pct_female", "pct_male",
    "pct_african_american", "pct_asian", "pct_hispanic",
    # Site characteristics
    "screen_count", "dma_rank", "active_months",
]

_MODEL_A_CATEGORICAL = [
    "network", "program", "experience_type", "hardware_type", "retailer",
    "brand_fuel", "brand_restaurant", "brand_c_store",
    "nearest_interstate",
]

# Boolean features are the same for both presets (all r_* and c_* flags)
_ALL_BOOLEAN = [
    "r_lottery_encoded", "r_government_encoded", "r_travel_and_tourism_encoded",
    "r_retail_car_wash_encoded", "r_cpg_beverage_beer_oof_encoded",
    "r_cpg_beverage_beer_vide_encoded", "r_cpg_beverage_wine_oof_encoded",
    "r_cpg_beverage_wine_video_encoded", "r_finance_credit_cards_encoded",
    "r_cpg_cbd_hemp_ingestibles_non_thc_encoded",
    "r_cpg_non_food_beverage_cannabis_medical_encoded",
    "r_cpg_non_food_beverage_cannabis_recreational_encoded",
    "r_cpg_non_food_beverage_cbd_hemp_non_thc_encoded",
    "r_alcohol_drink_responsibly_message_encoded", "r_alternative_transportation_encoded",
    "r_associations_and_npo_anti_smoking_encoded", "r_automotive_after_market_oil_encoded",
    "r_cpg_beverage_spirits_ooh_encoded", "r_cpg_beverage_spirits_video_encoded",
    "r_cpg_non_food_beverage_e_cigarette_encoded",
    "r_entertainment_casinos_and_gambling_encoded",
    "r_government_political_encoded", "r_automotive_electric_encoded",
    "r_recruitment_encoded", "r_restaurants_cdr_encoded", "r_restaurants_qsr_encoded",
    "r_retail_automotive_service_encoded", "r_retail_grocery_encoded",
    "r_retail_grocerty_with_fuel_encoded",
    "c_emv_enabled_encoded", "c_nfc_enabled_encoded", "c_open_24_hours_encoded",
    "c_sells_beer_encoded", "c_sells_diesel_fuel_encoded", "c_sells_lottery_encoded",
    "c_vistar_programmatic_enabled_encoded", "c_walk_up_enabled_encoded",
    "c_sells_wine_encoded",
    "schedule_site_encoded", "sellable_site_encoded",
]

MODEL_PRESETS: Dict[str, Dict] = {
    "model_a": {
        "name": "Train Model A",
        "description": "All available features from source datasets",
        "numeric": _MODEL_A_NUMERIC,
        "categorical": _MODEL_A_CATEGORICAL,
        "boolean": _ALL_BOOLEAN,
    },
    "model_b": {
        "name": "Train Model B",
        "description": "Curated features (no retailer, pct_male, nearest_interstate)",
        "numeric": _MODEL_B_NUMERIC,
        "categorical": _MODEL_B_CATEGORICAL,
        "boolean": _ALL_BOOLEAN,
    },
}


def get_model_preset(name: str) -> Dict:
    """Get a model preset by name. Raises ValueError if not found."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset '{name}'. Available: {', '.join(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[name]


def get_all_model_presets() -> Dict[str, Dict]:
    """Return all model presets with their metadata and feature counts."""
    result = {}
    for key, preset in MODEL_PRESETS.items():
        result[key] = {
            "name": preset["name"],
            "description": preset["description"],
            "counts": {
                "numeric": len(preset["numeric"]),
                "categorical": len(preset["categorical"]),
                "boolean": len(preset["boolean"]),
                "total": len(preset["numeric"]) + len(preset["categorical"]) + len(preset["boolean"]),
            },
        }
    return result


def get_all_available_features() -> Dict[str, List[str]]:
    """
    Return all available features across all presets (union of all features).
    This is useful for showing users what features can be selected.
    """
    all_numeric = set()
    all_categorical = set()
    all_boolean = set()

    for preset in MODEL_PRESETS.values():
        all_numeric.update(preset["numeric"])
        all_categorical.update(preset["categorical"])
        all_boolean.update(preset["boolean"])

    return {
        "numeric": sorted(list(all_numeric)),
        "categorical": sorted(list(all_categorical)),
        "boolean": sorted(list(all_boolean)),
    }


def filter_features_by_selection(
    preset_name: str,
    selected_features: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Filter preset features to only include selected features.

    Args:
        preset_name: The base preset to filter (model_a or model_b)
        selected_features: List of feature names to include. If None, returns all preset features.

    Returns:
        Dict with filtered numeric, categorical, and boolean feature lists.
    """
    preset = get_model_preset(preset_name)

    if selected_features is None:
        return {
            "numeric": list(preset["numeric"]),
            "categorical": list(preset["categorical"]),
            "boolean": list(preset["boolean"]),
        }

    selected_set = set(selected_features)

    return {
        "numeric": [f for f in preset["numeric"] if f in selected_set],
        "categorical": [f for f in preset["categorical"] if f in selected_set],
        "boolean": [f for f in preset["boolean"] if f in selected_set],
    }
