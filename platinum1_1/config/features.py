"""
Feature definitions and model configuration for the platinum1_1 backend.

Uses an extensible FeatureRegistry pattern: new features can be registered
without modifying existing code.  All feature lists are the canonical source
of truth for the ML pipeline and the web UI.

No Kroger features in this version.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# =============================================================================
# Feature type taxonomy
# =============================================================================

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


# =============================================================================
# Single feature definition
# =============================================================================

@dataclass(frozen=True)
class FeatureDefinition:
    """Immutable description of a single feature column."""

    name: str
    type: FeatureType
    description: str = ""
    group: str = "default"       # logical grouping: momentum, geospatial, ...
    default_enabled: bool = True


# =============================================================================
# Registry (class-level singleton)
# =============================================================================

class FeatureRegistry:
    """
    Extensible feature registry.

    Add new features by calling ``FeatureRegistry.register(FeatureDefinition(...))``
    anywhere at import time.  All query methods read from the same class-level
    dict so they always reflect the latest registrations.
    """

    _features: Dict[str, FeatureDefinition] = {}

    # -- mutators -------------------------------------------------------------

    @classmethod
    def register(cls, feature: FeatureDefinition) -> None:
        """Register a feature definition (overwrites if name exists)."""
        cls._features[feature.name] = feature

    @classmethod
    def register_many(cls, features: List[FeatureDefinition]) -> None:
        """Convenience: register a list of features at once."""
        for f in features:
            cls._features[f.name] = f

    # -- queries --------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> FeatureDefinition:
        """Look up a single feature by name. Raises KeyError if missing."""
        return cls._features[name]

    @classmethod
    def get_by_type(
        cls,
        ftype: FeatureType,
        enabled_only: bool = True,
    ) -> List[str]:
        """Return feature names matching *ftype*, optionally filtered."""
        return [
            f.name
            for f in cls._features.values()
            if f.type == ftype and (not enabled_only or f.default_enabled)
        ]

    @classmethod
    def get_by_group(cls, group: str) -> List[str]:
        """Return feature names belonging to *group*."""
        return [
            f.name for f in cls._features.values() if f.group == group
        ]

    @classmethod
    def all_features(cls) -> Dict[str, FeatureDefinition]:
        """Return a shallow copy of the full registry."""
        return dict(cls._features)

    @classmethod
    def summary(cls) -> Dict[str, int]:
        """Count of features per type."""
        counts: Dict[str, int] = {}
        for f in cls._features.values():
            counts[f.type.value] = counts.get(f.type.value, 0) + 1
        return counts


# =============================================================================
# Register all known features
# =============================================================================

def _register_defaults() -> None:
    """Populate the registry with all known columns from the training data."""

    # -- Numeric: Momentum indicators ----------------------------------------
    _momentum = [
        FeatureDefinition("rs_NVIs_95_185", FeatureType.NUMERIC,
                          "Relative strength NVIs 3/6 months", "momentum"),
        FeatureDefinition("rs_Revenue_95_185", FeatureType.NUMERIC,
                          "Relative strength Revenue 3/6 months", "momentum"),
        FeatureDefinition("rs_NVIs_185_370", FeatureType.NUMERIC,
                          "Relative strength NVIs 6/12 months", "momentum"),
        FeatureDefinition("rs_Revenue_185_370", FeatureType.NUMERIC,
                          "Relative strength Revenue 6/12 months", "momentum"),
        FeatureDefinition("rs_NVIs_370_740", FeatureType.NUMERIC,
                          "Relative strength NVIs 12/24 months", "momentum"),
        FeatureDefinition("rs_Revenue_370_740", FeatureType.NUMERIC,
                          "Relative strength Revenue 12/24 months", "momentum"),
    ]

    # -- Numeric: Revenue ----------------------------------------------------
    _revenue = [
        FeatureDefinition("log_total_revenue", FeatureType.NUMERIC,
                          "Log-transformed total revenue", "revenue"),
    ]

    # -- Numeric: Geospatial distances (log-transformed) ---------------------
    _geospatial = [
        FeatureDefinition("log_min_distance_to_nearest_site_mi", FeatureType.NUMERIC,
                          "Log distance to nearest competing site (mi)", "geospatial"),
        FeatureDefinition("log_min_distance_to_interstate_mi", FeatureType.NUMERIC,
                          "Log distance to nearest interstate (mi)", "geospatial"),
        FeatureDefinition("log_min_distance_to_mcdonalds_mi", FeatureType.NUMERIC,
                          "Log distance to nearest McDonald's (mi)", "geospatial"),
        FeatureDefinition("log_min_distance_to_walmart_mi", FeatureType.NUMERIC,
                          "Log distance to nearest Walmart (mi)", "geospatial"),
        FeatureDefinition("log_min_distance_to_target_mi", FeatureType.NUMERIC,
                          "Log distance to nearest Target (mi)", "geospatial"),
    ]

    # -- Numeric: Demographic ------------------------------------------------
    _demographic = [
        FeatureDefinition("log_avg_household_income", FeatureType.NUMERIC,
                          "Log average household income", "demographic"),
        FeatureDefinition("median_age", FeatureType.NUMERIC,
                          "Median age of surrounding area", "demographic"),
        FeatureDefinition("pct_female", FeatureType.NUMERIC,
                          "Percentage female population", "demographic"),
    ]

    # -- Categorical ---------------------------------------------------------
    _categorical = [
        FeatureDefinition("network", FeatureType.CATEGORICAL,
                          "Hardware network", "site_metadata"),
        FeatureDefinition("program", FeatureType.CATEGORICAL,
                          "Program type", "site_metadata"),
        FeatureDefinition("experience_type", FeatureType.CATEGORICAL,
                          "Screen experience type", "site_metadata"),
        FeatureDefinition("hardware_type", FeatureType.CATEGORICAL,
                          "Hardware model", "site_metadata"),
        FeatureDefinition("retailer", FeatureType.CATEGORICAL,
                          "Retailer name", "site_metadata"),
        FeatureDefinition("brand_fuel", FeatureType.CATEGORICAL,
                          "Fuel brand", "site_metadata"),
        FeatureDefinition("brand_c_store", FeatureType.CATEGORICAL,
                          "Convenience store brand", "site_metadata"),
    ]

    # -- Boolean: Restriction flags (r_*_encoded) ----------------------------
    _restrictions = [
        FeatureDefinition("r_retail_car_wash_encoded", FeatureType.BOOLEAN,
                          "Restriction: retail car wash", "restrictions"),
        FeatureDefinition("r_cpg_beverage_beer_oof_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage beer OOH", "restrictions"),
        FeatureDefinition("r_cpg_beverage_beer_vide_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage beer video", "restrictions"),
        FeatureDefinition("r_cpg_beverage_wine_oof_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage wine OOH", "restrictions"),
        FeatureDefinition("r_cpg_beverage_wine_video_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage wine video", "restrictions"),
        FeatureDefinition("r_finance_credit_cards_encoded", FeatureType.BOOLEAN,
                          "Restriction: finance credit cards", "restrictions"),
        FeatureDefinition("r_cpg_cbd_hemp_ingestibles_non_thc_encoded", FeatureType.BOOLEAN,
                          "Restriction: CBD hemp ingestibles (non-THC)", "restrictions"),
        FeatureDefinition("r_cpg_non_food_beverage_cannabis_medical_encoded", FeatureType.BOOLEAN,
                          "Restriction: cannabis medical", "restrictions"),
        FeatureDefinition("r_cpg_non_food_beverage_cannabis_recreational_encoded", FeatureType.BOOLEAN,
                          "Restriction: cannabis recreational", "restrictions"),
        FeatureDefinition("r_cpg_non_food_beverage_cbd_hemp_non_thc_encoded", FeatureType.BOOLEAN,
                          "Restriction: CBD hemp non-THC", "restrictions"),
        FeatureDefinition("r_automotive_after_market_oil_encoded", FeatureType.BOOLEAN,
                          "Restriction: automotive aftermarket oil", "restrictions"),
        FeatureDefinition("r_cpg_beverage_spirits_ooh_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage spirits OOH", "restrictions"),
        FeatureDefinition("r_cpg_beverage_spirits_video_encoded", FeatureType.BOOLEAN,
                          "Restriction: CPG beverage spirits video", "restrictions"),
        FeatureDefinition("r_cpg_non_food_beverage_e_cigarette_encoded", FeatureType.BOOLEAN,
                          "Restriction: e-cigarette", "restrictions"),
        FeatureDefinition("r_entertainment_casinos_and_gambling_encoded", FeatureType.BOOLEAN,
                          "Restriction: casinos and gambling", "restrictions"),
        FeatureDefinition("r_government_political_encoded", FeatureType.BOOLEAN,
                          "Restriction: government political", "restrictions"),
        FeatureDefinition("r_automotive_electric_encoded", FeatureType.BOOLEAN,
                          "Restriction: automotive electric", "restrictions"),
        FeatureDefinition("r_recruitment_encoded", FeatureType.BOOLEAN,
                          "Restriction: recruitment", "restrictions"),
        FeatureDefinition("r_restaurants_cdr_encoded", FeatureType.BOOLEAN,
                          "Restriction: restaurants CDR", "restrictions"),
        FeatureDefinition("r_restaurants_qsr_encoded", FeatureType.BOOLEAN,
                          "Restriction: restaurants QSR", "restrictions"),
        FeatureDefinition("r_retail_automotive_service_encoded", FeatureType.BOOLEAN,
                          "Restriction: retail automotive service", "restrictions"),
        FeatureDefinition("r_retail_grocery_encoded", FeatureType.BOOLEAN,
                          "Restriction: retail grocery", "restrictions"),
        FeatureDefinition("r_retail_grocerty_with_fuel_encoded", FeatureType.BOOLEAN,
                          "Restriction: retail grocery with fuel", "restrictions"),
    ]

    # -- Boolean: Capability flags (c_*_encoded) -----------------------------
    _capabilities = [
        FeatureDefinition("c_emv_enabled_encoded", FeatureType.BOOLEAN,
                          "Capability: EMV enabled", "capabilities"),
        FeatureDefinition("c_nfc_enabled_encoded", FeatureType.BOOLEAN,
                          "Capability: NFC enabled", "capabilities"),
        FeatureDefinition("c_open_24_hours_encoded", FeatureType.BOOLEAN,
                          "Capability: open 24 hours", "capabilities"),
        FeatureDefinition("c_sells_beer_encoded", FeatureType.BOOLEAN,
                          "Capability: sells beer", "capabilities"),
        FeatureDefinition("c_sells_diesel_fuel_encoded", FeatureType.BOOLEAN,
                          "Capability: sells diesel fuel", "capabilities"),
        FeatureDefinition("c_sells_lottery_encoded", FeatureType.BOOLEAN,
                          "Capability: sells lottery", "capabilities"),
        FeatureDefinition("c_vistar_programmatic_enabled_encoded", FeatureType.BOOLEAN,
                          "Capability: Vistar programmatic enabled", "capabilities"),
        FeatureDefinition("c_sells_wine_encoded", FeatureType.BOOLEAN,
                          "Capability: sells wine", "capabilities"),
    ]

    # -- Boolean: Other operational flags ------------------------------------
    _operational = [
        FeatureDefinition("schedule_site_encoded", FeatureType.BOOLEAN,
                          "Site is schedulable", "operational"),
        FeatureDefinition("sellable_site_encoded", FeatureType.BOOLEAN,
                          "Site is sellable", "operational"),
    ]

    # -- Register everything -------------------------------------------------
    all_features = (
        _momentum
        + _revenue
        + _geospatial
        + _demographic
        + _categorical
        + _restrictions
        + _capabilities
        + _operational
    )
    FeatureRegistry.register_many(all_features)


# Execute registration at import time
_register_defaults()


# =============================================================================
# Model configuration
# =============================================================================

@dataclass
class ModelConfig:
    """
    Full configuration for a training run.

    Pulls feature lists from the FeatureRegistry and applies leakage
    prevention automatically.
    """

    # Target and task
    target: str = "avg_monthly_revenue"
    task_type: str = "regression"              # "regression" or "lookalike"
    lookalike_lower_percentile: int = 90
    lookalike_upper_percentile: int = 100

    # Model architecture (neural network)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    embedding_dim: int = 16
    dropout: float = 0.2
    use_batch_norm: bool = True

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    early_stopping_patience: int = 10

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Network filter (None = all networks)
    network_filter: Optional[str] = None

    # Model type selector
    model_type: str = "neural_network"         # "neural_network" or "xgboost"

    # -----------------------------------------------------------------
    # Feature access (delegates to FeatureRegistry with leakage guard)
    # -----------------------------------------------------------------

    def get_numeric_features(self) -> List[str]:
        """Numeric features minus any that would leak the target."""
        features = FeatureRegistry.get_by_type(FeatureType.NUMERIC)
        leakage = self._get_leakage_columns()
        return [f for f in features if f not in leakage]

    def get_categorical_features(self) -> List[str]:
        """All registered categorical features."""
        return FeatureRegistry.get_by_type(FeatureType.CATEGORICAL)

    def get_boolean_features(self) -> List[str]:
        """All registered boolean features."""
        return FeatureRegistry.get_by_type(FeatureType.BOOLEAN)

    def get_all_features(self) -> List[str]:
        """Union of numeric + categorical + boolean (leakage-safe)."""
        return (
            self.get_numeric_features()
            + self.get_categorical_features()
            + self.get_boolean_features()
        )

    def _get_leakage_columns(self) -> set:
        """
        Columns that must be excluded to prevent data leakage.

        If predicting avg_monthly_revenue, total_revenue and its log
        are leaked; and vice-versa.
        """
        leakage = {self.target}
        if self.target == "total_revenue":
            leakage.update(["log_total_revenue", "avg_monthly_revenue"])
        elif self.target == "avg_monthly_revenue":
            leakage.update(["total_revenue", "log_total_revenue"])
        return leakage
