"""
Data lineage service for tracking column transformations through the ML pipeline.

Loads the auto-generated data ontology (docs/data_ontology.yaml) and provides
static column-level lineage mappings that describe how each input column
is transformed before reaching the training dataset (site_training_data.parquet).

The 5 pipeline-active datasets are those whose downstream includes
site_aggregated_precleaned (the first aggregation step).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ColumnLineage:
    """Describes how a single input column flows through the pipeline."""
    input_column: str
    input_dataset: str
    transformations: List[str]
    output_columns: List[str]
    dropped: bool = False
    notes: str = ""


def load_ontology(path: Path) -> dict:
    """Parse the data ontology YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_datasets(ontology: dict) -> list:
    """
    Return the 5 pipeline-active source datasets.

    These are source datasets whose downstream field contains
    'site_aggregated_precleaned', meaning they feed the ML pipeline.
    """
    datasets = ontology.get("datasets", {})
    result = []
    for ds_id, ds in datasets.items():
        if ds.get("role") != "source":
            continue
        downstream = ds.get("downstream", [])
        if "site_aggregated_precleaned" in downstream:
            result.append(ds)
    # Sort by dataset ID for stable ordering
    result.sort(key=lambda d: d["id"])
    return result


def get_dataset_info(ontology: dict, dataset_id: str) -> Optional[dict]:
    """Get metadata for a single dataset by ID."""
    return ontology.get("datasets", {}).get(dataset_id)


def get_training_schema(ontology: dict) -> dict:
    """Get the training dataset schema for cross-referencing."""
    training = ontology.get("datasets", {}).get("site_training_data", {})
    introspection = training.get("introspection", {})
    return introspection.get("schema", {})


# =============================================================================
# Static Lineage Mappings
# =============================================================================
# Defined statically because transformations carry semantic meaning
# ("momentum indicator", "geographic leakage prevention") that cannot
# be inferred from AST parsing of data_transform.py.
# =============================================================================

def _site_scores_monthly_lineages() -> List[ColumnLineage]:
    """Column lineages for site_scores_monthly (94 input columns)."""
    ds = "site_scores_monthly"
    lineages = []

    # --- Kept: Revenue aggregations ---
    lineages.append(ColumnLineage(
        input_column="revenue",
        input_dataset=ds,
        transformations=[
            "Group by id_gbase",
            "Sum across all months → total_revenue",
            "Log transform → log_total_revenue",
        ],
        output_columns=["log_total_revenue"],
        notes="Log transform reduces right-skew of revenue distribution.",
    ))
    lineages.append(ColumnLineage(
        input_column="revenue",
        input_dataset=ds,
        transformations=[
            "Group by id_gbase",
            "Average across months → avg_monthly_revenue",
        ],
        output_columns=["avg_monthly_revenue"],
        notes="Primary regression target variable.",
    ))

    # --- Kept: Relative strength (momentum) features ---
    for metric, label in [("NVIs", "Network Visit Impressions"), ("Revenue", "Revenue")]:
        for window, desc in [("95_185", "~3-6 months"), ("185_370", "~6-12 months"), ("370_740", "~12-24 months")]:
            col = f"rs_{metric}_{window}"
            lineages.append(ColumnLineage(
                input_column=f"monthly_{'nvis' if metric == 'NVIs' else 'impressions' if metric == 'Impressions' else 'revenue'}",
                input_dataset=ds,
                transformations=[
                    "Group by id_gbase",
                    f"Compute relative strength ratio ({desc} window)",
                    f"Recent avg / historical avg → {col}",
                ],
                output_columns=[col],
                notes=f"{label} momentum indicator over {desc}.",
            ))

    # --- Kept: Demographics (pass-through or log) ---
    lineages.append(ColumnLineage(
        input_column="avg_household_income",
        input_dataset=ds,
        transformations=[
            "Take last known value per site",
            "Log transform → log_avg_household_income",
        ],
        output_columns=["log_avg_household_income"],
        notes="Log transform normalizes income distribution.",
    ))
    for col in ["median_age", "pct_female"]:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=["Take last known value per site", "Pass-through (no transform)"],
            output_columns=[col],
        ))

    # --- Kept: Categorical pass-through ---
    for col in ["network", "program", "experience_type", "hardware_type"]:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=["Take last known value per site", "Pass-through categorical"],
            output_columns=[col],
        ))

    # --- Kept: Binned categoricals ---
    for col in ["retailer", "brand_fuel", "brand_c_store"]:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=[
                "Take last known value per site",
                "Bin to top N categories + 'Other'",
            ],
            output_columns=[col],
            notes="Low-frequency categories grouped as 'Other' to reduce cardinality.",
        ))

    # --- Kept: Capability flags → encoded ---
    c_flags = [
        "c_emv_enabled", "c_nfc_enabled", "c_open_24_hours",
        "c_sells_beer", "c_sells_diesel_fuel", "c_sells_lottery",
        "c_vistar_programmatic_enabled", "c_sells_wine",
    ]
    for col in c_flags:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=[
                "Take last known value per site",
                f"One-hot encode → {col}_encoded",
            ],
            output_columns=[f"{col}_encoded"],
        ))

    # --- Kept: Restriction flags → encoded ---
    r_flags = [
        "r_retail_car_wash", "r_cpg_beverage_beer_oof", "r_cpg_beverage_beer_vide",
        "r_cpg_beverage_wine_oof", "r_cpg_beverage_wine_video",
        "r_finance_credit_cards", "r_cpg_cbd_hemp_ingestibles_non_thc",
        "r_cpg_non_food_beverage_cannabis_medical",
        "r_cpg_non_food_beverage_cannabis_recreational",
        "r_cpg_non_food_beverage_cbd_hemp_non_thc",
        "r_automotive_after_market_oil", "r_cpg_beverage_spirits_ooh",
        "r_cpg_beverage_spirits_video", "r_cpg_non_food_beverage_e_cigarette",
        "r_entertainment_casinos_and_gambling", "r_government_political",
        "r_automotive_electric", "r_recruitment",
        "r_restaurants_cdr", "r_restaurants_qsr",
        "r_retail_automotive_service", "r_retail_grocery",
        "r_retail_grocerty_with_fuel",
    ]
    for col in r_flags:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=[
                "Take last known value per site",
                f"One-hot encode → {col}_encoded",
            ],
            output_columns=[f"{col}_encoded"],
        ))

    # --- Kept: Schedule/sellable → encoded ---
    for col in ["schedule_site", "sellable_site"]:
        lineages.append(ColumnLineage(
            input_column=col,
            input_dataset=ds,
            transformations=[
                "Take last known value per site",
                f"Boolean encode → {col}_encoded",
            ],
            output_columns=[f"{col}_encoded"],
        ))

    # --- Dropped columns ---
    dropped_grouping = [
        ("id_gbase", "Grouping/join key for site aggregation"),
        ("gtvid", "Join key for geospatial distance tables"),
        ("date", "Used for temporal aggregation, then dropped"),
        ("month", "Used for temporal aggregation, then dropped"),
        ("year", "Used for temporal aggregation, then dropped"),
    ]
    for col, reason in dropped_grouping:
        lineages.append(ColumnLineage(
            input_column=col, input_dataset=ds,
            transformations=["Used for grouping/joining"], output_columns=[],
            dropped=True, notes=reason,
        ))

    lineages.append(ColumnLineage(
        input_column="statuis",
        input_dataset=ds,
        transformations=["Filter to 'Active' sites only", "Then dropped"],
        output_columns=[],
        dropped=True,
        notes="Source has typo 'statuis' (not 'status'). Used to filter Active sites only.",
    ))

    dropped_geo_leakage = [
        ("state", "Dropped to prevent geographic leakage"),
        ("county", "Dropped to prevent geographic leakage"),
        ("dma", "Dropped to prevent geographic leakage"),
        ("zip", "Dropped to prevent geographic leakage"),
    ]
    for col, reason in dropped_geo_leakage:
        lineages.append(ColumnLineage(
            input_column=col, input_dataset=ds,
            transformations=["Dropped before training"], output_columns=[],
            dropped=True, notes=reason,
        ))

    dropped_webapp = [
        ("latitude", "Used in web app for map display, not in training"),
        ("longitude", "Used in web app for map display, not in training"),
    ]
    for col, reason in dropped_webapp:
        lineages.append(ColumnLineage(
            input_column=col, input_dataset=ds,
            transformations=["Passed to web app only"], output_columns=[],
            dropped=True, notes=reason,
        ))

    dropped_intermediate = [
        ("avg_daily_impressions", "Intermediate metric, superseded by monthly aggregation"),
        ("avg_daily_nvis", "Intermediate metric, superseded by monthly aggregation"),
        ("avg_latency", "Diagnostic metric, not predictive"),
        ("screen_count", "Used in per-screen calculations, then dropped"),
        ("site_activated_date", "Metadata field, not used as feature"),
        ("zip_4", "Extended ZIP, dropped with geographic columns"),
        ("dma_rank", "Kept in aggregated data but not a default training feature"),
        ("index_african_american", "Index version dropped; percentage version kept"),
        ("pct_african_american", "Available but not in default feature set"),
        ("index_asian", "Index version dropped; percentage version kept"),
        ("pct_asian", "Available but not in default feature set"),
        ("index_female", "Index version dropped; percentage version kept"),
        ("index_male", "Index version dropped"),
        ("pct_male", "Available but not in default feature set"),
        ("index_hispanic", "Index version dropped; percentage version kept"),
        ("pct_hispanic", "Available but not in default feature set"),
        ("gstv_fixed_fee_pct", "Fee structure field, not predictive"),
        ("gstv_fixed_fee_amt", "Fee structure field, not predictive"),
        ("ptd_fixed_fee_met", "Fee structure field, not predictive"),
        ("retailer_fixed_fee_pct", "Fee structure field, not predictive"),
        ("retailer_fixed_fee_amt", "Fee structure field, not predictive"),
        ("sellable_direct_plan", "Sellability sub-field, not used as feature"),
        ("sellable_direct_revision", "Sellability sub-field, not used as feature"),
        ("sellable_programmatic_plan", "Sellability sub-field, not used as feature"),
        ("sellable_programmatic_revision", "Sellability sub-field, not used as feature"),
        ("r_lottery", "Restriction dropped (low variance or redundant)"),
        ("r_government", "Restriction dropped (low variance or redundant)"),
        ("r_travel_and_tourism", "Restriction dropped (low variance or redundant)"),
        ("r_alcohol_drink_responsibly_message", "Restriction dropped (low variance)"),
        ("r_alternative_transportation", "Restriction dropped (low variance)"),
        ("r_associations_and_npo_anti_smoking", "Restriction dropped (low variance)"),
        ("brand_restaurant", "Categorical with too many unique values"),
        ("c_walk_up_enabled", "Capability flag not used in training"),
        ("monthly_impressions", "Raw monthly value; relative strength features derived instead"),
        ("monthly_nvis", "Raw monthly value; relative strength features derived instead"),
        ("monthly_impressions_per_screen", "Per-screen metric, not directly used"),
        ("monthly_nvis_per_screen", "Per-screen metric, not directly used"),
        ("monthly_revenue_per_screen", "Per-screen metric, not directly used"),
    ]
    for col, reason in dropped_intermediate:
        lineages.append(ColumnLineage(
            input_column=col, input_dataset=ds,
            transformations=["Not included in training features"], output_columns=[],
            dropped=True, notes=reason,
        ))

    return lineages


def _nearest_site_lineages() -> List[ColumnLineage]:
    """Column lineages for nearest_site_distances (8 input columns)."""
    ds = "nearest_site_distances"
    return [
        ColumnLineage(
            input_column="nearest_site_distance_mi",
            input_dataset=ds,
            transformations=[
                "Join on GTVID to aggregated site data",
                "Log transform → log_min_distance_to_nearest_site_mi",
            ],
            output_columns=["log_min_distance_to_nearest_site_mi"],
            notes="Log transform handles right-skewed distance distribution.",
        ),
        ColumnLineage(
            input_column="GTVID", input_dataset=ds,
            transformations=["Used as join key"], output_columns=[],
            dropped=True, notes="Join key for merging with site data.",
        ),
        ColumnLineage(
            input_column="Latitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="Longitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="nearest_site", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Identifier of nearest neighbor, not a training feature.",
        ),
        ColumnLineage(
            input_column="nearest_site_lat", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate of nearest neighbor.",
        ),
        ColumnLineage(
            input_column="nearest_site_lon", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate of nearest neighbor.",
        ),
        ColumnLineage(
            input_column="nearest_site_distance_m", input_dataset=ds,
            transformations=["Not used (miles version preferred)"], output_columns=[],
            dropped=True, notes="Distance in meters; miles version used instead.",
        ),
    ]


def _interstate_lineages() -> List[ColumnLineage]:
    """Column lineages for interstate_distances (5 input columns)."""
    ds = "interstate_distances"
    return [
        ColumnLineage(
            input_column="distance_to_interstate_mi",
            input_dataset=ds,
            transformations=[
                "Group by GTVID, take minimum distance",
                "Join to aggregated site data",
                "Log transform → log_min_distance_to_interstate_mi",
            ],
            output_columns=["log_min_distance_to_interstate_mi"],
            notes="Multiple rows per site (one per interstate); aggregated to min distance.",
        ),
        ColumnLineage(
            input_column="GTVID", input_dataset=ds,
            transformations=["Used as join key"], output_columns=[],
            dropped=True, notes="Join key for merging with site data.",
        ),
        ColumnLineage(
            input_column="Latitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="Longitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="nearest_interstate", input_dataset=ds,
            transformations=["Not used after aggregation"], output_columns=[],
            dropped=True, notes="Name of nearest interstate highway.",
        ),
    ]


def _kroger_lineages() -> List[ColumnLineage]:
    """Column lineages for kroger_distances (4 input columns)."""
    ds = "kroger_distances"
    return [
        ColumnLineage(
            input_column="min_distance_to_kroger_mi",
            input_dataset=ds,
            transformations=[
                "Join on GTVID to aggregated site data",
                "Log transform → log_min_distance_to_kroger_mi",
            ],
            output_columns=["log_min_distance_to_kroger_mi"],
            notes="Proximity to Kroger as a neighborhood quality signal.",
        ),
        ColumnLineage(
            input_column="GTVID", input_dataset=ds,
            transformations=["Used as join key"], output_columns=[],
            dropped=True, notes="Join key for merging with site data.",
        ),
        ColumnLineage(
            input_column="Latitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="Longitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
    ]


def _mcdonalds_lineages() -> List[ColumnLineage]:
    """Column lineages for mcdonalds_distances (4 input columns)."""
    ds = "mcdonalds_distances"
    return [
        ColumnLineage(
            input_column="min_distance_to_mcdonalds_mi",
            input_dataset=ds,
            transformations=[
                "Join on GTVID to aggregated site data",
                "Log transform → log_min_distance_to_mcdonalds_mi",
            ],
            output_columns=["log_min_distance_to_mcdonalds_mi"],
            notes="Proximity to McDonald's as a traffic/footfall signal.",
        ),
        ColumnLineage(
            input_column="GTVID", input_dataset=ds,
            transformations=["Used as join key"], output_columns=[],
            dropped=True, notes="Join key for merging with site data.",
        ),
        ColumnLineage(
            input_column="Latitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
        ColumnLineage(
            input_column="Longitude", input_dataset=ds,
            transformations=["Not used after join"], output_columns=[],
            dropped=True, notes="Coordinate used only for distance calculation.",
        ),
    ]


# Registry mapping dataset IDs to their lineage functions
_LINEAGE_REGISTRY = {
    "site_scores_monthly": _site_scores_monthly_lineages,
    "nearest_site_distances": _nearest_site_lineages,
    "interstate_distances": _interstate_lineages,
    "kroger_distances": _kroger_lineages,
    "mcdonalds_distances": _mcdonalds_lineages,
}


def get_lineage_for_dataset(dataset_id: str) -> List[ColumnLineage]:
    """Get all column lineages for a dataset."""
    builder = _LINEAGE_REGISTRY.get(dataset_id)
    if builder is None:
        return []
    return builder()
