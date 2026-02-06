#!/usr/bin/env python3
"""
Data Ontology Generator for Geospatial Site Analysis.

Introspects all data files (source, intermediate, training, result) and produces
a formalized YAML ontology at docs/data_ontology.yaml.

Catalogs:
  - File paths, sizes, modification times
  - Dataset roles (source / intermediate / training / result)
  - Schemas (column names, dtypes, null rates)
  - Row and column counts
  - Lineage relationships (upstream/downstream)
  - Key statistics (numeric summaries, cardinality for categoricals)

Usage:
    python scripts/generate_data_ontology.py
    python scripts/generate_data_ontology.py --quick   # Skip heavy stats
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    # Fallback: write YAML manually if PyYAML not installed
    yaml = None

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry: defines every known dataset and its metadata
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: List[Dict[str, Any]] = [
    # ── SOURCE datasets (data/input/) ──
    {
        "id": "site_scores_monthly",
        "path": "data/input/site_scores_revenue_and_diagnostics.csv",
        "role": "source",
        "description": "Primary monthly site-level revenue, impressions, and metadata. One row per site per month (~1.4M rows).",
        "key_columns": ["id_gbase", "gtvid", "date", "revenue", "monthly_impressions", "monthly_nvis", "statuis"],
        "join_key": "id_gbase",
        "upstream": [],
        "downstream": ["site_aggregated_precleaned"],
        "notes": "Source has 'statuis' typo (not 'status'). Contains 80+ columns including demographics, capabilities, restrictions.",
    },
    {
        "id": "sites_base",
        "path": "data/input/Sites - Base Data Set.csv",
        "role": "source",
        "description": "Site master data with names, coordinates, and network info (~60K sites).",
        "key_columns": ["site_name", "gtvid", "latitude", "longitude", "state", "network", "retailer"],
        "join_key": "gtvid",
        "upstream": [],
        "downstream": [],
        "notes": "Reference dataset. Not directly consumed by ML pipeline.",
    },
    {
        "id": "revenue_salesforce",
        "path": "data/input/Site Revenue - Salesforce.csv",
        "role": "source",
        "description": "Alternative revenue snapshots from Salesforce.",
        "key_columns": [],
        "join_key": None,
        "upstream": [],
        "downstream": [],
        "notes": "Rarely used. Primary revenue source is site_scores_monthly.",
    },
    {
        "id": "nearest_site_distances",
        "path": "data/input/nearest_site_distances.csv",
        "role": "source",
        "description": "Distance from each site to its nearest neighbor site.",
        "key_columns": ["GTVID", "nearest_site_distance_mi", "nearest_site"],
        "join_key": "GTVID",
        "upstream": [],
        "downstream": ["site_aggregated_precleaned"],
        "notes": "Pre-computed geospatial feature. One row per site.",
    },
    {
        "id": "interstate_distances",
        "path": "data/input/site_interstate_distances.csv",
        "role": "source",
        "description": "Distance from each site to nearest interstate highway. Multiple rows per site (one per interstate).",
        "key_columns": ["GTVID", "distance_to_interstate_mi", "nearest_interstate"],
        "join_key": "GTVID",
        "upstream": [],
        "downstream": ["site_aggregated_precleaned"],
        "notes": "Aggregated to min distance per site during ETL (data_transform.py).",
    },
    {
        "id": "kroger_distances",
        "path": "data/input/site_kroger_distances.csv",
        "role": "source",
        "description": "Distance from each site to nearest Kroger store.",
        "key_columns": ["GTVID", "min_distance_to_kroger_mi"],
        "join_key": "GTVID",
        "upstream": [],
        "downstream": ["site_aggregated_precleaned"],
        "notes": "Pre-aggregated. One row per site.",
    },
    {
        "id": "mcdonalds_distances",
        "path": "data/input/site_mcdonalds_distances.csv",
        "role": "source",
        "description": "Distance from each site to nearest McDonald's.",
        "key_columns": ["GTVID", "min_distance_to_mcdonalds_mi"],
        "join_key": "GTVID",
        "upstream": [],
        "downstream": ["site_aggregated_precleaned"],
        "notes": "Pre-aggregated. One row per site.",
    },
    {
        "id": "kroger_dist_legacy",
        "path": "data/input/kroger_dist_to_site_mi.csv",
        "role": "source",
        "description": "Legacy Kroger distance file (older format).",
        "key_columns": [],
        "join_key": None,
        "upstream": [],
        "downstream": [],
        "notes": "Superseded by site_kroger_distances.csv.",
    },
    {
        "id": "mcdonalds_dist_legacy",
        "path": "data/input/mcdonalds_dist_to_site_mi.csv",
        "role": "source",
        "description": "Legacy McDonald's distance file (older format).",
        "key_columns": [],
        "join_key": None,
        "upstream": [],
        "downstream": [],
        "notes": "Superseded by site_mcdonalds_distances.csv.",
    },

    # ── INTERMEDIATE datasets (data/processed/) ──
    {
        "id": "site_aggregated_precleaned",
        "path": "data/processed/site_aggregated_precleaned.parquet",
        "role": "intermediate",
        "description": "All sites aggregated to one row per site with geospatial joins, log transforms, and one-hot encoding. Includes all statuses (~57K sites).",
        "key_columns": ["id_gbase", "gtvid", "status", "active_months", "avg_monthly_revenue"],
        "join_key": "id_gbase",
        "upstream": ["site_scores_monthly", "nearest_site_distances", "interstate_distances", "kroger_distances", "mcdonalds_distances"],
        "downstream": ["site_training_data"],
        "produced_by": "site_scoring/data_transform.py::create_training_dataset()",
        "notes": "Also output as CSV. Contains ~100 columns after feature engineering.",
    },
    {
        "id": "site_aggregated_precleaned_csv",
        "path": "data/processed/site_aggregated_precleaned.csv",
        "role": "intermediate",
        "description": "CSV version of site_aggregated_precleaned for inspection/export.",
        "key_columns": [],
        "join_key": "id_gbase",
        "upstream": ["site_aggregated_precleaned"],
        "downstream": [],
        "notes": "Duplicate of parquet version in CSV format.",
    },

    # ── TRAINING datasets (data/processed/) ──
    {
        "id": "site_training_data",
        "path": "data/processed/site_training_data.parquet",
        "role": "training",
        "description": "Active-only sites filtered for ML training. Negative revenue removed. One-hot encoded categoricals added (~26K sites, ~106 columns).",
        "key_columns": ["id_gbase", "gtvid", "avg_monthly_revenue", "active_months"],
        "join_key": "id_gbase",
        "upstream": ["site_aggregated_precleaned"],
        "downstream": ["model_artifacts"],
        "produced_by": "site_scoring/data_transform.py::create_training_dataset()",
        "notes": "Primary input to site_scoring/data_loader.py. Features: 13 numeric + 7 categorical + 40 boolean.",
    },
    {
        "id": "site_training_data_csv",
        "path": "data/processed/site_training_data.csv",
        "role": "training",
        "description": "CSV version of training data for inspection/export.",
        "key_columns": [],
        "join_key": "id_gbase",
        "upstream": ["site_training_data"],
        "downstream": [],
        "notes": "Duplicate of parquet version in CSV format.",
    },

    # ── RESULT datasets (site_scoring/outputs/) ──
    {
        "id": "model_artifacts",
        "path": "site_scoring/outputs/",
        "role": "result",
        "description": "Trained model artifacts directory. Contains best models, preprocessors, SHAP values, and experiment history.",
        "key_columns": [],
        "join_key": None,
        "upstream": ["site_training_data"],
        "downstream": [],
        "produced_by": "src/services/training_service.py",
        "notes": "Max 10 experiments (FIFO cleanup). Subdirs: experiments/job_*/",
    },

    # ── SUMMARY reports (data/processed/) ──
    {
        "id": "precleaned_summary",
        "path": "data/processed/precleaned_summary.txt",
        "role": "result",
        "description": "Human-readable summary of the precleaned dataset (site counts, column counts, activity stats).",
        "key_columns": [],
        "join_key": None,
        "upstream": ["site_aggregated_precleaned"],
        "downstream": [],
        "produced_by": "site_scoring/data_transform.py",
    },
    {
        "id": "training_data_summary",
        "path": "data/processed/training_data_summary.txt",
        "role": "result",
        "description": "Human-readable summary of the training dataset (site counts, feature breakdown, encoding details).",
        "key_columns": [],
        "join_key": None,
        "upstream": ["site_training_data"],
        "downstream": [],
        "produced_by": "site_scoring/data_transform.py",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Lineage: full pipeline flow
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_LINEAGE = {
    "name": "Geospatial Site Analysis Pipeline",
    "stages": [
        {
            "stage": "1_ingest",
            "description": "Raw source data from external systems",
            "datasets": ["site_scores_monthly", "sites_base", "revenue_salesforce",
                         "nearest_site_distances", "interstate_distances",
                         "kroger_distances", "mcdonalds_distances"],
        },
        {
            "stage": "2_aggregate",
            "description": "Monthly data aggregated to one row per site with geospatial joins",
            "script": "site_scoring/data_transform.py",
            "datasets": ["site_aggregated_precleaned", "site_aggregated_precleaned_csv"],
        },
        {
            "stage": "3_filter_encode",
            "description": "Filter to Active sites, encode categoricals, remove leakage columns",
            "script": "site_scoring/data_transform.py",
            "datasets": ["site_training_data", "site_training_data_csv"],
        },
        {
            "stage": "4_train",
            "description": "ML training: tensor conversion, scaling, model fitting",
            "script": "site_scoring/data_loader.py + site_scoring/model.py",
            "datasets": ["model_artifacts"],
        },
        {
            "stage": "5_serve",
            "description": "Web visualization and inference",
            "script": "app.py + src/services/",
            "datasets": [],
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Introspection: read actual file metadata and schema
# ─────────────────────────────────────────────────────────────────────────────

def get_file_info(filepath: Path) -> Dict[str, Any]:
    """Get file size, modification time, and existence."""
    if not filepath.exists():
        return {"exists": False}

    stat = filepath.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_human": _human_size(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def introspect_csv(filepath: Path, quick: bool = False) -> Dict[str, Any]:
    """Read CSV schema and basic stats."""
    info: Dict[str, Any] = {}

    if pl is not None:
        try:
            df = pl.read_csv(filepath, infer_schema_length=1000, n_rows=0)
            info["columns"] = len(df.columns)
            info["schema"] = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

            if not quick:
                df_full = pl.read_csv(filepath, infer_schema_length=5000)
                info["rows"] = len(df_full)
                info["null_rates"] = {
                    col: round(df_full[col].null_count() / len(df_full), 4)
                    for col in df_full.columns
                    if df_full[col].null_count() > 0
                }
            return info
        except Exception as e:
            info["read_error"] = str(e)

    if pd is not None:
        try:
            df = pd.read_csv(filepath, nrows=0)
            info["columns"] = len(df.columns)
            info["schema"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            if not quick:
                df_full = pd.read_csv(filepath)
                info["rows"] = len(df_full)
            return info
        except Exception as e:
            info["read_error"] = str(e)

    return info


def introspect_parquet(filepath: Path, quick: bool = False) -> Dict[str, Any]:
    """Read Parquet schema and basic stats."""
    info: Dict[str, Any] = {}

    if pl is not None:
        try:
            df = pl.read_parquet(filepath)
            info["rows"] = len(df)
            info["columns"] = len(df.columns)
            info["schema"] = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

            if not quick:
                info["null_rates"] = {
                    col: round(df[col].null_count() / len(df), 4)
                    for col in df.columns
                    if df[col].null_count() > 0
                }
            return info
        except Exception as e:
            info["read_error"] = str(e)

    if pd is not None:
        try:
            df = pd.read_parquet(filepath)
            info["rows"] = len(df)
            info["columns"] = len(df.columns)
            info["schema"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            return info
        except Exception as e:
            info["read_error"] = str(e)

    return info


def introspect_directory(dirpath: Path) -> Dict[str, Any]:
    """Catalog contents of an output directory."""
    if not dirpath.exists():
        return {"exists": False}

    items = []
    total_size = 0
    for f in sorted(dirpath.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total_size += sz
            items.append({
                "path": str(f.relative_to(PROJECT_ROOT)),
                "size_human": _human_size(sz),
                "suffix": f.suffix,
            })

    return {
        "exists": True,
        "file_count": len(items),
        "total_size_human": _human_size(total_size),
        "files": items[:50],  # Cap to avoid massive output
    }


def introspect_dataset(entry: Dict[str, Any], quick: bool = False) -> Dict[str, Any]:
    """Introspect a single dataset entry."""
    filepath = PROJECT_ROOT / entry["path"]
    result = dict(entry)  # Copy registry metadata

    if filepath.is_dir():
        result["file_info"] = get_file_info(filepath) if filepath.exists() else {"exists": False}
        result["contents"] = introspect_directory(filepath)
    elif filepath.suffix == ".parquet":
        result["file_info"] = get_file_info(filepath)
        if filepath.exists():
            result["introspection"] = introspect_parquet(filepath, quick=quick)
    elif filepath.suffix == ".csv":
        result["file_info"] = get_file_info(filepath)
        if filepath.exists() and filepath.stat().st_size < 500_000_000:  # Skip files > 500MB for speed
            result["introspection"] = introspect_csv(filepath, quick=quick)
        elif filepath.exists():
            result["file_info"]["skipped_introspection"] = "File > 500MB, schema-only via n_rows=0"
            result["introspection"] = introspect_csv(filepath, quick=True)
    elif filepath.suffix == ".txt":
        result["file_info"] = get_file_info(filepath)
    else:
        result["file_info"] = get_file_info(filepath) if filepath.exists() else {"exists": False}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Feature catalog: from config.py
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_catalog() -> Dict[str, Any]:
    """Extract current feature configuration from site_scoring/config.py."""
    try:
        from site_scoring.config import Config, get_all_available_features
        config = Config()
        all_features = get_all_available_features()

        return {
            "default_target": config.target,
            "task_types": ["regression", "lookalike", "clustering"],
            "default_features": {
                "numeric": config.numeric_features,
                "categorical": config.categorical_features,
                "boolean": config.boolean_features,
                "total_count": len(config.numeric_features) + len(config.categorical_features) + len(config.boolean_features),
            },
            "all_available_features": {
                "numeric": all_features["numeric"],
                "categorical": all_features["categorical"],
                "boolean": all_features["boolean"],
                "total_count": len(all_features["numeric"]) + len(all_features["categorical"]) + len(all_features["boolean"]),
            },
        }
    except Exception as e:
        return {"error": f"Could not import config: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# YAML Output
# ─────────────────────────────────────────────────────────────────────────────

def build_ontology(quick: bool = False) -> Dict[str, Any]:
    """Build the complete data ontology."""
    print("Building data ontology...")

    datasets = {}
    for entry in DATASET_REGISTRY:
        dataset_id = entry["id"]
        print(f"  Introspecting: {dataset_id} ({entry['path']})")
        datasets[dataset_id] = introspect_dataset(entry, quick=quick)

    ontology = {
        "metadata": {
            "title": "Geospatial Site Analysis - Data Ontology",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/generate_data_ontology.py",
            "project_root": str(PROJECT_ROOT),
            "quick_mode": quick,
        },
        "pipeline": PIPELINE_LINEAGE,
        "feature_catalog": get_feature_catalog(),
        "datasets": datasets,
    }

    return ontology


def write_yaml(ontology: Dict[str, Any], output_path: Path) -> None:
    """Write ontology to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:
        with open(output_path, "w") as f:
            f.write("# Auto-generated Data Ontology\n")
            f.write(f"# Generated: {ontology['metadata']['generated_at']}\n")
            f.write("# Do not edit manually - regenerate with: python scripts/generate_data_ontology.py\n")
            f.write("# Hook: .claude/hooks/update-data-ontology.sh auto-triggers on data file changes\n\n")
            yaml.dump(ontology, f, default_flow_style=False, sort_keys=False, width=120, allow_unicode=True)
    else:
        # Fallback: basic YAML-like output without PyYAML
        with open(output_path, "w") as f:
            f.write("# Auto-generated Data Ontology\n")
            f.write(f"# Generated: {ontology['metadata']['generated_at']}\n")
            f.write("# Do not edit manually - regenerate with: python scripts/generate_data_ontology.py\n")
            f.write("# Note: Install PyYAML for proper YAML output (pip install pyyaml)\n\n")
            _write_dict_as_yaml(f, ontology, indent=0)

    print(f"\nOntology written to: {output_path}")


def _write_dict_as_yaml(f, obj: Any, indent: int = 0) -> None:
    """Minimal YAML writer fallback when PyYAML is not available."""
    prefix = "  " * indent
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, (dict, list)):
                f.write(f"{prefix}{key}:\n")
                _write_dict_as_yaml(f, val, indent + 1)
            else:
                f.write(f"{prefix}{key}: {_yaml_scalar(val)}\n")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                f.write(f"{prefix}-\n")
                _write_dict_as_yaml(f, item, indent + 1)
            else:
                f.write(f"{prefix}- {_yaml_scalar(item)}\n")
    else:
        f.write(f"{prefix}{_yaml_scalar(obj)}\n")


def _yaml_scalar(val: Any) -> str:
    """Format a scalar value for YAML."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        if any(c in val for c in ":#{}[]&*!|>'\"%@`"):
            return f'"{val}"'
        return val
    return str(val)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate data ontology for geospatial project")
    parser.add_argument("--quick", action="store_true", help="Skip heavy stats (row counts for large CSVs, null rates)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: docs/data_ontology.yaml)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "docs" / "data_ontology.yaml"

    ontology = build_ontology(quick=args.quick)
    write_yaml(ontology, output_path)

    # Summary
    roles = {}
    for ds in ontology["datasets"].values():
        role = ds.get("role", "unknown")
        roles[role] = roles.get(role, 0) + 1

    print(f"\nDatasets cataloged: {len(ontology['datasets'])}")
    for role, count in sorted(roles.items()):
        print(f"  {role}: {count}")


if __name__ == "__main__":
    main()
