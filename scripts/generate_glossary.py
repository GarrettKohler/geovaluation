#!/usr/bin/env python3
"""Generate the Data Pipeline Glossary HTML from codebase introspection.

Usage:
    python scripts/generate_glossary.py                        # Generate docs/pipeline_glossary.html
    python scripts/generate_glossary.py --output custom.html   # Custom output path
    python scripts/generate_glossary.py --check                # Validate docstrings only
    python scripts/generate_glossary.py --verbose              # Show introspection details
"""

import ast
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root — resolved relative to this script's location (scripts/ -> project root)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Source files to introspect
SOURCE_FILES = [
    "site_scoring/data_transform.py",
    "src/services/training_service.py",
    "site_scoring/model.py",
    "site_scoring/data_loader.py",
    "site_scoring/predict.py",
    "src/routes/prediction.py",
]

# Stage ordering
STAGE_ORDER = [
    ("collection", 0),
    ("cleaning", 1),
    ("combining", 2),
    ("modeling", 3),
    ("testing", 4),
    ("productionizing", 5),
]

# Color palette (must match CSS)
COLORS = {
    "accent": "#6366f1",
    "accentLt": "#818cf8",
    "green": "#34d399",
    "orange": "#fb923c",
    "pink": "#f472b6",
    "cyan": "#22d3ee",
    "yellow": "#facc15",
    "red": "#f87171",
    "border": "#334155",
    "surface": "#1e293b",
    "dim": "#64748b",
    "muted": "#94a3b8",
    "bg": "#0f172a",
    "text": "#e2e8f0",
}


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Detail:
    title: str
    body: str


@dataclass
class Step:
    id: str
    title: str
    step_order: int
    color: str = "accent"
    sub: str = ""
    analogy: Optional[str] = None
    why: Optional[str] = None
    details: list = field(default_factory=list)
    extra_html: str = ""
    source_function: str = ""
    source_file: str = ""
    source_line: int = 0


@dataclass
class Source:
    """Data source card for Collection stage."""
    id: str
    icon: str
    name: str
    desc: str
    source: str
    format: str
    rows: str
    cols: int
    color: str
    fields: list = field(default_factory=list)
    sample: list = field(default_factory=list)
    notes: str = ""


@dataclass
class StatCard:
    value: str
    label: str
    color: str


@dataclass
class Stage:
    id: str
    title: str
    category: str = "Pipeline Stages"
    question: str = ""
    intro: str = ""
    analogy: Optional[str] = None
    why: Optional[str] = None
    steps: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    stats_row: list = field(default_factory=list)
    flow_svg: str = ""
    stage_index: int = 0


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: INTROSPECT — Parse Code
# ═══════════════════════════════════════════════════════════════════════

def parse_glossary_tags(docstring: str) -> list:
    """Parse @glossary tags from a function/class docstring.

    Returns a list of dicts, one per @glossary entry found.
    Tags span multiple lines until the next @tag or end of docstring.
    A function can have multiple @glossary tags (appears in multiple stages).
    """
    if not docstring or "@glossary" not in docstring:
        return []

    entries = []
    current = None
    current_tag = None

    for line in docstring.split("\n"):
        stripped = line.strip()
        match = re.match(r"@(\w+)(?:\[([^\]]*)\])?:\s*(.*)", stripped)

        if match:
            tag_name = match.group(1)
            bracket = match.group(2)
            value = match.group(3).strip()

            if tag_name == "glossary":
                if current is not None:
                    entries.append(current)
                current = {"glossary": value, "details": []}
                current_tag = None
            elif current is not None:
                if tag_name == "detail":
                    detail = {"title": bracket or "Detail", "body": value}
                    current["details"].append(detail)
                    current_tag = ("detail", len(current["details"]) - 1)
                else:
                    current[tag_name] = value
                    current_tag = tag_name
        elif current is not None and current_tag and stripped:
            # Continuation line
            if isinstance(current_tag, tuple) and current_tag[0] == "detail":
                idx = current_tag[1]
                current["details"][idx]["body"] += " " + stripped
            elif isinstance(current_tag, str) and current_tag in current:
                current[current_tag] += " " + stripped

    if current is not None:
        entries.append(current)

    return entries


def _fix_surrogates(obj):
    """Convert surrogate pairs in strings to proper Unicode codepoints.

    Python source files may contain \\ud83d\\udcb0 style escapes which
    ast.literal_eval preserves as surrogate characters. These can't be
    encoded to UTF-8 directly, so we round-trip through UTF-16.
    """
    if isinstance(obj, str):
        try:
            return obj.encode("utf-16", "surrogatepass").decode("utf-16")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return obj
    elif isinstance(obj, dict):
        return {_fix_surrogates(k): _fix_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_surrogates(i) for i in obj]
    return obj


def _safe_parse_literal(node):
    """Safely parse an AST node as a Python literal (strings, numbers, lists, dicts).

    Uses ast.literal_eval which ONLY handles literals — no code execution.
    This is the stdlib-recommended safe alternative to eval().
    """
    try:
        # ast.literal_eval safely parses Python literal structures only
        result = ast.literal_eval(node)  # noqa: S307 — literal_eval is safe
        return _fix_surrogates(result)
    except (ValueError, TypeError):
        return None


def introspect_source_file(filepath: Path, project_root: Path) -> dict:
    """Parse a single Python file for @glossary docstrings and _GLOSSARY_* dicts."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))
    relative_path = str(filepath.relative_to(project_root))

    result = {"steps": [], "stages": {}, "sources": []}

    # Extract module-level _GLOSSARY_STAGES and _GLOSSARY_SOURCES
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "_GLOSSARY_STAGES":
                        val = _safe_parse_literal(node.value)
                        if val:
                            result["stages"] = val
                    elif target.id == "_GLOSSARY_SOURCES":
                        val = _safe_parse_literal(node.value)
                        if val:
                            result["sources"] = val

    # Walk all functions/classes for @glossary docstrings
    for node in ast.walk(tree):
        func_name = None
        lineno = 0

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            lineno = node.lineno
        elif isinstance(node, ast.ClassDef):
            func_name = node.name
            lineno = node.lineno

        if func_name:
            docstring = ast.get_docstring(node)
            if docstring and "@glossary" in docstring:
                tags = parse_glossary_tags(docstring)
                for tag in tags:
                    tag["source_function"] = func_name
                    tag["source_file"] = relative_path
                    tag["source_line"] = lineno
                    result["steps"].append(tag)

            # Also check __init__ for classes
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        doc = ast.get_docstring(item)
                        if doc and "@glossary" in doc:
                            tags = parse_glossary_tags(doc)
                            for tag in tags:
                                tag["source_function"] = f"{func_name}.__init__"
                                tag["source_file"] = relative_path
                                tag["source_line"] = item.lineno
                                result["steps"].append(tag)

    return result


def introspect_code(project_root: Path, verbose: bool = False) -> dict:
    """Parse all source files for @glossary tags and stage metadata."""
    all_steps = []
    all_stages = {}
    all_sources = []

    if verbose:
        print("Introspecting code...")

    for rel_path in SOURCE_FILES:
        filepath = project_root / rel_path
        if not filepath.exists():
            if verbose:
                print(f"  SKIP  {rel_path} (not found)")
            continue
        if verbose:
            print(f"  PARSE {rel_path}")

        result = introspect_source_file(filepath, project_root)
        all_steps.extend(result["steps"])
        all_stages.update(result["stages"])
        all_sources.extend(result["sources"])

        if verbose and result["steps"]:
            print(f"        found {len(result['steps'])} @glossary tags")
        if verbose and result["stages"]:
            print(f"        stages: {', '.join(result['stages'].keys())}")

    return {"steps": all_steps, "stages": all_stages, "sources": all_sources}


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: INTROSPECT — Read Data
# ═══════════════════════════════════════════════════════════════════════

def introspect_data(project_root: Path, verbose: bool = False) -> dict:
    """Read parquet metadata, glob files, parse config.py for feature lists."""
    facts = {}

    processed_dir = project_root / "data" / "processed"
    input_dir = project_root / "data" / "input"
    experiments_dir = project_root / "site_scoring" / "outputs" / "experiments"

    if verbose:
        print("\nIntrospecting data...")

    # ── Parquet metadata ──
    for name, filename in [
        ("precleaned", "site_aggregated_precleaned.parquet"),
        ("training", "site_training_data.parquet"),
    ]:
        path = processed_dir / filename
        if path.exists():
            try:
                import polars as pl

                lf = pl.scan_parquet(path)
                schema = lf.collect_schema()
                row_count = lf.select(pl.len()).collect().item()
                facts[f"{name}_rows"] = row_count
                facts[f"{name}_cols"] = len(schema)
                facts[f"{name}_columns"] = list(schema.names())
                if verbose:
                    print(f"  OK    {filename}: {row_count:,} rows x {len(schema)} cols")
            except Exception as e:
                if verbose:
                    print(f"  ERR   {filename}: {e}")
        else:
            if verbose:
                print(f"  MISS  {filename}")

    # ── Input file listing ──
    if input_dir.exists():
        csv_files = sorted(input_dir.glob("*.csv"))
        facts["input_files"] = [
            {"name": f.name, "size_mb": round(f.stat().st_size / 1_000_000, 1)}
            for f in csv_files
        ]
        facts["input_file_count"] = len(csv_files)
        if verbose:
            print(f"  OK    data/input/: {len(csv_files)} CSV files")
    else:
        facts["input_files"] = []
        facts["input_file_count"] = 0

    # ── Experiments ──
    if experiments_dir.exists():
        exp_dirs = sorted(
            [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("job_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        facts["experiment_count"] = len(exp_dirs)

        if exp_dirs:
            metadata_path = exp_dirs[0] / "model_metadata.json"
            if metadata_path.exists():
                try:
                    facts["latest_metrics"] = json.loads(metadata_path.read_text())
                except Exception:
                    pass
        if verbose:
            print(f"  OK    experiments/: {len(exp_dirs)} experiments")
    else:
        facts["experiment_count"] = 0

    # ── Feature lists from config.py ──
    config_path = project_root / "site_scoring" / "config.py"
    if config_path.exists():
        try:
            tree = ast.parse(config_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "Config":
                    for item in node.body:
                        if not isinstance(item, ast.AnnAssign):
                            continue
                        if not isinstance(item.target, ast.Name):
                            continue
                        name = item.target.id

                        # Extract list from field(default_factory=lambda: [...])
                        if name in ("numeric_features", "categorical_features",
                                    "boolean_features", "hidden_dims"):
                            if item.value and isinstance(item.value, ast.Call):
                                for kw in item.value.keywords:
                                    if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
                                        val = _safe_parse_literal(kw.value.body)
                                        if val is not None:
                                            facts[name] = val

                        # Extract simple literals
                        elif name in ("embedding_dim", "dropout", "batch_size",
                                      "train_ratio", "val_ratio", "test_ratio"):
                            if item.value:
                                val = _safe_parse_literal(item.value)
                                if val is not None:
                                    facts[name] = val

            for feat_type in ("numeric", "categorical", "boolean"):
                key = f"{feat_type}_features"
                if key in facts:
                    facts[f"n_{feat_type}"] = len(facts[key])

            if verbose:
                print(f"  OK    config.py: {facts.get('n_numeric', '?')} numeric, "
                      f"{facts.get('n_categorical', '?')} categorical, "
                      f"{facts.get('n_boolean', '?')} boolean features")
        except Exception as e:
            if verbose:
                print(f"  ERR   config.py: {e}")

    return facts


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: ASSEMBLE
# ═══════════════════════════════════════════════════════════════════════

def _build_stats_row(stage_id: str, facts: dict) -> list:
    """Build stat cards for a stage from data facts."""
    C = COLORS

    if stage_id == "collection":
        return [
            StatCard(f"~{facts.get('precleaned_rows', 57675):,}", "Unique sites", C["accent"]),
            StatCard(str(facts.get("input_file_count", 7)), "Source CSV files", C["green"]),
            StatCard("25K+", "Retailer locations", C["orange"]),
            StatCard("CSV", "Uniform format", C["cyan"]),
        ]
    elif stage_id == "cleaning":
        return [
            StatCard(f"{facts.get('precleaned_rows', 57675):,}", "Pre-cleaned sites", C["cyan"]),
            StatCard(f"{facts.get('training_rows', 26099):,}", "Training sites", C["green"]),
            StatCard(str(facts.get("training_cols", 111)), "Final columns", C["orange"]),
            StatCard("7", "Cleaning steps", C["accent"]),
        ]
    elif stage_id == "combining":
        return [
            StatCard("94", "Input columns", C["accent"]),
            StatCard(f"{facts.get('precleaned_rows', 57675):,}", "Sites after agg", C["cyan"]),
            StatCard("6", "Left joins", C["green"]),
            StatCard(f"~{facts.get('precleaned_cols', 102)}", "Output columns", C["orange"]),
        ]
    elif stage_id == "modeling":
        hd = facts.get("hidden_dims", [512, 256, 128, 64])
        return [
            StatCard(str(facts.get("n_numeric", 15)), "Numeric features", C["accent"]),
            StatCard(str(facts.get("n_categorical", 7)), "Categorical", C["cyan"]),
            StatCard(str(facts.get("n_boolean", 33)), "Boolean", C["green"]),
            StatCard("\u2192".join(str(d) for d in hd), "Hidden dims", C["orange"]),
        ]
    elif stage_id == "testing":
        metrics = facts.get("latest_metrics", {})
        if metrics and isinstance(metrics.get("test_r2"), (int, float)):
            return [
                StatCard(f"{metrics['test_r2']:.3f}", "Test R\u00b2", C["accent"]),
                StatCard(f"${metrics.get('test_mae', 0):,.0f}", "Test MAE", C["cyan"]),
                StatCard(f"{metrics.get('test_mape', 0):.1f}%", "Test MAPE", C["green"]),
                StatCard(str(facts.get("experiment_count", 0)), "Experiments", C["orange"]),
            ]
        return [
            StatCard("70/15/15", "Train/Val/Test", C["accent"]),
            StatCard("3", "Split sets", C["cyan"]),
            StatCard("SHAP", "Explainability", C["green"]),
            StatCard(str(facts.get("experiment_count", 0)), "Experiments", C["orange"]),
        ]
    elif stage_id == "productionizing":
        return [
            StatCard(f"{facts.get('precleaned_rows', 57675):,}", "Scoreable sites", C["accent"]),
            StatCard(str(facts.get("experiment_count", 0)), "Trained models", C["cyan"]),
            StatCard("4096", "Batch size", C["green"]),
            StatCard("CSV/XLSX", "Export formats", C["orange"]),
        ]
    return []


def _build_flow_svg(stage_id: str, facts: dict) -> str:
    """Build a flow diagram SVG for a stage."""
    C = COLORS
    pr = facts.get("precleaned_rows", 57675)
    pc = facts.get("precleaned_cols", 102)
    ifc = facts.get("input_file_count", 7)
    n_num = facts.get("n_numeric", 15)
    n_cat = facts.get("n_categorical", 7)
    n_bool = facts.get("n_boolean", 33)
    hd = facts.get("hidden_dims", [512, 256, 128, 64])
    emb_dim = facts.get("embedding_dim", 16)
    exp_count = facts.get("experiment_count", 0)
    tr_pct = int(facts.get("train_ratio", 0.7) * 100)
    va_pct = int(facts.get("val_ratio", 0.15) * 100)
    te_pct = int(facts.get("test_ratio", 0.15) * 100)

    if stage_id == "collection":
        return (
            f'<svg viewBox="0 0 640 170" xmlns="http://www.w3.org/2000/svg" style="background:{C["bg"]};border-radius:10px;border:1px solid {C["border"]}">'
            f'<text x="70" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">SALESFORCE</text>'
            f'<rect x="10" y="22" width="120" height="40" rx="8" fill="{C["accent"]}22" stroke="{C["accent"]}" stroke-width="1.5"/>'
            f'<text x="70" y="40" text-anchor="middle" fill="{C["accent"]}" font-size="9" font-weight="700">Sites Base Data Set</text>'
            f'<text x="70" y="53" text-anchor="middle" fill="{C["muted"]}" font-size="8">67,650 rows x 43 cols</text>'
            f'<rect x="10" y="70" width="120" height="40" rx="8" fill="{C["cyan"]}22" stroke="{C["cyan"]}" stroke-width="1.5"/>'
            f'<text x="70" y="88" text-anchor="middle" fill="{C["cyan"]}" font-size="9" font-weight="700">Site Revenue</text>'
            f'<text x="70" y="101" text-anchor="middle" fill="{C["muted"]}" font-size="8">67,604 rows x 7 cols</text>'
            f'<text x="210" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">GEOGRAPHIC</text>'
            f'<rect x="150" y="22" width="120" height="40" rx="8" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="1.5"/>'
            f'<text x="210" y="40" text-anchor="middle" fill="{C["green"]}" font-size="9" font-weight="700">Proximity Files (5)</text>'
            f'<text x="210" y="53" text-anchor="middle" fill="{C["muted"]}" font-size="8">~68K rows each</text>'
            f'<text x="210" y="78" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">3RD-PARTY</text>'
            f'<rect x="150" y="84" width="120" height="40" rx="8" fill="{C["orange"]}22" stroke="{C["orange"]}" stroke-width="1.5"/>'
            f'<text x="210" y="102" text-anchor="middle" fill="{C["orange"]}" font-size="9" font-weight="700">Retailer Locations</text>'
            f'<text x="210" y="115" text-anchor="middle" fill="{C["muted"]}" font-size="8">McD / Walmart / Target</text>'
            f'<line x1="130" y1="42" x2="330" y2="68" stroke="{C["border"]}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<line x1="130" y1="90" x2="330" y2="78" stroke="{C["border"]}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<line x1="270" y1="42" x2="330" y2="68" stroke="{C["border"]}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<line x1="270" y1="104" x2="330" y2="78" stroke="{C["border"]}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<rect x="330" y="48" width="130" height="50" rx="10" fill="{C["accentLt"]}22" stroke="{C["accentLt"]}" stroke-width="2"/>'
            f'<text x="395" y="69" text-anchor="middle" fill="{C["accentLt"]}" font-size="10" font-weight="700">Data Ingestion</text>'
            f'<text x="395" y="84" text-anchor="middle" fill="{C["muted"]}" font-size="8">Load via Polars ETL</text>'
            f'<line x1="460" y1="73" x2="498" y2="73" stroke="{C["accentLt"]}" stroke-width="2" marker-end="url(#ah)"/>'
            f'<rect x="498" y="48" width="130" height="50" rx="10" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="2"/>'
            f'<text x="563" y="69" text-anchor="middle" fill="{C["green"]}" font-size="10" font-weight="700">data/input/</text>'
            f'<text x="563" y="84" text-anchor="middle" fill="{C["muted"]}" font-size="8">~{pr:,} sites, {ifc} source files</text>'
            f'<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["accentLt"]}"/></marker></defs>'
            f'</svg>'
        )

    elif stage_id == "combining":
        return (
            f'<svg viewBox="0 0 640 200" xmlns="http://www.w3.org/2000/svg" style="background:{C["bg"]};border-radius:10px;border:1px solid {C["border"]}">'
            f'<defs><marker id="aj" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["green"]}"/></marker>'
            f'<marker id="aj2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["cyan"]}"/></marker>'
            f'<marker id="aj3" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["pink"]}"/></marker></defs>'
            f'<text x="75" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">RAW INPUT</text>'
            f'<rect x="10" y="22" width="130" height="44" rx="8" fill="{C["accent"]}22" stroke="{C["accent"]}" stroke-width="1.5"/>'
            f'<text x="75" y="40" text-anchor="middle" fill="{C["accent"]}" font-size="8" font-weight="700">site_scores_revenue_</text>'
            f'<text x="75" y="51" text-anchor="middle" fill="{C["accent"]}" font-size="8" font-weight="700">and_diagnostics.csv</text>'
            f'<text x="75" y="62" text-anchor="middle" fill="{C["muted"]}" font-size="7">1.4M rows x 94 cols</text>'
            f'<line x1="140" y1="44" x2="195" y2="44" stroke="{C["cyan"]}" stroke-width="1.5" marker-end="url(#aj2)"/>'
            f'<text x="280" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">AGGREGATED</text>'
            f'<rect x="195" y="22" width="170" height="44" rx="8" fill="{C["cyan"]}22" stroke="{C["cyan"]}" stroke-width="1.5"/>'
            f'<text x="280" y="40" text-anchor="middle" fill="{C["cyan"]}" font-size="9" font-weight="700">{pr:,} Sites</text>'
            f'<text x="280" y="53" text-anchor="middle" fill="{C["muted"]}" font-size="7">totals + averages + metadata</text>'
            f'<rect x="10" y="86" width="130" height="36" rx="8" fill="{C["pink"]}22" stroke="{C["pink"]}" stroke-width="1.5"/>'
            f'<text x="75" y="103" text-anchor="middle" fill="{C["pink"]}" font-size="8" font-weight="700">RS Features (12 cols)</text>'
            f'<line x1="140" y1="104" x2="195" y2="75" stroke="{C["pink"]}" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#aj3)"/>'
            f'<rect x="10" y="135" width="130" height="36" rx="8" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="1.5"/>'
            f'<text x="75" y="152" text-anchor="middle" fill="{C["green"]}" font-size="8" font-weight="700">6 Proximity CSVs</text>'
            f'<line x1="140" y1="153" x2="278" y2="70" stroke="{C["green"]}" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#aj)"/>'
            f'<line x1="365" y1="44" x2="425" y2="44" stroke="{C["orange"]}" stroke-width="2" marker-end="url(#aj)"/>'
            f'<rect x="425" y="18" width="200" height="52" rx="10" fill="{C["orange"]}22" stroke="{C["orange"]}" stroke-width="2"/>'
            f'<text x="525" y="38" text-anchor="middle" fill="{C["orange"]}" font-size="10" font-weight="700">Pre-Cleaned Dataset</text>'
            f'<text x="525" y="51" text-anchor="middle" fill="{C["muted"]}" font-size="8">{pr:,} rows x ~{pc} cols</text>'
            f'</svg>'
        )

    elif stage_id == "modeling":
        hd_str = " \u2192 ".join(str(d) for d in hd)
        total_f = n_num + n_cat + n_bool
        return (
            f'<svg viewBox="0 0 640 150" xmlns="http://www.w3.org/2000/svg" style="background:{C["bg"]};border-radius:10px;border:1px solid {C["border"]}">'
            f'<defs><marker id="am" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["accentLt"]}"/></marker></defs>'
            f'<text x="60" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">INPUT FEATURES</text>'
            f'<rect x="10" y="22" width="100" height="28" rx="6" fill="{C["accent"]}22" stroke="{C["accent"]}" stroke-width="1"/>'
            f'<text x="60" y="40" text-anchor="middle" fill="{C["accent"]}" font-size="8" font-weight="600">Numeric ({n_num})</text>'
            f'<rect x="10" y="56" width="100" height="28" rx="6" fill="{C["cyan"]}22" stroke="{C["cyan"]}" stroke-width="1"/>'
            f'<text x="60" y="74" text-anchor="middle" fill="{C["cyan"]}" font-size="8" font-weight="600">Categorical ({n_cat})</text>'
            f'<rect x="10" y="90" width="100" height="28" rx="6" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="1"/>'
            f'<text x="60" y="108" text-anchor="middle" fill="{C["green"]}" font-size="8" font-weight="600">Boolean ({n_bool})</text>'
            f'<text x="200" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">PROCESSING</text>'
            f'<rect x="130" y="22" width="140" height="28" rx="6" fill="{C["accent"]}22" stroke="{C["accent"]}" stroke-width="1"/>'
            f'<text x="200" y="40" text-anchor="middle" fill="{C["accent"]}" font-size="8">StandardScaler</text>'
            f'<rect x="130" y="56" width="140" height="28" rx="6" fill="{C["cyan"]}22" stroke="{C["cyan"]}" stroke-width="1"/>'
            f'<text x="200" y="74" text-anchor="middle" fill="{C["cyan"]}" font-size="8">Embedding (dim={emb_dim})</text>'
            f'<rect x="130" y="90" width="140" height="28" rx="6" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="1"/>'
            f'<text x="200" y="108" text-anchor="middle" fill="{C["green"]}" font-size="8">Cast to Float</text>'
            f'<line x1="270" y1="70" x2="300" y2="70" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#am)"/>'
            f'<rect x="300" y="40" width="140" height="56" rx="10" fill="{C["accentLt"]}22" stroke="{C["accentLt"]}" stroke-width="2"/>'
            f'<text x="370" y="60" text-anchor="middle" fill="{C["accentLt"]}" font-size="9" font-weight="700">Residual MLP</text>'
            f'<text x="370" y="74" text-anchor="middle" fill="{C["muted"]}" font-size="7">{hd_str}</text>'
            f'<text x="370" y="86" text-anchor="middle" fill="{C["muted"]}" font-size="7">BatchNorm + Dropout</text>'
            f'<line x1="440" y1="68" x2="470" y2="68" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#am)"/>'
            f'<rect x="470" y="48" width="100" height="40" rx="8" fill="{C["orange"]}22" stroke="{C["orange"]}" stroke-width="1.5"/>'
            f'<text x="520" y="65" text-anchor="middle" fill="{C["orange"]}" font-size="9" font-weight="700">Output</text>'
            f'<text x="520" y="78" text-anchor="middle" fill="{C["muted"]}" font-size="7">Linear(64, 1)</text>'
            f'<text x="370" y="130" text-anchor="middle" fill="{C["dim"]}" font-size="8">{total_f} total features \u2192 1 prediction</text>'
            f'</svg>'
        )

    elif stage_id == "testing":
        return (
            f'<svg viewBox="0 0 640 95" xmlns="http://www.w3.org/2000/svg" style="background:{C["bg"]};border-radius:10px;border:1px solid {C["border"]}">'
            f'<text x="320" y="16" text-anchor="middle" fill="{C["muted"]}" font-size="9" font-weight="600">DATA SPLIT STRATEGY</text>'
            f'<rect x="20" y="28" width="{tr_pct * 5.6}" height="30" rx="6" fill="{C["accent"]}44" stroke="{C["accent"]}" stroke-width="1.5"/>'
            f'<text x="{20 + tr_pct * 2.8}" y="47" text-anchor="middle" fill="{C["accent"]}" font-size="10" font-weight="700">Train ({tr_pct}%)</text>'
            f'<rect x="{20 + tr_pct * 5.6 + 4}" y="28" width="{va_pct * 5.6}" height="30" rx="6" fill="{C["cyan"]}44" stroke="{C["cyan"]}" stroke-width="1.5"/>'
            f'<text x="{20 + tr_pct * 5.6 + 4 + va_pct * 2.8}" y="47" text-anchor="middle" fill="{C["cyan"]}" font-size="10" font-weight="700">Val ({va_pct}%)</text>'
            f'<rect x="{20 + (tr_pct + va_pct) * 5.6 + 8}" y="28" width="{te_pct * 5.6}" height="30" rx="6" fill="{C["green"]}44" stroke="{C["green"]}" stroke-width="1.5"/>'
            f'<text x="{20 + (tr_pct + va_pct) * 5.6 + 8 + te_pct * 2.8}" y="47" text-anchor="middle" fill="{C["green"]}" font-size="10" font-weight="700">Test ({te_pct}%)</text>'
            f'<text x="20" y="78" fill="{C["accent"]}" font-size="8">Trains the model</text>'
            f'<text x="{20 + tr_pct * 5.6 + 4}" y="78" fill="{C["cyan"]}" font-size="8">Tunes + early stop</text>'
            f'<text x="{20 + (tr_pct + va_pct) * 5.6 + 8}" y="78" fill="{C["green"]}" font-size="8">Final unbiased eval</text>'
            f'</svg>'
        )

    elif stage_id == "productionizing":
        return (
            f'<svg viewBox="0 0 640 110" xmlns="http://www.w3.org/2000/svg" style="background:{C["bg"]};border-radius:10px;border:1px solid {C["border"]}">'
            f'<defs><marker id="ap" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{C["accentLt"]}"/></marker></defs>'
            f'<rect x="10" y="25" width="100" height="50" rx="8" fill="{C["accent"]}22" stroke="{C["accent"]}" stroke-width="1.5"/>'
            f'<text x="60" y="45" text-anchor="middle" fill="{C["accent"]}" font-size="8" font-weight="700">Experiment</text>'
            f'<text x="60" y="58" text-anchor="middle" fill="{C["muted"]}" font-size="7">{exp_count} available</text>'
            f'<line x1="110" y1="50" x2="135" y2="50" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#ap)"/>'
            f'<rect x="135" y="25" width="110" height="50" rx="8" fill="{C["cyan"]}22" stroke="{C["cyan"]}" stroke-width="1.5"/>'
            f'<text x="190" y="45" text-anchor="middle" fill="{C["cyan"]}" font-size="8" font-weight="700">BatchPredictor</text>'
            f'<text x="190" y="58" text-anchor="middle" fill="{C["muted"]}" font-size="7">Load + reconstruct</text>'
            f'<line x1="245" y1="50" x2="270" y2="50" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#ap)"/>'
            f'<rect x="270" y="25" width="110" height="50" rx="8" fill="{C["green"]}22" stroke="{C["green"]}" stroke-width="1.5"/>'
            f'<text x="325" y="45" text-anchor="middle" fill="{C["green"]}" font-size="8" font-weight="700">Score Sites</text>'
            f'<text x="325" y="58" text-anchor="middle" fill="{C["muted"]}" font-size="7">{pr:,} sites</text>'
            f'<line x1="380" y1="50" x2="405" y2="50" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#ap)"/>'
            f'<rect x="405" y="25" width="100" height="50" rx="8" fill="{C["orange"]}22" stroke="{C["orange"]}" stroke-width="1.5"/>'
            f'<text x="455" y="45" text-anchor="middle" fill="{C["orange"]}" font-size="8" font-weight="700">Results</text>'
            f'<text x="455" y="58" text-anchor="middle" fill="{C["muted"]}" font-size="7">rank + percentile</text>'
            f'<line x1="505" y1="50" x2="530" y2="50" stroke="{C["accentLt"]}" stroke-width="1.5" marker-end="url(#ap)"/>'
            f'<rect x="530" y="25" width="90" height="50" rx="8" fill="{C["pink"]}22" stroke="{C["pink"]}" stroke-width="1.5"/>'
            f'<text x="575" y="45" text-anchor="middle" fill="{C["pink"]}" font-size="8" font-weight="700">Export</text>'
            f'<text x="575" y="58" text-anchor="middle" fill="{C["muted"]}" font-size="7">CSV / XLSX</text>'
            f'</svg>'
        )

    return ""


def assemble_stages(code_data: dict, data_facts: dict) -> list:
    """Merge code introspection with data facts into Stage objects."""
    stages_meta = code_data["stages"]
    steps_raw = code_data["steps"]
    sources_raw = code_data["sources"]

    # Group steps by stage_id
    stage_steps = {}
    for step_raw in steps_raw:
        glossary_path = step_raw.get("glossary", "")
        if "/" not in glossary_path:
            continue
        stage_id, step_id = glossary_path.split("/", 1)

        if stage_id not in stage_steps:
            stage_steps[stage_id] = []

        details = [Detail(d["title"], d["body"]) for d in step_raw.get("details", [])]

        step = Step(
            id=step_id,
            title=step_raw.get("title", step_id.replace("-", " ").title()),
            step_order=int(step_raw.get("step", 0)),
            color=step_raw.get("color", "accent"),
            sub=step_raw.get("sub", ""),
            analogy=step_raw.get("analogy"),
            why=step_raw.get("why"),
            details=details,
            source_function=step_raw.get("source_function", ""),
            source_file=step_raw.get("source_file", ""),
            source_line=step_raw.get("source_line", 0),
        )
        stage_steps[stage_id].append(step)

    # Sort steps within each stage
    for stage_id in stage_steps:
        stage_steps[stage_id].sort(key=lambda s: s.step_order)

    # Build stages
    stages = []
    for stage_id, stage_index in STAGE_ORDER:
        meta = stages_meta.get(stage_id, {})

        sources = []
        if stage_id == "collection":
            for s in sources_raw:
                sources.append(Source(**s))

        stage = Stage(
            id=stage_id,
            title=meta.get("title", f"{stage_index + 1}. {stage_id.title()}"),
            category=meta.get("category", "Pipeline Stages"),
            question=meta.get("question", ""),
            intro=meta.get("intro", ""),
            analogy=meta.get("analogy"),
            why=meta.get("why"),
            steps=stage_steps.get(stage_id, []),
            sources=sources,
            stats_row=_build_stats_row(stage_id, data_facts),
            flow_svg=_build_flow_svg(stage_id, data_facts),
            stage_index=stage_index,
        )
        stages.append(stage)

    return stages


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: RENDER — HTML Output
# ═══════════════════════════════════════════════════════════════════════

CSS = """*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; min-height: 100vh; }
#sidebar { width: 260px; flex-shrink: 0; border-right: 1px solid #334155; padding: 24px 0; overflow-y: auto; }
.sidebar-header { padding: 0 20px 20px; border-bottom: 1px solid #334155; }
.sidebar-header h1 { font-size: 17px; font-weight: 700; }
.sidebar-header p { font-size: 12px; color: #64748b; margin-top: 4px; }
.cat-label { padding: 0 20px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #64748b; margin: 16px 0 6px; }
.nav-btn { display: block; width: 100%; text-align: left; border: none; cursor: pointer; background: transparent; border-left: 3px solid transparent; padding: 8px 20px; font-family: inherit; font-size: 13px; color: #94a3b8; transition: all 0.15s; }
.nav-btn:hover { background: rgba(99,102,241,0.06); }
.nav-btn.active { background: rgba(99,102,241,0.09); border-left-color: #6366f1; color: #818cf8; font-weight: 600; }
.sidebar-footer { padding: 20px; margin-top: 20px; border-top: 1px solid #334155; font-size: 11px; color: #64748b; line-height: 1.5; }
.pipeline-progress { padding: 16px 20px; border-bottom: 1px solid #334155; }
.pip-step { display: flex; align-items: center; gap: 10px; padding: 4px 0; font-size: 12px; color: #64748b; }
.pip-dot { width: 10px; height: 10px; border-radius: 50%; border: 2px solid #334155; flex-shrink: 0; }
.pip-line { width: 2px; height: 12px; background: #334155; margin-left: 4px; }
#main { flex: 1; padding: 32px 40px; max-width: 740px; overflow-y: auto; }
.pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 10px; border-radius: 99px; background: rgba(99,102,241,0.13); color: #6366f1; border: 1px solid rgba(99,102,241,0.27); }
#entry-title { font-size: 24px; font-weight: 700; margin: 8px 0 4px; }
#entry-question { font-size: 15px; color: #818cf8; font-style: italic; margin-bottom: 24px; }
.etext { font-size: 14px; line-height: 1.65; margin-bottom: 14px; }
.analogy-box, .why-box, .detail-box { border-radius: 10px; padding: 14px 18px; margin-top: 14px; }
.analogy-box { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.2); }
.why-box { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.2); margin-top: 16px; }
.detail-box { background: rgba(251,146,60,0.07); border: 1px solid rgba(251,146,60,0.2); margin-top: 14px; }
.box-label { font-size: 13px; font-weight: 700; margin-bottom: 5px; }
.analogy-box .box-label { color: #34d399; }
.why-box .box-label { color: #818cf8; }
.detail-box .box-label { color: #fb923c; }
.box-body { font-size: 13px; line-height: 1.6; }
.stats-row { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
.stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; flex: 1; min-width: 100px; }
.stat-val { font-size: 20px; font-weight: 700; }
.stat-label { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.cap { font-size: 12px; color: #64748b; margin-bottom: 8px; }
.flow-wrap { margin: 20px 0; }
.flow-wrap svg { width: 100%; display: block; }
table.tt { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 14px; }
table.tt th { padding: 6px 10px; text-align: left; color: #64748b; font-weight: 500; border-bottom: 1px solid #334155; }
table.tt td { padding: 6px 10px; }
table.tt tr:nth-child(even) td { background: #111827; }
code { background: #1e293b; padding: 1px 5px; border-radius: 4px; font-size: 12px; font-family: "SF Mono", Monaco, Consolas, monospace; }
.source-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 16px 0; }
.source-card { background: #1e293b; border: 2px solid #334155; border-radius: 10px; padding: 14px 16px; cursor: pointer; transition: all 0.15s; }
.source-card:hover { border-color: #475569; }
.source-card.active { border-color: #6366f1; background: rgba(99,102,241,0.08); }
.source-icon { font-size: 22px; margin-bottom: 6px; }
.source-name { font-size: 14px; font-weight: 700; margin-bottom: 2px; }
.source-desc { font-size: 11px; color: #94a3b8; line-height: 1.4; }
.source-meta { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.source-meta span { font-size: 10px; padding: 2px 8px; border-radius: 6px; background: rgba(99,102,241,0.1); color: #94a3b8; }
.source-detail { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 18px; margin-top: 12px; display: none; }
.source-detail.visible { display: block; }
.detail-header { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
.detail-section { margin-top: 14px; }
.detail-section h4 { font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.sample-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.sample-table th { padding: 5px 8px; text-align: left; color: #64748b; font-weight: 600; border-bottom: 1px solid #334155; background: #0f172a; }
.sample-table td { padding: 5px 8px; border-bottom: 1px solid #1e293b; font-family: "SF Mono", Monaco, Consolas, monospace; }
.sample-wrap { max-height: 180px; overflow-y: auto; border-radius: 8px; border: 1px solid #334155; }
.step-card { background: #1e293b; border: 2px solid #334155; border-radius: 10px; padding: 16px; margin-top: 12px; cursor: pointer; transition: all 0.15s; }
.step-card:hover { border-color: #475569; }
.step-card.open { border-color: #6366f1; }
.step-num { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%; font-size: 12px; font-weight: 700; margin-right: 8px; flex-shrink: 0; }
.step-title { font-size: 14px; font-weight: 700; }
.step-sub { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.step-detail { margin-top: 12px; display: none; font-size: 13px; line-height: 1.6; }
.step-card.open .step-detail { display: block; }
.before-after { display: grid; grid-template-columns: 1fr 30px 1fr; gap: 8px; align-items: start; margin: 14px 0; }
.ba-col { background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 10px; font-size: 12px; }
.ba-col h5 { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; }
.ba-arrow { display: flex; align-items: center; justify-content: center; color: #6366f1; font-size: 18px; padding-top: 24px; }
.funnel-row { margin: 4px 0; display: flex; align-items: center; gap: 10px; }
.funnel-bar { height: 26px; border-radius: 5px; display: flex; align-items: center; padding: 0 10px; font-size: 11px; font-weight: 600; color: #0f172a; min-width: 40px; }
.funnel-label { font-size: 12px; color: #94a3b8; min-width: 160px; }
.placeholder { text-align: center; padding: 60px 20px; color: #475569; }
.placeholder-icon { font-size: 48px; margin-bottom: 12px; }
.placeholder-text { font-size: 15px; line-height: 1.6; }
.src-ref { font-size: 11px; color: #475569; margin-top: 10px; font-family: "SF Mono", Monaco, Consolas, monospace; }"""


def _serialize_stages(stages: list) -> str:
    """Serialize stages to JSON for embedding in HTML."""

    def _to_dict(obj):
        if hasattr(obj, "__dict__") and hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        return obj

    data = [_to_dict(s) for s in stages]
    return json.dumps(data, ensure_ascii=False)


JS_RUNTIME = r'''const C={accent:"#6366f1",accentLt:"#818cf8",green:"#34d399",orange:"#fb923c",pink:"#f472b6",cyan:"#22d3ee",yellow:"#facc15",red:"#f87171",border:"#334155",surface:"#1e293b",dim:"#64748b",muted:"#94a3b8",bg:"#0f172a",text:"#e2e8f0"};
const colorMap={accent:C.accent,accentLt:C.accentLt,cyan:C.cyan,green:C.green,orange:C.orange,pink:C.pink,yellow:C.yellow,red:C.red};
function analogyHTML(t){return`<div class="analogy-box"><div class="box-label">Plain-language analogy</div><div class="box-body">${t}</div></div>`;}
function whyHTML(t,title){return`<div class="why-box"><div class="box-label">${title||"How this works in our pipeline"}</div><div class="box-body">${t}</div></div>`;}
function detailHTML(t,title){return`<div class="detail-box"><div class="box-label">${title||"Key detail"}</div><div class="box-body">${t}</div></div>`;}
function renderStatsRow(stats){if(!stats||!stats.length)return'';return`<div class="stats-row">${stats.map(s=>`<div class="stat-card"><div class="stat-val" style="color:${s.color}">${s.value}</div><div class="stat-label">${s.label}</div></div>`).join('')}</div>`;}
function renderSourceCards(sources,stageId){let h=`<div class="source-grid" id="${stageId}-sources">`;sources.forEach(s=>{h+=`<div class="source-card" data-src="${s.id}"><div class="source-icon">${s.icon}</div><div class="source-name">${s.name}</div><div class="source-desc">${s.desc}</div><div class="source-meta"><span>${s.format}</span><span>${s.rows} rows</span><span>${s.cols} cols</span></div></div>`;});h+=`</div>`;return h;}
function renderStepCards(steps,stageId){let h=`<div id="${stageId}-steps">`;steps.forEach(step=>{const c=colorMap[step.color]||C.accent;let det='';if(step.analogy)det+=analogyHTML(step.analogy);if(step.extra_html)det+=step.extra_html;if(step.details)step.details.forEach(d=>{det+=detailHTML(d.body,d.title);});if(step.why)det+=whyHTML(step.why);if(step.source_function)det+=`<div class="src-ref">\u2192 ${step.source_function}() \u2014 ${step.source_file}:${step.source_line}</div>`;h+=`<div class="step-card" data-step="${step.step_order}"><div style="display:flex;align-items:center"><span class="step-num" style="background:${c}22;color:${c}">${step.step_order}</span><span class="step-title" style="color:${c}">${step.title}</span></div><div class="step-sub">${step.sub}</div><div class="step-detail">${det}</div></div>`;});h+=`</div>`;return h;}
function renderStageContent(stage){let h='';if(stage.intro)h+=`<p class="etext">${stage.intro}</p>`;if(stage.analogy)h+=analogyHTML(stage.analogy);if(stage.flow_svg)h+=`<div class="flow-wrap">${stage.flow_svg}</div>`;h+=renderStatsRow(stage.stats_row);if(stage.sources&&stage.sources.length>0){h+=`<p class="cap" style="margin-top:20px">Click a source to explore its fields, sample data, and collection details:</p>`;h+=renderSourceCards(stage.sources,stage.id);h+=`<div class="source-detail" id="source-detail-${stage.id}"></div>`;}if(stage.steps&&stage.steps.length>0){h+=`<p class="cap" style="margin-top:20px">Click each step to expand the details:</p>`;h+=renderStepCards(stage.steps,stage.id);}if((!stage.steps||!stage.steps.length)&&(!stage.sources||!stage.sources.length)&&!stage.intro){h+=`<div class="placeholder"><div class="placeholder-icon">\ud83d\udccb</div><div class="placeholder-text">The <strong>${stage.title}</strong> entry needs <code>@glossary</code> docstrings.<br>Run <code>python scripts/generate_glossary.py --check</code> to see what\u2019s missing.</div></div>`;}if(stage.why)h+=whyHTML(stage.why);return h;}
function initStage(stage){const sc=document.getElementById(stage.id+'-steps');if(sc){sc.querySelectorAll('.step-card').forEach(card=>{card.addEventListener('click',()=>{const w=card.classList.contains('open');sc.querySelectorAll('.step-card').forEach(c=>c.classList.remove('open'));if(!w)card.classList.add('open');});});}const sr=document.getElementById(stage.id+'-sources');if(sr&&stage.sources){sr.querySelectorAll('.source-card').forEach(card=>{card.addEventListener('click',()=>{const s=stage.sources.find(d=>d.id===card.dataset.src);if(!s)return;sr.querySelectorAll('.source-card').forEach(c=>c.classList.remove('active'));card.classList.add('active');const det=document.getElementById('source-detail-'+stage.id);const cc=colorMap[s.color]||C.accent;det.innerHTML=`<div class="detail-header" style="color:${cc}">${s.icon} ${s.name}</div><div class="detail-section"><h4>Key Fields</h4><div style="display:flex;flex-wrap:wrap;gap:6px">${s.fields.map(f=>`<code>${f}</code>`).join('')}</div></div><div class="detail-section"><h4>Sample Rows</h4><div class="sample-wrap"><table class="sample-table"><thead><tr>${s.fields.map(f=>`<th>${f}</th>`).join('')}</tr></thead><tbody>${s.sample.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div><div class="detail-section"><h4>Notes</h4><p style="font-size:13px;line-height:1.6">${s.notes}</p></div>`;det.classList.add('visible');});});}}
const ENTRIES=STAGE_DATA.map(stage=>({id:stage.id,title:stage.title,category:stage.category,question:stage.question,entry:{render(){return renderStageContent(stage);},init(){initStage(stage);}},stage:stage.stage_index}));
let activeId=ENTRIES[0].id;
const stageLabels=["Collect","Clean","Combine","Model","Test","Deploy"];
const stageColors=[C.accent,C.cyan,C.green,C.orange,C.pink,C.accentLt];
function renderPipelineViz(){const a=ENTRIES.find(e=>e.id===activeId);const as=a?a.stage:0;let h='<div class="pipeline-progress">';stageLabels.forEach((l,i)=>{const ds=i<as?`background:${C.green};border-color:${C.green}`:i===as?`background:${stageColors[i]};border-color:${stageColors[i]};box-shadow:0 0 6px ${stageColors[i]}44`:"";h+=`<div class="pip-step" style="${i===as?"color:"+C.text+";font-weight:600":""}"><div class="pip-dot" style="${ds}"></div>${l}</div>`;if(i<5)h+=`<div class="pip-line" style="${i<as?`background:${C.green}`:""}"></div>`;});h+='</div>';document.getElementById("pipeline-viz").innerHTML=h;}
function renderNav(){document.getElementById("count-label").textContent=`Interactive reference \u00b7 ${ENTRIES.length} stages`;const cats=[...new Set(ENTRIES.map(e=>e.category))];let h="";cats.forEach(cat=>{h+=`<div class="cat-label">${cat}</div>`;ENTRIES.filter(e=>e.category===cat).forEach(e=>{h+=`<button class="nav-btn${e.id===activeId?" active":""}" data-id="${e.id}">${e.title}</button>`;});});document.getElementById("nav-list").innerHTML=h;document.querySelectorAll(".nav-btn").forEach(b=>{b.addEventListener("click",()=>{activeId=b.dataset.id;renderPipelineViz();renderNav();showEntry();});});}
function showEntry(){const e=ENTRIES.find(e=>e.id===activeId);document.getElementById("ep").innerHTML=`<span class="pill">${e.category}</span>`;document.getElementById("entry-title").textContent=e.title;document.getElementById("entry-question").textContent=e.question;document.getElementById("ec").innerHTML=e.entry.render();if(e.entry.init)e.entry.init();}
renderPipelineViz();renderNav();showEntry();'''


def render_html(stages: list) -> str:
    """Render the complete HTML document."""
    stage_json = _serialize_stages(stages)

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>Data Pipeline Glossary</title>\n'
        f'<style>\n{CSS}\n</style>\n'
        '</head>\n<body>\n'
        '<nav id="sidebar">\n'
        '  <div class="sidebar-header"><h1>Data Pipeline</h1><p id="count-label"></p></div>\n'
        '  <div id="pipeline-viz"></div>\n'
        '  <div id="nav-list"></div>\n'
        '  <div class="sidebar-footer"><strong style="color:#94a3b8">Interactive reference</strong><br>'
        'Generated from codebase introspection. Click any stage to explore.</div>\n'
        '</nav>\n'
        '<main id="main">\n'
        '  <div id="ep"></div><h2 id="entry-title"></h2><p id="entry-question"></p><div id="ec"></div>\n'
        '</main>\n'
        f'<script>\nconst STAGE_DATA = {stage_json};\n{JS_RUNTIME}\n</script>\n'
        '</body>\n</html>'
    )


# ═══════════════════════════════════════════════════════════════════════
# CHECK MODE
# ═══════════════════════════════════════════════════════════════════════

def check_coverage(code_data: dict, data_facts: dict) -> int:
    """Report on glossary coverage. Returns number of issues found."""
    issues = 0
    stages_meta = code_data["stages"]
    steps_raw = code_data["steps"]

    print("Checking glossary coverage...\n")

    # Group steps by stage
    stage_steps = {}
    for step in steps_raw:
        path = step.get("glossary", "")
        if "/" in path:
            sid = path.split("/")[0]
            stage_steps.setdefault(sid, []).append(step)

    for stage_id, stage_index in STAGE_ORDER:
        meta = stages_meta.get(stage_id)
        steps = stage_steps.get(stage_id, [])

        if meta:
            print(f"  \033[32mOK\033[0m    Stage: {stage_id} ({len(steps)} steps)")
            if stage_id == "collection" and code_data["sources"]:
                print(f"         + {len(code_data['sources'])} source cards")
        else:
            print(f"  \033[33mMISS\033[0m  Stage: {stage_id} (no _GLOSSARY_STAGES entry)")
            issues += 1

        for step in sorted(steps, key=lambda s: int(s.get("step", 0))):
            fn = step.get("source_function", "?")
            fl = step.get("source_file", "?")
            ln = step.get("source_line", 0)
            title = step.get("title", "?")
            print(f"         step {step.get('step', '?')}: {title}  ({fn} @ {fl}:{ln})")

        if not steps and stage_id != "collection":
            print(f"         \033[33m(no @glossary docstrings found)\033[0m")
            issues += 1

    print("\nData files:")
    for name, label in [("precleaned", "site_aggregated_precleaned.parquet"),
                        ("training", "site_training_data.parquet")]:
        rows = data_facts.get(f"{name}_rows")
        cols = data_facts.get(f"{name}_cols")
        if rows is not None:
            print(f"  \033[32mOK\033[0m    {label}: {rows:,} rows x {cols} cols")
        else:
            print(f"  \033[33mMISS\033[0m  {label}")
            issues += 1

    ifc = data_facts.get("input_file_count", 0)
    print(f"  {'\\033[32mOK\\033[0m' if ifc else '\\033[33mMISS\\033[0m'}    data/input/: {ifc} CSV files")
    print(f"\n{issues} issue{'s' if issues != 1 else ''} found.")
    return issues


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate the Data Pipeline Glossary HTML from codebase introspection."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "docs" / "pipeline_glossary.html",
        help="Output HTML file path (default: docs/pipeline_glossary.html)",
    )
    parser.add_argument("--check", "-c", action="store_true",
                        help="Check docstring coverage without generating HTML")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show introspection details")
    args = parser.parse_args()

    # Phase 1: Introspect
    code_data = introspect_code(PROJECT_ROOT, verbose=args.verbose)
    data_facts = introspect_data(PROJECT_ROOT, verbose=args.verbose)

    if args.verbose:
        print(f"\nFound {len(code_data['steps'])} @glossary tags across "
              f"{len(code_data['stages'])} stages")

    # Check mode
    if args.check:
        issues = check_coverage(code_data, data_facts)
        sys.exit(2 if issues > 0 else 0)

    # Phase 2: Assemble
    stages = assemble_stages(code_data, data_facts)

    # Phase 3: Render
    html = render_html(stages)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(f"Generated {args.output} ({len(html):,} bytes, {len(stages)} stages)")


if __name__ == "__main__":
    main()
