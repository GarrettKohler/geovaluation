# Dynamic Pipeline Glossary — Design Spec

**Date:** 2026-04-14
**Status:** Approved
**Approach:** AST parsing + structured docstrings → single self-contained HTML file

---

## Problem

The Data Pipeline Glossary is a standalone interactive HTML document that explains the ML pipeline across 6 stages. Currently, all content — row counts, column counts, feature lists, function references, editorial explanations — is hardcoded. When the code changes, the glossary drifts. Three stages (Modeling, Testing & Validation, Productionizing) are empty placeholders.

## Solution

A build-time generator script that:
1. Parses structured `@glossary` docstrings from pipeline source files via Python's `ast` module
2. Reads parquet file metadata and filesystem state for dynamic facts (row counts, column counts, file sizes)
3. Renders a single self-contained HTML file with the same interactive dark-theme design

Editorial content (analogies, explanations) lives **in the code** as structured docstrings. Factual content (counts, feature lists) is introspected at generation time. The output is a standalone HTML file — no server required.

---

## Docstring Format

### Step-Level Tags

Each pipeline function that should appear in the glossary gets `@glossary` tags in its docstring:

```python
def add_log_transformations(df):
    """Apply sign-preserving log transforms to skewed numeric features.

    @glossary: cleaning/log-transforms
    @title: Log Transformations
    @step: 4
    @color: pink
    @sub: 13 right-skewed features transformed to normalize distributions
    @analogy: Revenue and distances are like earthquake magnitudes —
        the difference between $100 and $1,000 matters more than
        between $100,000 and $101,000. Log compression makes the
        model treat differences proportionally.
    @why: Revenue and distance metrics are heavily right-skewed.
        Log transformation compresses the long tail so the model
        doesn't overweight outliers.
    @detail[Why sign-preserving?]: The formula uses sign(x) x log(1+|x|)
        for signed types and log(x+1) for unsigned. This handles rare
        negative revenue records without losing sign information.
    """
```

### Tag Reference

| Tag | Required | Type | Purpose |
|-----|----------|------|---------|
| `@glossary` | Yes | `stage_id/step_id` | Maps function to a stage and step |
| `@title` | Yes | string | Accordion card title |
| `@step` | Yes | int | Sort order within the stage (0-indexed) |
| `@color` | No | color key | Theme color. One of: `accent`, `cyan`, `green`, `orange`, `pink`, `yellow`, `red`. Default: `accent` |
| `@sub` | Yes | string | One-line subtitle below the title |
| `@analogy` | No | multiline | Plain-language analogy (green box) |
| `@why` | No | multiline | "How this works in our pipeline" (purple box) |
| `@detail[Title]` | No | multiline | Key detail box (orange box). Multiple allowed |

Tags span multiple lines until the next `@` tag or end of docstring. A function can have multiple `@glossary` tags to appear in multiple stages.

### Stage-Level Metadata

Module-level `_GLOSSARY_STAGES` dicts define stage-level content:

```python
_GLOSSARY_STAGES = {
    "cleaning": {
        "title": "2. Data Cleaning",
        "question": "How do we fix errors, standardize formats, and prepare data for modeling?",
        "intro": "Before any modeling can happen, raw data needs to be...",
        "analogy": "Imagine you received a massive stack of monthly reports...",
        "why": "The full cleaning pipeline runs via python3 -m site_scoring.data_transform...",
    },
}
```

### Data Source Cards (Collection Stage)

The Collection stage uses interactive source cards. These are defined via a `_GLOSSARY_SOURCES` list:

```python
_GLOSSARY_SOURCES = [
    {
        "id": "sites_base",
        "icon": "\u26fd",
        "name": "Sites - Base Data Set",
        "desc": "Site attributes, capabilities, ad eligibility, demographics",
        "source": "Salesforce",
        "format": "CSV",
        "cols": 43,
        "color": "accent",
        "fields": ["ID - Gbase", "Avg Daily Impressions", ...],
        "sample": [["5fa964f8...", "121", ...], ...],
        "notes": "Exported from Salesforce. Each row is a gas station..."
    },
    ...
]
```

Row counts for source cards are introspected from actual files when available; otherwise the value in the dict is used as fallback.

---

## Generator Architecture

### File: `scripts/generate_glossary.py`

Three phases: Introspect → Assemble → Render.

### Phase 1: Introspect

**Code introspection** via `ast` module (no imports of heavy dependencies):

```
introspect_code(source_files: list[Path]) -> dict
├── For each source file:
│   ├── ast.parse() the file
│   ├── Walk AST for FunctionDef / AsyncFunctionDef / ClassDef nodes
│   ├── Extract docstrings, parse @glossary tags
│   ├── Record function name, file path, line number
│   └── Extract module-level _GLOSSARY_STAGES and _GLOSSARY_SOURCES via ast.literal_eval
└── Return: {stage_id: {steps: [...], stage_meta: {...}}}
```

**Data introspection** via filesystem + Polars:

```
introspect_data(data_dir: Path, experiments_dir: Path) -> dict
├── Parquet metadata (schema + row count, no full load):
│   ├── site_aggregated_precleaned.parquet → rows, column names
│   └── site_training_data.parquet → rows, column names
├── Glob data/input/*.csv → file names, sizes
├── Glob experiments/ → experiment count
├── AST-parse config.py for feature lists:
│   ├── NUMERIC_FEATURES → names + count
│   ├── CATEGORICAL_FEATURES → names + count
│   └── BOOLEAN_FEATURES → names + count
└── Return: {precleaned_rows, training_rows, precleaned_cols, training_cols, ...}
```

### Phase 2: Assemble

Build a structured data model:

```python
@dataclass
class Detail:
    title: str
    body: str

@dataclass
class Step:
    id: str
    title: str
    step_order: int
    color: str
    sub: str
    analogy: str | None
    why: str | None
    details: list[Detail]
    source_function: str
    source_file: str
    source_line: int

@dataclass
class StatCard:
    value: str
    label: str
    color: str

@dataclass
class Stage:
    id: str
    title: str
    category: str
    question: str
    intro: str
    analogy: str | None
    why: str | None
    steps: list[Step]
    stats_row: list[StatCard]
    flow_svg: str | None
    stage_index: int
```

Assembly merges code-introspected docstrings with data-introspected facts. Stat card values and SVG placeholder values are filled from the data introspection results.

### Phase 3: Render

Serialize the data model into a single HTML file:

- **CSS:** Embedded in `<style>`, identical to current glossary styling
- **HTML structure:** `<nav id="sidebar">` + `<main id="main">` — static shell
- **JavaScript:** Generated `ENTRIES` array from the assembled data model, plus static helper functions (`analogyHTML()`, `whyHTML()`, `detailHTML()`) and shell functions (`renderPipelineViz()`, `renderNav()`, `renderEntry()`)

Each `ENTRIES[i].render()` function returns HTML strings built from the Step data. Each `ENTRIES[i].init()` function wires up accordion click handlers.

Output: `docs/pipeline_glossary.html`

---

## Source Files Modified

### Docstrings Added (no logic changes)

| File | Changes | Functions Tagged |
|------|---------|-----------------|
| `site_scoring/data_transform.py` | Add `_GLOSSARY_STAGES` (collection, cleaning, combining), `_GLOSSARY_SOURCES`, and `@glossary` docstrings | `load_site_scores()`, `load_auxiliary_data()`, `load_daily_transactions()`, `load_daily_status()`, `aggregate_site_metrics()`, `calculate_all_relative_strength_features()`, `join_geospatial_features()`, `add_log_transformations()`, `one_hot_encode_flags()`, `bin_high_cardinality()`, `prepare_training_dataset()`, `transform_data()` |
| `src/services/training_service.py` | Add `_GLOSSARY_STAGES` (modeling, testing), `@glossary` docstrings | `run_training_logic()` (data loading step, training loop step, test eval step, artifact saving step, SHAP step, classification export step), `_run_tree_training()`, `_run_revenue_prediction_phase()` |
| `site_scoring/model.py` | Add `@glossary` docstrings | `SiteScoringModel.__init__()` |
| `site_scoring/data_loader.py` | Add `@glossary` docstrings | `DataProcessor.load_and_process()`, `_process_numeric()`, `_process_categorical()`, `_process_boolean()`, `_process_target()` |
| `site_scoring/predict.py` | Add `_GLOSSARY_STAGES` (productionizing), `@glossary` docstrings | `BatchPredictor.__init__()`, `BatchPredictor.predict()`, `BatchPredictor.predict_with_metadata()`, `get_all_sites_for_prediction()` |
| `src/routes/prediction.py` | Add `@glossary` docstrings | `api_predict_batch()`, `api_predict_export()`, `api_predict_filtered()` |

### Files Created

| File | Purpose |
|------|---------|
| `scripts/generate_glossary.py` | Generator script (~600-800 lines) |
| `docs/pipeline_glossary.html` | Generated output (single self-contained HTML) |

### Files Read Only (not modified)

| File | What's Extracted |
|------|-----------------|
| `site_scoring/config.py` | Feature lists (NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES) via AST |
| `data/processed/*.parquet` | Row counts, column schemas via Polars metadata read |
| `data/input/*.csv` | File existence, sizes via filesystem |
| `site_scoring/outputs/experiments/` | Experiment count, latest experiment metrics via glob + JSON read |

---

## Stage Coverage

### Stage 1: Data Collection

**Source:** `_GLOSSARY_SOURCES` in `data_transform.py`, `load_*()` functions

**Steps:**
- Stage-level content only (no accordion steps) — uses interactive source cards instead
- 4 source cards: Sites Base Data Set, Site Revenue, Geographic Proximity Files, Retailer Location Data
- Each card: icon, name, description, source, format, fields, sample rows, notes
- Row counts dynamically read from CSV files when available

**Stats row:** Unique sites (from precleaned parquet rows), source file count (from glob), retailer locations count (hardcoded in source metadata), format label

**Flow SVG:** Source files → Data Ingestion → data/input/ — row counts injected

### Stage 2: Data Cleaning

**Source:** `_GLOSSARY_STAGES["cleaning"]` + 7 tagged functions in `data_transform.py`

**Steps:**
0. Null Value Handling — `load_site_scores()` with before/after view
1. Monthly to Site-Level Aggregation — `aggregate_site_metrics()` with strategy table and before/after
2. Geospatial Feature Joining — `join_geospatial_features()` with match rate table
3. Log Transformations — `add_log_transformations()` with column table and formula
4. Flag Encoding — `one_hot_encode_flags()` with encoding strategy table and before/after
5. High-Cardinality Binning — `bin_high_cardinality()` with top-N table
6. Training Set Filtering — `prepare_training_dataset()` with funnel chart (dynamic row counts)

**Stats row:** Dynamic from parquet metadata

**Funnel chart:** Row counts from actual parquet files (precleaned rows, training rows)

### Stage 3: Data Combining

**Source:** `_GLOSSARY_STAGES["combining"]` + tagged functions

**Steps:**
0. Pre-Combined Salesforce Data — stage-level note about upstream merge
1. Temporal Aggregation — `aggregate_site_metrics()` (second `@glossary` tag) with strategy table
2. Relative Strength Feature Join — `calculate_all_relative_strength_features()` with horizon table
3. Geospatial Feature Joins — `join_geospatial_features()` (second tag) with join diagram
4. Combined Output — `transform_data()` with column growth bar chart

**Stats row:** Input columns (94), sites after aggregation (from parquet), join count (6), output columns (from parquet schema)

**Column growth chart:** Column counts from parquet schemas at each stage

### Stage 4: Modeling (NEW)

**Source:** `_GLOSSARY_STAGES["modeling"]` in `training_service.py`, tagged functions across `training_service.py`, `model.py`, `data_loader.py`

**Steps:**
0. Data Loading & Splitting — `DataProcessor.load_and_process()` — parquet load, network filter, history filter, 70/15/15 split
1. Feature Processing — `_process_numeric/categorical/boolean()` — StandardScaler, LabelEncoder, 0/1 cast. Before/after view
2. Target Preparation — `_process_target()` — regression: scale revenue. Lookalike: binarize by percentile. Before/after
3. Neural Network Architecture — `SiteScoringModel.__init__()` — architecture SVG with layer dims from config.py
4. XGBoost Alternative — `_run_tree_training()` — gradient boosted trees, when/why to use each
5. Training Loop — epoch loop in `run_training_logic()` — forward/backward/optimize, early stopping, LR scheduling, SSE streaming
6. Experiment Artifacts — save block — what gets saved and directory structure diagram

**Stats row:** Feature counts from config.py (numeric count, categorical count, boolean count), hidden_dims from config.py

**Flow SVG:** Data → Split → Train Loop (with epoch detail) → Save Artifacts — dims and counts injected from config.py

**Architecture SVG:** Input layers (with feature counts) → Embedding + BatchNorm → Concat → Residual MLP blocks (with hidden_dims) → Output

### Stage 5: Testing & Validation (NEW)

**Source:** `_GLOSSARY_STAGES["testing"]` in `training_service.py`, tagged blocks

**Steps:**
0. Train/Val/Test Split Philosophy — stage-level intro about 3 splits and the "never report val as test" lesson
1. Validation During Training — val loop — per-epoch eval, early stopping, LR scheduling
2. Test Set Evaluation — test eval block — load best model, score held-out set, compute metrics, inverse transform
3. Regression Metrics — MAE, RMSE, R-squared, MAPE, SMAPE — what each means and which to trust
4. Classification Metrics — AUC-ROC, F1, Log Loss — threshold selection and class imbalance handling
5. SHAP Feature Importance — `compute_shap_values()` — background sampling, explain set, top-20 features
6. Classification Exports — `_export_classification_results()` — training_sites.csv, test_predictions.csv, non_active_classification.csv

**Stats row:** If latest experiment has model_metadata.json, show its test metrics. Otherwise show descriptive placeholders.

### Stage 6: Productionizing (NEW)

**Source:** `_GLOSSARY_STAGES["productionizing"]` in `predict.py`, tagged functions across `predict.py` and `src/routes/prediction.py`

**Steps:**
0. Experiment Discovery — `_find_latest_experiment()` — mtime sort, artifact validation (config.json + preprocessor.pkl + model file)
1. Model Loading — `BatchPredictor.__init__()` — load config, preprocessor, reconstruct model, eval mode. Caching strategy
2. Inference Data Preparation — `get_all_sites_for_prediction()` — load precleaned parquet (all statuses), apply same transforms, module-level cache
3. Batch Prediction — `BatchPredictor.predict()` — process features, route to XGB or NN, 4096-batch inference, inverse transform
4. Filtered Scoring — `/api/predict/filtered` — apply map filter state, score subset, summary stats
5. Export Pipeline — `/api/predict/export` — join predictions + metadata, rank/percentile, CSV/XLSX download

**Stats row:** Total scoreable sites (from precleaned parquet rows), experiment count (from experiments glob), model types available

**Flow SVG:** Experiment folder → BatchPredictor (load) → Process Features → Model Inference → {gtvid: score} → Export — site counts injected

---

## SVG Diagrams

### Approach

Each stage has 1-2 SVG flow diagrams stored as Python template strings in the generator. Layout and styling are static; numeric labels are injected from introspection results.

### Template Inventory

| Stage | Diagram | Dynamic Placeholders |
|-------|---------|---------------------|
| Collection | Source → Ingestion → data/input/ | `{sites_base_rows}`, `{revenue_rows}`, `{unique_sites}`, `{source_file_count}` |
| Cleaning | Funnel chart (records → sites → active → clean) | `{raw_rows}`, `{precleaned_rows}`, `{active_rows}`, `{training_rows}` |
| Combining | Raw → Aggregated → Joins → Output | `{input_cols}`, `{precleaned_rows}`, `{precleaned_cols}`, `{rs_feature_count}`, `{distance_join_count}` |
| Combining | Column growth bar chart | `{raw_cols}`, `{after_agg_cols}`, `{after_rs_cols}`, `{after_geo_cols}`, `{training_cols}` |
| Modeling | Feature Input → Model → Output | `{n_numeric}`, `{n_categorical}`, `{n_boolean}`, `{hidden_dims}`, `{embedding_dim}` |
| Modeling | Training flow | `{train_split}`, `{val_split}`, `{test_split}` |
| Testing | Train/Val/Test split bars | `{train_count}`, `{val_count}`, `{test_count}` |
| Productionizing | Experiment → Load → Predict → Export | `{total_sites}`, `{experiment_count}` |

### What Stays Hardcoded in Templates

- SVG viewBox, box positions, line paths, arrow markers
- Color assignments (which box gets which theme color)
- Font sizes and text anchoring
- Overall layout structure

---

## CLI Interface

```bash
# Generate the glossary (default output: docs/pipeline_glossary.html)
python scripts/generate_glossary.py

# Custom output path
python scripts/generate_glossary.py --output /path/to/glossary.html

# Check mode — report missing @glossary docstrings, validate tags, don't generate
python scripts/generate_glossary.py --check

# Verbose — log each introspection step
python scripts/generate_glossary.py --verbose
```

### Exit Codes

- `0` — success
- `1` — missing required files (source files or data files)
- `2` — docstring parse errors (malformed @glossary tags)

### Check Mode Output

```
Checking glossary coverage...

Stage: collection (4 sources, 0 steps)
  OK  load_site_scores          data_transform.py:28
  OK  load_auxiliary_data        data_transform.py:40

Stage: cleaning (7 steps)
  OK  null-handling              data_transform.py:28    step=0
  OK  aggregation                data_transform.py:402   step=1
  ...

Stage: modeling (7 steps)
  MISSING  _run_tree_training    training_service.py:477  (no @glossary tag)

Data files:
  OK  site_aggregated_precleaned.parquet  57,675 rows x 102 cols
  OK  site_training_data.parquet          26,099 rows x 111 cols
  MISSING  data/input/site_scores_revenue_and_diagnostics.csv

2 issues found.
```

---

## Dependencies

The generator uses only:
- `ast`, `pathlib`, `json`, `dataclasses`, `argparse`, `html` — stdlib
- `polars` — read parquet schema and row counts (already a project dependency)

No torch, sklearn, or xgboost imports. Generation runs in < 2 seconds.

---

## Output Characteristics

- Single self-contained HTML file (~80-100KB)
- No external dependencies (CSS, JS, fonts all inline)
- Works offline — can be opened directly in any browser
- Same interactive behavior as current glossary: sidebar nav, pipeline progress indicator, accordion steps, source cards, before/after views, funnel/bar charts, flow diagrams
- Identical visual design (dark theme, color palette, typography)
