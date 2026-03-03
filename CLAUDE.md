# Geospatial Site Analysis

> ML-powered site scoring and visualization platform for 60K+ gas station advertising sites.

## Quick Context

Flask web app with PyTorch ML pipeline. Visualizes site performance on Leaflet map, trains revenue prediction and lookalike classification models. Optimized for Apple Silicon (MPS).

## Project Structure

```
/                           # Flask app root
├── app.py                  # Flask routes, API endpoints
├── templates/              # Jinja2 templates
│   └── index.html          # Main UI (map + training panel)
├── src/services/           # Web service layer
│   ├── data_service.py     # Site data loading for web (pandas)
│   └── training_service.py # ML training orchestration
├── site_scoring/           # ML pipeline
│   ├── data/               # Standardized data loading
│   │   ├── registry.py     # DataRegistry singleton
│   │   └── outputs/tensor.py # FeatureProcessor
│   ├── config.py           # Model configuration, feature lists
│   ├── model.py            # XGBoost + PyTorch neural network
│   ├── trainer.py          # Training loop
│   ├── data_loader.py      # Tensor conversion
│   └── data_transform.py   # ETL pipeline (polars)
├── data/
│   ├── input/              # Source CSVs
│   └── processed/          # Parquet files
└── docs/
    └── knowledge-core.md   # Accumulated learnings
```

## Key Patterns

### Data Loading

**Use DataRegistry for new code** (ADR-007):
```python
from site_scoring.data import DataRegistry
registry = DataRegistry()
training_data = registry.get_training_data()  # Active, 12+ months
```

### Feature Processing

**Use FeatureProcessor for train/inference**:
```python
from site_scoring.data import FeatureProcessor
processor = FeatureProcessor(config)
bundle = processor.fit_transform(df)
processor.save("processor.pkl")
```

### Status Column Typo

Source data has `statuis` (not `status`). The DataRegistry handles both, but be aware when writing raw queries.

### Datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `site_scores_revenue_and_diagnostics.csv` | 1.4M | 94 | Primary source — one row per site-month with revenue, diagnostics, and metadata |
| `site_transactions_daily.csv` | 1M | 4 | Daily transaction counts per site (`ID - Gbase`, `Date`, `Daily Transactions`, `GTVID`). Covers ~31.5K unique GTVIDs, date range 2022-01-01 to 2022-02-03 |
| `site_status_daily.csv` | 1M | 4 | Daily status snapshots per site (`Date`, `Status`, `GTVID`, `ID - Gbase`). Covers ~53.4K unique GTVIDs, date range 2022-01-01 to 2022-01-19. Statuses: Active, Deactivated, Temporarily Deactivated, Awaiting Deactivation, Awaiting Installation, Cancelled, Contract Signed |
| `site_aggregated_precleaned.parquet` | 57,675 | 106 | ETL output — one row per site, aggregated from primary source |
| `site_training_data.parquet` | 26,099 | 106 | Active sites with 12+ months history, used for model training |

## Development

### Run App
```bash
python3 app.py  # Serves on http://localhost:8080
```

### Run Tests
```bash
pytest tests/ -v --ignore=tests/slow
```

### Generate Training Data
```bash
python3 -m site_scoring.data_transform
```

## Architecture Decisions

- **ADR-003**: Polars for large CSV (10-20x faster than pandas)
- **ADR-005**: SSE for training progress (simpler than WebSocket)
- **ADR-007**: DataRegistry singleton for consistent data access
- **ADR-008**: Transformer chain pattern (planned)

See `docs/knowledge-core.md` for full decision log.

## Current Task Context

### Recently Completed (2026-02-05)
- Dynamic "Huber Loss" chart title for NN regression (vs generic "Loss Curve")
- MAPE metric (Mean Absolute Percentage Error) for regression models
- Live weight/bias distribution histograms during NN training
- Experiment folder FIFO cleanup (max 10 models in `site_scoring/outputs/experiments`)

### Previously Completed (2026-02-01)
- Added live training charts (Chart.js) to sidebar
- Implemented configurable percentile bounds for lookalike classifier
- Created DataRegistry and FeatureProcessor patterns
- Revenue percentile calculation now uses active sites only (p20-p95)

### ML Configuration

| Task | Target | Notes |
|------|--------|-------|
| Regression | `avg_monthly_revenue` | XGBoost or Neural Network |
| Lookalike | Binarized by percentile | p90-p100 default (configurable) |

### Model Presets — Ignore

Model A / Model B presets in `config.py` were for demo purposes only. Do not reference or build on them. There is one feature set; use user-selected features via the UI.

## Conventions

- **Indentation**: 4 spaces (Python), 4 spaces (JS)
- **Imports**: stdlib → third-party → local (isort compatible)
- **Commits**: Co-authored with Claude
- **API responses**: Always clean NaN/Inf for JSON serialization

## Known Gotchas / Past Bugs

### Test vs Validation Metrics (Fixed 2026-02-04)

**Problem**: Training service was reporting validation metrics as "test" metrics in the UI.

**Root Cause**: After the training loop, the code saved the best model but never ran evaluation on the held-out **test set**. It was reusing the last epoch's validation metrics, which are biased because validation loss influenced early stopping.

**Fix**: Added proper test set evaluation after loading the best model:
```python
# CORRECT: Evaluate on test_loader AFTER loading best_model_state
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    for batch in test_loader:  # NOT val_loader
        # ... compute predictions and metrics
```

**Key Insight**:
- **Validation set** is used during training (early stopping, hyperparameter tuning)
- **Test set** is used ONCE at the end for unbiased final metrics
- Never report validation metrics as test metrics - they're optimistically biased

### sklearn Scalers Need 2D Arrays (Fixed 2026-02-03)

**Problem**: Neural network test metrics showed catastrophic values (loss: 3.37e+19, R²: -Infinity).

**Root Cause**: `StandardScaler.inverse_transform()` expects 2D arrays. The code passed 1D arrays for NN predictions but correctly reshaped for gradient boosting.

**Fix**: Always reshape before inverse transform:
```python
# CORRECT
preds_orig = scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()

# WRONG - causes silent corruption
preds_orig = scaler.inverse_transform(predictions_np)  # 1D array
```

### JSON Serialization of Infinity (Fixed 2026-02-05)

**Problem**: Browser SSE handler silently failed — UI stuck at "Training job started..." with no progress updates.

**Root Cause**: `TrainingProgress.best_val_loss` defaults to `float('inf')`. Python's `json.dumps()` outputs `Infinity`, which is NOT valid JSON (RFC 8259). Browser `JSON.parse()` throws `SyntaxError`.

**Fix**: Added `_sanitize_for_json()` recursive function in `training_service.py` that replaces `inf`/`NaN` with `None` before serialization:
```python
def _sanitize_for_json(obj):
    if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    # ... recurse for dict/list
```

### XGBoost 2.0+ Callbacks API Change (Fixed 2026-02-05)

**Problem**: `TypeError: XGBModel.fit() got an unexpected keyword argument 'callbacks'`

**Root Cause**: XGBoost 2.0+ removed `callbacks` from sklearn `.fit()` method.

**Fix**: Use `model.set_params(callbacks=callbacks)` before `.fit()` in `site_scoring/model.py`. Also clear callbacks before pickling: `model.model.set_params(callbacks=None)` before `save_model()`.

### Experiment Folder Limit

The `site_scoring/outputs/experiments` folder is limited to 10 models (FIFO). Controlled by `MAX_EXPERIMENTS` constant in `training_service.py`. Cleanup runs automatically when creating new `TrainingJob`.

## Planned Features: Inference & Export Pipeline

### Business Context

This platform serves internal teams (sales, ops, analytics) for site selection, portfolio optimization, and sales enablement across 60K+ gas station advertising sites. Revenue is actual ad dollars. The lookalike model supports prospect expansion and tiering. There is currently no external-facing output — scored site lists need to get into sales reps' hands.

### Architecture Snapshot (Current State)

```
UI (index.html)                    API (app.py)                     ML Pipeline
─────────────────                  ──────────────                   ──────────────
Leaflet map + sidebar    ──POST──▶ /api/training/start              training_service.py
Feature selector                   /api/training/stream  ◀──SSE──   └── trainer.py
Hyperparameter controls            /api/predict/batch               └── model.py (NN, XGB, Clustering)
                                   /api/experiments                 └── data_loader.py (DataProcessor)
                                   /api/sites, /api/shap/*

Experiment Output (per job):
  site_scoring/outputs/experiments/job_xxx/
  ├── config.json              # Full config + training_features dict
  ├── preprocessor.pkl         # StandardScaler, LabelEncoders, vocab sizes
  ├── best_model.pt            # NN checkpoint (state_dict + Config)
  ├── model_wrapper.pkl        # XGBoost pickled wrapper (tree models only)
  ├── shap_cache.npz           # SHAP values
  ├── shap_importance.json     # Top-N feature importance
  ├── training_sites.csv       # Sites used for training
  ├── test_predictions.csv     # Test set predictions with actuals
  └── non_active_classification.csv  # Lookalike scores for non-active sites
```

### Key Existing Classes for Inference

- **`BatchPredictor`** (`site_scoring/predict.py`): Loads any experiment folder, reconstructs model (NN or XGB), processes features using saved scaler/encoders, returns `{gtvid: score}` dict. Supports both regression and classification.
- **`get_all_sites_for_prediction()`** (`site_scoring/data_transform.py`): Loads all 57K sites from `site_aggregated_precleaned.parquet`, applies same transforms as training (log, one-hot, binning) but without active-only filter. Module-level cache.
- **`/api/predict/batch`** (`app.py`): POST endpoint, accepts optional `experiment_dir`, finds latest experiment by default, creates `BatchPredictor`, scores all sites, returns JSON.

---

### Feature 1: Experiment Registry & Catalog

**Gap**: `get_all_jobs()` only returns in-memory active jobs. Once the Flask process restarts, all experiment history is lost. The experiments folder on disk has full artifacts but no browsable catalog.

**Goal**: List all experiments with their configs, metrics, and status — persisted across restarts.

#### API Design

```
GET /api/experiments/catalog
Response: {
  "experiments": [
    {
      "job_id": "job_1771882528_ae3755b1",
      "created_at": "2026-02-23T21:35:00Z",
      "model_type": "neural_network",
      "task_type": "lookalike",
      "target": "avg_monthly_revenue",
      "feature_count": { "numeric": 19, "categorical": 7, "boolean": 33 },
      "training_features": { ... },  // from config.json
      "test_metrics": { "auc": 0.87, "accuracy": 0.91, ... },  // from model_metadata or test_predictions
      "has_shap": true,
      "has_predictions": true,
      "artifacts": ["config.json", "best_model.pt", "preprocessor.pkl", ...]
    },
    ...
  ]
}
```

#### Implementation Plan

1. **Scan disk** — iterate `site_scoring/outputs/experiments/job_*/`, parse each `config.json`
2. **Extract metrics** — check for `model_metadata.json` first (if present), else parse `test_predictions.csv` header or compute from predictions
3. **Cache** — module-level `_experiment_catalog_cache` dict, invalidated when `experiments/` mtime changes
4. **Sort** — by created_at desc (newest first), parsed from job_id timestamp component

#### Files to Modify

| File | Change |
|------|--------|
| `app.py` | Add `GET /api/experiments/catalog` route |
| `src/services/training_service.py` | Add `scan_experiment_folders()` function that reads disk artifacts |
| `src/services/training_service.py` | Extend `_save_experiment_artifacts()` to always write `model_metadata.json` with test metrics, timestamps |
| `templates/index.html` | Add experiment browser panel (sidebar tab or modal) |

#### Validation & Testing

- **Unit test**: Create a mock experiment folder with `config.json` + `model_metadata.json`, verify `scan_experiment_folders()` returns correct structure
- **Integration test**: POST `/api/training/start` with small config (5 epochs), wait for completion, then GET `/api/experiments/catalog` and verify new experiment appears
- **Edge cases**: Experiment folder with missing `config.json` (skip gracefully), corrupted JSON (log warning, skip), empty experiments dir (return empty list)
- **Manual QA**: Restart Flask, verify catalog still shows all experiments from disk

---

### Feature 2: Inference UI (Score Sites from Trained Model)

**Gap**: `/api/predict/batch` exists but there's no button in the UI to trigger it, select an experiment, or view results.

**Goal**: After training (or selecting a past experiment), users can click "Score All Sites" and see results on the map and in a table.

#### UI Design

```
Sidebar: "Scoring" tab
├── Experiment Selector (dropdown, populated from /api/experiments/catalog)
├── "Score All Sites" button
├── "Score Filtered Only" button (uses current map filter state)
├── Progress indicator (sites scored / total)
└── Results summary:
    ├── Scored: 57,234 sites
    ├── Mean predicted revenue: $X,XXX
    ├── Distribution histogram (Chart.js)
    └── "Export CSV" button
```

#### API Design

```
POST /api/predict/batch     (already exists — extend response)
Request:  { "experiment_dir": "job_xxx", "filter": { "network": "Wayne", "state": "TX" } }
Response: {
  "predictions": { "gtvid_1": 0.87, "gtvid_2": 0.34, ... },
  "model_type": "neural_network",
  "task_type": "lookalike",
  "count": 57234,
  "summary": {
    "mean": 0.62,
    "median": 0.58,
    "p10": 0.12,
    "p90": 0.91,
    "histogram": { "bins": [...], "counts": [...] }
  }
}
```

#### Implementation Plan

1. **Extend `/api/predict/batch`** — add optional `filter` param, add `summary` stats to response
2. **Filter support** — when `filter` is provided, apply to the `get_all_sites_for_prediction()` DataFrame before scoring (network, state, status, etc.)
3. **Map integration** — after scoring, push predictions to the Leaflet map as a choropleth layer (color sites by score). Use existing `updateSiteMarkers()` pattern.
4. **Table view** — sortable DataTable below map showing `gtvid | name | city | state | score | rank`

#### Files to Modify

| File | Change |
|------|--------|
| `app.py` | Extend `/api/predict/batch` with filter param and summary stats |
| `site_scoring/predict.py` | Add filter-then-predict path to `BatchPredictor` |
| `templates/index.html` | Add "Scoring" sidebar tab with experiment dropdown, score button, results panel |
| `src/services/data_service.py` | Add helper to map gtvid predictions back to site lat/lng for map rendering |

#### Validation & Testing

- **Unit test**: `BatchPredictor.predict()` with a filtered DataFrame (e.g., 100 sites) returns correct count and score range
- **Integration test**: POST `/api/predict/batch` with `filter: { "network": "Wayne" }`, verify response count < total sites
- **UI smoke test**: Select experiment → click "Score All Sites" → verify progress indicator → verify map markers update with score colors → verify summary stats render
- **Edge case**: Experiment with missing `preprocessor.pkl` (graceful error), experiment trained on features not present in prediction data (log warning, fill zeros per existing DataProcessor pattern)

---

### Feature 3: Scored Output Export (CSV/Excel)

**Gap**: Predictions return as JSON to the browser. No download endpoint for scored site lists.

**Goal**: Export scored results as CSV or Excel with site metadata, usable by sales reps and external stakeholders.

#### API Design

```
POST /api/predict/export
Request:  {
  "experiment_dir": "job_xxx",
  "format": "csv" | "xlsx",
  "filter": { ... },           // optional
  "columns": ["gtvid", "name", "city", "state", "score", "rank", "percentile"]  // optional
}
Response: File download (Content-Disposition: attachment)

Filename pattern: site_scores_{task_type}_{model_type}_{YYYY-MM-DD}.csv
```

#### Export Column Spec

| Column | Source | Notes |
|--------|--------|-------|
| `gtvid` | prediction DataFrame | Primary site ID |
| `name` | site_aggregated_precleaned.parquet | Site name |
| `city`, `state` | site_aggregated_precleaned.parquet | Location |
| `network` | site_aggregated_precleaned.parquet | Network affiliation |
| `status` | site_aggregated_precleaned.parquet | Active/Inactive/etc |
| `actual_revenue` | `avg_monthly_revenue` column | Current actual (if available) |
| `predicted_score` | BatchPredictor output | Model prediction |
| `rank` | Computed | 1 = highest score |
| `percentile` | Computed | 0-100, what % of sites score below this |
| `model_type` | config.json | "neural_network" or "xgboost" |
| `experiment_id` | folder name | Traceability |
| `scored_at` | Timestamp | When inference ran |

#### Implementation Plan

1. **New endpoint** — `/api/predict/export` that reuses `BatchPredictor` internally
2. **Join metadata** — after scoring, join predictions back to site metadata from the prediction DataFrame (it already has gtvid, name, city, state, network, status columns)
3. **Compute rank/percentile** — `scipy.stats.percentileofscore` or simple `rank(ascending=False)` on score column
4. **CSV path** — use Polars `write_csv()` to a BytesIO, return via `send_file()`
5. **Excel path** — use `openpyxl` or Polars `write_excel()` for xlsx, add header formatting

#### Files to Modify

| File | Change |
|------|--------|
| `app.py` | Add `POST /api/predict/export` route |
| `site_scoring/predict.py` | Add `BatchPredictor.predict_with_metadata()` that returns a full DataFrame (predictions + site info + rank + percentile) |
| `templates/index.html` | Add "Export CSV" / "Export Excel" buttons to Scoring tab, wired to POST then download |

#### Validation & Testing

- **Unit test**: `predict_with_metadata()` returns DataFrame with all expected columns, no NaN in gtvid/score/rank
- **Integration test**: POST `/api/predict/export` with `format=csv`, verify response Content-Type is `text/csv`, verify file parses correctly with pandas
- **Data integrity**: Verify rank 1 has the highest score, percentile 99+ for rank 1, sum of sites in export == total sites in prediction set
- **Edge case**: Sites with identical scores get same rank (dense ranking), missing metadata columns degrade gracefully (fill "Unknown")

---

### Feature 4: Model Comparison

**Gap**: Multiple experiments can be trained but there's no way to compare them side-by-side.

**Goal**: Compare any two (or more) experiments across metrics, feature importance, and per-site score differences.

#### API Design

```
POST /api/experiments/compare
Request: {
  "experiment_ids": ["job_xxx", "job_yyy"],
  "comparison_type": "metrics" | "scores" | "features" | "all"
}
Response: {
  "metrics_comparison": {
    "job_xxx": { "auc": 0.87, "accuracy": 0.91, "f1": 0.84 },
    "job_yyy": { "rmse": 12450, "r2": 0.73, "mape": 18.2 }
  },
  "feature_comparison": {
    "job_xxx": { "top_features": [...], "feature_count": 59 },
    "job_yyy": { "top_features": [...], "feature_count": 42 }
  },
  "score_comparison": {  // only when comparison_type is "scores" or "all"
    "correlation": 0.82,
    "mean_diff": 0.03,
    "sites_with_large_diff": [
      { "gtvid": "xxx", "score_a": 0.92, "score_b": 0.41, "diff": 0.51 }
    ],
    "agreement_pct": 87.3  // % of sites ranked in same quintile by both models
  }
}
```

#### Implementation Plan

1. **Metrics comparison** — load `config.json` from each experiment, load `model_metadata.json` or recompute from `test_predictions.csv` if needed. Return side-by-side metrics.
2. **Feature comparison** — load `shap_importance.json` from each experiment, align feature names, compute overlap and divergence.
3. **Score comparison** — run `BatchPredictor` for each experiment on same site set, compute Pearson correlation, rank agreement (what % of sites fall in same quintile), identify sites with largest score disagreements.
4. **UI** — comparison modal or dedicated page: side-by-side metric cards, overlaid feature importance bar chart (Chart.js), scatter plot of score_A vs score_B.

#### Files to Modify

| File | Change |
|------|--------|
| `app.py` | Add `POST /api/experiments/compare` route |
| `src/services/training_service.py` | Add `compare_experiments()` function |
| `site_scoring/predict.py` | Add `BatchPredictor.predict_all()` class method that scores from multiple experiments and returns aligned DataFrame |
| `templates/index.html` | Add comparison UI (modal with Chart.js visualizations) |

#### Validation & Testing

- **Unit test**: `compare_experiments()` with two mock experiment folders returns correct structure, handles mismatched task types gracefully (regression vs classification → warn, still compare where possible)
- **Integration test**: Train two experiments (XGB + NN), call compare endpoint, verify correlation is computed, verify feature overlap count
- **Edge case**: Comparing experiments trained on different feature sets (features unique to one model show as "N/A" in the other), comparing regression vs classification (return metrics for each but skip score_comparison scatter)
- **Sanity check**: Same experiment compared to itself → correlation = 1.0, agreement = 100%, diff = 0

---

### Feature 5: Score-on-Demand for Filtered Subsets

**Gap**: Batch predict scores all 57K sites. No way to score just "Gilbarco sites in Texas" or "sites within 5 miles of an interstate."

**Goal**: Apply map/filter state to the scoring pipeline so users score exactly the sites they're looking at.

#### API Design

```
POST /api/predict/filtered
Request: {
  "experiment_dir": "job_xxx",         // optional, defaults to latest
  "filters": {
    "network": ["Wayne", "Dover"],     // multi-select
    "state": ["TX", "CA"],
    "status": ["Active"],
    "min_revenue": 500,
    "max_revenue": 50000,
    "retailer": ["7-Eleven"],
    "gtvids": ["GT001", "GT002", ...]  // explicit site list
  }
}
Response: {
  "predictions": { ... },
  "count": 1234,
  "filter_applied": { ... },
  "summary": { ... }
}
```

#### Implementation Plan

1. **Filter engine** — create `FilterEngine` class in `site_scoring/predict.py` that takes a Polars DataFrame + filter dict and returns filtered DataFrame. Reuse filter logic from `data_service.py` where applicable.
2. **Wire to BatchPredictor** — `BatchPredictor.predict_filtered(df, filters)` applies `FilterEngine`, then runs prediction on result.
3. **Sync with map state** — the Leaflet map already has filter state (network dropdown, status toggle, etc). When user clicks "Score Filtered," JS collects current filter state and sends as POST body.
4. **Complement existing batch endpoint** — `/api/predict/filtered` is separate from `/api/predict/batch` to keep the batch endpoint simple. Alternatively, extend batch with optional filters (see Feature 2 design).

#### Files to Modify

| File | Change |
|------|--------|
| `app.py` | Add `POST /api/predict/filtered` route (or extend `/api/predict/batch`) |
| `site_scoring/predict.py` | Add `FilterEngine` class, add `BatchPredictor.predict_filtered()` method |
| `templates/index.html` | Wire "Score Filtered" button to collect current map filter state and POST to endpoint |

#### Validation & Testing

- **Unit test**: `FilterEngine` with `{"network": ["Wayne"]}` reduces DataFrame to only Wayne rows, returns correct count
- **Unit test**: `FilterEngine` with `{"gtvids": ["GT001"]}` returns exactly 1 row
- **Integration test**: POST `/api/predict/filtered` with `{"filters": {"state": ["TX"]}}`, verify count matches number of TX sites in parquet
- **Edge case**: Empty filter (scores all sites, same as batch), filter that matches 0 sites (return empty predictions + count: 0), filter on column not in data (ignore, log warning)
- **Performance test**: Scoring 1,000 filtered sites should return in <2 seconds (vs ~10s for full 57K batch)

---

### Implementation Order (Recommended)

```
Phase 1 — Foundation
  Feature 1: Experiment Registry     (required by all other features)
  ↓
Phase 2 — Core Value
  Feature 2: Inference UI            (depends on Feature 1 for experiment selector)
  Feature 3: Export                   (depends on Feature 2 for scoring results)
  ↓
Phase 3 — Power Features
  Feature 5: Filtered Scoring        (extends Feature 2 with filter param)
  Feature 4: Model Comparison        (depends on Feature 1 for catalog, Feature 2 for scoring)
```

### Shared Conventions for All Features

- All new API endpoints must use `_sanitize_for_json()` before returning (see Known Gotchas)
- All sklearn inverse transforms must use `.reshape(-1, 1)` (see Known Gotchas)
- New endpoints follow existing pattern: route in `app.py`, logic in `src/services/`, ML in `site_scoring/`
- All predictions include `scored_at` ISO timestamp for cache/staleness tracking
- Export filenames include experiment_id for traceability
- Tests go in `tests/` matching source structure: `tests/test_predict.py`, `tests/test_training_service.py`

---

## Don't

- ❌ Commit `.env`, credentials, or large binaries
- ❌ Use `git push --force` without asking
- ❌ Add features beyond what was requested
- ❌ Create documentation files unless explicitly asked
- ❌ Report validation metrics as test metrics
- ❌ Pass 1D arrays to sklearn scalers (always reshape to 2D)
- ❌ When asked to remove something, do only the removal — no commentary, no refactoring, no related changes
