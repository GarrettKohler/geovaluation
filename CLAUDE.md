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

## Don't

- ❌ Commit `.env`, credentials, or large binaries
- ❌ Use `git push --force` without asking
- ❌ Add features beyond what was requested
- ❌ Create documentation files unless explicitly asked
- ❌ Report validation metrics as test metrics
- ❌ Pass 1D arrays to sklearn scalers (always reshape to 2D)
- ❌ When asked to remove something, do only the removal — no commentary, no refactoring, no related changes
