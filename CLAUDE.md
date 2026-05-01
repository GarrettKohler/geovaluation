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
| `site_transactions_daily.csv` | 1M | 4 | Daily transaction counts per site |
| `site_status_daily.csv` | 1M | 4 | Daily status snapshots per site |
| `site_aggregated_precleaned.parquet` | 57,675 | 104 | ETL output — one row per site, aggregated from primary source |
| `site_training_data.parquet` | 26,096 | 114 | Active sites with 12+ months history, used for model training |

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

## ML Configuration

| Task | Target | Notes |
|------|--------|-------|
| Regression | `avg_daily_revenue` | XGBoost or Neural Network |
| Lookalike | Binarized by percentile | p90-p100 default (configurable) |
| Clustering | Deep Embedded Clustering | Segment top performers |

Features: 16 numeric (15 after target-leakage filtering removes `log_total_revenue` when training on `avg_daily_revenue`) + 7 categorical + 33 boolean (defined in `config.py`)

## Conventions

- **Indentation**: 4 spaces (Python), 4 spaces (JS)
- **Imports**: stdlib → third-party → local (isort compatible)
- **Commits**: Co-authored with Claude
- **API responses**: Always clean NaN/Inf for JSON serialization

## Known Gotchas

- **Never report validation metrics as test metrics** — validation loss influences early stopping, so val metrics are optimistically biased. Always evaluate on held-out test set after loading best model.
- **sklearn scalers need 2D arrays** — always `scaler.inverse_transform(arr.reshape(-1, 1))` not `scaler.inverse_transform(arr)`
- **Always `_sanitize_for_json()` before returning API responses** — `float('inf')` and `NaN` break `JSON.parse()`
- **XGBoost 2.0+**: use `model.set_params(callbacks=...)` not `fit(callbacks=...)`. Clear callbacks before pickling.
- **Experiment folder limit**: max 10 experiments (FIFO), controlled by `MAX_EXPERIMENTS` in `training_service.py`

## Planned Features

See `docs/planned-features.md` for detailed specs on Inference & Export Pipeline (Features 1-5).

## Don't

- ❌ Commit `.env`, credentials, or large binaries
- ❌ Use `git push --force` without asking
- ❌ Add features beyond what was requested
- ❌ Create documentation files unless explicitly asked
- ❌ Report validation metrics as test metrics
- ❌ Pass 1D arrays to sklearn scalers (always reshape to 2D)
- ❌ When asked to remove something, do only the removal — no commentary, no refactoring, no related changes
