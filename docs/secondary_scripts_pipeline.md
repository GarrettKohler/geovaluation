# scripts/ — Data Preparation Pipeline

This document describes how the `scripts/` directory connects `src/services/` (geospatial calculations) to `site_scoring/` (ML training), serving as the **feature generation** step in the data pipeline.

## Role in the System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                   │
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐  │
│  │  Raw Data    │     │   scripts/   │     │   Processed Data        │  │
│  │  data/input/ │ ──► │  (Feature    │ ──► │   data/processed/       │  │
│  │  (CSVs)      │     │  Generation) │     │   (parquet)             │  │
│  └──────────────┘     └──────┬───────┘     └────────────┬────────────┘  │
│                              │                           │               │
│                     Uses src/services/          Consumed by site_scoring/ │
└─────────────────────────────────────────────────────────────────────────┘
```

The scripts are **offline batch jobs** — they run once (or periodically) to compute derived features from raw site data. Those features then become columns in the training dataset used by the ML model.

---

## Scripts Overview

### scripts/demo.py — Interstate Distance Calculator

**Purpose**: Calculates the distance from every site to its nearest US Interstate highway.

**Usage**:
```bash
python -m scripts.demo
```

**What it does**:
1. Loads `data/input/Sites - Base Data Set.csv` (57K+ sites)
2. Calls `preload_highway_data()` — downloads TIGER/Line shapefiles on first run (~15MB)
3. Calls `batch_distance_to_interstate()` — projects to EPSG:5070 and finds nearest road
4. Saves results to `data/output/site_interstate_distances.csv`

**Output columns**:
| Column | Type | Description |
|--------|------|-------------|
| `GTVID` | string | Site identifier |
| `Latitude` | float | Site latitude |
| `Longitude` | float | Site longitude |
| `nearest_interstate` | string | Name of closest Interstate (e.g., "I-95") |
| `distance_to_interstate_mi` | float | Distance in miles |

**Connection to site_scoring**: These values become the following model features in `site_scoring/config.py`:
- `min_distance_to_interstate_mi` (numeric)
- `log_min_distance_to_interstate_mi` (numeric, log-transformed)
- `nearest_interstate` (categorical, embedded)

**Imports from**: `src.services.interstate_distance`
- `batch_distance_to_interstate()` — processes all sites in one call using vectorized operations
- `preload_highway_data()` — loads shapefile into memory

---

### scripts/site_walkability.py — EPA Walkability Scorer

**Purpose**: Calculates EPA National Walkability Index scores for sites using Census block group geometry.

**Usage**:
```bash
python -m scripts.site_walkability
```

**What it does**:
1. Loads `data/input/Sites - Base Data Set.csv`
2. Calls `batch_walkability_scores()` — downloads EPA Smart Location Database (~220MB) and Census TIGER/Line block group shapefiles
3. Performs spatial join of site points against block group polygons
4. Returns walkability metrics per site

**Output columns** (returned in-memory, not saved to file):
| Column | Type | Description |
|--------|------|-------------|
| `walkability_index` | float | National Walkability Index (1-20 scale) |
| `intersection_density` | float | Road intersection density |
| Additional EPA metrics | float | Various Smart Location Database fields |

**Connection to site_scoring**: Not currently included as model features, but could be added to the numeric features list in `config.py` for future model iterations.

**Imports from**: `src.services.epa_walkability`
- `batch_walkability_scores()` — batch spatial join against EPA data

---

## How Scripts Feed the ML Model

The scripts generate the geospatial features that the model uses for prediction. Here's the complete data flow:

```
                    ┌─────────────────────────────┐
                    │   scripts/demo.py           │
                    │   (Interstate distances)    │
                    └────────────┬────────────────┘
                                 │
                                 ▼
┌──────────────┐    ┌─────────────────────────────┐    ┌──────────────────┐
│ Sites CSV    │───►│  data/output/               │───►│ data/processed/  │
│ (raw coords) │    │  site_interstate_distances  │    │ site_training_   │
└──────────────┘    │  .csv                       │    │ data.parquet     │
                    └─────────────────────────────┘    └────────┬─────────┘
                                                                │
                    ┌─────────────────────────────┐             │
                    │  src/services/nearest_site  │             │
                    │  (Nearest neighbor dists)   │─────────────┘
                    └─────────────────────────────┘             │
                                                                ▼
                                                    ┌──────────────────────┐
                                                    │ site_scoring/        │
                                                    │ data_loader.py       │
                                                    │                      │
                                                    │ Reads parquet,       │
                                                    │ encodes features,    │
                                                    │ creates tensors      │
                                                    └──────────────────────┘
```

### Features Traced Back to Scripts

| Model Feature | Generated By | Script/Service |
|---------------|-------------|----------------|
| `min_distance_to_interstate_mi` | Interstate distance calc | `scripts/demo.py` |
| `log_min_distance_to_interstate_mi` | Log transform of above | Derived in processing |
| `nearest_interstate` | Nearest highway name | `scripts/demo.py` |
| `nearest_site_distance_mi` | KDTree nearest neighbor | `src/services/nearest_site.py` |
| `log_nearest_site_distance_mi` | Log transform of above | Derived in processing |

---

## Scripts vs. Web App: Two Ways to Trigger the Same Logic

The geospatial calculations in `src/services/` can be invoked two ways:

| Method | Entry Point | Scope | Output |
|--------|------------|-------|--------|
| **Batch (scripts)** | `python -m scripts.demo` | All 57K+ sites at once | CSV file |
| **On-demand (web)** | `GET /api/site/<id>` | Single site per request | JSON response |

The scripts run the calculations in bulk for pre-computing training features, while the web app runs them individually for real-time user queries.

```
┌──────────────────────────────────────────────────────────────┐
│            src/services/interstate_distance.py               │
│                                                              │
│  ┌────────────────────────────┐  ┌────────────────────────┐  │
│  │  batch_distance_to_        │  │  distance_to_nearest_  │  │
│  │  interstate()              │  │  interstate()          │  │
│  │                            │  │                        │  │
│  │  Called by: scripts/demo   │  │  Called by: app.py     │  │
│  │  Input: DataFrame (57K)   │  │  Input: single lat/lon │  │
│  │  Output: DataFrame        │  │  Output: dict          │  │
│  └────────────────────────────┘  └────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Scripts vs. site_scoring CLI: Two Ways to Train

There are also two ways to trigger model training:

| Method | Command | Apple Silicon Opt | Progress UI | Used By |
|--------|---------|-------------------|-------------|---------|
| **CLI** | `python -m site_scoring.run` | Manual (`--device mps`) | Terminal print | Developers |
| **Web UI** | POST `/api/training/start` | Auto-detected per chip | SSE streaming | End users |

Both ultimately use the same `site_scoring/` module (Config, Model, DataProcessor), but the web path adds:
- Automatic Apple Silicon chip detection and optimization
- Background thread execution
- Real-time progress via Server-Sent Events
- Graceful stop capability

---

## Known Issues

### Broken Import Paths

Both scripts have incorrect import paths that will cause `ModuleNotFoundError` at runtime:

**scripts/demo.py** (line 16):
```python
# Current (BROKEN):
from src.interstate_distance import batch_distance_to_interstate, preload_highway_data

# Correct:
from src.services.interstate_distance import batch_distance_to_interstate, preload_highway_data
```

**scripts/site_walkability.py** (line 15):
```python
# Current (BROKEN):
from src.epa_walkability import batch_walkability_scores

# Correct:
from src.services.epa_walkability import batch_walkability_scores
```

These paths reference the old module structure (before `services/` was introduced). The functions are re-exported via `src/__init__.py`, so an alternative fix is:
```python
from src import batch_distance_to_interstate, preload_highway_data
from src import batch_walkability_scores
```

---

## Execution Order for Full Pipeline

To regenerate training data from scratch:

```bash
# Step 1: Calculate interstate distances for all sites
python -m scripts.demo
# Output: data/output/site_interstate_distances.csv

# Step 2: (Optional) Calculate walkability scores
python -m scripts.site_walkability
# Output: in-memory only (would need saving)

# Step 3: (External) Aggregate and merge all features into training data
# This step combines site data + distances + other features into:
# data/processed/site_training_data.parquet

# Step 4a: Train via CLI
python -m site_scoring.run --epochs 50 --batch-size 4096

# Step 4b: Or train via web UI
python app.py  # Then use the Training panel in the browser
```

---

## File Summary

```
scripts/
├── __init__.py              # Package marker
├── demo.py                  # Interstate distance batch calculator
└── site_walkability.py      # EPA walkability score calculator
```

| File | Imports From | Writes To | Feeds Into |
|------|-------------|-----------|------------|
| `demo.py` | `src/services/interstate_distance` | `data/output/site_interstate_distances.csv` | `site_scoring` features |
| `site_walkability.py` | `src/services/epa_walkability` | (in-memory only) | Not yet used |
