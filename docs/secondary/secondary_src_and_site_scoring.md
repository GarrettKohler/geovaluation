# src/ and site_scoring/ — Architecture & Data Flow

This document describes how the `src/` service layer and `site_scoring/` ML module work together to power the Geospatial Site Analysis application.

## Overview

The application follows a three-tier architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask Controller                          │
│                           app.py                                 │
│         (API routes, request handling, SSE streaming)            │
└──────────────┬──────────────────┬──────────────────┬────────────┘
               │                  │                  │
               ▼                  ▼                  ▼
┌──────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
│  data_service.py │ │interstate_distance │ │ training_service.py │
│                  │ │       .py          │ │                     │
│ Site data loading│ │ Highway distance   │ │ Training lifecycle  │
│ Revenue metrics  │ │ calculations       │ │ Apple Silicon opts  │
│ Filtering        │ │ Spatial indexing   │ │ Progress streaming  │
└──────────────────┘ └────────────────────┘ └─────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │   site_scoring/     │
                                            │                     │
                                            │ Config, Model,      │
                                            │ DataLoader,         │
                                            │ DataProcessor       │
                                            └─────────────────────┘
```

## src/services/ — The Service Layer

### data_service.py

**Purpose**: Loads and caches all site data for the web visualization.

**Key functions used by app.py**:
| Function | API Route | Description |
|----------|-----------|-------------|
| `load_sites()` | `/api/sites` | Returns all 57K+ sites with coordinates and revenue |
| `load_revenue_metrics()` | `/api/sites` | Revenue scoring and color mapping |
| `load_site_details()` | `/api/bulk-site-details` | Full detail records for selected sites |
| `get_site_details_for_display()` | `/api/site-details/<id>` | Formatted details for side panel |
| `get_filter_options()` | `/api/filter-options` | Unique values for categorical dropdowns |
| `get_filtered_site_ids()` | `/api/filtered-sites` | Sites matching filter criteria |
| `preload_all_data()` | Startup | Caches all data in memory on boot |
| `_clean_nan_values()` | Multiple | Sanitizes NaN/Inf for JSON responses |

**Data source**: `data/input/` CSV files (Sites, Revenue, Scores)

---

### interstate_distance.py

**Purpose**: Calculates the distance from any geographic coordinate to the nearest US Interstate highway using TIGER/Line shapefiles and spatial indexing.

**Key functions used by app.py**:
| Function | API Route | Description |
|----------|-----------|-------------|
| `distance_to_nearest_interstate()` | `/api/site/<id>`, `/api/highway-connections` | Single-point distance query |
| `preload_highway_data()` | Startup | Loads shapefile into memory |

**Data source**: `data/shapefiles/tl_2024_us_primaryroads/`

**Connection to site_scoring**: The calculated distances (`min_distance_to_interstate_mi`, `nearest_interstate`) are stored in processed training data and used as **input features** for the ML model (see `site_scoring/config.py` lines 58-59, 69).

---

### training_service.py

**Purpose**: The bridge between the web UI and the `site_scoring/` ML module. Manages the full training lifecycle with real-time progress streaming.

**Key functions used by app.py**:
| Function | API Route | Description |
|----------|-----------|-------------|
| `get_system_info()` | `/api/training/system-info` | Detects Apple Silicon chip, GPU availability |
| `start_training()` | `/api/training/start` | Launches background training with chip optimizations |
| `stop_training()` | `/api/training/stop` | Gracefully stops training |
| `get_training_status()` | `/api/training/status` | Current epoch, metrics, completion state |
| `stream_training_progress()` | `/api/training/stream` | SSE generator for real-time UI updates |

**What it imports from site_scoring/**:
```python
from site_scoring.config import Config
from site_scoring.model import SiteScoringModel
from site_scoring.data_loader import DataProcessor, create_data_loaders
```

---

### Additional services (available but not used by app.py)

| File | Purpose | Used By |
|------|---------|---------|
| `nearest_site.py` | KDTree-based nearest neighbor site distance | `scripts/demo.py` |
| `epa_walkability.py` | EPA National Walkability Index lookups | `scripts/site_walkability.py` |

These are re-exported via `src/__init__.py` for library usage but not called by the web application.

---

## site_scoring/ — The ML Module

A standalone PyTorch module for site revenue prediction, optimized for Apple Silicon MPS acceleration.

### Files

| File | Purpose |
|------|---------|
| `config.py` | Model configuration: features, hyperparameters, paths, device settings |
| `model.py` | `SiteScoringModel` — Embedding + Residual MLP neural network |
| `data_loader.py` | `DataProcessor` and `SiteDataset` — Polars-based data loading, encoding, tensor creation |
| `data_transform.py` | Data transformation utilities |
| `train.py` | Standalone training loop (CLI usage) |
| `predict.py` | Inference module for generating predictions |
| `run.py` | CLI entry point for standalone training/prediction |

### Model Architecture

```
Input Features (94 total)
├── Numeric (30+): lat, lon, revenue metrics, distances, demographics
├── Categorical (9): network, retailer, hardware_type, nearest_interstate, etc.
└── Boolean (39): capability flags, restriction flags, schedule/sellable
         │
         ▼
┌─────────────────────────────┐
│   Categorical Embeddings    │  (dim=16 per feature)
│   + Numeric Passthrough     │
│   + Boolean Passthrough     │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│   Residual MLP              │
│   512 → 256 → 128 → 64     │
│   (BatchNorm + Dropout 0.2) │
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│   Output: Revenue Prediction│
│   (avg_monthly_revenue)     │
└─────────────────────────────┘
```

### Key Configuration (config.py)

- **Training data**: `data/processed/site_training_data.parquet` (~26K active sites, one row per site)
- **Target variable**: `avg_monthly_revenue` (recommended) or `total_revenue`
- **Device**: Auto-detects Apple Silicon MPS, falls back to CPU
- **Leakage prevention**: Automatically removes target-derived columns from features

---

## How They Connect: The Training Flow

When a user clicks "Train Model" in the web UI:

```
1. Browser POST /api/training/start
   └── { epochs: 50, batch_size: 4096, device: "mps", apple_chip: "auto" }

2. app.py → start_training(config_dict)
   └── training_service.py:
       a. detect_apple_chip() → identifies M1/M2/M3/M4 variant
       b. get_optimized_training_params() → adjusts batch size, workers for chip
       c. Creates TrainingJob with optimized TrainingConfig
       d. Spawns background thread

3. TrainingJob._run_training() (background thread):
       a. Maps TrainingConfig → site_scoring.Config
       b. create_data_loaders(config) → loads parquet, encodes, splits train/val/test
       c. SiteScoringModel(...) → creates neural network
       d. Training loop:
          - Forward pass → Huber loss
          - AdamW optimizer + gradient clipping
          - ReduceLROnPlateau scheduler
          - Early stopping (patience=10)
          - Progress updates → Queue

4. Browser connects GET /api/training/stream (SSE)
   └── stream_training_progress() yields real-time updates:
       { epoch, train_loss, val_loss, val_mae, val_r2, elapsed_time, status }

5. On completion:
   └── Saves best_model.pt + preprocessor.pkl to site_scoring/outputs/
```

---

## Data Flow: From Raw CSV to Model Features

The geospatial calculations from `src/services/` become input features for the ML model:

```
data/input/Sites - Base Data Set.csv
     │
     ├── interstate_distance.py ──→ min_distance_to_interstate_mi
     │                               nearest_interstate (categorical)
     │
     ├── nearest_site.py ──────────→ nearest_site_distance_mi
     │
     └── (external processing) ────→ demographics, revenue, capabilities
                                      │
                                      ▼
                          data/processed/site_training_data.parquet
                                      │
                                      ▼
                          site_scoring/data_loader.py
                                      │
                                      ▼
                          PyTorch Tensors → Model Training
```

The features `nearest_site_distance_mi`, `min_distance_to_interstate_mi`, `log_nearest_site_distance_mi`, `log_min_distance_to_interstate_mi`, and `nearest_interstate` in the model config all originate from calculations performed by `src/services/`.

---

## Apple Silicon Optimization

The `training_service.py` contains a chip specification table that optimizes training parameters based on the detected Apple Silicon variant:

| Chip | GPU Cores | Max Batch | Workers | Tier |
|------|-----------|-----------|---------|------|
| M1 | 8 | 4,096 | 2 | 1 |
| M1 Pro | 16 | 8,192 | 4 | 2 |
| M1 Max | 32 | 16,384 | 6 | 3 |
| M2 | 10 | 4,096 | 2 | 1 |
| M3 Pro | 18 | 8,192 | 4 | 2 |
| M4 | 10 | 8,192 | 4 | 2 |
| M4 Pro | 20 | 16,384 | 6 | 3 |
| M4 Max | 40 | 32,768 | 8 | 4 |

The chip is auto-detected via `sysctl` system calls and parameters are adjusted accordingly.

---

## File Dependency Map

```
app.py
 ├── src/services/data_service.py        (8 imports)
 ├── src/services/interstate_distance.py  (2 imports)
 └── src/services/training_service.py     (5 imports)
      ├── site_scoring/config.py          (Config)
      ├── site_scoring/model.py           (SiteScoringModel)
      └── site_scoring/data_loader.py     (DataProcessor, create_data_loaders)
           └── site_scoring/config.py     (Config)
```

No circular dependencies exist. The dependency flow is strictly:
`app.py` → `src/services/` → `site_scoring/`
