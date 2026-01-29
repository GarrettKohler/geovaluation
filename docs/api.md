# API Reference

Complete API documentation for the GSTV Geospatial Site Visualization application.

## Table of Contents

- [Flask Web API](#flask-web-api)
  - [Page Routes](#page-routes)
  - [Sites Data Endpoints](#sites-data-endpoints)
  - [Highway Connection Endpoints](#highway-connection-endpoints)
  - [Filtering Endpoints](#filtering-endpoints)
  - [Model Training Endpoints](#model-training-endpoints)
  - [SHAP Feature Importance Endpoints](#shap-feature-importance-endpoints)
- [Python Library Modules](#python-library-modules)
  - [interstate_distance Module](#interstate_distance-module)
  - [nearest_site Module](#nearest_site-module)
  - [Feature Selection Module](#feature-selection-module)
- [Constants](#constants)

---

## Flask Web API

The application runs a Flask server (default port 8080) providing REST API endpoints and Server-Sent Events for real-time updates.

### Page Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main map visualization page |
| `/training-details` | GET | Training details page with site records |
| `/glossary` | GET | ML/statistics glossary page |
| `/shap-values` | GET | SHAP feature importance visualization page |

---

### Sites Data Endpoints

#### GET /api/sites

Get all sites with coordinates, revenue metrics, and status.

**Response:**
```json
[
  {
    "GTVID": "SFR001",
    "Latitude": 37.7749,
    "Longitude": -122.4194,
    "revenueScore": 0.85,
    "avgMonthlyRevenue": 2500.00,
    "totalRevenue": 75000.00,
    "activeMonths": 30,
    "status": "Active"
  }
]
```

---

#### GET /api/site/{site_id}

Get basic site information including highway distance.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `site_id` | string | The site GTVID |

**Response:**
```json
{
  "site_id": "SFR001",
  "latitude": 37.7749,
  "longitude": -122.4194,
  "nearest_highway": "I- 80",
  "distance_miles": 2.45,
  "highway_point": {
    "lat": 37.8012,
    "lon": -122.4089
  }
}
```

**Error Response (404):**
```json
{"error": "Site not found"}
```

---

#### GET /api/site-details/{site_id}

Get comprehensive site details for side panel display.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `site_id` | string | The site GTVID |

**Response:**
```json
{
  "site_id": "SFR001",
  "categories": {
    "Location": {"state": "CA", "county": "San Francisco", ...},
    "Site Info": {"network": "Gilbarco", "hardware_type": "DCR", ...},
    "Brands": {"brand_fuel": "Shell", "brand_c_store": "7-Eleven", ...},
    "Revenue": {"avg_monthly_revenue": 2500.00, "total_revenue": 75000.00, ...}
  }
}
```

---

#### POST /api/bulk-site-details

Get detailed information for multiple sites at once.

**Request Body:**
```json
{
  "site_ids": ["SFR001", "GHR001", "SFR002"]
}
```

**Response:**
```json
{
  "SFR001": {"gtvid": "SFR001", "state": "CA", ...},
  "GHR001": {"gtvid": "GHR001", "state": "TX", ...},
  "SFR002": {"gtvid": "SFR002", "state": "CA", ...}
}
```

---

### Highway Connection Endpoints

#### POST /api/highway-connections

Calculate highway connections for selected sites.

**Request Body:**
```json
{
  "site_ids": ["SFR001", "GHR001"]
}
```

**Response:**
```json
{
  "connections": [
    {
      "site_id": "SFR001",
      "site_lat": 37.7749,
      "site_lon": -122.4194,
      "highway_lat": 37.8012,
      "highway_lon": -122.4089,
      "highway_name": "I- 80",
      "highway_segment": [[37.79, -122.41], [37.80, -122.40], ...],
      "distance_miles": 2.45
    }
  ]
}
```

---

### Filtering Endpoints

#### GET /api/filter-options

Get unique values for all categorical fields that can be used as filters.

**Response:**
```json
{
  "State": ["CA", "TX", "FL", "NY", ...],
  "Network": ["Gilbarco", "Verifone", "Wayne", ...],
  "Retailer": ["Shell", "Chevron", "ExxonMobil", ...],
  "DMA": ["Houston", "Los Angeles", "New York", ...],
  ...
}
```

---

#### POST /api/filtered-sites

Get sites matching the specified filters.

**Request Body:**
```json
{
  "filters": {
    "State": "TX",
    "Network": "Gilbarco",
    "Retailer": "Shell"
  }
}
```

**Response:**
```json
{
  "site_ids": ["GHR001", "GHR002", "GHR003"],
  "count": 3
}
```

---

### Model Training Endpoints

#### GET /api/training/system-info

Get system information for training (GPU availability, Apple Silicon detection).

**Response:**
```json
{
  "pytorch_version": "2.1.0",
  "cuda_available": false,
  "mps_available": true,
  "recommended_device": "mps",
  "mps_device": "Apple Silicon GPU (MPS)",
  "detected_chip": "m4_pro",
  "chip_name": "Apple M4 Pro",
  "gpu_cores": 20,
  "total_memory": "48 GB"
}
```

---

#### POST /api/training/start

Start a new model training job.

**Request Body:**
```json
{
  "model_type": "neural_network",
  "task_type": "regression",
  "target": "avg_monthly_revenue",
  "epochs": 50,
  "batch_size": 4096,
  "learning_rate": 0.0001,
  "dropout": 0.2,
  "hidden_layers": [512, 256, 128, 64],
  "device": "mps",
  "apple_chip": "auto",
  "feature_selection_method": "stg_light",
  "stg_lambda": 0.1,
  "stg_sigma": 0.5,
  "run_shap_validation": false,
  "track_gradients": false
}
```

**Feature Selection Methods:**
| Method | Description |
|--------|-------------|
| `none` | No feature selection |
| `stg_light` | Light Stochastic Gates regularization |
| `stg_aggressive` | Aggressive STG with SHAP validation |
| `lassonet_standard` | Standard LassoNet configuration |
| `lassonet_path` | LassoNet with full lambda path |
| `shap_only` | Post-training SHAP-Select only |
| `tabnet` | TabNet with sparsemax attention |
| `hybrid_stg_shap` | STG during training + SHAP post-training |

**Response (Success):**
```json
{
  "success": true,
  "job_id": "job_1705849200"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "A training job is already running"
}
```

---

#### POST /api/training/stop

Stop the current training job.

**Response:**
```json
{
  "success": true,
  "message": "Training stop requested"
}
```

---

#### GET /api/training/status

Get current training job status.

**Response (Running):**
```json
{
  "job_id": "job_1705849200",
  "is_running": true,
  "epoch": 25,
  "total_epochs": 50,
  "train_loss": 0.0234,
  "val_loss": 0.0256,
  "val_mae": 1234.56,
  "val_smape": 15.2,
  "val_rmse": 1567.89,
  "val_r2": 0.8523,
  "learning_rate": 0.0001,
  "elapsed_time": 45.2,
  "status": "running",
  "message": "Epoch 25/50 | 180 features active",
  "best_val_loss": 0.0245
}
```

**Response (No Job):**
```json
{
  "status": "no_job",
  "message": "No training job exists"
}
```

---

#### GET /api/training/stream

Server-Sent Events stream for real-time training progress.

**Response (SSE Stream):**
```
data: {"epoch": 25, "total_epochs": 50, "train_loss": 0.0234, "val_loss": 0.0256, "val_mae": 1234.56, "val_smape": 15.2, "val_rmse": 1567.89, "val_r2": 0.8523, "learning_rate": 0.0001, "elapsed_time": 45.2, "status": "running", "message": "Epoch 25/50", "best_val_loss": 0.0245, "n_active_features": 180, "fs_reg_loss": 0.0012}

data: {"epoch": 26, "total_epochs": 50, ...}

data: {"status": "completed", "final_metrics": {...}}

data: {"status": "stream_end"}
```

---

### SHAP Feature Importance Endpoints

#### GET /api/shap/available

Check if SHAP data is available from the last training run.

**Response (Available):**
```json
{
  "available": true,
  "n_samples": 200,
  "n_features": 84
}
```

**Response (Not Available):**
```json
{
  "available": false
}
```

---

#### GET /api/shap/summary

Get SHAP feature importance summary data.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | int | 30 | Number of top features to return |

**Response:**
```json
{
  "features": [
    {
      "feature": "avg_monthly_revenue",
      "importance": 0.456,
      "mean_shap": 0.234,
      "std_shap": 0.089
    },
    {
      "feature": "screen_count",
      "importance": 0.234,
      "mean_shap": 0.123,
      "std_shap": 0.045
    }
  ],
  "base_value": 2345.67,
  "n_samples": 200,
  "n_features": 84,
  "total_features_available": 84
}
```

**Error Response (404):**
```json
{
  "error": "No SHAP data available. Train a model first."
}
```

---

#### GET /api/shap/plots

Get SHAP visualization plots as base64-encoded PNG images.

**Response:**
```json
{
  "bar_plot": "iVBORw0KGgoAAAANSUhEUgAA...",
  "summary_plot": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Error Response (404):**
```json
{
  "error": "No SHAP plots available. Train a model first."
}
```

---

## Python Library Modules

---

## interstate_distance Module

Calculate distance from coordinates to the nearest US Interstate highway.

This module downloads and caches Census Bureau TIGER/Line Primary Roads data, filters to Interstate highways only, and uses EPSG:5070 projection for accurate distance calculations.

### Module Constants

```python
TIGER_PRIMARY_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
METERS_PER_MILE = 1609.344
```

---

### distance_to_nearest_interstate

Calculate the distance from a single point to the nearest US Interstate highway.

```python
def distance_to_nearest_interstate(
    latitude: float,
    longitude: float
) -> dict
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `latitude` | `float` | Latitude in decimal degrees (WGS84). Valid range: -90 to 90. |
| `longitude` | `float` | Longitude in decimal degrees (WGS84). Valid range: -180 to 180. |

#### Returns

| Key | Type | Description |
|-----|------|-------------|
| `distance_miles` | `float` | Distance to nearest Interstate in miles |
| `distance_meters` | `float` | Distance to nearest Interstate in meters |
| `nearest_highway` | `str` | Full name of the nearest Interstate (e.g., "I- 95", "I- 80") |

#### Example

```python
from interstate_distance import distance_to_nearest_interstate

# San Francisco coordinates
result = distance_to_nearest_interstate(37.7749, -122.4194)

print(f"Distance: {result['distance_miles']:.2f} miles")
print(f"Distance: {result['distance_meters']:.0f} meters")
print(f"Nearest: {result['nearest_highway']}")
```

Output:
```
Distance: 2.45 miles
Distance: 3943 meters
Nearest: I- 80
```

#### Notes

- The first call will download highway data (~15MB) from the Census Bureau
- Subsequent calls use cached data for faster response
- Uses STRtree spatial indexing for efficient nearest-neighbor queries
- Coordinates are internally projected to EPSG:5070 for accurate distance calculation

---

### batch_distance_to_interstate

Calculate distance to nearest Interstate for multiple points efficiently.

```python
def batch_distance_to_interstate(
    points_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> gpd.GeoDataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points_df` | `pd.DataFrame` | *required* | DataFrame containing latitude and longitude columns |
| `lat_col` | `str` | `'latitude'` | Name of the column containing latitude values |
| `lon_col` | `str` | `'longitude'` | Name of the column containing longitude values |

#### Returns

Returns a `gpd.GeoDataFrame` with all original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `distance_to_interstate_m` | `float` | Distance in meters |
| `distance_to_interstate_mi` | `float` | Distance in miles |
| `nearest_interstate` | `str` | Name of nearest Interstate highway |
| `geometry` | `Point` | Point geometry in WGS84 (EPSG:4326) |

#### Example

```python
import pandas as pd
from interstate_distance import batch_distance_to_interstate

# Create sample data
sites = pd.DataFrame({
    'site_id': ['A', 'B', 'C'],
    'Latitude': [37.7749, 39.7392, 40.7128],
    'Longitude': [-122.4194, -104.9903, -74.0060]
})

# Calculate distances
result = batch_distance_to_interstate(
    sites,
    lat_col='Latitude',
    lon_col='Longitude'
)

# View results
print(result[['site_id', 'nearest_interstate', 'distance_to_interstate_mi']])
```

Output:
```
  site_id nearest_interstate  distance_to_interstate_mi
0       A              I- 80                       2.45
1       B             I- 25                       0.89
2       C             I- 95                       1.23
```

#### Notes

- More efficient than calling `distance_to_nearest_interstate` repeatedly
- Uses GeoPandas `sjoin_nearest` for optimized spatial join operations
- Preserves all original DataFrame columns in the output

---

### preload_highway_data

Pre-load highway data into memory for faster subsequent queries.

```python
def preload_highway_data() -> None
```

#### Parameters

None

#### Returns

None. Prints status messages to stdout.

#### Example

```python
from interstate_distance import preload_highway_data

# Call at application startup
preload_highway_data()
# Output: Loading US Interstate highway data from Census Bureau...
#         Loaded 12,345 Interstate highway segments
#         Highway data pre-loaded and indexed.
```

#### Notes

- Call this at application startup to avoid delays on first distance query
- Downloads and caches highway data
- Builds spatial index (STRtree) for efficient queries
- Subsequent calls to distance functions will use cached data

---

## nearest_site Module

Calculate distance from each site to its nearest neighboring site.

This module processes a CSV file of site locations and calculates the distance to each site's nearest neighbor using KDTree spatial indexing.

### Module Constants

```python
PROJECT_DIR = Path(__file__).parent
INPUT_FILE = PROJECT_DIR / "Sites - Base Data Set.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
METERS_PER_MILE = 1609.344
```

---

### calculate_nearest_site_distances

Calculate distance from each site to its nearest neighboring site.

```python
def calculate_nearest_site_distances(
    input_file: Path = INPUT_FILE
) -> pd.DataFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | `Path` | `INPUT_FILE` | Path to CSV file containing site data. Must have columns: `GTVID`, `Latitude`, `Longitude` |

#### Returns

Returns a `pd.DataFrame` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `GTVID` | `str` | Site identifier |
| `Latitude` | `float` | Original latitude (WGS84) |
| `Longitude` | `float` | Original longitude (WGS84) |
| `nearest_site` | `str` | GTVID of the nearest neighboring site |
| `nearest_site_lat` | `float` | Latitude of the nearest site |
| `nearest_site_lon` | `float` | Longitude of the nearest site |
| `nearest_site_distance_m` | `float` | Distance to nearest site in meters |
| `nearest_site_distance_mi` | `float` | Distance to nearest site in miles |

#### Input File Format

The input CSV must contain:

| Column | Required | Description |
|--------|----------|-------------|
| `GTVID` | Yes | Unique site identifier |
| `Latitude` | Yes | Latitude in decimal degrees (WGS84) |
| `Longitude` | Yes | Longitude in decimal degrees (WGS84) |

#### Example

```python
from pathlib import Path
from nearest_site import calculate_nearest_site_distances

# Use default input file
result = calculate_nearest_site_distances()

# Or specify a custom input file
result = calculate_nearest_site_distances(
    input_file=Path("/path/to/custom_sites.csv")
)

# View results
print(result[['GTVID', 'nearest_site', 'nearest_site_distance_mi']].head())
```

Output:
```
   GTVID nearest_site  nearest_site_distance_mi
0  S0001        S0042                      0.35
1  S0002        S0089                      1.24
2  S0003        S0015                      0.78
3  S0004        S0067                      2.15
4  S0005        S0023                      0.56
```

#### Algorithm

1. Load sites from CSV file
2. Convert to GeoDataFrame with point geometries
3. Project coordinates to EPSG:5070 for accurate distance calculation
4. Build KDTree spatial index from projected coordinates
5. Query for k=3 nearest neighbors (self, nearest, 2nd nearest)
6. Extract nearest neighbor (index 1, since index 0 is self)
7. Return DataFrame with distances

#### Notes

- Uses `scipy.spatial.cKDTree` for O(n log n) nearest neighbor queries
- Coordinates are projected to EPSG:5070 before distance calculation
- Automatically removes unnamed columns from input CSV
- The nearest neighbor excludes the point itself

---

## Constants

### Unit Conversion

```python
METERS_PER_MILE = 1609.344
```

Used for converting distances between meters and miles.

### Data Sources

```python
TIGER_PRIMARY_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER2024/PRIMARYROADS/tl_2024_us_primaryroads.zip"
```

Census Bureau TIGER/Line Primary Roads shapefile URL (2024 edition).

---

## Coordinate Reference Systems

### EPSG:4326 (WGS84)

- **Usage**: Input and output coordinates
- **Description**: World Geodetic System 1984, the standard for GPS coordinates
- **Units**: Decimal degrees

### EPSG:5070 (NAD83 / Conus Albers)

- **Usage**: Internal distance calculations
- **Description**: NAD83 / Conus Albers Equal Area Conic projection
- **Units**: Meters
- **Best for**: Continental United States measurements

The library automatically handles projection conversions. All input should be in WGS84 (standard GPS coordinates), and all output coordinates are returned in WGS84.

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'Latitude'` | Column name mismatch | Specify correct column names in `lat_col`/`lon_col` parameters |
| `ConnectionError` | Cannot download Census data | Check internet connection; data is cached after first download |
| `FileNotFoundError` | Input CSV not found | Verify the path to your input file |

### Example Error Handling

```python
from interstate_distance import batch_distance_to_interstate
import pandas as pd

try:
    result = batch_distance_to_interstate(
        df,
        lat_col='lat',  # Use your column name
        lon_col='lng'   # Use your column name
    )
except KeyError as e:
    print(f"Column not found: {e}")
    print(f"Available columns: {df.columns.tolist()}")
```

---

## Performance Considerations

### Memory Usage

- Highway data: ~50-100 MB in memory after loading
- Spatial index: Additional ~20-50 MB for STRtree
- KDTree: Scales with number of input points

### Processing Speed

| Operation | Typical Performance |
|-----------|---------------------|
| First query (with download) | 30-60 seconds |
| Subsequent single queries | <100 ms |
| Batch processing (1000 points) | 2-5 seconds |
| Batch processing (100,000 points) | 30-60 seconds |

### Optimization Tips

1. Call `preload_highway_data()` at application startup
2. Use `batch_distance_to_interstate()` for multiple points
3. For very large datasets, consider chunking into batches of 50,000-100,000 points

---

## Feature Selection Module

Dynamic feature selection techniques for neural network training. Located in `site_scoring/feature_selection/`.

### Available Methods

The module implements 5 techniques for feature selection during or after training:

| Method | Type | Description |
|--------|------|-------------|
| **Stochastic Gates (STG)** | During training | Learnable Bernoulli gates with L0 regularization |
| **LassoNet** | During training | Hierarchical L1 constraints with skip connections |
| **SHAP-Select** | Post-training | Iterative elimination using statistical significance |
| **TabNet** | During training | Instance-wise sparsemax attention (replaces MLP) |
| **Gradient Analysis** | During training | Weight/gradient profile tracking |

### Quick Start

```python
from site_scoring.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    FeatureSelectionTrainer,
    create_feature_selection_model,
    get_preset,
)

# Quick start with preset
config = get_preset('stg_light')
model, fs_trainer = create_feature_selection_model(
    config=config,
    base_model=base_model,
    n_numeric=24,
    n_boolean=48,
    categorical_vocab_sizes={'state': 50, 'network': 10},
    feature_names=feature_names,
    device='mps'
)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        loss, fs_loss = model(batch)
        total_loss = loss + fs_loss
        total_loss.backward()
        optimizer.step()

    # End of epoch - record feature selection stats
    stats = fs_trainer.on_epoch_end(epoch)
    print(f"Active features: {stats.get('n_active_features')}")

# Get final selection summary
summary = fs_trainer.get_selection_summary()
print(f"Selected {summary['n_selected']}/{summary['n_total_features']} features")
```

---

### Configuration

```python
@dataclass
class FeatureSelectionConfig:
    # Primary method to use during training
    method: FeatureSelectionMethod = FeatureSelectionMethod.NONE

    # Post-training SHAP validation (can combine with any method)
    run_shap_validation: bool = False

    # Track gradient/weight profiles (can combine with any method)
    track_gradients: bool = False

    # Stochastic Gates (STG) Parameters
    stg_sigma: float = 0.5          # Gate distribution spread
    stg_lambda: float = 0.1         # L0 regularization weight
    stg_init_mean: float = 0.5      # Initial gate activation
    stg_threshold: float = 0.5      # Selection threshold

    # LassoNet Parameters
    lassonet_M: float = 10.0        # Hierarchy coefficient
    lassonet_lambda: float = 0.01   # L1 regularization strength
    lassonet_lambda_path: bool = False

    # SHAP-Select Parameters
    shap_significance_level: float = 0.05
    shap_n_background: int = 100
    shap_n_samples: int = 100

    # TabNet Parameters
    tabnet_n_d: int = 64
    tabnet_n_a: int = 64
    tabnet_n_steps: int = 5
    tabnet_gamma: float = 1.5

    # Gradient Analysis Parameters
    gradient_analysis_interval: int = 5
    gradient_elimination_interval: int = 10
    gradient_elimination_percentile: float = 5
    gradient_min_features: int = 5
```

---

### Presets

```python
from site_scoring.feature_selection import get_preset

# Available presets
config = get_preset('none')              # No feature selection
config = get_preset('stg_light')         # Light STG (keeps most features)
config = get_preset('stg_aggressive')    # Aggressive STG + SHAP validation
config = get_preset('lassonet_standard') # Standard LassoNet
config = get_preset('lassonet_path')     # LassoNet with full lambda path
config = get_preset('shap_only')         # Post-training SHAP-Select only
config = get_preset('tabnet')            # TabNet with sparsemax
config = get_preset('hybrid_stg_shap')   # STG + SHAP post-training
```

---

### Stochastic Gates (STG)

Implements the method from "Feature Selection using Stochastic Gates" (ICML 2020).

**Key idea:** Attach a learnable gate z_d to each input feature. The gate is drawn from a continuous relaxation of the Bernoulli distribution, enabling gradient-based optimization of L0 regularization.

**Mathematical formulation:**
```
z_d = clamp(0.5 + mu_d + epsilon_d, 0, 1)   where epsilon_d ~ N(0, sigma^2)
Loss = L(f(z * x), y) + lambda * sum_d P(z_d > 0)
```

```python
from site_scoring.feature_selection import StochasticGates

stg = StochasticGates(
    n_features=200,
    sigma=0.5,
    reg_weight=0.1,
    init_mean=0.5
)

# Forward pass
gated_x, reg_loss = stg(x)  # gated_x has features weighted by gates

# Get feature importance
importance = stg.get_feature_importance()  # tensor of gate probabilities
mask = stg.get_feature_mask(threshold=0.5)  # boolean mask
n_active = stg.get_n_active_features()  # count of active features
```

---

### LassoNet

Implements the method from "LassoNet: A Neural Network with Feature Sparsity" (JMLR 2021).

**Key idea:** Enforce a hierarchy constraint where a feature can only participate in hidden layers if its linear (skip connection) representative is active.

**Mathematical formulation:**
```
minimize  L(theta, W) + lambda * ||theta||_1
s.t.      ||W_j^(1)||_inf <= M * |theta_j|,  j=1,...,d
```

When theta_j = 0, the constraint forces W_j = 0, completely removing feature j.

```python
from site_scoring.feature_selection import LassoNetModel, HierProx

# Create LassoNet model
model = LassoNetModel(
    n_numeric=24,
    n_boolean=48,
    categorical_vocab_sizes={'state': 50},
    hidden_dims=[512, 256, 128],
    M=10.0  # hierarchy coefficient
)

# Apply hierarchical proximal operator after optimizer step
theta_new, W_new = HierProx.apply(
    theta=model.lassonet_layer.theta,
    W=model.lassonet_layer.linear.weight,
    lam=0.01,
    M=10.0
)
```

---

### SHAP-Select

Post-training feature elimination using SHAP values and statistical significance testing.

```python
from site_scoring.feature_selection import apply_shap_select

results = apply_shap_select(
    model_predict_fn=model.predict,
    X_val=X_val,
    y_val=y_val,
    feature_names=feature_names,
    n_background=100,
    n_shap_samples=100,
    significance_level=0.05,
    verbose=True
)

# Results include:
# - significant_features: list of statistically significant features
# - eliminated_features: list of features to eliminate
# - p_values: dict of feature -> p-value
# - shap_values: raw SHAP values matrix
```

---

### FeatureSelectionTrainer

Unified trainer that integrates feature selection with model training.

```python
from site_scoring.feature_selection import FeatureSelectionTrainer

fs_trainer = FeatureSelectionTrainer(
    config=config,
    model=model,
    feature_names=feature_names,
    input_dim=200,
    device='mps'
)

# During training
for epoch in range(epochs):
    # Training step
    gated_x, reg_loss = fs_trainer.apply_feature_gating(x)
    predictions = model.mlp(gated_x)
    total_loss = criterion(predictions, targets) + reg_loss

    # End of epoch
    stats = fs_trainer.on_epoch_end(epoch)

# After training
summary = fs_trainer.get_selection_summary()
# {
#     'method': 'Stochastic Gates (STG)',
#     'n_total_features': 200,
#     'n_selected': 150,
#     'n_eliminated': 50,
#     'selection_rate': 0.75,
#     'selected_features': [...],
#     'eliminated_features': [...],
#     'importance_scores': {...},
#     'top_10_features': [...],
#     'bottom_10_features': [...]
# }

# Save results
fs_trainer.save_results(output_dir)
```

---

### Integration with Training Service

The training service automatically integrates feature selection when configured:

```python
# Via API
POST /api/training/start
{
    "feature_selection_method": "stg_light",
    "stg_lambda": 0.1,
    "stg_sigma": 0.5
}

# Progress updates include feature selection stats
{
    "n_active_features": 150,
    "fs_reg_loss": 0.0012
}

# Final metrics include feature selection summary
{
    "feature_selection": {
        "n_selected": 150,
        "n_eliminated": 50,
        "selected_features": [...],
        "top_10_features": [...]
    }
}
```
