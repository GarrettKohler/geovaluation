# site_scoring.data_transform

ETL (Extract, Transform, Load) pipeline that converts raw monthly time-series data into a static, site-level dataset suitable for ML training. Uses **Polars** for high-performance data manipulation.

---

## Pipeline Stages

### 1. aggregate_site_metrics

```python
aggregate_site_metrics(df) → pl.DataFrame
```

Collapses monthly records (e.g., 24 rows for a 2-year-old site) into a single row per site.

**Aggregations:**

- **Sums** – `total_revenue`, `total_monthly_impressions`, `total_monthly_nvis`
- **Averages** – `avg_monthly_revenue`, `avg_monthly_impressions`
- **Counts** – `active_months` (number of months with data)
- **Last values** – Metadata fields (e.g., current `hardware_type`, `network`, `retailer`)

---

### 2. join_geospatial_features

```python
join_geospatial_features(df) → pl.DataFrame
```

Merges the site data with pre-computed geospatial distance features.

**Sources joined:**

- **nearest_site_distances.csv** – Distance to nearest GSTV site (competitive density)
- **site_interstate_distances.csv** – Distance to nearest US Interstate highway (traffic proxy)

---

### 3. Feature Engineering

**add_log_transformations** *(df)*

Applies `log1p` (natural log of x + 1) to right-skewed distributions like revenue and distance. See [features.LogTransforms](DATA_SOURCES.md#featureslogtransforms).

**one_hot_encode_flags** *(df)*

Converts capability flags (e.g., `c_sells_beer`) and restriction flags (e.g., `r_lottery`) from string `"Yes"` / `"No"` into machine-readable binary 0/1 columns. See [features.BooleanEncoding](DATA_SOURCES.md#featuresbooleanencoding).

---

### 4. prepare_training_dataset

```python
prepare_training_dataset(df, active_only=True, drop_geo_ids=True) → pl.DataFrame
```

Applies final filters and cleanup for model consumption.

**Parameters:**

- **active_only** (*bool*) – If `True`, filters to `status == 'Active'` (26,099 sites). If `False`, keeps all 57,675 sites (used for prediction/scoring).
- **drop_geo_ids** (*bool*) – If `True`, removes explicit geographic identifiers (Zip, DMA) to prevent overfitting to specific locations.

**Additional filters:**

- Drops rows with negative revenue (data errors)

---

## Key Outputs

- **site_aggregated_precleaned.parquet** (*57,675 rows*) – Full dataset, all sites, all statuses. Used by `get_all_sites_for_prediction()`.
- **site_training_data.parquet** (*26,099 rows*) – Sanitized, Active-only, ready-to-train dataset.
- **training_data_summary.txt** – Statistical report of the processed data.

---

## get_all_sites_for_prediction

```python
get_all_sites_for_prediction() → pl.DataFrame
```

Loads `site_aggregated_precleaned.parquet` and applies the same transforms as training (`prepare_training_dataset(active_only=False)`), returning all 57,675 sites ready for model inference.

> **Note**
> Cached at module level. First call loads from disk; subsequent calls return the cached DataFrame.

**Example:**

```python
from site_scoring.data_transform import get_all_sites_for_prediction

all_sites = get_all_sites_for_prediction()
print(all_sites.shape)  # (57675, ~94)
```

See also: [`site_scoring.data_loader.create_data_loaders`](data_loading_and_processing.md#site_scoringdata_loadercreate_data_loaders)

---

## Usage

Run as a standalone script to regenerate datasets from raw CSVs:

```bash
python -m site_scoring.data_transform
```
