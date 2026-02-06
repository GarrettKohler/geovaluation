# Data Sources, Assumptions & Considerations

> **GSTV Site Visualization & ML Pipeline**
> *Comprehensive documentation of all datasets, their origins, processing assumptions, and unique considerations*

---

## Executive Summary

This system processes **57,675 unique DOOH (Digital Out-of-Home) advertising sites** across the United States, with **1.47M monthly performance records** spanning ~4 years of history. The data pipeline transforms raw operational data into ML-ready features for revenue prediction and lookalike modeling.

| Metric | Value |
|--------|-------|
| Total Sites | 57,675 |
| Active Sites (Training) | 26,099 (45.3%) |
| Monthly Records | 1,475,761 |
| Total Revenue | $405.5M |
| Time Span | ~47 months |

---

## Table of Contents

1. [Data Architecture Overview](#1-data-architecture-overview)
2. [Primary Data Sources](#2-primary-data-sources)
3. [Geospatial Data Sources](#3-geospatial-data-sources)
4. [Processed Datasets](#4-processed-datasets)
5. [Feature Engineering](#5-feature-engineering)
6. [Assumptions & Reasoning](#6-assumptions--reasoning)
7. [Data Quality & Limitations](#7-data-quality--limitations)
8. [Quick Reference](#8-quick-reference)

---

## 1. Data Architecture Overview

### Directory Structure

```
data/
├── input/                              # Raw source data
│   ├── Site Scores - Site Revenue...   # Primary operational data (927 MB)
│   ├── Sites - Base Data Set.csv       # Static site metadata (17 MB)
│   ├── Site Revenue - Salesforce.csv   # CRM revenue data (5.1 MB)
│   ├── nearest_site_distances.csv      # Pre-computed site proximity (6.5 MB)
│   └── site_interstate_distances.csv   # Highway distances (3.8 MB)
│
├── processed/                          # ML-ready datasets
│   ├── site_aggregated_precleaned.*    # All sites, aggregated (57,675 rows)
│   └── site_training_data.*            # Active sites only (26,099 rows)
│
└── shapefiles/                         # Geographic boundaries
    └── US Interstate highway data (Census TIGER)
```

### Processing Pipeline

```
Raw Monthly Data (1.47M rows)
         │
         ▼
┌─────────────────────────────┐
│  Aggregate by Site          │  Group 47 months → 1 row per site
│  - Sum: revenue, impressions│
│  - Last: metadata fields    │
│  - Count: active_months     │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Feature Engineering        │  Add derived features
│  - Relative strength (RS)   │
│  - Log transformations      │
│  - One-hot encoding         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Join Geospatial Features   │  Merge distance calculations
│  - Nearest site distance    │
│  - Interstate proximity     │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Filter for Training        │  Active status only
│  - 26,099 / 57,675 sites    │
│  - Positive revenue         │
└─────────────────────────────┘
         │
         ▼
    ML Model Training
```

---

## 2. Primary Data Sources

### 2.1 Site Scores - Site Revenue, Impressions, and Diagnostics

| Property | Value |
|----------|-------|
| **File** | `data/input/Site Scores - Site Revenue, Impressions, and Diagnostics.csv` |
| **Size** | 927 MB |
| **Rows** | 1,475,761 (monthly records) |
| **Unique Sites** | 57,675 |
| **Columns** | 94 |

#### Purpose
Primary operational data source containing monthly performance metrics for every GSTV site. Each row represents one site's performance for one month. This is from Salesforce Sites and Site Revenue data, and Vistar Revenue and Diagnostic Insights data.

#### Schema Categories

**Time & Identification (5 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Month-year of record (YYYY-MM-DD) |
| `month`, `year` | Int | Extracted time components |
| `id_gbase` | String | Internal unique site hash |
| `gtvid` | String | Public site ID (e.g., "DCW126") |

**Location (7 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `latitude`, `longitude` | Float | WGS84 coordinates |
| `state`, `county`, `zip` | String | Geographic identifiers |
| `dma`, `dma_rank` | String/Int | Designated Market Area |

**Performance Metrics (5 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `revenue` | Float | Monthly revenue in USD (nullable) |
| `monthly_impressions` | Int | Total ad impressions |
| `monthly_nvis` | Int | Network video impressions |
| `monthly_impressions_per_screen` | Float | Impressions / screen_count |
| `monthly_nvis_per_screen` | Float | NVIs / screen_count |

**Site Configuration (9 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `network` | String | Retailer network (e.g., "Wayne", "Dover") |
| `program` | String | Program type |
| `experience_type` | String | Content type |
| `hardware_type` | String | Hardware model |
| `retailer` | String | Retailer brand (e.g., "Speedway") |
| `screen_count` | Int | Physical screens at site |
| `statuis` | String | Status (note: typo in source) |
| `schedule_site`, `sellable_site` | String | Boolean flags ("Yes"/"No") |

**Demographics (7 columns)**
| Column | Type | Description |
|--------|------|-------------|
| `avg_household_income` | Float | Area median income |
| `median_age` | Float | Area median age |
| `pct_african_american`, `pct_asian`, `pct_hispanic` | Float | Racial/ethnic % |
| `pct_female`, `pct_male` | Float | Gender breakdown |

**Capability Flags (9 columns)** — Boolean as "Yes"/"No"/"Unknown"
| Column | Description |
|--------|-------------|
| `c_emv_enabled` | EMV payment capable |
| `c_nfc_enabled` | Contactless payment |
| `c_open_24_hours` | 24/7 operation |
| `c_sells_beer`, `c_sells_wine`, `c_sells_diesel_fuel` | Product availability |
| `c_sells_lottery` | Lottery tickets |
| `c_vistar_programmatic_enabled` | Programmatic ads |
| `c_walk_up_enabled` | Pedestrian access |

**Restriction Flags (31 columns)** — Content category restrictions
| Prefix | Examples |
|--------|----------|
| `r_lottery`, `r_government` | Category-wide restrictions |
| `r_cpg_beverage_beer_*` | Alcohol advertising |
| `r_cpg_cbd_*`, `r_cpg_cannabis_*` | Cannabis-related |
| `r_government_political` | Political ads (55.8% restricted) |

**Brand Information (3 columns)**
| Column | Description |
|--------|-------------|
| `brand_fuel` | Fuel brand (Shell, ExxonMobil, etc.) |
| `brand_restaurant` | Restaurant brand |
| `brand_c_store` | Convenience store brand |

#### Unique Considerations

1. **Monthly Granularity**: Data is reported monthly, not daily. Intra-month patterns are not captured.

2. **Revenue Nullability**: Some months have null/zero revenue due to:
   - New site ramp-up period
   - Temporary deactivation
   - Reporting delays

3. **Column Typo**: The status column is named `statuis` (not `status`) in the source data.

4. **Flag Encoding**: Boolean fields use strings ("Yes"/"No"/"Unknown"), not native booleans.

---

### 2.2 Site Revenue - Salesforce

| Property | Value |
|----------|-------|
| **File** | `data/input/Site Revenue - Salesforce.csv` |
| **Size** | 5.1 MB |
| **Rows** | 67,604 |

#### Purpose
Alternative revenue source from Salesforce CRM. Used for cross-validation and reconciliation.

#### Schema
| Column | Type | Description |
|--------|------|-------------|
| `ID - Gbase` | String | Site identifier |
| `Sellable` | Boolean | Can site be sold |
| `Schedulable` | Boolean | Can ads be scheduled |
| `Program`, `Network` | String | Site classification |
| `Average Revenue` | Float | Average monthly revenue |
| `Average DMA Rank` | Int | Market ranking |

#### Unique Considerations

1. **Different Aggregation**: Salesforce provides pre-aggregated averages, not monthly records.

2. **Reconciliation Use**: Primarily used to validate Site Scores revenue data, not for ML training.

---

### 2.3 Sites - Base Data Set

| Property | Value |
|----------|-------|
| **File** | `data/input/Sites - Base Data Set.csv` |
| **Size** | 17 MB |
| **Rows** | 67,650 |

#### Purpose
Static site metadata with different source/coverage than Site Scores. Used in early preprocessing stages.

#### Unique Considerations

1. **Higher Row Count**: Contains 67,650 sites vs. 57,675 in Site Scores (includes historical/removed sites).

2. **Metadata Focus**: Emphasizes site configuration over performance metrics.

---

## 3. Geospatial Data Sources

### 3.1 Nearest Site Distances

| Property | Value |
|----------|-------|
| **File** | `data/input/nearest_site_distances.csv` |
| **Size** | 6.5 MB |
| **Rows** | 67,644 |

#### Purpose
Pre-computed distances to the nearest neighboring GSTV site for competitive density analysis.

#### Schema
| Column | Type | Description |
|--------|------|-------------|
| `GTVID` | String | Site identifier |
| `Latitude`, `Longitude` | Float | Site coordinates |
| `nearest_site` | String | GTVID of nearest neighbor |
| `nearest_site_lat`, `nearest_site_lon` | Float | Neighbor coordinates |
| `nearest_site_distance_m` | Float | Distance in meters |
| `nearest_site_distance_mi` | Float | Distance in miles |

#### Calculation Method
- **Algorithm**: KDTree spatial search (O(n log n) complexity)
- **Metric**: Haversine (great circle) distance on WGS84 ellipsoid
- **Accuracy**: ~0.1% error vs. true geodesic distance

#### Reasoning
**Why include nearest site distance?**
- Sites in dense urban areas compete for advertiser attention
- Isolated rural sites may have different revenue characteristics
- Proxy for market saturation and local competition

#### Unique Considerations

1. **Self-Exclusion**: Each site's distance is to a *different* site (not itself).

2. **Log Transformation**: Converted to `log_nearest_site_distance_mi` for ML:
   - Range: 0.0 to 3.92 (log miles)
   - Median: 0.59 (~1.8 miles)

3. **Coverage**: 99.96% of sites have distance data (24 sites missing).

---

### 3.2 Interstate Highway Distances

| Property | Value |
|----------|-------|
| **File** | `data/input/site_interstate_distances.csv` |
| **Size** | 3.8 MB |
| **Rows** | 70,300 |

#### Purpose
Distance from each site to the nearest US Interstate highway. Highway proximity correlates with traffic volume and visibility.

#### Schema
| Column | Type | Description |
|--------|------|-------------|
| `GTVID` | String | Site identifier |
| `Latitude`, `Longitude` | Float | Site coordinates |
| `nearest_interstate` | String | Highway ID (e.g., "I-280") |
| `distance_to_interstate_mi` | Float | Distance in miles |

#### Data Source
- **Highway Geometry**: US Census Bureau TIGER/Line Primary Roads shapefile
- **Calculation**: Point-to-polyline minimum distance using projected coordinates

#### Reasoning
**Why include interstate distance?**
- Highway-adjacent sites capture commuter and long-haul trucker traffic
- Travel centers near interstates have different revenue profiles
- Proxy for site accessibility and traffic volume

#### Unique Considerations

1. **Multiple Entries**: Some sites have multiple rows (one per nearby interstate). Aggregation takes the minimum distance.

2. **Categorical Feature**: `nearest_interstate` is also used as a categorical feature (which highway is closest).

3. **Log Transformation**: Converted to `log_min_distance_to_interstate_mi`:
   - Range: 0.0 to 7.27 (log miles)
   - Sites at 0 are directly on interstate exit ramps

4. **Row Count Mismatch**: 70,300 rows > 57,675 sites due to multiple-interstate proximity.

---

### 3.3 Census TIGER Shapefiles

| Property | Value |
|----------|-------|
| **Directory** | `data/shapefiles/` |
| **Source** | US Census Bureau TIGER/Line |

#### Purpose
Geographic boundary files for highway network analysis.

#### Contents
- Primary and Secondary Roads (US Interstates, US Routes)
- State boundaries for clipping
- Road attribute metadata (route number, road class)

#### Unique Considerations

1. **Coordinate System**: NAD83 (EPSG:4269), converted to WGS84 for distance calculations.

2. **Vintage**: Should match the time period of site data for accuracy.

---

## 4. Processed Datasets

### 4.1 Site Aggregated Precleaned

| Property | Value |
|----------|-------|
| **Files** | `site_aggregated_precleaned.parquet`, `.csv` |
| **Size** | 8.5 MB (parquet), 41 MB (csv) |
| **Rows** | 57,675 (one per site) |
| **Columns** | 90 |

#### Purpose
Intermediate dataset with all sites aggregated from monthly to site-level, before training filters.

#### Transformations Applied

1. **Temporal Aggregation**
   - **Sums**: `total_revenue`, `total_monthly_impressions`, `total_monthly_nvis`
   - **Averages**: `avg_monthly_revenue`, `avg_monthly_impressions`
   - **Counts**: `active_months` (number of months with data)
   - **Last Values**: Metadata fields (network, retailer, status)

2. **Relative Strength Indicators** (see Section 5.1)

3. **Geospatial Joins** (nearest site + interstate distances)

4. **Log Transformations** (see Section 5.2)

5. **One-Hot Encoding** (see Section 5.3)

#### Site Status Distribution
| Status | Count | Percentage |
|--------|-------|------------|
| Active | 26,101 | 45.3% |
| Temporarily Deactivated | 23,374 | 40.5% |
| Awaiting Installation | 4,417 | 7.7% |
| Deactivated | 3,094 | 5.4% |
| Awaiting Reactivation | 627 | 1.1% |
| Cancelled | 60 | 0.1% |

---

### 4.2 Site Training Data

| Property | Value |
|----------|-------|
| **Files** | `site_training_data.parquet`, `.csv` |
| **Size** | 5.6 MB (parquet), 19 MB (csv) |
| **Rows** | 26,099 (Active sites only) |
| **Columns** | 94 |

#### Purpose
Final ML-ready dataset with only Active sites and sufficient history for training.

#### Filters Applied

```python
# Filter 1: Active status only
df = df.filter(status == 'Active')  # 26,099 sites

# Filter 2: Positive revenue
df = df.filter(total_revenue >= 0)

# Filter 3: Sufficient history (applied at training time)
df = df.filter(active_months > 11)  # >1 year of data
```

#### Feature Categories

**Numeric Features (12)**
```
rs_Impressions, rs_NVIs, rs_Revenue, rs_RevenuePerScreen  # Momentum
avg_monthly_revenue, log_total_revenue                     # Revenue
log_nearest_site_distance_mi, log_min_distance_to_interstate_mi  # Geospatial
avg_household_income, median_age, pct_female, pct_male     # Demographics
```

**Categorical Features (8)**
```
network, program, experience_type, hardware_type, retailer
brand_fuel, brand_restaurant, brand_c_store, nearest_interstate
```

**Boolean Features (40)**
```
c_*_encoded (9 capability flags)
r_*_encoded (29 restriction flags)
schedule_site_encoded, sellable_site_encoded
```

#### Reasoning for Active-Only Filter

1. **Prediction Target**: We predict revenue for operational sites, not historical/deactivated ones.

2. **Data Quality**: Deactivated sites often have incomplete recent data.

3. **Business Relevance**: Lookalike models should identify sites similar to *current* top performers.

---

## 5. Feature Engineering

### 5.1 Relative Strength Indicators

#### Definition
Momentum indicators comparing recent performance to historical baseline.

#### Formula
```
RS = (30-day average + ε) / (90-day average + ε)

Where:
- RS > 1.0: Trending upward (recent > historical)
- RS = 1.0: Stable performance
- RS < 1.0: Trending downward
- ε = 1.0 (smoothing constant)
```

#### Features Created
| Feature | Source Metric | Interpretation |
|---------|---------------|----------------|
| `rs_Impressions` | `monthly_impressions` | Impression volume trend |
| `rs_NVIs` | `monthly_nvis` | Network video impression trend |
| `rs_Revenue` | `revenue` | Revenue trend |
| `rs_RevenuePerScreen` | `revenue / screen_count` | Per-screen efficiency trend |

#### Reasoning

1. **Why momentum matters**: A site trending upward may outperform its historical average in future months.

2. **Why 30/90 day windows**: 30 days captures recent changes; 90 days provides stable baseline.

3. **Why smoothing (ε=1.0)**: Prevents division by zero for new sites with limited history.

#### Assumptions

- **Missing data**: Filled with 1.0 (neutral trend) if insufficient history
- **Stationarity**: Assumes trends persist into the future
- **Seasonality**: Not explicitly modeled (30/90 day comparison may capture some)

---

### 5.2 Log Transformations

#### Formula
```python
log_value = sign(x) * log(1 + |x|)
```

#### Features Transformed
| Original Column | Log Column | Range (log units) |
|-----------------|------------|-------------------|
| `total_revenue` | `log_total_revenue` | 0.0 - 12.14 |
| `total_monthly_impressions` | `log_total_monthly_impressions` | 0.0 - 14.76 |
| `nearest_site_distance_mi` | `log_nearest_site_distance_mi` | 0.0 - 3.92 |
| `min_distance_to_interstate_mi` | `log_min_distance_to_interstate_mi` | 0.0 - 7.27 |

#### Reasoning

1. **Right-skewed distributions**: Revenue and impressions follow power-law distributions with long right tails.

2. **Variance stabilization**: Log transform makes variance more uniform across the range.

3. **Outlier reduction**: Extreme values (very high-revenue sites) have less impact on gradients.

4. **Interpretability**: Coefficients represent multiplicative effects (e.g., "2x distance → Y% revenue change").

#### Assumptions

- **Non-negativity**: All transformed values are ≥ 0 (distances, revenue)
- **Zero handling**: `log(1 + x)` ensures `log(0) = 0`, not undefined

---

### 5.3 One-Hot Encoding of Boolean Flags

#### Input Format
Source data stores booleans as strings: `"Yes"`, `"No"`, `"Unknown"`

#### Encoding Logic
```python
"Yes"     → 1
"No"      → 0
"Unknown" → 0  # Treated as missing capability
NULL      → 0
```

#### Output Columns
| Category | Count | Examples |
|----------|-------|----------|
| Capability (`c_*`) | 9 | `c_emv_enabled_encoded`, `c_nfc_enabled_encoded` |
| Restriction (`r_*`) | 29 | `r_lottery_encoded`, `r_government_political_encoded` |
| Sales | 2 | `schedule_site_encoded`, `sellable_site_encoded` |
| **Total** | **40** | |

#### Reasoning

1. **Neural network compatibility**: NNs require numeric inputs; strings must be encoded.

2. **Unknown → 0**: Conservative assumption that missing capability = no capability.

3. **Separate from categoricals**: Boolean features get direct 0/1 encoding, not embedding.

#### Assumptions

- **Static flags**: Capabilities and restrictions are assumed constant over time (using most recent snapshot)
- **Unknown handling**: "Unknown" treated same as "No" (may undercount some capabilities)

---

### 5.4 Missing Value Handling

| Data Type | Strategy | Rationale |
|-----------|----------|-----------|
| **Numeric** | Fill with 0 | Neutral after standardization |
| **Categorical** | Fill with `"__MISSING__"` | Creates explicit "unknown" category |
| **Boolean** | Fill with 0 | Assumes missing = False |
| **Target** | Fill with median | Preserves target distribution |

#### High-Missing Columns
| Column | Missing % | Reason |
|--------|-----------|--------|
| `c_emv_enabled_encoded` | 91.7% | Legacy feature, not tracked historically |
| `c_walk_up_enabled_encoded` | 98.1% | Rare capability, seldom recorded |

---

### 5.5 Outlier Treatment

#### Method: Percentile Clipping (Winsorization)

```python
# Step 1: Clip to 1st-99th percentile
numeric_array = np.clip(col_data, p1, p99)

# Step 2: Standardize (zero mean, unit variance)
numeric_scaled = StandardScaler().fit_transform(numeric_array)

# Step 3: Final clip to ±10 standard deviations
numeric_scaled = np.clip(numeric_scaled, -10, 10)
```

#### Reasoning

1. **Preserves information**: Unlike removal, clipping keeps extreme cases in training.

2. **Gradient stability**: Prevents extreme values from dominating loss gradients.

3. **Conservative bounds**: ±10σ is generous; true outliers are ~0.01% of data.

---

## 6. Assumptions & Reasoning

### 6.1 Data Aggregation Assumptions

| Assumption | Reasoning | Risk |
|------------|-----------|------|
| **Most Recent Metadata** | Sites rarely change network/retailer | May miss recent transitions |
| **Revenue Summing** | Monthly revenues are independent | May include test periods |
| **Active Months = Row Count** | Proxy for site tenure | Doesn't capture partial months |
| **GTVID Uniqueness** | IDs are stable over time | Some sites may have ID changes |

### 6.2 Feature Engineering Assumptions

| Feature | Assumption | Risk |
|---------|------------|------|
| **Relative Strength** | 30/90 day windows capture trends | May miss longer cycles |
| **Log Distances** | All sites have valid coordinates | 0.04% missing |
| **Demographics** | Census block group ≈ site location | ~500m radius approximation |
| **Capabilities** | Static over time | Sites may add EMV/NFC |

### 6.3 Model Training Assumptions

| Assumption | Implementation | Reasoning |
|------------|----------------|-----------|
| **Data Leakage Prevention** | Exclude target-derived features | `avg_monthly_revenue` target → exclude `total_revenue` |
| **Sufficient History** | Filter to `active_months > 11` | Ensures >1 year of data |
| **Active Sites Only** | Filter to `status == 'Active'` | Predict for operational sites |
| **Class Balance** | `pos_weight=9.0` for lookalike | Compensates for 90/10 imbalance |

### 6.4 Why These Specific Features?

**Included:**
- **Momentum (RS) features**: Capture trajectory, not just level
- **Log distances**: Normalize skewed distributions
- **Demographics**: Area-level proxies for customer base
- **Capabilities**: Direct business relevance (payment types, products)

**Excluded (in Model B preset):**
- **`retailer`**: High cardinality, may overfit to specific chains
- **`pct_male`**: Collinear with `pct_female` (always sum to ~100%)
- **`nearest_interstate`**: Categorical with 200+ values, may not generalize

---

## 7. Data Quality & Limitations

### 7.1 Coverage Statistics

| Metric | Value |
|--------|-------|
| Sites with nearest_site_distance | 99.96% (57,651 / 57,675) |
| Sites with interstate_distance | 99.96% |
| Active sites in training | 45.3% (26,099 / 57,675) |
| Months per site | 1-47 (median: 28) |

### 7.2 Known Issues

1. **Column Typo**: Source data has `statuis` instead of `status`

2. **Boolean Strings**: Flags stored as "Yes"/"No" strings, not native booleans

3. **EMV/Walk-up Sparsity**: >90% missing values for these capabilities

4. **Deactivated Sites**: 40.5% of sites are "Temporarily Deactivated" and excluded from training

### 7.3 Temporal Limitations

1. **No Intra-Month Data**: Daily patterns not captured in monthly aggregates

2. **Seasonality**: Not explicitly modeled; 30/90 day RS may partially capture

3. **COVID Impact**: Data likely spans pre/during/post-COVID periods with different patterns

### 7.4 Geographic Limitations

1. **US Only**: All sites are in the United States

2. **Census Data Vintage**: Demographics may lag current population

3. **Highway Network Changes**: New interstates not reflected if shapefiles are outdated

---

## 8. Quick Reference

### Key Files

| File | Purpose | Rows |
|------|---------|------|
| `Site Scores - ...Diagnostics.csv` | Primary source (monthly) | 1.47M |
| `site_training_data.parquet` | ML-ready (Active only) | 26,099 |
| `site_aggregated_precleaned.parquet` | Intermediate (all sites) | 57,675 |
| `nearest_site_distances.csv` | Geospatial feature | 67,644 |
| `site_interstate_distances.csv` | Highway distances | 70,300 |

### Key Code Files

| File | Purpose |
|------|---------|
| `src/services/data_service.py` | Data loading API |
| `site_scoring/data_transform.py` | ETL pipeline |
| `site_scoring/data_loader.py` | PyTorch data preparation |
| `site_scoring/config.py` | Feature definitions & model presets |

### Feature Counts by Type

| Type | Count | Examples |
|------|-------|----------|
| Numeric | 12 | `rs_Revenue`, `avg_household_income` |
| Categorical | 8 | `network`, `retailer`, `nearest_interstate` |
| Boolean | 40 | `c_nfc_enabled_encoded`, `r_lottery_encoded` |
| **Total** | **60** | |

### Target Variable Options

| Target | Task | Metric |
|--------|------|--------|
| `avg_monthly_revenue` | Regression | MAE, SMAPE, R² |
| `total_revenue` | Regression | MAE, RMSE |
| Top 10% (binary) | Classification (Lookalike) | AUC, Accuracy |

---

## Appendix: Data Provenance

### Internal Sources
- **Site Scores CSV**: GSTV operations database (monthly snapshots)
- **Salesforce Revenue**: Sales CRM system

### External Sources
- **Census TIGER/Line**: US Census Bureau (highway shapefiles)
- **Demographics**: US Census Bureau (block group data)

### Computed Features
- **Nearest Site Distance**: KDTree spatial search (haversine metric)
- **Interstate Distance**: Point-to-polyline geodesic distance
- **Relative Strength**: Rolling window calculations (30/90 days)

---

*Document generated: 2026-01-28*
*Data vintage: ~47 months of operational history*
