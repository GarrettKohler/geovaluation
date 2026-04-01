# data

GSTV Site Visualization & ML Pipeline — Data Sources Reference

This system processes **57,675 unique DOOH advertising sites** across the United States, with **1.47M monthly performance records** spanning ~4 years of history.

- **Total Sites** – 57,675
- **Active Sites (Training)** – 26,099 (45.3%)
- **Monthly Records** – 1,475,761
- **Total Revenue** – $405.5M
- **Time Span** – ~47 months

---

## Processing Pipeline

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

```
data/
├── input/                              # Raw source data
│   ├── site_scores_revenue_and_diagnostics.csv  # Primary operational data (927 MB)
│   ├── sites_base_data_set.csv         # Static site metadata (17 MB)
│   ├── salesforce_site_revenue.csv     # CRM revenue data (5.1 MB)
│   ├── nearest_site_distances.csv      # Pre-computed site proximity (6.5 MB)
│   ├── site_interstate_distances.csv   # Highway distances (3.8 MB)
│   ├── site_kroger_distances.csv       # Nearest Kroger distance (2.7 MB)
│   ├── site_mcdonalds_distances.csv    # Nearest McDonald's distance (2.8 MB)
│   ├── site_walmart_distances.csv      # Nearest Walmart distance (2.7 MB)
│   ├── site_target_distances.csv       # Nearest Target distance (2.7 MB)
│   ├── site_transactions_daily.csv     # Daily transaction counts (45 MB)
│   ├── site_status_daily.csv           # Daily status snapshots (54 MB)
│   └── active_days_per_month.csv       # Monthly active day counts (114 MB)
│
├── processed/                          # ML-ready datasets
│   ├── site_aggregated_precleaned.*    # All sites, aggregated (57,675 rows)
│   ├── site_training_data.*            # Active sites only (26,099 rows)
│   ├── site_monthly_activity.parquet   # Monthly activity merge (83,874 rows)
│   ├── precleaned_summary.txt          # ETL summary report (all sites)
│   └── training_data_summary.txt       # ETL summary report (training sites)
│
└── shapefiles/                         # Geographic boundaries
    └── US Interstate highway data (Census TIGER)
```

---

# Source Datasets

## data.input.SiteScoresRevenue

```
data/input/Site Scores - Site Revenue, Impressions, and Diagnostics.csv
```

Primary operational data source containing monthly performance metrics for every GSTV site. Each row represents one site's performance for one month. Derived from Salesforce Sites/Revenue data and Vistar Revenue/Diagnostic Insights data.

**Properties:**

- **size** (*bytes*) – 927 MB
- **rows** (*int*) – 1,475,761 (monthly records)
- **unique_sites** (*int*) – 57,675
- **columns** (*int*) – 94

### Attributes

*Time & Identification (5 columns)*

- **date** (*Date*) – Month-year of record (YYYY-MM-DD)
- **month** (*int*) – Extracted month component
- **year** (*int*) – Extracted year component
- **id_gbase** (*str*) – Internal unique site hash
- **gtvid** (*str*) – Public site ID (e.g., `"DCW126"`)

*Location (7 columns)*

- **latitude** (*float*) – WGS84 latitude
- **longitude** (*float*) – WGS84 longitude
- **state** (*str*) – US state
- **county** (*str*) – County name
- **zip** (*str*) – ZIP code
- **dma** (*str*) – Designated Market Area
- **dma_rank** (*int*) – DMA ranking

*Performance Metrics (5 columns)*

- **revenue** (*float, nullable*) – Monthly revenue in USD
- **monthly_impressions** (*int*) – Total ad impressions
- **monthly_nvis** (*int*) – Network video impressions
- **monthly_impressions_per_screen** (*float*) – Impressions / screen_count
- **monthly_nvis_per_screen** (*float*) – NVIs / screen_count

*Site Configuration (9 columns)*

- **network** (*str*) – Retailer network (e.g., `"Wayne"`, `"Dover"`)
- **program** (*str*) – Program type
- **experience_type** (*str*) – Content type
- **hardware_type** (*str*) – Hardware model
- **retailer** (*str*) – Retailer brand (e.g., `"Speedway"`)
- **screen_count** (*int*) – Physical screens at site
- **statuis** (*str*) – Status (see Note below)
- **schedule_site** (*str*) – Whether site is schedulable (`"Yes"` / `"No"`)
- **sellable_site** (*str*) – Whether site is sellable (`"Yes"` / `"No"`)

*Demographics (7 columns)*

- **avg_household_income** (*float*) – Area median income
- **median_age** (*float*) – Area median age
- **pct_african_american** (*float*) – Racial/ethnic percentage
- **pct_asian** (*float*) – Racial/ethnic percentage
- **pct_hispanic** (*float*) – Racial/ethnic percentage
- **pct_female** (*float*) – Gender breakdown
- **pct_male** (*float*) – Gender breakdown

*Capability Flags (9 columns)* — Encoded as `"Yes"` / `"No"` / `"Unknown"`

- **c_emv_enabled** – EMV payment capable
- **c_nfc_enabled** – Contactless payment
- **c_open_24_hours** – 24/7 operation
- **c_sells_beer** – Beer availability
- **c_sells_wine** – Wine availability
- **c_sells_diesel_fuel** – Diesel fuel availability
- **c_sells_lottery** – Lottery tickets
- **c_vistar_programmatic_enabled** – Programmatic ads
- **c_walk_up_enabled** – Pedestrian access

*Restriction Flags (31 columns)* — Content category advertising restrictions

- **r_lottery**, **r_government** – Category-wide restrictions
- **r_cpg_beverage_beer_\*** – Alcohol advertising (multiple sub-flags)
- **r_cpg_cbd_\***, **r_cpg_cannabis_\*** – Cannabis-related (multiple sub-flags)
- **r_government_political** – Political ads (55.8% restricted)

*Brand Information (3 columns)*

- **brand_fuel** (*str*) – Fuel brand (Shell, ExxonMobil, etc.)
- **brand_restaurant** (*str*) – Restaurant brand
- **brand_c_store** (*str*) – Convenience store brand

> **Note**
> The status column is named `statuis` (not `status`) in the source data. The `DataRegistry` handles both spellings, but be aware when writing raw queries.

> **Warning**
> Revenue values may be null due to new site ramp-up, temporary deactivation, or reporting delays. Boolean flags use strings (`"Yes"` / `"No"` / `"Unknown"`), not native booleans.

---

## data.input.SiteRevenueSalesforce

```
data/input/Site Revenue - Salesforce.csv
```

Alternative revenue source from Salesforce CRM. Used for cross-validation and reconciliation, not for ML training.

**Properties:**

- **size** (*bytes*) – 5.1 MB
- **rows** (*int*) – 67,604

### Attributes

- **ID - Gbase** (*str*) – Site identifier
- **Sellable** (*bool*) – Can site be sold
- **Schedulable** (*bool*) – Can ads be scheduled
- **Program** (*str*) – Site classification
- **Network** (*str*) – Network affiliation
- **Average Revenue** (*float*) – Average monthly revenue
- **Average DMA Rank** (*int*) – Market ranking

> **Note**
> Provides pre-aggregated averages, not monthly records. Primarily used to validate Site Scores revenue data.

---

## data.input.SitesBaseDataSet

```
data/input/Sites - Base Data Set.csv
```

Static site metadata with different source/coverage than Site Scores. Used in early preprocessing stages.

**Properties:**

- **size** (*bytes*) – 17 MB
- **rows** (*int*) – 67,650

> **Note**
> Contains 67,650 sites vs. 57,675 in Site Scores (includes historical/removed sites). Emphasizes site configuration over performance metrics.

---

# Geospatial Datasets

## data.input.NearestSiteDistances

```
data/input/nearest_site_distances.csv
```

Pre-computed distances to the nearest neighboring GSTV site for competitive density analysis.

**Properties:**

- **size** (*bytes*) – 6.5 MB
- **rows** (*int*) – 67,644
- **algorithm** (*str*) – KDTree spatial search (O(n log n))
- **metric** (*str*) – Haversine (great circle) distance on WGS84 ellipsoid
- **accuracy** (*float*) – ~0.1% error vs. true geodesic distance

### Attributes

- **GTVID** (*str*) – Site identifier
- **Latitude** (*float*) – Site latitude
- **Longitude** (*float*) – Site longitude
- **nearest_site** (*str*) – GTVID of nearest neighbor
- **nearest_site_lat** (*float*) – Neighbor latitude
- **nearest_site_lon** (*float*) – Neighbor longitude
- **nearest_site_distance_m** (*float*) – Distance in meters
- **nearest_site_distance_mi** (*float*) – Distance in miles

### Derived Feature

- **log_nearest_site_distance_mi** (*float*) – `sign(x) * log(1 + |x|)` of distance in miles. Range: 0.0–3.92, median: 0.59 (~1.8 miles).

### Reasoning

Sites in dense urban areas compete for advertiser attention. Isolated rural sites have different revenue characteristics. This feature serves as a proxy for market saturation and local competition.

> **Note**
> Each site's distance is to a *different* site (self-excluded). Coverage: 99.96% of sites have distance data (24 sites missing).

---

## data.input.InterstateDistances

```
data/input/site_interstate_distances.csv
```

Distance from each site to the nearest US Interstate highway. Highway proximity correlates with traffic volume and visibility.

**Properties:**

- **size** (*bytes*) – 3.8 MB
- **rows** (*int*) – 70,300
- **source** (*str*) – US Census Bureau TIGER/Line Primary Roads shapefile
- **method** (*str*) – Point-to-polyline minimum distance using projected coordinates

### Attributes

- **GTVID** (*str*) – Site identifier
- **Latitude** (*float*) – Site latitude
- **Longitude** (*float*) – Site longitude
- **nearest_interstate** (*str*) – Highway ID (e.g., `"I-280"`)
- **distance_to_interstate_mi** (*float*) – Distance in miles

### Derived Feature

- **log_min_distance_to_interstate_mi** (*float*) – `sign(x) * log(1 + |x|)` of minimum distance. Range: 0.0–7.27. Zero means directly on interstate exit ramp.

### Reasoning

Highway-adjacent sites capture commuter and long-haul trucker traffic. Travel centers near interstates have different revenue profiles. Proxy for site accessibility and traffic volume.

> **Note**
> Some sites have multiple rows (one per nearby interstate). Aggregation takes the minimum distance. Row count (70,300) exceeds site count (57,675) due to this. `nearest_interstate` is also used as a categorical feature.

---

## data.input.RetailerDistances

```
data/input/site_kroger_distances.csv
data/input/site_mcdonalds_distances.csv
data/input/site_walmart_distances.csv
data/input/site_target_distances.csv
```

Pre-computed minimum distance from each GSTV site to the nearest location of four major retailers: Kroger, McDonald's, Walmart, and Target. Proximity to high-traffic retail anchors serves as a proxy for commercial area density and foot traffic potential.

**Properties:**

- **rows** (*int*) – 57,675 per file (one per site)
- **columns** (*int*) – 4 per file
- **sizes** (*bytes*) – Kroger: 2.7 MB, McDonald's: 2.8 MB, Walmart: 2.7 MB, Target: 2.7 MB
- **algorithm** (*str*) – KDTree spatial search (haversine metric), same as nearest site distances

### Attributes (identical schema across all four files)

- **GTVID** (*str*) – Site identifier
- **Latitude** (*float*) – Site latitude
- **Longitude** (*float*) – Site longitude
- **min_distance_to_{retailer}_mi** (*float*) – Distance in miles to nearest retailer location (e.g., `min_distance_to_kroger_mi`)

### Derived Features

- **log_min_distance_to_kroger_mi** (*float*) – `sign(x) * log(1 + |x|)` of Kroger distance
- **log_min_distance_to_mcdonalds_mi** (*float*) – `sign(x) * log(1 + |x|)` of McDonald's distance
- **log_min_distance_to_walmart_mi** (*float*) – `sign(x) * log(1 + |x|)` of Walmart distance
- **log_min_distance_to_target_mi** (*float*) – `sign(x) * log(1 + |x|)` of Target distance

### Reasoning

1. **Foot traffic proxy** – Sites near major retailers benefit from shared customer traffic.
2. **Commercial density indicator** – Proximity to multiple retail brands signals a developed commercial corridor.
3. **Revenue correlation** – Preliminary analysis showed significant correlation between retailer proximity and site revenue, especially for McDonald's (fast-food adjacency) and Walmart (high daily visitor count).
4. **Complementary to interstate distance** – Interstate distance captures highway traffic; retailer distance captures local commercial activity.

> **Note**
> Source geodata files for each retailer (e.g., `mcdonalds_geodata.csv`, `walmart_geodata.csv`) are also present in `data/input/` and contain the full location listings used to compute these distances.

---

## data.input.CensusTIGERShapefiles

```
data/shapefiles/
```

Geographic boundary files for highway network analysis from the US Census Bureau TIGER/Line dataset.

**Properties:**

- **contents** – Primary and Secondary Roads (US Interstates, US Routes), State boundaries, Road attribute metadata (route number, road class)
- **coordinate_system** – NAD83 (EPSG:4269), converted to WGS84 for distance calculations

> **Note**
> Shapefile vintage should match the time period of site data for accuracy.

---

# Processed Datasets

## data.processed.SiteAggregatedPrecleaned

```
data/processed/site_aggregated_precleaned.parquet
```

Intermediate dataset with all sites aggregated from monthly to site-level, before training filters are applied. This is the dataset used by `get_all_sites_for_prediction()` to score all 57K sites.

**Properties:**

- **size** (*bytes*) – 8.5 MB (parquet), 41 MB (csv)
- **rows** (*int*) – 57,675 (one per site)
- **columns** (*int*) – 90

### Transformations Applied

1. **Temporal Aggregation** – Sums: `total_revenue`, `total_monthly_impressions`, `total_monthly_nvis`. Averages: `avg_monthly_revenue`, `avg_monthly_impressions`. Counts: `active_months`. Last values: metadata fields (network, retailer, status).
2. **Relative Strength Indicators** – See [Feature Engineering](#feature-engineering).
3. **Geospatial Joins** – Nearest site + interstate distances merged.
4. **Log Transformations** – Applied to revenue, impressions, distances.
5. **One-Hot Encoding** – Boolean flags converted to 0/1.

### Site Status Distribution

- **Active** (*int*) – 26,101 (45.3%)
- **Temporarily Deactivated** (*int*) – 23,374 (40.5%)
- **Awaiting Installation** (*int*) – 4,417 (7.7%)
- **Deactivated** (*int*) – 3,094 (5.4%)
- **Awaiting Reactivation** (*int*) – 627 (1.1%)
- **Cancelled** (*int*) – 60 (0.1%)

**Example:**

```python
import polars as pl
df = pl.read_parquet("data/processed/site_aggregated_precleaned.parquet")
print(df.shape)  # (57675, 90)
```

See also: [`data.processed.SiteTrainingData`](#dataprocessedsitetrainingdata)

---

## data.processed.SiteTrainingData

```
data/processed/site_training_data.parquet
```

Final ML-ready dataset with only Active sites and sufficient history for training. Created by `prepare_training_dataset(active_only=True)`.

**Properties:**

- **size** (*bytes*) – 5.6 MB (parquet), 19 MB (csv)
- **rows** (*int*) – 26,099 (Active sites only)
- **columns** (*int*) – 94

### Filters Applied

```python
# Filter 1: Active status only
df = df.filter(status == 'Active')  # 26,099 sites

# Filter 2: Positive revenue
df = df.filter(total_revenue >= 0)

# Filter 3: Sufficient history (applied at training time)
df = df.filter(active_months > 11)  # >1 year of data
```

### Feature Categories

*Numeric Features (12)*

- **rs_Impressions**, **rs_NVIs**, **rs_Revenue**, **rs_RevenuePerScreen** – Momentum indicators
- **avg_monthly_revenue**, **log_total_revenue** – Revenue metrics
- **log_nearest_site_distance_mi**, **log_min_distance_to_interstate_mi** – Geospatial
- **avg_household_income**, **median_age**, **pct_female**, **pct_male** – Demographics

*Categorical Features (8)*

- **network**, **program**, **experience_type**, **hardware_type**, **retailer** – Site config
- **brand_fuel**, **brand_restaurant**, **brand_c_store** – Brand information

*Boolean Features (40)*

- **c_\*_encoded** – 9 capability flags
- **r_\*_encoded** – 29 restriction flags
- **schedule_site_encoded**, **sellable_site_encoded** – Sales flags

### Reasoning

1. **Active-only filter** – We predict revenue for operational sites, not historical/deactivated ones.
2. **Data quality** – Deactivated sites often have incomplete recent data.
3. **Business relevance** – Lookalike models should identify sites similar to *current* top performers.

**Example:**

```python
from site_scoring.data_loader import create_data_loaders
from site_scoring.config import Config

config = Config()
train_loader, val_loader, test_loader, processor = create_data_loaders(config)

for numeric, categorical, boolean, target in train_loader:
    # numeric.shape    → [batch_size, 12]
    # categorical.shape → [batch_size, 8]
    # boolean.shape    → [batch_size, 40]
    # target.shape     → [batch_size, 1]
    break
```

See also: [`data.processed.SiteAggregatedPrecleaned`](#dataprocessedsiteaggregatedprecleaned)

---

## data.processed.SiteMonthlyActivity

```
data/processed/site_monthly_activity.parquet
```

Intermediate dataset that merges daily transaction counts (`site_transactions_daily.csv`) and daily status snapshots (`site_status_daily.csv`) into monthly aggregates per site. Each row represents one site's activity summary for one month.

**Properties:**

- **size** (*bytes*) – 987 KB (parquet)
- **rows** (*int*) – 83,874 (site-month combinations)
- **columns** (*int*) – 6

### Attributes

- **id_gbase** (*str*) – Internal unique site hash
- **year_month** (*str*) – Month period (e.g., `"2022-01"`)
- **gtvid** (*str*) – Public site ID
- **active_days** (*int*) – Number of days the site had Active status in that month
- **transaction_days** (*int*) – Number of days the site recorded at least one transaction
- **total_days_in_data** (*int*) – Total days of data coverage for that month

### Reasoning

1. **Operational consistency** – Sites that are active and transacting every day of the month are more reliable revenue generators than sites with intermittent activity.
2. **Data quality signal** – Low `transaction_days` relative to `active_days` may indicate reporting gaps or hardware issues.
3. **Deduplication** – Daily status data can have duplicate rows; the aggregation step handles deduplication before counting.

> **Note**
> This file is generated by the ETL pipeline and joined into the aggregated dataset. It is not used directly for model training.

---

## data.processed.ETLSummaryReports

```
data/processed/precleaned_summary.txt
data/processed/training_data_summary.txt
```

Human-readable text reports generated by the ETL pipeline after each run. They provide a quick overview of dataset dimensions, status distributions, column breakdowns, and key statistics without needing to load the full parquet files.

**Properties:**

- **precleaned_summary.txt** – Covers all 57,675 sites (site_aggregated_precleaned output)
- **training_data_summary.txt** – Covers 26,099 Active training sites (site_training_data output)

### Contents

Both reports include:

- Dataset dimensions (rows, columns)
- Site status distribution with counts and percentages
- Column breakdown by type (log-transformed, one-hot encoded, metric, average)
- Generation timestamp

### Reasoning

These reports serve as a quick sanity check after running the ETL pipeline. If the site count, status distribution, or column count changes unexpectedly, the reports make it immediately visible without writing ad-hoc queries.

> **Note**
> Regenerated on every ETL run (`python3 -m site_scoring.data_transform`). Not versioned — always reflects the latest pipeline output.

---

# Feature Engineering

## features.RelativeStrength

Momentum indicators comparing recent performance to historical baseline.

**Formula:**

```
RS = (30-day average + ε) / (90-day average + ε)

Where:
- RS > 1.0: Trending upward (recent > historical)
- RS = 1.0: Stable performance
- RS < 1.0: Trending downward
- ε = 1.0 (smoothing constant)
```

### Attributes

- **rs_Impressions** (*float*) – Impression volume trend. Source: `monthly_impressions`
- **rs_NVIs** (*float*) – Network video impression trend. Source: `monthly_nvis`
- **rs_Revenue** (*float*) – Revenue trend. Source: `revenue`
- **rs_RevenuePerScreen** (*float*) – Per-screen efficiency trend. Source: `revenue / screen_count`

### Reasoning

1. **Why momentum matters** – A site trending upward may outperform its historical average in future months.
2. **Why 30/90 day windows** – 30 days captures recent changes; 90 days provides stable baseline.
3. **Why smoothing (ε=1.0)** – Prevents division by zero for new sites with limited history.

### Assumptions

- **Missing data** – Filled with 1.0 (neutral trend) if insufficient history.
- **Stationarity** – Assumes trends persist into the future.
- **Seasonality** – Not explicitly modeled (30/90 day comparison may capture some).

---

## features.LogTransforms

Signed log transformation applied to right-skewed distributions.

**Formula:**

```python
log_value = sign(x) * log(1 + |x|)
```

### Attributes

- **log_total_revenue** (*float*) – Range: 0.0–12.14. Source: `total_revenue`
- **log_total_monthly_impressions** (*float*) – Range: 0.0–14.76. Source: `total_monthly_impressions`
- **log_nearest_site_distance_mi** (*float*) – Range: 0.0–3.92. Source: `nearest_site_distance_mi`
- **log_min_distance_to_interstate_mi** (*float*) – Range: 0.0–7.27. Source: `min_distance_to_interstate_mi`

### Reasoning

1. **Right-skewed distributions** – Revenue and impressions follow power-law distributions with long right tails.
2. **Variance stabilization** – Log transform makes variance more uniform across the range.
3. **Outlier reduction** – Extreme values (very high-revenue sites) have less impact on gradients.
4. **Interpretability** – Coefficients represent multiplicative effects.

> **Note**
> Zero handling: `log(1 + x)` ensures `log(0) = 0`, not undefined. All transformed values are ≥ 0.

---

## features.BooleanEncoding

One-hot encoding of string boolean flags to numeric 0/1 values.

**Encoding Logic:**

```python
"Yes"     → 1
"No"      → 0
"Unknown" → 0  # Treated as missing capability
NULL      → 0
```

### Attributes

- **Capability flags (c_\*)** (*int*, 9 columns) – `c_emv_enabled_encoded`, `c_nfc_enabled_encoded`, etc.
- **Restriction flags (r_\*)** (*int*, 29 columns) – `r_lottery_encoded`, `r_government_political_encoded`, etc.
- **Sales flags** (*int*, 2 columns) – `schedule_site_encoded`, `sellable_site_encoded`

### Assumptions

- **Static flags** – Capabilities and restrictions are assumed constant over time (using most recent snapshot).
- **Unknown → 0** – Conservative assumption that missing capability = no capability. Separate from categoricals: boolean features get direct 0/1 encoding, not embedding.

---

## features.MissingValueHandling

| Data Type | Strategy | Rationale |
|-----------|----------|-----------|
| Numeric | Fill with 0 | Neutral after standardization |
| Categorical | Fill with `"__MISSING__"` | Creates explicit "unknown" category |
| Boolean | Fill with 0 | Assumes missing = False |
| Target | Fill with median | Preserves target distribution |

### High-Missing Columns

- **c_emv_enabled_encoded** – 91.7% missing. Legacy feature, not tracked historically.
- **c_walk_up_enabled_encoded** – 98.1% missing. Rare capability, seldom recorded.

---

## features.OutlierTreatment

Percentile clipping (Winsorization) applied during `DataProcessor._process_numeric()`.

```python
# Step 1: Clip to 1st-99th percentile
numeric_array = np.clip(col_data, p1, p99)

# Step 2: Standardize (zero mean, unit variance)
numeric_scaled = StandardScaler().fit_transform(numeric_array)

# Step 3: Final clip to ±10 standard deviations
numeric_scaled = np.clip(numeric_scaled, -10, 10)
```

### Reasoning

1. **Preserves information** – Unlike removal, clipping keeps extreme cases in training.
2. **Gradient stability** – Prevents extreme values from dominating loss gradients.
3. **Conservative bounds** – ±10σ is generous; true outliers are ~0.01% of data.

---

# Assumptions & Reasoning

## Data Aggregation

- **Most Recent Metadata** – Sites rarely change network/retailer. Risk: may miss recent transitions.
- **Revenue Summing** – Monthly revenues are independent. Risk: may include test periods.
- **Active Months = Row Count** – Proxy for site tenure. Risk: doesn't capture partial months.
- **GTVID Uniqueness** – IDs are stable over time. Risk: some sites may have ID changes.

## Feature Engineering

- **Relative Strength** – 30/90 day windows capture trends. Risk: may miss longer cycles.
- **Log Distances** – All sites have valid coordinates. Risk: 0.04% missing.
- **Demographics** – Census block group ≈ site location. Risk: ~500m radius approximation.
- **Capabilities** – Static over time. Risk: sites may add EMV/NFC.

## Model Training

- **Data Leakage Prevention** – Exclude target-derived features. `avg_monthly_revenue` target → exclude `total_revenue`.
- **Sufficient History** – Filter to `active_months > 11` ensures >1 year of data.
- **Active Sites Only** – Filter to `status == 'Active'`. Predict for operational sites.
- **Class Balance** – `pos_weight=9.0` for lookalike. Compensates for ~90/10 imbalance.

### Feature Selection Rationale

*Included:*

- **Momentum (RS) features** – Capture trajectory, not just level.
- **Log distances** – Normalize skewed distributions.
- **Demographics** – Area-level proxies for customer base.
- **Capabilities** – Direct business relevance (payment types, products).

*Excluded:*

- **retailer** – High cardinality, may overfit to specific chains.
- **pct_male** – Collinear with `pct_female` (always sum to ~100%).
- **nearest_interstate** – Categorical with 200+ values, may not generalize.

---

# Data Quality & Limitations

## Coverage

- **Sites with nearest_site_distance** – 99.96% (57,651 / 57,675)
- **Sites with interstate_distance** – 99.96%
- **Active sites in training** – 45.3% (26,099 / 57,675)
- **Months per site** – 1–47 (median: 28)

## Known Issues

1. **Column Typo** – Source data has `statuis` instead of `status`.
2. **Boolean Strings** – Flags stored as `"Yes"` / `"No"` strings, not native booleans.
3. **EMV/Walk-up Sparsity** – >90% missing values for these capabilities.
4. **Deactivated Sites** – 40.5% of sites are "Temporarily Deactivated" and excluded from training.

## Temporal Limitations

1. **No Intra-Month Data** – Daily patterns not captured in monthly aggregates.
2. **Seasonality** – Not explicitly modeled; 30/90 day RS may partially capture.
3. **COVID Impact** – Data likely spans pre/during/post-COVID periods with different patterns.

## Geographic Limitations

1. **US Only** – All sites are in the United States.
2. **Census Data Vintage** – Demographics may lag current population.
3. **Highway Network Changes** – New interstates not reflected if shapefiles are outdated.

---

# Quick Reference

## Key Files

- **`Site Scores - ...Diagnostics.csv`** (*1.47M rows*) – Primary source (monthly)
- **`site_training_data.parquet`** (*26,099 rows*) – ML-ready (Active only)
- **`site_aggregated_precleaned.parquet`** (*57,675 rows*) – Intermediate (all sites)
- **`nearest_site_distances.csv`** (*67,644 rows*) – Geospatial feature
- **`site_interstate_distances.csv`** (*70,300 rows*) – Highway distances

## Key Code

- **`src/services/data_service.py`** – Data loading API
- **`site_scoring/data_transform.py`** – ETL pipeline
- **`site_scoring/data_loader.py`** – PyTorch data preparation
- **`site_scoring/config.py`** – Feature definitions & model configuration

## Feature Counts

- **Numeric** (*int*, 12) – `rs_Revenue`, `avg_household_income`, etc.
- **Categorical** (*int*, 8) – `network`, `retailer`, `nearest_interstate`, etc.
- **Boolean** (*int*, 40) – `c_nfc_enabled_encoded`, `r_lottery_encoded`, etc.
- **Total** – 60

## Target Variables

- **avg_monthly_revenue** – Regression task. Metrics: MAE, SMAPE, R²
- **total_revenue** – Regression task. Metrics: MAE, RMSE
- **Top percentile (binary)** – Classification (Lookalike) task. Metrics: AUC, F1

---

# Data Provenance

## Internal Sources

- **Site Scores CSV** – GSTV operations database (monthly snapshots)
- **Salesforce Revenue** – Sales CRM system

## External Sources

- **Census TIGER/Line** – US Census Bureau (highway shapefiles)
- **Demographics** – US Census Bureau (block group data)

## Computed Features

- **Nearest Site Distance** – KDTree spatial search (haversine metric)
- **Interstate Distance** – Point-to-polyline geodesic distance
- **Relative Strength** – Rolling window calculations (30/90 days)

---

*Document generated: 2026-01-28 | Data vintage: ~47 months of operational history*
