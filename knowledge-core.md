# Knowledge Core - Geospatial Site Analysis

> Accumulated learnings, patterns, architectural decisions, and ML strategy for this project.
> Updated automatically by Claude during development sessions.

---

## Table of Contents

1. [Discovered Patterns](#discovered-patterns)
2. [Architectural Decisions](#architectural-decisions)
3. [ML Strategy & Architecture](#ml-strategy--architecture)
4. [Code Conventions](#code-conventions)
5. [Edge Cases & Gotchas](#edge-cases--gotchas)

---

## Discovered Patterns

### Data Loading & Performance

**Polars over Pandas for Large CSV**
- Dataset: 1.47M rows of site scores
- Polars streaming scan is 10-20x faster than pandas
- Use `pl.scan_csv()` for lazy evaluation, collect only needed columns

**Module Singleton Caching**
- Data loaded once, cached at module level in `data_service.py`
- Avoids repeated CSV parsing on every API call
- Pattern: `_cached_df = None` at module top, lazy load in getter function

**Revenue Metrics Aggregation**
- Pre-compute `avgMonthlyRevenue`, `totalRevenue`, `revenueScore` at load time
- Score normalized to 0-1 using p10/p90 percentiles (robust to outliers)

### Frontend Architecture

**Hybrid Filtering Strategy**
- Categorical filters: Server-side via `/api/filtered-sites` (SQL-like efficiency)
- Range filters: Client-side JavaScript (instant response, no network latency)
- Intersection happens in browser for best UX

**WebGL Point Rendering**
- Leaflet.glify handles 57K+ points smoothly
- Points colored by revenue score (white → green gradient)
- Selected sites highlighted in blue, unselected in grey

**Lasso Selection Algorithm**
- Ray-casting point-in-polygon test
- SVG overlay for drawing path
- Disable map panning during lasso mode

### ML Pipeline

**Apple Silicon MPS Optimization**
- Batch sizes tiered by chip: M1=4096, M1 Pro=8192, M1 Max=16384, etc.
- Worker threads and prefetch also scale with GPU cores
- Auto-detection via `platform.processor()` and sysctl

**Feature Selection Hierarchy**
1. STG (Stochastic Gates) - During training, L0 regularization
2. LassoNet - Hierarchical L1 with skip connections
3. SHAP-Select - Post-training statistical significance
4. Can combine: `hybrid_stg_shap` preset

**Training Progress via SSE**
- Server-Sent Events for real-time epoch updates
- Includes: loss, MAE, R2, learning rate, active features
- Frontend updates progress bar and metric cards live

### API Design

**Site Details Categorization**
- 8 categories: Location, Site Info, Brands, Revenue, Demographics, Performance, Capabilities, Sales
- Single endpoint `/api/site-details/<id>` returns all categories
- Bulk endpoint `/api/bulk-site-details` for multi-select efficiency

**Highway Distance Calculation**
- Census TIGER/Line data auto-downloaded on first use (~15MB)
- Spatial index (STRtree) for fast nearest-line queries
- Result includes distance_miles and nearest_highway name

---

## Architectural Decisions

### ADR-001: Flask over FastAPI
- **Decision**: Use Flask for web framework
- **Context**: Simple REST API, no async requirements, team familiarity
- **Consequence**: Straightforward debugging, extensive ecosystem

### ADR-002: PyTorch over TensorFlow
- **Decision**: Use PyTorch for ML pipeline
- **Context**: Better Apple Silicon MPS support, dynamic graphs for debugging
- **Consequence**: Native MPS acceleration, easier model inspection

### ADR-003: Polars for Data Loading
- **Decision**: Use Polars instead of Pandas for large CSV operations
- **Context**: 1.47M row dataset, need fast startup
- **Consequence**: 10-20x faster load times, streaming capability

### ADR-004: Client-Side Range Filtering
- **Decision**: Range filters applied in browser, not server
- **Context**: Categorical filters benefit from server-side indexing; range filters need instant feedback
- **Consequence**: Better UX for slider-based filtering, reduced API calls

### ADR-005: SSE over WebSocket for Training
- **Decision**: Server-Sent Events for training progress
- **Context**: One-way data flow (server → client), simpler than WebSocket
- **Consequence**: Auto-reconnection, native browser support, simpler server code

### ADR-006: Three-Model ML Architecture
- **Decision**: Combine similarity scoring, causal inference, and classification
- **Context**: Traditional regression models fail because they predict revenue without explaining *why* or *what to change*
- **Consequence**: Answers business questions directly—lookalike identification, causal attribution, and activation prediction

---

## ML Strategy & Architecture

### Overview: Three-Model Framework

For 60,000 gas station advertising sites, this architecture delivers both accurate predictions and actionable recommendations:

| Model | Question Answered | Method |
|-------|-------------------|--------|
| **Similarity** | "Which inactive sites resemble top performers?" | Gower distance lookalike |
| **Causal** | "What hardware changes will increase revenue?" | Double Machine Learning |
| **Classification** | "Will this site succeed within a year?" | CatBoost with uncertainty |

### 1. Similarity Modeling (Lookalike Scoring)

**Gower distance** handles mixed continuous/categorical features natively—optimal for geospatial metrics alongside categorical hardware specs.

```python
import gower
import numpy as np

# Define high performers (top 25% by revenue)
threshold = df_active['revenue'].quantile(0.75)
high_performers = df_active[df_active['revenue'] >= threshold]

# Compute Gower distance from inactive sites to high performers
feature_cols = ['distance_to_highway', 'poi_density', 'traffic_volume',
                'hardware_type', 'content_type', 'market_region']

distances = gower.gower_matrix(
    df_inactive[feature_cols],
    high_performers[feature_cols]
)

# Score by average distance to k nearest high-performers
k = 10
k_nearest_dist = np.partition(distances, k, axis=1)[:, :k].mean(axis=1)
df_inactive['lookalike_score'] = 1 / (1 + k_nearest_dist)
```

**Alternatives:**
- **K-prototypes**: Extends k-means for mixed data; use `gamma=0.5` to reduce categorical dominance
- **Entity embeddings**: Neural network-learned categorical representations for larger scale

### 2. Causal Inference (Treatment Effect Estimation)

The core challenge: determining whether "good sites get good hardware" or "good hardware makes sites good."

**Double Machine Learning** removes confounding bias through two-stage residualization:

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Y: Revenue | T: Treatment (hardware type) | X: Effect modifiers | W: Confounders
est = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=200),
    model_t=GradientBoostingClassifier(n_estimators=200),
    discrete_treatment=True,
    n_estimators=500,
    min_samples_leaf=50,
    cv=5  # Cross-fitting prevents overfitting bias
)
est.fit(Y=revenue, T=hardware_type, X=site_features, W=confounders)

# Get treatment effects with confidence intervals
effects = est.effect(site_features)
lower, upper = est.effect_interval(site_features, alpha=0.05)
```

**Confounders (W)**: Historical performance, traffic, geography, competitor presence, regional budgets
**Effect modifiers (X)**: Urban vs. rural, mall vs. transit locations

**Robustness validation** with DoWhy sensitivity analysis:

```python
import dowhy

model = CausalModel(data=df, treatment='hardware_type', outcome='revenue',
                    graph=causal_graph_gml)
refutation = model.refute_estimate(
    identified_estimand, estimate,
    method_name="add_unobserved_common_cause",
    effect_strength_on_treatment=[0.01, 0.05, 0.1],
    effect_strength_on_outcome=[0.01, 0.05, 0.1]
)
```

Report the **robustness value**: minimum confounding strength to nullify the effect.

### 3. Classification with Uncertainty Quantification

**CatBoost with class weights** outperforms SMOTE for imbalanced data (~10-20% positive rate):

```python
from catboost import CatBoostClassifier
import numpy as np

scale_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    scale_pos_weight=scale_weight,
    cat_features=['hardware_type', 'content_category', 'market_region'],
    eval_metric='AUC',
    early_stopping_rounds=50
)
```

**Evaluation metrics** for imbalanced data:
- **PR-AUC** (primary): Focuses on positive class performance
- **Matthews Correlation Coefficient**: Balanced single metric
- **Lift in top decile**: Business-relevant prioritization quality

#### Conformal Prediction for Guaranteed Coverage

**MAPIE** provides distribution-free coverage guarantee: true label falls within prediction set with probability ≥ 1-α.

```python
from mapie.classification import MapieClassifier

# Wrap trained CatBoost with MAPIE
mapie_clf = MapieClassifier(
    estimator=catboost_clf,
    cv="prefit",
    method="aps"  # Adaptive Prediction Sets - avoids empty sets
)
mapie_clf.fit(X_calibration, y_calibration)  # Requires held-out calibration data

# Generate prediction sets at 90% confidence
y_pred, y_sets = mapie_clf.predict(X_test, alpha=[0.10])
```

**Method selection:**
- **LAC**: Uses softmax scores directly
- **APS**: Guarantees non-empty sets (recommended for business)
- **RAPS**: Produces smaller, more actionable sets

#### Probability Calibration

Ensures "75% confident" means 75% historical success rate. Use manual isotonic regression (sklearn's `CalibratedClassifierCV` has CatBoost compatibility issues):

```python
from sklearn.isotonic import IsotonicRegression

y_proba_cal = model.predict_proba(X_cal)[:, 1]
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_proba_cal, y_cal)

y_proba_calibrated = calibrator.predict(model.predict_proba(X_test)[:, 1])
```

### 4. Counterfactual Explanations (Actionable Recommendations)

**DiCE with genetic search** generates "what would need to change" recommendations with explicit feature constraints:

```python
import dice_ml

# Define actionable vs immutable features
ACTIONABLE = ['operational_hours', 'screen_size_sqm', 'content_category', 'maintenance_visits']
# Immutable: site location, building type, demographics, installation date

d = dice_ml.Data(
    dataframe=train_df,
    continuous_features=['operational_hours', 'screen_size_sqm', 'maintenance_visits'],
    outcome_name='is_premium_site'
)
m = dice_ml.Model(model=catboost_model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="genetic")  # genetic required for tree models

counterfactuals = exp.generate_counterfactuals(
    query_instance=site_to_evaluate,
    total_CFs=5,
    desired_class=1,
    features_to_vary=ACTIONABLE,
    permitted_range={
        'operational_hours': [current_hours, 24],  # Can only increase
        'screen_size_sqm': [current_size, 100],
        'maintenance_visits': [current_visits, 12]
    }
)
```

**Fleet-wide patterns** via counterfactual clustering:

```python
from sklearn.cluster import KMeans

# Generate counterfactuals for all low-value sites
all_changes = []
for site in low_value_sites:
    cfs = exp.generate_counterfactuals(site, total_CFs=3, features_to_vary=ACTIONABLE)
    for cf_row in cfs.cf_examples_list[0].final_cfs_df.values:
        change_vector = cf_row[:-1] - site.values
        all_changes.append(change_vector)

# Cluster to find common upgrade patterns
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(np.array(all_changes))
upgrade_patterns = pd.DataFrame(kmeans.cluster_centers_, columns=ACTIONABLE)
```

Each cluster center = typical intervention pattern (e.g., "extend hours by 4+" vs "upgrade screen + maintenance").

### 5. SHAP for Feature Attribution

**Critical distinction**: SHAP explains *model predictions*, not *causal effects*. Use SHAP for "why did the model predict this?" and CATE for "what would happen if we changed X?"

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Global importance
shap.plots.bar(shap_values)

# Individual site explanation
shap.plots.waterfall(shap_values[0])  # "LED display adds $3K, loop_length adds $1.5K..."
```

### 6. Data Pipeline: Preventing Leakage

**Temporal splitting with gap periods**:

```python
def temporal_split_with_gap(df, date_col='campaign_date',
                            train_end='2024-06-15',
                            val_start='2024-07-01',  # 2-week gap
                            val_end='2024-09-15',
                            test_start='2024-10-01'):
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]
    test = df[df[date_col] >= test_start]
    return train, val, test
```

**Group-based splitting** keeps site observations together:

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in sgkf.split(X, y, groups=site_ids):
    pass  # All observations for each site stay in same split
```

**Feature engineering golden rule**: Only use data available at prediction time.

```python
# WRONG: Uses all-time average (includes future)
df['avg_revenue'] = df.groupby('site_id')['revenue'].transform('mean')

# CORRECT: Uses only past data via shift
df['revenue_rolling_30d'] = df.groupby('site_id')['revenue'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
)
```

### 7. Executive Communication

**Categorical tiers** replace raw probability scores:

| Tier | Model Score | Business Label | Historical Accuracy | Action |
|------|-------------|----------------|---------------------|--------|
| 1 | >0.85 | **Recommended** | 88% succeeded | Proceed to contract |
| 2 | 0.65–0.85 | **Promising** | 76% accuracy | Site visit required |
| 3 | 0.50–0.65 | **Review Required** | 62% accuracy | Detailed assessment needed |
| 4 | <0.50 | **Not Recommended** | N/A | Do not pursue |

**Language patterns:**
- Instead of "the model predicts 0.78 probability," say "**this site has characteristics of our successful locations—similar sites in this confidence range succeed about 8 out of 10 times**"
- Frame counterfactuals as "upgrade paths" not "counterfactual explanations"

### 8. Combined Prioritization Framework

```python
def prioritize_sites(inactive_df, lookalike_model, classifier, cate_model):
    scores = inactive_df.copy()
    X = inactive_df[feature_cols]

    # Three scoring dimensions
    scores['lookalike_score'] = lookalike_model.score(X)
    scores['success_prob'] = classifier.predict_proba(X)[:, 1]
    scores['expected_uplift'] = cate_model.effect(X)

    # Confidence from causal model
    lb, _ = cate_model.effect_interval(X, alpha=0.05)
    scores['uplift_confident'] = lb > 0

    # Combined priority (weights adjustable per business need)
    scores['priority'] = (
        0.2 * scores['lookalike_score'].rank(pct=True) +
        0.3 * scores['success_prob'].rank(pct=True) +
        0.5 * scores['expected_uplift'].rank(pct=True)
    )

    return scores.sort_values('priority', ascending=False)
```

### Library Summary

| Purpose | Libraries |
|---------|-----------|
| Core ML | scikit-learn, CatBoost, XGBoost, LightGBM |
| Causal inference | EconML (CausalForestDML), CausalML (meta-learners), DoWhy |
| Uncertainty | MAPIE (conformal prediction) |
| Interpretability | SHAP, DiCE |
| Survival analysis | lifelines, scikit-survival |

```bash
pip install scikit-learn catboost xgboost lightgbm
pip install econml causalml dowhy mapie
pip install shap dice-ml lifelines scikit-survival
```

---

## Code Conventions

### Python
- Type hints for all public functions
- Docstrings: Google style
- pytest for testing
- Keep Flask routes thin, delegate to `src/services/`

### JavaScript (Frontend)
- Vanilla JS (no React/Vue) - single `index.html` SPA
- Global state: `map`, `allSites`, `selectedSites`, `activeFilters`
- Event delegation for dynamic elements

### File Organization
```
src/services/     - Business logic (data, training, SHAP)
site_scoring/     - ML pipeline (model, data_loader, feature_selection)
templates/        - Flask templates (index.html)
data/input/       - Source CSVs
data/output/      - Generated results
```

---

## Edge Cases & Gotchas

### Data Quality
- Some sites have NULL revenue (exclude from score calculation)
- `statuis` column has typo (not `status`) - handle in code
- Boolean columns may have mixed types (string "True" vs boolean)

### MPS Training
- MPS doesn't support all PyTorch operations (fallback to CPU for unsupported)
- Memory pressure on smaller chips - reduce batch size if OOM
- `torch.mps.synchronize()` needed for accurate timing

### Highway Distance
- Alaska/Hawaii sites have no Interstate connections (handle gracefully)
- TIGER shapefile CRS is EPSG:4269 (NAD83), convert to EPSG:4326 for web

### Conformal Prediction
- Requires held-out calibration set (15-20% of data)
- sklearn's `CalibratedClassifierCV` has CatBoost compatibility issues—use manual isotonic regression

### DiCE Counterfactuals
- Genetic method required for tree models (non-differentiable)
- Processing ~500ms per site with genetic search
- For 1,000 sites, expect ~10 minutes for full counterfactual generation

---

## Historical

<!-- Archived patterns and decisions that are no longer active but preserved for reference -->

*No archived items yet.*

---

## Session Notes

<!-- Temporary notes from current development session. Review and promote to appropriate sections. -->

*Last updated: 2026-01-28*
