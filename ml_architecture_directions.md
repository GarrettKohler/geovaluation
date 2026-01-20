# Machine Learning Architecture for DOOH Site Optimization

**For 60,000 gas station advertising sites, a three-model architecture combining similarity scoring, causal inference, and classification delivers both accurate predictions and actionable recommendations.** The key insight: traditional regression models fail because they predict revenue without explaining *why* or *what to change*. This research presents a production-ready framework using Gower distance for lookalike modeling, Double Machine Learning for causal attribution, and CatBoost for activation classification—each chosen for interpretability alongside performance.

The recommended approach prioritizes methods that answer business questions directly: "Which inactive sites resemble our top performers?" (lookalike), "What hardware changes will increase revenue?" (causal), and "Will this site succeed within a year?" (classification). Implementation requires careful temporal splitting to prevent data leakage and group-based validation to keep site observations together.

---

## Similarity modeling identifies high-potential sites through mixed-feature distance metrics

For identifying inactive sites resembling high-revenue performers, **Gower distance** emerges as the optimal approach for mixed continuous/categorical features. Unlike k-means (which requires encoding categorical variables), Gower natively handles your feature types: continuous geospatial metrics (distance to highway, POI proximity) alongside categorical hardware specifications.

```python
import gower
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

**K-prototypes** extends k-means for mixed data, combining squared Euclidean distance for numeric features with simple matching for categorical. The `gamma` parameter controls relative weighting—lower values reduce categorical feature dominance:

```python
from kmodes.kprototypes import KPrototypes

cat_indices = [df.columns.get_loc(c) for c in ['hardware_type', 'content_type']]
kproto = KPrototypes(n_clusters=5, init='Cao', gamma=0.5)
clusters = kproto.fit_predict(df.values, categorical=cat_indices)
```

For larger scale deployments, **entity embeddings** learned through neural networks capture semantic relationships between categorical values. FastAI's tabular module trains embeddings during revenue prediction, which can then be extracted for similarity calculations. The embedding dimension follows a heuristic: `min(600, round(1.6 * n_categories ** 0.56))`.

Cluster quality should be evaluated with **silhouette scores** (target >0.5) and **Davies-Bouldin index** (lower is better). Visualize clusters with UMAP to validate that high-scoring inactive sites cluster near established performers.

---

## Causal inference separates hardware effects from site selection bias

The core challenge—determining whether "good sites get good hardware" or "good hardware makes sites good"—requires causal methods that address confounding. Standard regression coefficients conflate correlation with causation; a site with LED displays may outperform LCD sites because LEDs were deployed at already-promising locations.

**Double Machine Learning (DML)** addresses this through a two-stage approach that removes confounding bias:

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

The **theoretical foundation**: DML first predicts the outcome and treatment from confounders, then regresses residualized outcomes on residualized treatment. This isolates the "as-if-random" variation in treatment assignment after controlling for observable confounders.

**Confounders (W)** should include everything affecting both hardware placement and revenue: historical site performance, traffic metrics, geographic indicators, competitor presence, and regional budgets. **Effect modifiers (X)** are features where you expect treatment effects to vary: urban vs. rural, mall vs. transit locations.

For multi-treatment scenarios (comparing LCD, LED, OLED, and interactive displays), **meta-learners** estimate relative effects:

```python
from causalml.inference.meta import BaseXRegressor
from lightgbm import LGBMRegressor

learner = BaseXRegressor(learner=LGBMRegressor(), control_name='LCD')
learner.fit(X=site_features, treatment=hardware_type, y=revenue)
tau = learner.predict(X=site_features)  # Effects vs LCD baseline
```

**DoWhy** provides crucial validation through refutation tests. Sensitivity analysis reveals how strong unobserved confounding would need to be to nullify your findings:

```python
import dowhy

model = CausalModel(data=df, treatment='hardware_type', outcome='revenue', 
                    graph=causal_graph_gml)
identified_estimand = model.identify_effect()

# Sensitivity analysis: How robust to unmeasured confounding?
refutation = model.refute_estimate(
    identified_estimand, estimate,
    method_name="add_unobserved_common_cause",
    effect_strength_on_treatment=[0.01, 0.05, 0.1],
    effect_strength_on_outcome=[0.01, 0.05, 0.1]
)
```

Report the **robustness value**: the minimum confounding strength that would nullify your effect estimate. If a confounder would need to explain 30% of residual variance to change conclusions, stakeholders can judge whether such strong unobserved confounding is plausible.

---

## Classification for activation prediction favors gradient boosting with class weights

For predicting first-year success from ~5,000 historical activations with 10-20% positive rate, **CatBoost with class weights** outperforms SMOTE-based resampling. Recent benchmarks show CatBoost's native categorical handling and ordered boosting provide 6-20% improvement on categorical-heavy datasets with limited samples.

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
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

**Evaluation metrics** must account for imbalance. Accuracy misleads—a model predicting all negatives achieves 85% accuracy with 15% positive rate. Use:

- **PR-AUC** (primary): Focuses on positive class performance
- **Matthews Correlation Coefficient**: Balanced single metric requiring threshold
- **Lift in top decile**: Business-relevant measure of prioritization quality

```python
from sklearn.metrics import average_precision_score, matthews_corrcoef

pr_auc = average_precision_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, (y_pred_proba >= optimal_threshold).astype(int))
```

**Probability calibration** ensures predicted probabilities reflect true likelihood—critical for business decisions:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
```

For inactive sites lacking transaction history, engineer **proxy features** from nearby active sites:

```python
def create_market_proxies(new_site, active_sites, radius_km=10):
    nearby = find_sites_in_radius(new_site, active_sites, radius_km)
    return {
        'nearby_avg_revenue': nearby['revenue'].mean(),
        'nearby_high_revenue_pct': (nearby['revenue'] > threshold).mean(),
        'nearby_site_count': len(nearby)
    }
```

**Survival analysis** provides an alternative when time-to-threshold matters or data includes right-censored observations (sites still being observed):

```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

y_surv = Surv.from_arrays(event=reached_threshold, time=months_to_threshold)
rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=5)
rsf.fit(X_train, y_surv)

# Predict probability of reaching threshold within 12 months
surv_funcs = rsf.predict_survival_function(X_test)
prob_high_revenue_12mo = 1 - np.array([fn(12) for fn in surv_funcs])
```

---

## Pipeline architecture prevents leakage through temporal and group-aware splitting

Your campaign and revenue data contains temporal dependencies requiring **time-based splits with gap periods**. Random splitting risks training on future campaigns and evaluating on past performance—a recipe for production failures.

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

**Group-based splitting** keeps all observations for a site in the same split, preventing the model from "memorizing" site-specific patterns:

```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in sgkf.split(X, y, groups=site_ids):
    # All observations for each site stay together
    pass
```

**Feature engineering timing** is the most common leakage source. The golden rule: only use data available at prediction time.

```python
# WRONG: Uses all-time average (includes future)
df['avg_revenue'] = df.groupby('site_id')['revenue'].transform('mean')

# CORRECT: Uses only past data via shift
df['revenue_rolling_30d'] = df.groupby('site_id')['revenue'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean()
)
```

Use **scikit-learn Pipeline** to ensure preprocessing fits only on training data:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Fit only on training data
    ('classifier', CatBoostClassifier())
])
pipeline.fit(X_train, y_train)  # Preprocessing learns from train only
```

Output train/val/test as separate files with reproducibility metadata:

```python
def save_splits(df, train_idx, val_idx, test_idx, output_dir='data/processed'):
    df.iloc[train_idx].to_parquet(f'{output_dir}/train.parquet')
    df.iloc[val_idx].to_parquet(f'{output_dir}/val.parquet')
    df.iloc[test_idx].to_parquet(f'{output_dir}/test.parquet')
    
    # Save split metadata
    metadata = {
        'train_sites': df.iloc[train_idx]['site_id'].unique().tolist(),
        'val_sites': df.iloc[val_idx]['site_id'].unique().tolist(),
        'test_sites': df.iloc[test_idx]['site_id'].unique().tolist(),
        'random_state': 42,
        'created': datetime.now().isoformat()
    }
    json.dump(metadata, open(f'{output_dir}/split_metadata.json', 'w'))
```

---

## Interpretability transforms predictions into actionable business recommendations

**SHAP's TreeExplainer** provides exact feature attributions for gradient boosting models, enabling both global importance rankings and individual site explanations:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Global importance: which features drive predictions overall
shap.plots.bar(shap_values)

# Individual site explanation (waterfall plot)
shap.plots.waterfall(shap_values[0])  # "Site #123: LED display adds $3K, loop_length adds $1.5K..."
```

**Critical limitation**: SHAP explains model predictions, not causal effects. Use SHAP for "why did the model predict this?" and CATE from causal models for "what would happen if we changed X?"

**DiCE counterfactuals** answer "what minimal changes would make this site high-revenue?" while constraining to controllable features:

```python
import dice_ml

exp = dice_ml.Dice(d, m, method="genetic")
counterfactuals = exp.generate_counterfactuals(
    query_instance=site_features,
    total_CFs=4,
    desired_class=1,  # High revenue
    features_to_vary=['display_technology', 'content_type', 
                      'loop_length', 'cpm_floor'],  # Only controllable
    permitted_range={
        'loop_length': [10, 60],
        'cpm_floor': [5, 50]
    }
)
```

For **causal recommendations**, convert CATE estimates to business language with confidence intervals:

```python
def generate_recommendation(site, cate_model):
    effect = cate_model.effect(site[features])[0]
    lb, ub = cate_model.effect_interval(site[features], alpha=0.05)
    
    if lb > 0:  # Confident positive effect
        return {
            'site_id': site['site_id'],
            'action': 'Upgrade to LED display',
            'expected_uplift': f'${effect:,.0f}/month',
            'confidence_interval': f'${lb:,.0f} to ${ub:,.0f}',
            'confidence': 'High' if lb > effect * 0.3 else 'Medium'
        }
```

---

## Complete implementation combines all models into a prioritization framework

The final architecture scores inactive sites on three dimensions—lookalike similarity, predicted success probability, and expected treatment uplift—producing ranked recommendations:

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

**Library summary** for the complete stack:

| Purpose | Libraries |
|---------|-----------|
| Core ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Causal inference | EconML (CausalForestDML, LinearDML), CausalML (meta-learners), DoWhy |
| Interpretability | SHAP, LIME, DiCE |
| Imbalanced data | imbalanced-learn, class weights in boosting libraries |
| Survival analysis | lifelines, scikit-survival |

```bash
pip install scikit-learn xgboost lightgbm catboost
pip install econml causalml dowhy
pip install shap lime dice-ml imbalanced-learn lifelines scikit-survival
```

---

## Conclusion

This three-model architecture addresses the fundamental limitation of regression-only approaches: prediction without prescription. **Gower distance-based lookalike modeling** identifies inactive sites resembling top performers without requiring extensive feature engineering for mixed types. **Double Machine Learning through CausalForestDML** isolates true hardware effects from site selection bias, enabling confident "change X to increase revenue by Y" recommendations with robustness bounds. **CatBoost with class weights** predicts activation success while handling categorical features natively—superior to SMOTE for your sample size.

The critical implementation details that determine production success: **temporal splitting with gap periods** prevents future data leakage, **group-aware cross-validation** keeps site observations together, and **Pipeline objects** ensure preprocessing fits only on training data. For interpretability, combine SHAP waterfall plots (explaining predictions) with DiCE counterfactuals (suggesting changes) and CATE confidence intervals (quantifying expected impact).

The combined priority score—weighting similarity, predicted success, and expected uplift—transforms these models into an actionable site ranking. Sites with high lookalike scores, strong success predictions, *and* confident positive treatment effects represent the highest-value activation targets, while the causal analysis provides specific hardware recommendations for each.