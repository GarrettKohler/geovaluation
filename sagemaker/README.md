# SageMaker Notebook Pipeline Documentation

> ML pipeline for site revenue prediction and lookalike classification on 60K+ gas station advertising sites.

## Pipeline Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  01_data_loading    │───▶│  02_preprocessing   │───▶│  03_model_training  │
│  (Raw CSVs → EDA)   │    │  (Tensors + Split)  │    │  (Neural Network)   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
                                                               │
┌─────────────────────┐    ┌─────────────────────┐             │
│  06_explainability  │◀───│  05_model_compare   │◀────────────┘
│  (SHAP + Tiers)     │    │  (XGBoost + NN)     │
└─────────────────────┘    └─────────────────────┘
                                     │
                           ┌─────────┴─────────┐
                           ▼                   ▼
                   ┌─────────────────┐  ┌─────────────────┐
                   │ 04_evaluation   │  │ xgboost_revenue │
                   │ (Test + Infer)  │  │ (Alt workflow)  │
                   └─────────────────┘  └─────────────────┘
```

---

## Notebook Details

### 01_data_loading_exploration.ipynb

**Purpose:** Load raw data, aggregate monthly records to site-level, join geospatial features, and perform exploratory data analysis.

#### Input Data

| File | Size | Description |
|------|------|-------------|
| `site_scores_revenue_and_diagnostics.csv` | 926.7 MB | Monthly revenue records (~500K rows) |
| `Sites - Base Data Set.csv` | 16.9 MB | Site metadata, capabilities, restrictions |
| `Site Revenue - Salesforce.csv` | 5.1 MB | Revenue summary from Salesforce |
| `nearest_site_distances.csv` | 6.5 MB | Distance to nearest GSTV site |
| `site_interstate_distances.csv` | 3.8 MB | Distance to nearest interstate |
| `site_kroger_distances.csv` | 2.7 MB | Distance to nearest Kroger |
| `site_mcdonalds_distances.csv` | 2.8 MB | Distance to nearest McDonald's |

#### Processing Steps

1. Load 7 raw CSV files using Polars (fast CSV parsing)
2. Aggregate monthly data to site-level (1 row per site)
3. Join geospatial distance features on `GTVID`
4. Filter to Active sites only (~26K of ~58K total)
5. Analyze target distribution (`avg_monthly_revenue`)
6. Compute feature correlations with target

#### Outputs

| File | Description |
|------|-------------|
| `outputs/processed_data.parquet` | Joined dataset (26K sites × 44 columns) |
| `outputs/data_summary.json` | Summary statistics and metadata |

#### Key Findings

- **~500K monthly records** aggregated to **~26K site-level rows**
- Target variable (`avg_monthly_revenue`): Mean $268, Median $176
- Top correlated feature: `active_months` (r=0.34)
- Class imbalance for lookalike: ~9:1 ratio

---

### 02_data_preprocessing.ipynb

**Purpose:** Feature engineering, scaling, encoding, and train/val/test split creation.

#### Input Data

| File | Description |
|------|-------------|
| `../data/processed/site_training_data.parquet` | ~22K sites with >11 months history |

#### Processing Steps

1. Define feature groups:
   - **Numeric (5-11):** `log_min_distance_to_interstate_mi`, `avg_household_income`, `median_age`, `pct_female`, `pct_male`
   - **Categorical (9):** `network`, `program`, `experience_type`, `hardware_type`, `retailer`, `brand_fuel`, `brand_restaurant`, `brand_c_store`, `nearest_interstate`
   - **Boolean (40):** Capability and restriction flags (`r_lottery_encoded`, `c_sells_beer_encoded`, etc.)
2. Apply StandardScaler to numeric features (clip to 1st-99th percentile)
3. Apply LabelEncoder to categorical features (creates vocabulary sizes)
4. Convert boolean features to Float32 tensors
5. Create train/val/test split (70/15/15)
6. Build PyTorch DataLoaders (batch_size=4096)

#### Outputs

| File | Description |
|------|-------------|
| `outputs/preprocessor.pkl` | Fitted StandardScaler, LabelEncoders, feature lists |
| `outputs/processed_data.pt` | PyTorch tensors + train/val/test indices |
| `outputs/model_config.json` | Architecture configuration for model building |

#### Model Configuration

```json
{
  "n_numeric": 5,
  "n_boolean": 40,
  "n_categorical": 9,
  "categorical_vocab_sizes": {
    "retailer": 9736,
    "brand_c_store": 8139,
    "nearest_interstate": 227,
    "brand_restaurant": 168,
    "brand_fuel": 139
  },
  "task_type": "regression",
  "target": "avg_monthly_revenue"
}
```

---

### 03_model_training.ipynb

**Purpose:** Define and train the neural network model for revenue prediction.

#### Input Data

| File | Description |
|------|-------------|
| `outputs/processed_data.pt` | Preprocessed PyTorch tensors |
| `outputs/model_config.json` | Model architecture configuration |
| `outputs/preprocessor.pkl` | For target inverse transform |

#### Model Architecture

```
SiteScoringModel (1,146,515 parameters)
├── CategoricalEmbedding
│   ├── network: Embedding(5, 4)
│   ├── program: Embedding(14, 7)
│   ├── retailer: Embedding(9737, 16)
│   ├── brand_c_store: Embedding(8140, 16)
│   └── ... (9 total, output_dim=104)
├── BatchNorm1d(5) for numeric features
├── ResidualBlock [149→512]
├── ResidualBlock [512→256]
├── ResidualBlock [256→128]
├── ResidualBlock [128→64]
└── Linear [64→1] (output)
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 500 (with early stopping) |
| Learning Rate | 1e-4 |
| Batch Size | 4096 |
| Loss Function | HuberLoss (delta=1.0) |
| Optimizer | AdamW (weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Early Stopping | Patience=10 |

#### Outputs

| File | Description |
|------|-------------|
| `outputs/best_model.pt` | Model weights + config checkpoint |
| `outputs/training_curves.png` | Loss/metric visualizations |
| `outputs/training_summary.json` | Final training metrics |

#### Typical Results

| Metric | Value |
|--------|-------|
| Test MAE | ~$132 |
| Test R² | ~0.46 |
| Training Time | ~15 seconds (MPS) |

---

### 04_evaluation_inference.ipynb

**Purpose:** Detailed model evaluation, batch inference, and predictions for inactive sites.

#### Input Data

| File | Description |
|------|-------------|
| `outputs/best_model.pt` | Trained model checkpoint |
| `outputs/preprocessor.pkl` | For inverse transform |
| `outputs/processed_data.pt` | Test set tensors |
| `../data/processed/site_aggregated_precleaned.parquet` | Full dataset (all sites) |

#### Processing Steps

1. Load model and generate test set predictions
2. Calculate comprehensive metrics (MAE, RMSE, SMAPE, R², Median AE)
3. Create Actual vs Predicted scatter plots
4. Analyze errors by revenue range
5. Build production-ready `SiteScorer` class
6. **Predict revenue for ALL inactive sites** (~31K sites)

#### Outputs

| File | Description |
|------|-------------|
| `outputs/evaluation_plots.png` | Performance visualizations |
| `outputs/evaluation_results.json` | Detailed test metrics |
| `outputs/inactive_site_predictions.csv` | Predictions for 31,574 inactive sites |
| `outputs/high_value_inactive_sites.csv` | Top 10% reactivation candidates (3,158 sites) |

#### Inactive Site Predictions Summary

| Status | Count | Mean Predicted | Total Potential |
|--------|-------|----------------|-----------------|
| Temporarily Deactivated | 23,374 | $210/month | $4.9M/month |
| Awaiting Installation | 4,417 | $151/month | $668K/month |
| Deactivated | 3,094 | $307/month | $951K/month |
| Awaiting Reactivation | 627 | $187/month | $118K/month |
| **Total** | **31,574** | **$211/month** | **$6.66M/month** |

---

### 05_model_comparison.ipynb

**Purpose:** Compare XGBoost regression with Neural Network classification (Lookalike model).

#### Input Data

| File | Description |
|------|-------------|
| `outputs/processed_data.pt` | Preprocessed tensors |
| `outputs/model_config.json` | Configuration |
| `outputs/preprocessor.pkl` | Scalers and encoders |

#### Part 1: XGBoost Regression

**Configuration:**
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
```

**Results:**

| Metric | Value |
|--------|-------|
| MAE | ~$130 |
| RMSE | ~$175 |
| R² | ~0.45 |
| SMAPE | ~45% |

#### Part 2: Neural Network Classification (Lookalike)

**Target:** Binary classification - Top 10% revenue sites vs. rest

**Results:**

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.85 |
| Accuracy | ~0.90 |
| Precision | ~0.50 |
| Recall | ~0.45 |
| F1 Score | ~0.47 |

#### Outputs

| File | Description |
|------|-------------|
| `outputs/xgboost_regression.json` | Trained XGBoost model |
| `outputs/lookalike_classifier.pt` | NN classifier checkpoint |
| `outputs/xgboost_regression_results.png` | XGBoost visualizations |
| `outputs/nn_classification_results.png` | Classification metrics |
| `outputs/model_comparison.json` | Side-by-side comparison |

#### Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Direct revenue predictions | XGBoost Regression |
| Understanding feature importance | XGBoost (gain-based) |
| Identifying potential high performers | Neural Network Classifier |
| Lead scoring and prioritization | Neural Network Classifier |

---

### 06_explainability.ipynb

**Purpose:** Probability calibration, business tier classification, and SHAP feature importance.

#### Input Data

| File | Description |
|------|-------------|
| `outputs/best_model.pt` | Trained model |
| `outputs/preprocessor.pkl` | Feature names and scalers |
| `outputs/processed_data.pt` | Validation data |

#### Processing Steps

1. Create sklearn-compatible model wrapper (`SklearnModelWrapper`)
2. **Probability Calibration** with Isotonic Regression
3. **Tier Classification** based on calibrated probabilities
4. **SHAP Analysis** using KernelExplainer
5. Generate individual site explanations

#### Probability Calibration

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Brier Score | 0.1795 | 0.0280 | 84.4% |

#### Business Tier Classification

| Tier | Probability Threshold | Label | Recommended Action |
|------|----------------------|-------|-------------------|
| 1 | ≥75% | Premium | Invest: High priority for expansion and premium pricing |
| 2 | ≥50% | Strong | Optimize: Good performer with growth potential |
| 3 | ≥25% | Standard | Monitor: Average performance, watch for changes |
| 4 | <25% | Review | Flag: Below average, consider intervention or exit |

#### Top SHAP Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `log_total_revenue` | 0.1878 |
| 2 | `rs_Impressions` | 0.0219 |
| 3 | `rs_Revenue` | 0.0099 |
| 4 | `rs_NVIs` | 0.0094 |
| 5 | `c_open_24_hours_encoded` | 0.0080 |

#### Outputs

| File | Description |
|------|-------------|
| `outputs/calibrator.pkl` | Fitted isotonic calibrator |
| `outputs/tier_classifier.json` | Tier threshold definitions |
| `outputs/feature_importance.csv` | SHAP importance rankings |
| `outputs/calibration_diagram.png` | Reliability diagram |
| `outputs/tier_distribution.png` | Tier breakdown visualization |
| `outputs/shap_importance.png` | Feature importance plot |
| `outputs/explainability_summary.json` | Summary statistics |

---

## Complete Artifact Reference

### Data Artifacts

| File | Created By | Used By |
|------|------------|---------|
| `processed_data.parquet` | 01 | 02 |
| `processed_data.pt` | 02 | 03, 04, 05, 06 |
| `preprocessor.pkl` | 02 | 03, 04, 05, 06 |
| `model_config.json` | 02 | 03, 05 |

### Model Artifacts

| File | Created By | Description |
|------|------------|-------------|
| `best_model.pt` | 03 | Neural network checkpoint |
| `xgboost_regression.json` | 05 | XGBoost model |
| `lookalike_classifier.pt` | 05 | Classification model |
| `calibrator.pkl` | 06 | Probability calibrator |

### Evaluation Artifacts

| File | Created By | Description |
|------|------------|-------------|
| `training_summary.json` | 03 | Training metrics |
| `evaluation_results.json` | 04 | Test set metrics |
| `model_comparison.json` | 05 | Model comparison |
| `feature_importance.csv` | 06 | SHAP rankings |
| `inactive_site_predictions.csv` | 04 | Revenue predictions for 31K sites |

### Visualization Artifacts

| File | Created By |
|------|------------|
| `training_curves.png` | 03 |
| `evaluation_plots.png` | 04 |
| `xgboost_regression_results.png` | 05 |
| `nn_classification_results.png` | 05 |
| `calibration_diagram.png` | 06 |
| `tier_distribution.png` | 06 |
| `shap_importance.png` | 06 |

---

## Running the Pipeline

### Prerequisites

```bash
pip install torch polars pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### Execution Order

```bash
# 1. Data Loading & EDA
jupyter notebook 01_data_loading_exploration.ipynb

# 2. Preprocessing
jupyter notebook 02_data_preprocessing.ipynb

# 3. Model Training
jupyter notebook 03_model_training.ipynb

# 4. Evaluation & Inference
jupyter notebook 04_evaluation_inference.ipynb

# 5. Model Comparison (optional)
jupyter notebook 05_model_comparison.ipynb

# 6. Explainability (optional)
jupyter notebook 06_explainability.ipynb
```

### SageMaker Instance Recommendations

| Notebook | Recommended Instance | Notes |
|----------|---------------------|-------|
| 01-02 | `ml.t3.medium` | Data exploration |
| 03 | `ml.g4dn.xlarge` | GPU training |
| 04 | `ml.m5.large` | Batch inference |
| 05 | `ml.p3.2xlarge` | XGBoost + NN comparison |
| 06 | `ml.m5.xlarge` | SHAP analysis |

---

## SageMaker Deployment

To deploy as a SageMaker endpoint:

1. **Package model artifacts:**
   ```bash
   cd outputs
   tar -czvf model.tar.gz best_model.pt preprocessor.pkl model_config.json
   aws s3 cp model.tar.gz s3://bucket/site-scoring/model/
   ```

2. **Create inference script** (see `04_evaluation_inference.ipynb` for `SiteScorer` class)

3. **Deploy endpoint:**
   ```python
   from sagemaker.pytorch import PyTorchModel

   model = PyTorchModel(
       model_data='s3://bucket/site-scoring/model/model.tar.gz',
       role=role,
       framework_version='2.0.0',
       py_version='py310',
       entry_point='inference.py',
   )

   predictor = model.deploy(
       instance_type='ml.m5.large',
       initial_instance_count=1,
   )
   ```

---

## Key Business Insights

### Revenue Prediction Performance

- **MAE of ~$132** means predictions are typically within $132 of actual monthly revenue
- **R² of ~0.46** indicates the model explains 46% of revenue variance
- Best predictions are for mid-range revenue sites ($100-$400/month)

### Reactivation Opportunities

- **31,574 inactive sites** have been scored for potential reactivation
- **$6.66M/month** total predicted revenue opportunity (~$80M/year)
- **Top 10% candidates** (3,158 sites) have predicted revenue ≥$375/month
- **Deactivated** status shows highest mean predicted revenue ($307/month)

### Feature Importance

1. Historical revenue metrics dominate (`log_total_revenue`, `rs_Impressions`)
2. Location factors matter (`distance_to_interstate`, `nearest_site_distance`)
3. Demographics contribute (`avg_household_income`, `median_age`)
4. Capabilities provide signal (`c_open_24_hours`, `c_sells_beer`)

---

## Notes

- Notebooks are self-contained and can be run independently if artifacts exist
- GPU training is optional but recommended for faster experimentation
- SHAP analysis requires the `shap` package (install separately)
- macOS users should set `NUM_WORKERS=0` in DataLoaders to avoid fork issues
