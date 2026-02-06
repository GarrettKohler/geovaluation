# Site Scoring Model: Artifact & Deployment Checklist

This checklist identifies the essential files and metadata required to move a trained model from development to a production or inference environment.

---

## 1. The Core Artifact Bundle
To run predictions on a new system, you MUST have these three files located in `site_scoring/outputs/`:

*   [ ] **`best_model.pt` (The Brain)**: 
    *   Contains the trained weights (tensors) of the neural network.
    *   Includes the serialized `Config` object (architecture, feature lists, and hyperparameters).
*   [ ] **`preprocessor.pkl` (The Translator)**: 
    *   Contains the `StandardScaler` for numeric features.
    *   Contains the `LabelEncoders` for categorical features.
    *   **CRITICAL**: Without this, the model cannot interpret raw input data (e.g., it won't know that "TX" maps to index 42).
*   [ ] **`feature_selection_summary.json` (The Map)**: 
    *   A record of which features were kept or eliminated during training.
    *   Essential for verifying that the input data matches the model's expected input shape.

---

## 2. Explainability & Validation (Optional but Recommended)
For high-trust environments, include these diagnostic artifacts:

*   [ ] **`shap_cache.npz`**: Stores the pre-calculated Shapley values for the training/test set to show feature importance immediately without re-calculating.
*   [ ] **`explainability/` Folder**: 
    *   `calibrator.pkl`: Ensures predicted probabilities (0.0 to 1.0) reflect real-world frequencies.
    *   `conformal.pkl`: Provides rigorous "confidence sets" (e.g., "We are 90% sure this site is a top performer").
*   [ ] **`training_history.json`**: Documentation of the loss curves and metrics (MAE, R²) achieved during development.

---

## 3. Environment & Runtime Requirements
Before loading the model, ensure the target environment meets these specs:

*   [ ] **Python 3.8+**
*   [ ] **PyTorch 2.0+**: (Preferably matching the version used during training).
*   [ ] **Hardware**: 
    *   If `device="mps"` was used, an Apple Silicon (M1-M4) chip is required for hardware acceleration.
    *   The `SiteScorer` will fallback to `cpu` if necessary, but inference will be slower.
*   [ ] **Dependencies**: `polars`, `numpy`, `scikit-learn` (for the preprocessor), and `torch`.

---

## 4. Deployment Verification Steps
Once the files are moved, run these checks:

1.  **Load Check**: Instantiate `SiteScorer()` in `predict.py`. It should print "Model loaded successfully on [device]".
2.  **Schema Check**: Pass a single row of the original `site_training_data.parquet` through `scorer.predict()`. The output should match the expected revenue.
3.  **Unknown Category Test**: Pass a categorical value the model has never seen (e.g., a new Retailer). The `SiteScorer` should handle this by mapping it to index `0` (Unknown) rather than crashing.

---

## 5. Lifecycle Management
*   **Version Control**: Tag your `.pt` files with the date or a version number (e.g., `best_model_v1_2026-02-04.pt`).
*   **Retraining Trigger**: If the **R² score** on new monthly data drops by more than 15%, or if **SMAPE** exceeds 25%, the model is likely experiencing "Data Drift" and requires a new training cycle.
