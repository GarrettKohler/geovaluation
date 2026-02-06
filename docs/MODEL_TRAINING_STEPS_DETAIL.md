# Detailed Model Training Workflow: Step-by-Step Breakdown

This document provides a technical deep-dive into each stage of the `site_scoring` model training pipeline.

---

## Step A: Configuration (`config.py`)
**Objective**: Establish the "Source of Truth" for features, hardware optimization, and experiment logic.

*   **Dynamic Path Resolution**: Automatically identifies the project root and points to aggregated Parquet data (`site_training_data.parquet`), ensuring consistent data access across different environments.
*   **Hardware Profiling**: Auto-detects Apple Silicon (`mps`) or CUDA. It configures high-throughput settings for the M4 GPU, including a `batch_size` of 4096 and optimized `num_workers`.
*   **Feature Taxonomy**:
    *   **Numeric**: Includes "Relative Strength" (RS) indicators—multi-horizon momentum features (3, 6, 12, 24 months) that compare recent performance against historical baselines.
    *   **Categorical**: Flags high-cardinality metadata (Retailer, Network, Program) for learned embedding representations.
    *   **Boolean**: Maps 40+ domain-specific restriction (`r_`) and capability (`c_`) flags.
*   **Leakage Prevention**: A `__post_init__` hook automatically scrubs any feature that could "leak" the answer into the training set (e.g., if predicting `avg_monthly_revenue`, it removes `total_revenue` and other highly correlated targets).
*   **Model Presets**: Provides `model_a` (kitchen sink) and `model_b` (curated/stable) presets to allow for rapid A/B testing of feature sets.

---

## Step B: Data Loading & Preprocessing (`data_loader.py`)
**Objective**: Rapidly ingest data and transform it into optimized GPU-ready tensors.

*   **Polars Engine**: Uses the Polars `scan_csv` or `read_parquet` engine for sub-second ingestion of large datasets, bypassing the overhead of standard Pandas.
*   **Performance Stability Filter**: Automatically excludes sites with less than 11 active months of history. This ensures the model learns from sites with established, representative performance profiles.
*   **Robust Scaling**:
    *   Numeric features are clipped at the 1st and 99th percentiles to mitigate the impact of extreme outliers.
    *   `StandardScaler` is applied to bring all numeric inputs to a zero-mean, unit-variance scale.
*   **Target Processing**: 
    *   **Regression**: Continuous targets are scaled.
    *   **Classification**: Revenue is binarized based on the 90th percentile threshold (identifying the top 10% "Lookalike" performers).
*   **Tensor Layout**: Data is converted into **contiguous memory tensors**, which is critical for maximum performance on Apple's unified memory architecture.

---

## Step C: Model Architecture (`model.py`)
**Objective**: Build a flexible, deep neural network capable of capturing complex geospatial and attribute-based interactions.

*   **Residual MLP Design**: The core architecture utilizes `ResidualBlocks`. Each block includes:
    *   Linear Layers + Batch Normalization + ReLU Activation + Dropout.
    *   **Skip Connections**: Adds the original input back to the output of the block, allowing the network to train much deeper (512 -> 256 -> 128 -> 64 layers) without the signal degrading.
*   **Entity Embeddings**: Uses a `CategoricalEmbedding` layer that learns a dense, multi-dimensional representation for every unique category (e.g., specific Retailers). This allows the model to learn that "Shell" might be more similar to "Chevron" than to an independent network.
*   **Weight Initialization**: Employs **Xavier/Glorot Initialization** for linear layers to maintain stable variance of activations through the deep stack, and specialized Normal initialization for embeddings.
*   **Clustering Support**: Includes a `ClusteringModel` (Deep Embedded Clustering) that can segment top performers into distinct "tiers" using a Student’s t-distribution kernel.

---

## Step D: Training & Optimization (`train.py`)
**Objective**: Execute the learning process with stability and precision.

*   **Huber Loss**: Utilizes Huber Loss (with delta=1.0) which acts like MSE (squared error) for small errors but like MAE (absolute error) for large ones. This makes the training extremely robust to outliers in site revenue.
*   **AdamW Optimizer**: Uses Adam with Weight Decay, providing better regularization than standard Adam by decoupling the weight decay from the gradient update.
*   **Adaptive Learning Rates**: The `ReduceLROnPlateau` scheduler monitors validation loss. If progress stalls, it automatically cuts the learning rate by 50% to "fine-tune" the weights.
*   **Gradient Clipping**: Forces gradients to a maximum norm of 1.0, preventing "exploding gradients" which can often happen in deep tabular networks.
*   **Early Stopping**: A patience-based system (10 epochs) monitors for overfitting, stopping the process and reverting to the `best_model_state` if validation performance begins to degrade.

---

## Step E: Evaluation & Explainability
**Objective**: Verify the model's predictive power and understand its decision-making.

*   **Inverse Transformation**: Predictions are automatically converted from their scaled "tensor" values back into original dollar amounts for interpretable error reporting.
*   **Standard Metrics**: Calculates **MAE (Mean Absolute Error)** and **R² (Coefficient of Determination)** on a completely unseen 15% test set.
*   **Feature Selection Integration**: (via `feature_selection/`)
    *   **Stochastic Gates (STG)**: Can be enabled to learn which features are redundant and "shut them off" during training.
    *   **SHAP Validation**: Uses Shapley values to rank features by their actual contribution to the final prediction, allowing users to see exactly which site attributes drive revenue.
*   **Artifact Export**: Saves a complete package—`best_model.pt` (weights), `preprocessor.pkl` (scalers/encoders), and `config` metadata—enabling one-click deployment for inference.
