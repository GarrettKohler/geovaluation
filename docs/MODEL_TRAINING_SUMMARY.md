# Site Scoring Model Training: E2E Workflow Summary

## Overview
The `site_scoring` pipeline is a high-performance Machine Learning system optimized for Apple Silicon (M4). It transitions from raw geospatial site data to a production-ready PyTorch model using a Residual MLP architecture.

## 1. Step A: Configuration (`config.py`)
The "Brain" of the operation. It defines the entire environment:
*   **Hardware Optimization**: Detects Apple M-series GPUs (`mps`) and scales batch sizes (4096) to maximize throughput.
*   **Feature Taxonomy**: Segregates data into **Numeric** (RS indicators, demographics), **Categorical** (Retailer, Network), and **Boolean** (40+ capability flags).
*   **Leakage Protection**: Automatically scrubs features that overlap with the target variable (e.g., removing `total_revenue` if predicting `avg_monthly_revenue`).
*   **Presets**: Allows switching between "Model A" (comprehensive) and "Model B" (curated/stable).

## 2. Step B: Data Loading & Preprocessing (`data_loader.py`)
*   **Polars Integration**: Uses high-speed Parquet/CSV scanning for sub-second data ingestion.
*   **Cleaning**: Filters out "unstable" sites with less than 11 months of history.
*   **Normalization**: Applies `StandardScaler` to numeric values and handles categorical embeddings.

## 3. Step C: Model Architecture (`model.py`)
*   **Residual MLP**: Uses skip-connections to allow for a deeper network (512 -> 256 -> 128 -> 64) without losing signal.
*   **Entity Embeddings**: Transforms categorical strings into 16-dimensional dense vectors, allowing the model to "learn" relationships between different retailers or regions.

## 4. Step D: Training & Optimization (`train.py`)
*   **Huber Loss**: Provides robustness against revenue outliers that would normally skew a standard MSE loss.
*   **Adaptive Learning**: Uses `ReduceLROnPlateau` to squeeze out performance when the model begins to converge.
*   **Early Stopping**: Prevents overfitting by monitoring validation loss and stopping training automatically.

## 5. Step E: Evaluation & Explainability
*   **Test Metrics**: Final validation against a 15% holdout set reporting MAE (Mean Absolute Error) and R².
*   **SHAP Integration**: (Optional) Calculates feature importance to explain *why* a specific site received its score.
