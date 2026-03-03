# site_scoring.data_loader

Data loading and PyTorch tensor preparation for the ML training pipeline.

Optimized for Apple Silicon (M4) by utilizing **Polars** for multi-threaded CSV/Parquet parsing and generating contiguous memory arrays for efficient transfer to the MPS (Metal Performance Shaders) backend.

---

## site_scoring.data_loader.SiteDataset

```python
class SiteDataset(numeric, categorical, boolean, target)
```

A custom PyTorch `Dataset` that holds data in memory as tensors. Stores data in four distinct tensor groups to maximize cache locality during training.

**Parameters:**

- **numeric** (*torch.Tensor*) – Scaled numeric features. Shape: `(N, n_numeric)`
- **categorical** (*torch.Tensor*) – Label-encoded categorical features. Shape: `(N, n_categorical)`
- **boolean** (*torch.Tensor*) – Binary boolean features. Shape: `(N, n_boolean)`
- **target** (*torch.Tensor*) – Target variable. Shape: `(N, 1)`

**Returns:**

- **tuple** – `(numeric, categorical, boolean, target)` for each index

---

## site_scoring.data_loader.DataProcessor

```python
class DataProcessor(config)
```

Handles the transformation of raw DataFrames into neural-network-ready tensors. Can save/load its state (scalers and encoders) via pickle to ensure consistency between training and inference.

**Parameters:**

- **config** (*Config*) – Model configuration containing feature lists, task type, and target column.

### Methods

**load_and_process** *(path)*

Loads a parquet/CSV file, fits scalers on the data, and returns four tensor arrays. Sets `_fitted = True` on completion.

- **Numeric Processing** – Applies `StandardScaler`, clips outliers (1st/99th percentile), handles NaN values.
- **Categorical Processing** – Uses `LabelEncoder` to convert string categories into integer indices for embedding layers.
- **Boolean Processing** – Standardizes various boolean formats (`True`/`False`, `1`/`0`, `"Yes"`/`"No"`) into float tensors.
- **Target Processing** – Scales the target variable using a separate `StandardScaler` (regression) or binarizes by percentile threshold (lookalike classification).

**save** *(path)*

Persists fitted state to a pickle file for inference.

- **Contents** – `{scaler, label_encoders, target_scaler, categorical_vocab_sizes}`

> **Note**
> The processor stores `source_gtvids`, `source_statuses`, and `source_revenues` during `load_and_process()` for post-training export of classification results.

---

## site_scoring.data_loader.create_data_loaders

```python
create_data_loaders(config, processor=None) → (train_loader, val_loader, test_loader, processor)
```

Primary entry point for the training pipeline. Creates a `DataProcessor` (or reuses one), loads and processes data, splits into train/val/test sets, and returns PyTorch `DataLoader` objects.

**Parameters:**

- **config** (*Config*) – Model configuration object.
- **processor** (*DataProcessor, optional*) – Pre-fitted processor. If `None`, a new one is created and fitted.

**Returns:**

- **train_loader** (*DataLoader*) – Training set batches
- **val_loader** (*DataLoader*) – Validation set batches
- **test_loader** (*DataLoader*) – Held-out test set batches
- **processor** (*DataProcessor*) – Fitted processor (for saving to `preprocessor.pkl`)

> **Note**
> Configures `num_workers`, `prefetch_factor`, and `pin_memory` based on hardware configuration in Config for M4 optimization.

**Example:**

```python
from site_scoring.data_loader import create_data_loaders
from site_scoring.config import Config

config = Config()
train_loader, val_loader, test_loader, processor = create_data_loaders(config)

for numeric, cat, bools, target in train_loader:
    # numeric.shape → [batch_size, n_numeric]
    # cat.shape     → [batch_size, n_categorical]
    # bools.shape   → [batch_size, n_boolean]
    # target.shape  → [batch_size, 1]
    pass

# Save for inference
processor.save("preprocessor.pkl")
```
