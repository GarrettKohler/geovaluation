# **Parquet-to-Model Pipeline: Complete Code Trace**

## **Overview**

This document traces every line of code that operates on `site_training_data.parquet` (26,096 Active sites, 114 columns) for all three model training paths:

| Model | Task | Loss Function | Target Format |
|-------|------|---------------|---------------|
| **MLP Neural Network** | Classification (lookalike) | `BCEWithLogitsLoss(pos_weight=9.0)` | Binary 0/1 |
| **MLP Neural Network** | Regression (revenue) | `HuberLoss(delta=1.0)` | StandardScaler-normalized |
| **XGBoost** | Regression (revenue) | `reg:squarederror` | Raw dollar values |

All three paths share the same data loading and feature processing pipeline. They diverge at target preparation and model training.

---

## **Layer 1: Configuration — `site_scoring/config.py`**

### **Default Data Path and Target**

```python
# site_scoring/config.py:16-20

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "site_training_data.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "site_scoring" / "outputs"
```

### **Feature Definitions**

The `Config` dataclass defines exactly which columns from the parquet become model features:

```python
# site_scoring/config.py:24-134

@dataclass
class Config:
    # Paths
    data_path: Path = field(default_factory=lambda: DEFAULT_DATA_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    # Network filter: None = all networks, or "Gilbarco", "Speedway", "Wayne/Dover"
    network_filter: Optional[str] = None

    # Target variable
    target: str = "avg_daily_revenue"

    # Task type: "regression" (predict revenue) or "lookalike" (classify top performers)
    task_type: str = "regression"

    # Lookalike classifier percentile bounds
    lookalike_lower_percentile: int = 90
    lookalike_upper_percentile: int = 100

    # Standard deviation threshold mode (alternative to percentile)
    lookalike_threshold_mode: str = "percentile"  # "percentile" | "stddev"
    lookalike_lower_sigma: float = 1.0
    lookalike_upper_sigma: float = float('inf')

    # Device - auto-detect M4 MPS
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")

    # Data loading
    batch_size: int = 4096
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 16 Numeric Features
    numeric_features: List[str] = field(default_factory=lambda: [
        # Multi-horizon relative strength indicators (momentum features)
        "rs_NVIs_95_185", "rs_Revenue_95_185",
        "rs_NVIs_185_370", "rs_Revenue_185_370",
        "rs_NVIs_370_740", "rs_Revenue_370_740",
        # Revenue metrics
        "log_total_revenue",
        # Geospatial distances (log-transformed)
        "log_min_distance_to_nearest_site_mi", "log_min_distance_to_interstate_mi",
        "log_min_distance_to_kroger_mi", "log_min_distance_to_mcdonalds_mi",
        "log_min_distance_to_walmart_mi", "log_min_distance_to_target_mi",
        # Demographics
        "log_avg_household_income", "median_age",
        "pct_female",
    ])

    # 7 Categorical Features
    categorical_features: List[str] = field(default_factory=lambda: [
        "network", "program", "experience_type", "hardware_type", "retailer",
        "brand_fuel", "brand_c_store",
    ])

    # 33 Boolean Features
    boolean_features: List[str] = field(default_factory=lambda: [
        "r_retail_car_wash_encoded", "r_cpg_beverage_beer_oof_encoded",
        "r_cpg_beverage_beer_vide_encoded", "r_cpg_beverage_wine_oof_encoded",
        "r_cpg_beverage_wine_video_encoded", "r_finance_credit_cards_encoded",
        "r_cpg_cbd_hemp_ingestibles_non_thc_encoded",
        "r_cpg_non_food_beverage_cannabis_medical_encoded",
        "r_cpg_non_food_beverage_cannabis_recreational_encoded",
        "r_cpg_non_food_beverage_cbd_hemp_non_thc_encoded",
        "r_automotive_after_market_oil_encoded",
        "r_cpg_beverage_spirits_ooh_encoded", "r_cpg_beverage_spirits_video_encoded",
        "r_cpg_non_food_beverage_e_cigarette_encoded",
        "r_entertainment_casinos_and_gambling_encoded",
        "r_government_political_encoded", "r_automotive_electric_encoded",
        "r_recruitment_encoded", "r_restaurants_cdr_encoded", "r_restaurants_qsr_encoded",
        "r_retail_automotive_service_encoded", "r_retail_grocery_encoded",
        "r_retail_grocerty_with_fuel_encoded",
        "c_emv_enabled_encoded", "c_nfc_enabled_encoded", "c_open_24_hours_encoded",
        "c_sells_beer_encoded", "c_sells_diesel_fuel_encoded", "c_sells_lottery_encoded",
        "c_vistar_programmatic_enabled_encoded",
        "c_sells_wine_encoded",
        "schedule_site_encoded", "sellable_site_encoded",
    ])

    # Model architecture
    embedding_dim: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.2
    use_batch_norm: bool = True

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    early_stopping_patience: int = 10
```

### **Data Leakage Prevention**

The `__post_init__` method removes target-derived columns to prevent data leakage:

```python
# site_scoring/config.py:158-170

def __post_init__(self):
    self.output_dir.mkdir(parents=True, exist_ok=True)

    # Remove target and derived columns from features to prevent data leakage
    leakage_columns = {self.target}
    if self.target == "total_revenue":
        leakage_columns.update(["log_total_revenue", "avg_monthly_revenue",
                                "avg_daily_revenue", "log_avg_daily_revenue"])
    elif self.target == "avg_monthly_revenue":
        leakage_columns.update(["total_revenue", "log_total_revenue",
                                "avg_daily_revenue", "log_avg_daily_revenue"])
    elif self.target == "avg_daily_revenue":
        leakage_columns.update(["total_revenue", "log_total_revenue",
                                "avg_monthly_revenue", "log_avg_daily_revenue"])

    self.numeric_features = [f for f in self.numeric_features if f not in leakage_columns]
```

### **Model Presets**

Two preset configurations control which features each experiment uses:

```python
# site_scoring/config.py:278-293

MODEL_PRESETS: Dict[str, Dict] = {
    "model_a": {
        "name": "Train Model A",
        "description": "All available features from source datasets",
        "numeric": _MODEL_A_NUMERIC,      # 16 + demographics (pct_african_american, etc.)
        "categorical": _MODEL_A_CATEGORICAL,  # includes retailer
        "boolean": _ALL_BOOLEAN,
    },
    "model_b": {
        "name": "Train Model B",
        "description": "Curated features (no retailer, pct_male, nearest_interstate)",
        "numeric": _MODEL_B_NUMERIC,       # 16 features
        "categorical": _MODEL_B_CATEGORICAL,  # excludes retailer
        "boolean": _ALL_BOOLEAN,
    },
}
```

---

## **Layer 2: Data Loading & Feature Processing — `site_scoring/data_loader.py`**

### **DataProcessor: The Single Entry Point**

All model types share this processor. It reads the parquet, processes each feature type, and returns four tensors:

```python
# site_scoring/data_loader.py:47-136

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.categorical_vocab_sizes: Dict[str, int] = {}
        self._fitted = False
        self.n_numeric_features = 0
        self.n_boolean_features = 0

    def load_and_process(self):
        # Detect file type and load
        data_path = str(self.config.data_path)
        if data_path.endswith('.parquet'):
            df = pl.read_parquet(self.config.data_path)
        else:
            df = pl.scan_csv(
                self.config.data_path,
                infer_schema_length=10000,
                null_values=["", "NA", "null", "Unknown"],
            ).collect(streaming=True)

        # Filter by network if specified
        if self.config.network_filter:
            if self.config.network_filter == "Wayne/Dover":
                df = df.filter(pl.col('network').is_in(['Wayne', 'Dover']))
            else:
                df = df.filter(pl.col('network') == self.config.network_filter)

        # Filter to sites with sufficient history (more than 11 active months)
        if 'active_months' in df.columns:
            df = df.filter(pl.col('active_months') > 11)

        # Store site metadata for post-training exports
        self.source_gtvids = df['gtvid'].to_list() if 'gtvid' in df.columns else None

        # Process features
        numeric_data = self._process_numeric(df)
        categorical_data = self._process_categorical(df)
        boolean_data = self._process_boolean(df)
        target_data = self._process_target(df)

        self._fitted = True
        return numeric_data, categorical_data, boolean_data, target_data
```

### **Numeric Feature Processing**

Percentile clipping prevents outliers from dominating the scaler:

```python
# site_scoring/data_loader.py:138-175

def _process_numeric(self, df: pl.DataFrame) -> torch.Tensor:
    available = []
    for c in self.config.numeric_features:
        if c in df.columns:
            dtype = df[c].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                available.append(c)

    self.n_numeric_features = len(available)

    processed_cols = []
    for col in available:
        col_data = df[col].cast(pl.Float64).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
        processed_cols.append(col_data)

    numeric_array = (
        np.column_stack(processed_cols) if processed_cols
        else np.zeros((len(df), 1), dtype=np.float32)
    )

    # Clip extreme values to [1st, 99th] percentile
    for i in range(numeric_array.shape[1]):
        col_data = numeric_array[:, i]
        p1, p99 = np.percentile(col_data, [1, 99])
        numeric_array[:, i] = np.clip(col_data, p1, p99)

    # StandardScaler: mean=0, std=1
    numeric_scaled = self.scaler.fit_transform(numeric_array)

    # Final safety clip
    numeric_scaled = np.clip(numeric_scaled, -10, 10)

    return torch.from_numpy(np.ascontiguousarray(numeric_scaled, dtype=np.float32))
```

### **Categorical Feature Processing**

Each categorical column gets its own `LabelEncoder`. The vocab sizes are stored for the embedding layer:

```python
# site_scoring/data_loader.py:177-196

def _process_categorical(self, df: pl.DataFrame) -> torch.Tensor:
    available = [c for c in self.config.categorical_features if c in df.columns]

    encoded_cols = []
    for col in available:
        col_data = df.select(col).fill_null("__MISSING__").to_series().to_list()
        le = LabelEncoder()
        encoded = le.fit_transform(col_data)
        self.label_encoders[col] = le
        self.categorical_vocab_sizes[col] = len(le.classes_)
        encoded_cols.append(encoded)

    categorical_array = np.column_stack(encoded_cols).astype(np.int64)
    return torch.from_numpy(np.ascontiguousarray(categorical_array))
```

### **Boolean Feature Processing**

Handles native booleans, pre-encoded integers, and string representations:

```python
# site_scoring/data_loader.py:198-226

def _process_boolean(self, df: pl.DataFrame) -> torch.Tensor:
    available = [c for c in self.config.boolean_features if c in df.columns]
    self.n_boolean_features = len(available) if available else 1

    bool_cols = []
    for col in available:
        col_data = df.select(col).to_series()
        if col_data.dtype == pl.Boolean:
            values = col_data.fill_null(False).to_numpy().astype(np.float32)
        elif col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            values = col_data.fill_null(0).to_numpy().astype(np.float32)
        else:
            values = (
                col_data.fill_null("false")
                .str.to_lowercase()
                .is_in(["true", "1", "yes", "t"])
                .to_numpy()
                .astype(np.float32)
            )
        bool_cols.append(values)

    boolean_array = (
        np.column_stack(bool_cols) if bool_cols
        else np.zeros((len(df), 1), dtype=np.float32)
    )
    return torch.from_numpy(np.ascontiguousarray(boolean_array, dtype=np.float32))
```

### **Target Preparation — The Divergence Point**

This is where the pipeline splits based on task type:

```python
# site_scoring/data_loader.py:228-350

def _process_target(self, df: pl.DataFrame) -> torch.Tensor:
    target_col = df[self.config.target]
    median_val = target_col.median()
    if median_val is None:
        median_val = 0

    target_data = (
        target_col.fill_null(median_val).fill_nan(median_val)
        .to_numpy().astype(np.float32).reshape(-1, 1)
    )

    # Clip extreme target values (1st and 99th percentile)
    p1, p99 = np.percentile(target_data, [1, 99])
    target_data = np.clip(target_data, p1, p99)

    if self.config.task_type == "lookalike":
        # ═══════════════════════════════════════════════════════
        # CLASSIFICATION PATH: Binarize by percentile or stddev
        # ═══════════════════════════════════════════════════════
        threshold_mode = getattr(self.config, 'lookalike_threshold_mode', 'percentile')

        if threshold_mode == "stddev":
            lower_sigma = getattr(self.config, 'lookalike_lower_sigma', 1.0)
            upper_sigma = getattr(self.config, 'lookalike_upper_sigma', float('inf'))
            mean_val = float(np.mean(target_data))
            std_val = float(np.std(target_data))
            lower_threshold = mean_val + lower_sigma * std_val
            upper_threshold = (
                (mean_val + upper_sigma * std_val)
                if np.isfinite(upper_sigma) else float('inf')
            )
            self.top_performer_threshold = lower_threshold

            if np.isfinite(upper_threshold):
                binary_labels = (
                    (target_data >= lower_threshold) & (target_data <= upper_threshold)
                ).astype(np.float32)
            else:
                binary_labels = (target_data >= lower_threshold).astype(np.float32)

        else:
            # Percentile mode (default)
            lower_pct = getattr(self.config, 'lookalike_lower_percentile', 90)
            upper_pct = getattr(self.config, 'lookalike_upper_percentile', 100)
            lower_threshold = float(np.percentile(target_data, lower_pct))
            upper_threshold = (
                float(np.percentile(target_data, upper_pct))
                if upper_pct < 100 else float('inf')
            )
            self.top_performer_threshold = lower_threshold

            if upper_pct >= 100:
                binary_labels = (target_data >= lower_threshold).astype(np.float32)
            else:
                binary_labels = (
                    (target_data >= lower_threshold) & (target_data <= upper_threshold)
                ).astype(np.float32)

        self.target_scaler = None  # No inverse transform for binary labels
        return torch.from_numpy(np.ascontiguousarray(binary_labels, dtype=np.float32))

    else:
        # ═══════════════════════════════════════════════════════
        # REGRESSION PATH: Scale continuous target
        # ═══════════════════════════════════════════════════════
        target_scaled = self.target_scaler.fit_transform(target_data)
        return torch.from_numpy(np.ascontiguousarray(target_scaled, dtype=np.float32))
```

### **Train/Val/Test Split and DataLoader Creation**

Random permutation splits the data 70/15/15 and wraps each split in a PyTorch DataLoader:

```python
# site_scoring/data_loader.py:373-477

def create_data_loaders(config, processor=None):
    if processor is None:
        processor = DataProcessor(config)

    numeric, categorical, boolean, target = processor.load_and_process()

    n_samples = len(target)

    # Random split
    indices = torch.randperm(n_samples)
    n_train = int(n_samples * config.train_ratio)
    n_val = int(n_samples * config.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Store split indices for post-training export reconstruction
    processor.train_indices = train_idx.tolist()
    processor.val_indices = val_idx.tolist()
    processor.test_indices = test_idx.tolist()

    # Create SiteDataset per split
    train_dataset = SiteDataset(
        numeric[train_idx], categorical[train_idx],
        boolean[train_idx], target[train_idx],
    )
    val_dataset = SiteDataset(
        numeric[val_idx], categorical[val_idx],
        boolean[val_idx], target[val_idx],
    )
    test_dataset = SiteDataset(
        numeric[test_idx], categorical[test_idx],
        boolean[test_idx], target[test_idx],
    )

    # Cap batch_size to training set size
    effective_batch_size = min(config.batch_size, len(train_dataset))
    use_drop_last = len(train_dataset) > effective_batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=use_drop_last,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, ...)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, ...)

    return train_loader, val_loader, test_loader, processor
```

### **SiteDataset: The Tensor Container**

Each sample is a tuple of `(numeric, categorical, boolean, target)`:

```python
# site_scoring/data_loader.py:17-44

class SiteDataset(Dataset):
    def __init__(self, numeric_tensor, categorical_tensor, boolean_tensor, target_tensor):
        self.numeric = numeric_tensor
        self.categorical = categorical_tensor
        self.boolean = boolean_tensor
        self.target = target_tensor

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return (
            self.numeric[idx],       # float32 (n_numeric,)
            self.categorical[idx],   # int64 (n_categorical,)
            self.boolean[idx],       # float32 (n_boolean,)
            self.target[idx],        # float32 (1,)
        )
```

---

## **Layer 3: Preprocessor Persistence**

The fitted scaler and label encoders are saved alongside the model so inference uses identical transformations:

```python
# site_scoring/data_loader.py:352-370

def save(self, path: Path):
    with open(path, "wb") as f:
        # NOTE: pickle is used to match the project's existing model serialization pattern.
        # The preprocessor contains fitted sklearn objects (StandardScaler, LabelEncoder)
        # which require pickle for serialization.
        import pickle
        pickle.dump({
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "target_scaler": self.target_scaler,
            "categorical_vocab_sizes": self.categorical_vocab_sizes,
        }, f)
```

---

## **Summary: Data Flow Diagram**

```
site_training_data.parquet (26,096 sites, 114 cols)
    |
    +-- pl.read_parquet()
    +-- filter: network (optional)
    +-- filter: active_months > 11
    |
    +--- _process_numeric()
    |       -> cast Float64 -> fill_null(0) -> percentile clip [1%, 99%]
    |       -> StandardScaler.fit_transform() -> clip [-10, 10]
    |       -> float32 tensor (n_sites, 16)
    |
    +--- _process_categorical()
    |       -> fill_null("__MISSING__") -> LabelEncoder per column
    |       -> stores vocab_sizes for embedding layer
    |       -> int64 tensor (n_sites, 7)
    |
    +--- _process_boolean()
    |       -> handle bool/int/string types -> float 0.0/1.0
    |       -> float32 tensor (n_sites, 33)
    |
    +--- _process_target()
            -> fill_null(median) -> percentile clip [1%, 99%]
            |
            +-- REGRESSION: StandardScaler.fit_transform()
            |       -> float32 tensor (n_sites, 1)  [normalized]
            |
            +-- CLASSIFICATION: binarize by percentile or stddev
                    -> float32 tensor (n_sites, 1)  [0.0 or 1.0]

    ============================================

    torch.randperm(n) -> 70/15/15 split
        -> SiteDataset(numeric, categorical, boolean, target) per split
        -> DataLoader(batch_size=4096, shuffle=True for train)

    ============================================

    DISPATCH by model_type:
        +-- "neural_network" -> tensors consumed directly
        +-- "xgboost" -> _dataloaders_to_numpy() -> flat np.ndarray
```
