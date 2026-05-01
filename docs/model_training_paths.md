# **Model Training Paths: NN Classification, NN Regression, XGBoost**

## **Overview**

This document covers the three model training paths and their inference counterparts. All three share the data loading pipeline documented in `parquet_to_model_pipeline.md`.

| Path | Model | Task | Entry Point | Key Difference |
|------|-------|------|-------------|----------------|
| **A** | `SiteScoringModel` (MLP NN) | Lookalike classification | `run_training_logic()` inline loop | `BCEWithLogitsLoss`, binary target, sigmoid post-processing |
| **B** | `SiteScoringModel` (MLP NN) | Revenue regression | `run_training_logic()` inline loop | `HuberLoss`, scaled target, inverse-transform post-processing |
| **C** | `XGBoostModel` | Revenue regression | `_run_tree_training()` | Flat numpy arrays, raw dollar targets, TreeExplainer SHAP |

---

## **Model Architectures -- `site_scoring/model.py`**

### **SiteScoringModel (Paths A and B)**

The same architecture handles both classification and regression. Only the loss function and post-processing differ.

```python
# site_scoring/model.py:36-59

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions change
        self.projection = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        out = torch.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        return torch.relu(out + residual)
```

```python
# site_scoring/model.py:62-94

class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical features with learned representations."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.feature_names = list(vocab_sizes.keys())

        for name, vocab_size in vocab_sizes.items():
            # Embedding dimension based on cardinality
            dim = min(embedding_dim, (vocab_size + 1) // 2)
            dim = max(dim, 4)  # Minimum dimension
            self.embeddings[name] = nn.Embedding(vocab_size + 1, dim, padding_idx=0)

        # Total output dimension
        self.output_dim = sum(
            self.embeddings[name].embedding_dim for name in self.feature_names
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i, name in enumerate(self.feature_names):
            idx = x[:, i].clamp(0, self.embeddings[name].num_embeddings - 1)
            embeddings.append(self.embeddings[name](idx))
        return torch.cat(embeddings, dim=1)
```

```python
# site_scoring/model.py:96-227

class SiteScoringModel(nn.Module):
    """
    Neural network for site revenue/impression prediction.

    Architecture:
    1. Categorical features -> Embeddings -> Dense representation
    2. Numeric features -> BatchNorm -> Scaled representation
    3. Boolean features -> Direct concatenation
    4. Concatenate all -> Residual MLP blocks -> Output
    """

    def __init__(
        self,
        n_numeric: int,
        n_boolean: int,
        categorical_vocab_sizes: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]

        # Categorical embeddings
        self.cat_embedding = CategoricalEmbedding(categorical_vocab_sizes, embedding_dim)

        # Numeric feature normalization
        self.numeric_bn = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None
        self.n_numeric = n_numeric
        self.n_boolean = n_boolean

        # Calculate input dimension
        total_input_dim = self.cat_embedding.output_dim + n_numeric + n_boolean

        # Build residual MLP
        layers = []
        in_dim = total_input_dim
        for out_dim in hidden_dims:
            layers.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer: single scalar
        self.output = nn.Linear(hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, numeric, categorical, boolean) -> torch.Tensor:
        # Process categorical through embeddings
        cat_embedded = self.cat_embedding(categorical)

        # Normalize numeric features
        if self.numeric_bn is not None and self.n_numeric > 0:
            numeric = self.numeric_bn(numeric)

        # Concatenate all features
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        # Pass through MLP
        x = self.mlp(x)

        # Output: (batch, 1)
        return self.output(x)

    @classmethod
    def from_config(cls, config: Config, categorical_vocab_sizes: Dict[str, int]):
        return cls(
            n_numeric=len(config.numeric_features),
            n_boolean=len(config.boolean_features),
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
        )
```

### **XGBoostModel (Path C)**

A wrapper around the scikit-learn-compatible XGBoost API:

```python
# site_scoring/model.py:552-688

class XGBoostModel:
    def __init__(
        self,
        task_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        early_stopping_rounds: int = 50,
        verbosity: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.task_type = task_type
        self.feature_names = feature_names
        self.early_stopping_rounds = early_stopping_rounds
        self.is_fitted = False

        common_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'verbosity': verbosity,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'tree_method': 'hist',
            'enable_categorical': True,
        }

        if task_type == "regression":
            self.model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                **common_params
            )
        else:  # lookalike / classification
            self.model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                **common_params
            )

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            progress_callback=None, callbacks=None):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = 100

        # XGBoost 2.0+: set callbacks via set_params, not fit()
        if callbacks:
            self.model.set_params(callbacks=callbacks)

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type == "regression":
            raise ValueError("predict_proba not available for regression")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        importance = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importance):
            return {name: float(val) for name, val in zip(self.feature_names, importance)}
        return {i: float(val) for i, val in enumerate(importance)}
```

### **Factory Function**

```python
# site_scoring/model.py:690-747

def create_model(model_type, task_type, n_numeric, n_boolean,
                 categorical_vocab_sizes, feature_names=None, config=None, **kwargs):
    if model_type == "neural_network":
        if config is not None:
            return SiteScoringModel.from_config(config, categorical_vocab_sizes)
        return SiteScoringModel(
            n_numeric=n_numeric, n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes, **kwargs
        )
    elif model_type == "xgboost":
        return XGBoostModel(
            task_type=task_type, feature_names=feature_names,
            n_estimators=kwargs.get('epochs', 1000),
            learning_rate=kwargs.get('learning_rate', 0.03),
            max_depth=kwargs.get('max_depth', 6),
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
        )
    elif model_type == "clustering":
        return ClusteringModel(
            n_numeric=n_numeric, n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes, ...
        )
```

---

## **Training Orchestration -- `src/services/training_service.py`**

### **Entry Point: `run_training_logic()`**

This function receives a `TrainingJob` and a progress callback, creates the PyTorch config, loads data, and dispatches to the correct training path:

```python
# src/services/training_service.py:1476-1646

def run_training_logic(job: TrainingJob, report_callback):
    config = job.config

    # Create PyTorch config from user config
    pytorch_config = Config()
    pytorch_config.output_dir = job.output_dir
    pytorch_config.target = config.target
    pytorch_config.epochs = config.epochs
    pytorch_config.batch_size = config.batch_size
    pytorch_config.learning_rate = config.learning_rate
    pytorch_config.weight_decay = config.weight_decay
    pytorch_config.dropout = config.dropout
    pytorch_config.hidden_dims = config.hidden_layers
    pytorch_config.embedding_dim = config.embedding_dim
    pytorch_config.early_stopping_patience = config.early_stopping_patience
    pytorch_config.scheduler_patience = config.scheduler_patience
    pytorch_config.device = config.device
    pytorch_config.task_type = config.task_type
    pytorch_config.lookalike_lower_percentile = config.lookalike_lower_percentile
    pytorch_config.lookalike_upper_percentile = config.lookalike_upper_percentile
    pytorch_config.lookalike_threshold_mode = config.lookalike_threshold_mode
    pytorch_config.lookalike_lower_sigma = config.lookalike_lower_sigma
    pytorch_config.lookalike_upper_sigma = config.lookalike_upper_sigma
    pytorch_config.network_filter = config.network_filter

    # Apply model preset and optional user feature selection
    if config.model_preset:
        pytorch_config.apply_model_preset(config.model_preset)
        if config.selected_features:
            from site_scoring.config import filter_features_by_selection
            filtered = filter_features_by_selection(
                config.model_preset, config.selected_features
            )
            pytorch_config.numeric_features = filtered["numeric"]
            pytorch_config.categorical_features = filtered["categorical"]
            pytorch_config.boolean_features = filtered["boolean"]

    # Save config to experiment directory
    with open(job.output_dir / "config.json", "w") as f:
        cfg_dict = asdict(config)
        cfg_dict['output_dir'] = str(cfg_dict['output_dir'])
        cfg_dict['training_features'] = {
            'numeric': pytorch_config.numeric_features,
            'categorical': pytorch_config.categorical_features,
            'boolean': pytorch_config.boolean_features,
        }
        json.dump(_sanitize_for_json(cfg_dict), f, indent=2)

    # Load data (shared by all paths)
    train_loader, val_loader, test_loader, processor = create_data_loaders(pytorch_config)

    # DISPATCH based on model_type
    if config.model_type == "xgboost":
        _run_tree_training(
            job, config, pytorch_config, train_loader, val_loader,
            test_loader, processor, report_callback, start_time
        )
        return

    if config.task_type == "clustering":
        _run_clustering_training(...)
        return

    # Otherwise: Neural Network training (Paths A and B)
    # ... (continues below)
```

---

### **Path A and B: Neural Network Training (inline in `run_training_logic`)**

Both classification and regression use the same training loop. The only differences are the loss function and metric computation:

```python
# src/services/training_service.py:1664-1875

# Create model
model = SiteScoringModel(
    n_numeric=processor.n_numeric_features,
    n_boolean=processor.n_boolean_features,
    categorical_vocab_sizes=processor.categorical_vocab_sizes,
    embedding_dim=pytorch_config.embedding_dim,
    hidden_dims=pytorch_config.hidden_dims,
    dropout=pytorch_config.dropout,
    use_batch_norm=True,
)
model = model.to(torch.device(config.device))

# LOSS FUNCTION: The key difference between Path A and B
device = torch.device(config.device)
if config.task_type == "lookalike":
    # PATH A: Classification
    # pos_weight=9.0 compensates for ~10% positive class ratio (p90 threshold)
    pos_weight = torch.tensor([9.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    # PATH B: Regression
    # HuberLoss is robust to outliers (delta=1.0)
    criterion = nn.HuberLoss(delta=1.0)

optimizer = AdamW(model.parameters(), lr=pytorch_config.learning_rate,
                  weight_decay=pytorch_config.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                              patience=pytorch_config.scheduler_patience)

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# TRAINING LOOP (same structure for both paths)
for epoch in range(1, config.epochs + 1):
    if job.should_stop:
        return

    # --- Train ---
    model.train()
    total_train_loss = 0.0
    n_batches = 0

    for numeric, categorical, boolean, target in train_loader:
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        boolean = boolean.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(numeric, categorical, boolean)
        task_loss = criterion(predictions, target)

        loss = task_loss
        # Optional feature selection regularization
        if hasattr(model, 'get_regularization_loss'):
            fs_reg_loss = model.get_regularization_loss()
            loss += fs_reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += task_loss.item()
        n_batches += 1

    train_loss = total_train_loss / n_batches

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_targs = []

    with torch.no_grad():
        for numeric, categorical, boolean, target in val_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            boolean = boolean.to(device)
            target = target.to(device)
            preds = model(numeric, categorical, boolean)
            loss = criterion(preds, target)
            total_val_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targs.append(target.cpu())

    val_loss = total_val_loss / len(val_loader)
    predictions_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targs).numpy()

    # METRICS: Different computation per task type
    if config.task_type == "lookalike":
        # PATH A: Classification metrics
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        probs = 1 / (1 + np.exp(-predictions_np.flatten()))  # sigmoid
        r2 = float(roc_auc_score(targets_np.flatten(), probs))       # AUC-ROC
        binary_preds = (probs >= 0.5).astype(int)
        f1 = float(f1_score(targets_np.flatten().astype(int), binary_preds, zero_division=0))
        logloss = float(log_loss(targets_np.flatten(), probs))
    else:
        # PATH B: Regression metrics (inverse-transform to dollar scale)
        preds_orig = processor.target_scaler.inverse_transform(
            predictions_np.reshape(-1, 1)
        ).flatten()
        targs_orig = processor.target_scaler.inverse_transform(
            targets_np.reshape(-1, 1)
        ).flatten()
        errors = preds_orig - targs_orig
        mae = float(np.mean(np.abs(errors)))
        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((targs_orig - np.mean(targs_orig))**2)
        r2 = float(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)

    # --- Early Stopping ---
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if patience_counter >= pytorch_config.early_stopping_patience:
        break

# Restore best model and run test set
if best_model_state:
    model.load_state_dict(best_model_state)

# TEST SET: unbiased final metrics
model.eval()
with torch.no_grad():
    for numeric, categorical, boolean, target in test_loader:
        # ... same forward pass, compute test metrics ...
```

---

### **Path C: XGBoost Training -- `_run_tree_training()`**

#### **Step 1: Convert DataLoaders to Numpy**

XGBoost needs flat numpy arrays, not PyTorch tensors:

```python
# src/services/training_service.py:371-416

def _dataloaders_to_numpy(train_loader, val_loader, test_loader, processor, config):
    """
    Convert PyTorch DataLoaders to numpy arrays for tree-based models.

    DataLoaders yield: (numeric_tensor, categorical_tensor, boolean_tensor, target_tensor)
    Tree models need: X (all features concatenated), y (target)
    """
    def _loader_to_arrays(loader):
        all_numeric, all_cat, all_bool, all_targets = [], [], [], []
        for numeric, categorical, boolean, target in loader:
            all_numeric.append(numeric)
            all_cat.append(categorical)
            all_bool.append(boolean)
            all_targets.append(target)
        numeric_np = torch.cat(all_numeric).numpy()
        cat_np = torch.cat(all_cat).numpy().astype(np.float64)
        bool_np = torch.cat(all_bool).numpy()
        target_np = torch.cat(all_targets).numpy()
        X = np.concatenate([numeric_np, cat_np, bool_np], axis=1)
        return X, target_np

    X_train, y_train = _loader_to_arrays(train_loader)
    X_val, y_val = _loader_to_arrays(val_loader)
    X_test, y_test = _loader_to_arrays(test_loader)

    # KEY DIFFERENCE: For regression, inverse-transform targets to original $ scale
    # Tree models train on raw revenue, not standardized values
    if config.task_type != "lookalike" and processor.target_scaler is not None:
        y_train = processor.target_scaler.inverse_transform(
            y_train.reshape(-1, 1)
        ).flatten()
        y_val = processor.target_scaler.inverse_transform(
            y_val.reshape(-1, 1)
        ).flatten()
        y_test = processor.target_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

    # Feature names matching concatenation order: numeric + categorical + boolean
    feature_names = (
        list(config.numeric_features) +
        list(config.categorical_features) +
        list(config.boolean_features)
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
```

#### **Step 2: Train XGBoost with Progress Callbacks**

```python
# src/services/training_service.py:495-571

def _run_tree_training(job, config, pytorch_config, train_loader, val_loader,
                       test_loader, processor, report_callback, start_time):
    # Convert DataLoaders to numpy
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = _dataloaders_to_numpy(
        train_loader, val_loader, test_loader, processor, pytorch_config
    )

    # Create model via factory
    n_estimators = config.epochs
    if n_estimators < 100:
        n_estimators = 500  # Sensible default for tree models

    model = create_model(
        model_type=config.model_type,
        task_type=config.task_type,
        n_numeric=processor.n_numeric_features,
        n_boolean=processor.n_boolean_features,
        categorical_vocab_sizes=processor.categorical_vocab_sizes,
        feature_names=feature_names,
        epochs=n_estimators,
        learning_rate=config.learning_rate if config.learning_rate > 0.001 else 0.03,
        early_stopping_rounds=config.early_stopping_patience * 5,
    )

    # Progress callback for SSE streaming
    callbacks = []
    if config.model_type == "xgboost":
        callbacks.append(_create_xgboost_progress_callback(
            report_fn=report_callback, job=job, config=config,
            X_val=X_val, y_val=y_val, total_rounds=n_estimators,
            start_time=start_time, report_interval=10,
        ))

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, callbacks=callbacks)
```

#### **Step 3: Test Set Assessment and SHAP**

```python
# src/services/training_service.py:573-684

    # Test set assessment
    test_preds = model.predict(X_test)

    if config.task_type == "lookalike":
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        probs = model.predict_proba(X_test)[:, 1]
        test_r2 = float(roc_auc_score(y_test, probs))
        binary_preds = (probs >= 0.5).astype(int)
        test_f1 = float(f1_score(y_test.astype(int), binary_preds, zero_division=0))
        test_logloss = float(log_loss(y_test, probs))
    else:
        errors = test_preds - y_test
        test_mae = float(np.mean(np.abs(errors)))
        test_rmse = float(np.sqrt(np.mean(errors**2)))
        # SMAPE
        denominator = (np.abs(test_preds) + np.abs(y_test)) / 2
        denominator = np.where(denominator == 0, 1, denominator)
        test_smape = float(np.mean(np.abs(errors) / denominator) * 100)
        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        test_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # SHAP feature importance (TreeExplainer -- fast for tree models)
    shap_success = compute_shap_values_tree(
        model=model, X_test=X_test, feature_names=feature_names,
        output_dir=job.output_dir, n_explain=500,
        task_type=config.task_type, progress_callback=shap_progress,
    )

    # Save model artifacts
    if config.model_type == "xgboost":
        model.model.set_params(callbacks=None)  # Clear unpicklable callbacks
        model.model.save_model(str(job.output_dir / "best_model.json"))

    processor.save(job.output_dir / "preprocessor.pkl")
```

---

### **Phase 2: Post-Classification Revenue Prediction**

After a lookalike classification job completes, a separate XGBoost regressor is automatically trained to predict revenue for all sites:

```python
# src/services/training_service.py:758-846

def _run_revenue_prediction_phase(job, pytorch_config, report_callback, start_time):
    """
    After lookalike classification, train a quick XGBoost regression model
    to predict revenue for ALL sites, then export combined predictions.
    """
    # Create regression config (same features, regression task)
    reg_config = Config()
    reg_config.output_dir = job.output_dir
    reg_config.target = job.config.target
    reg_config.task_type = "regression"
    reg_config.network_filter = getattr(pytorch_config, "network_filter", None)
    reg_config.numeric_features = pytorch_config.numeric_features
    reg_config.categorical_features = pytorch_config.categorical_features
    reg_config.boolean_features = pytorch_config.boolean_features

    # Load data with regression targets (continuous revenue, not binary)
    reg_train_loader, reg_val_loader, reg_test_loader, reg_processor = create_data_loaders(reg_config)

    # Convert to numpy for XGBoost
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = _dataloaders_to_numpy(
        reg_train_loader, reg_val_loader, reg_test_loader, reg_processor, reg_config
    )

    # Train XGBoost regression model (fast -- typically <30 seconds)
    model = create_model(
        model_type="xgboost", task_type="regression",
        n_numeric=reg_processor.n_numeric_features,
        n_boolean=reg_processor.n_boolean_features,
        categorical_vocab_sizes=reg_processor.categorical_vocab_sizes,
        feature_names=feature_names,
        epochs=500, learning_rate=0.03, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, callbacks=[])

    # Test set metrics
    test_preds = model.predict(X_test)
    errors = test_preds - y_test
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    reg_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    reg_mae = float(np.mean(np.abs(errors)))
    reg_rmse = float(np.sqrt(np.mean(errors**2)))

    # Save regression model artifacts in a subdirectory
    reg_dir = experiment_dir / "regression"
    # ... saves config.json, preprocessor.pkl, model_wrapper.pkl ...
```

---

## **Inference -- `site_scoring/predict.py`**

### **BatchPredictor: Load Any Experiment and Score Sites**

```python
# site_scoring/predict.py:200-262

class BatchPredictor:
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

        # Load config.json
        with open(self.experiment_dir / "config.json") as f:
            self.config = json.load(f)

        self.model_type = self.config["model_type"]
        self.task_type = self.config["task_type"]
        self.training_features = self.config["training_features"]

        # Load preprocessor (fitted scaler + label_encoders)
        with open(self.experiment_dir / "preprocessor.pkl", "rb") as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data["scaler"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.target_scaler = preprocessor_data["target_scaler"]
        self.categorical_vocab_sizes = preprocessor_data["categorical_vocab_sizes"]

        # Load model (XGBoost or NN)
        if self.model_type == "xgboost":
            with open(self.experiment_dir / "model_wrapper.pkl", "rb") as f:
                self.model = pickle.load(f)
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = torch.device(device)
            checkpoint = torch.load(
                self.experiment_dir / "best_model.pt",
                map_location=self.device, weights_only=False,
            )
            nn_config = checkpoint["config"]
            self.nn_model = SiteScoringModel.from_config(
                nn_config, self.categorical_vocab_sizes
            )
            self.nn_model.load_state_dict(checkpoint["model_state_dict"])
            self.nn_model.to(self.device)
            self.nn_model.eval()
```

### **XGBoost Inference Path**

```python
# site_scoring/predict.py:485-499

def _predict_xgboost(self, numeric, categorical, boolean) -> np.ndarray:
    X = np.hstack([numeric, categorical.astype(np.float32), boolean])

    if self.task_type == "lookalike":
        return self.model.predict_proba(X)[:, 1]  # Probability of class 1
    else:
        preds = self.model.predict(X)
        if self.target_scaler is not None:
            preds = self.target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).flatten()
        return preds
```

### **Neural Network Inference Path**

```python
# site_scoring/predict.py:501-529

@torch.no_grad()
def _predict_nn(self, numeric, categorical, boolean) -> np.ndarray:
    batch_size = 4096
    n_samples = len(numeric)
    all_scores = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        num_t = torch.from_numpy(numeric[start:end]).to(self.device)
        cat_t = torch.from_numpy(categorical[start:end]).to(self.device)
        bool_t = torch.from_numpy(boolean[start:end]).to(self.device)

        output = self.nn_model(num_t, cat_t, bool_t)

        if self.task_type == "lookalike":
            # Classification: apply sigmoid to raw logits
            scores = torch.sigmoid(output).cpu().numpy().flatten()
        else:
            # Regression: inverse-transform to dollar scale
            scores = output.cpu().numpy()
            if self.target_scaler is not None:
                scores = self.target_scaler.inverse_transform(scores).flatten()
            else:
                scores = scores.flatten()

        all_scores.append(scores)

    return np.concatenate(all_scores)
```

---

## **Standalone Training -- `site_scoring/train.py`**

The `Trainer` class provides a simpler training loop for CLI usage (without SSE progress streaming):

```python
# site_scoring/train.py:32-256

class Trainer:
    def __init__(self, model: SiteScoringModel, config: Config, processor: DataProcessor):
        self.model = model.to(torch.device(config.device))
        self.config = config
        self.processor = processor
        self.device = torch.device(config.device)

        # Huber loss is robust to outliers
        self.criterion = nn.HuberLoss(delta=1.0)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=config.scheduler_patience,
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (numeric, categorical, boolean, target) in enumerate(train_loader):
            numeric = numeric.to(self.device, non_blocking=True)
            categorical = categorical.to(self.device, non_blocking=True)
            boolean = boolean.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(numeric, categorical, boolean)
            loss = self.criterion(predictions, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for numeric, categorical, boolean, target in val_loader:
            numeric = numeric.to(self.device, non_blocking=True)
            categorical = categorical.to(self.device, non_blocking=True)
            boolean = boolean.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            predictions = self.model(numeric, categorical, boolean)
            loss = self.criterion(predictions, target)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())

        # Inverse transform to original scale for interpretable metrics
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        predictions_orig = self.processor.target_scaler.inverse_transform(predictions)
        targets_orig = self.processor.target_scaler.inverse_transform(targets)

        mae = np.mean(np.abs(predictions_orig - targets_orig))
        ss_res = np.sum((targets_orig - predictions_orig) ** 2)
        ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return total_loss / len(val_loader), mae, r2

    def train(self, train_loader, val_loader):
        best_model_state = None

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae, val_r2 = self.evaluate(val_loader)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = self.config.output_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, checkpoint_path)
        self.processor.save(self.config.output_dir / "preprocessor.pkl")
```

```python
# site_scoring/train.py:259-318

def run_training(config: Optional[Config] = None):
    """CLI entry point for standalone training."""
    from .data_loader import create_data_loaders

    if config is None:
        config = Config()

    train_loader, val_loader, test_loader, processor = create_data_loaders(config)

    model = SiteScoringModel(
        n_numeric=processor.n_numeric_features,
        n_boolean=processor.n_boolean_features,
        categorical_vocab_sizes=processor.categorical_vocab_sizes,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_batch_norm=config.use_batch_norm,
    )

    trainer = Trainer(model, config, processor)
    trained_model = trainer.train(train_loader, val_loader)

    # Final evaluation on test set (unbiased)
    test_loss, test_mae, test_r2 = trainer.evaluate(test_loader)

    return trained_model, processor, {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
```

---

## **API Routes -- `src/routes/prediction.py`**

### **Batch Prediction Endpoint**

```python
# src/routes/prediction.py:92-175

@prediction_bp.route('/predict/batch', methods=['POST'])
def api_predict_batch():
    """Run batch prediction using a trained model on all (or filtered) sites."""
    data = request.get_json() or {}
    experiment_name = data.get('experiment_dir')
    filters = data.get('filter')

    # Find experiment directory
    if experiment_name:
        experiment_dir = DEFAULT_OUTPUT_DIR / "experiments" / experiment_name
    else:
        experiment_dir = _find_latest_experiment()

    # Load predictor (cached at module level)
    predictor = _get_cached_predictor(experiment_dir)

    # Load all sites for prediction
    from site_scoring.data_transform import get_all_sites_for_prediction
    all_sites_df = get_all_sites_for_prediction()

    # Apply filters if provided
    if filters:
        matching_ids = get_filtered_site_ids(filters)
        all_sites_df = all_sites_df.filter(pl.col("gtvid").is_in(matching_ids))

    # Run prediction
    predictions = predictor.predict(all_sites_df)

    # Summary statistics
    scores = np.array(list(predictions.values()))
    summary = {
        'mean': float(np.mean(scores)),
        'median': float(np.median(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'p10': float(np.percentile(scores, 10)),
        'p25': float(np.percentile(scores, 25)),
        'p75': float(np.percentile(scores, 75)),
        'p90': float(np.percentile(scores, 90)),
    }

    return jsonify(_clean_nan_values({
        'predictions': _clean_nan_values(predictions),
        'model_type': predictor.model_type,
        'task_type': predictor.task_type,
        'count': len(predictions),
        'experiment_dir': experiment_dir.name,
        'summary': summary,
    }))
```

### **Module-Level Predictor Cache**

```python
# src/routes/prediction.py:58-68

_cached_predictor = None
_cached_predictor_experiment = None

def _get_cached_predictor(experiment_dir: Path):
    global _cached_predictor, _cached_predictor_experiment

    if (_cached_predictor is not None
            and _cached_predictor_experiment == str(experiment_dir)):
        return _cached_predictor

    from site_scoring.predict import BatchPredictor
    _cached_predictor = BatchPredictor(experiment_dir)
    _cached_predictor_experiment = str(experiment_dir)
    return _cached_predictor
```

---

## **Key Design Decisions**

| Decision | Rationale |
|----------|-----------|
| Same model architecture for classification and regression | Simplifies codebase; only loss function and post-processing differ |
| XGBoost trains on raw dollar values; NN trains on standardized targets | Tree models are scale-invariant; standardization adds no benefit |
| `pos_weight=9.0` for classification | Compensates for ~10% positive class ratio at default p90 threshold |
| Percentile clipping [1%, 99%] before scaling | Prevents extreme outliers from dominating StandardScaler |
| HuberLoss instead of MSE for regression | Robust to revenue outliers (high-revenue sites can be 10x median) |
| Module-level predictor cache | Avoids reloading model on every API request |
| Inverse-transform for XGBoost targets | Tree models need raw dollar values; transforms would distort splits |
