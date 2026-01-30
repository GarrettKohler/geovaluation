"""
TRAINING FOR BINARY CLASSIFICATION FOR LOOKALIKE CALCULATION
Training service for model training with GPU acceleration.
Provides async training with progress tracking via Server-Sent Events.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
import time
import json
import threading
from queue import Queue
import traceback

# Import from site_scoring module (at project root)
from site_scoring.config import Config, DEFAULT_OUTPUT_DIR
from site_scoring.model import (
    SiteScoringModel,
    CatBoostModel,
    XGBoostModel,
    create_model,
    CATBOOST_AVAILABLE,
    XGBOOST_AVAILABLE,
)
from site_scoring.data_loader import DataProcessor, create_data_loaders
from src.services.shap_service import compute_shap_values, compute_shap_values_tree, ShapCache

# Feature selection imports
from site_scoring.feature_selection import (
    FeatureSelectionConfig,
    FeatureSelectionMethod,
    FeatureSelectionTrainer,
    create_feature_selection_model,
    STGAugmentedModel,
    get_preset,
)

# Explainability imports (Phase 3 integration)
from site_scoring.explainability import (
    ExplainabilityPipeline,
    ProbabilityCalibrator,
    ConformalClassifier,
    TierClassifier,
)


@dataclass
class TrainingConfig:
    """User-configurable training parameters."""
    # Model type
    model_type: str = "neural_network"  # neural_network, gradient_boosting, random_forest

    # Task type: "regression" (predict revenue) or "lookalike" (classify top performers)
    task_type: str = "regression"

    # Target variable
    target: str = "avg_monthly_revenue"  # avg_monthly_revenue (recommended), total_revenue

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 4096
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.2

    # Architecture
    hidden_layers: list = field(default_factory=lambda: [512, 256, 128, 64])
    embedding_dim: int = 16

    # Training behavior
    early_stopping_patience: int = 10
    scheduler_patience: int = 5

    # Device and hardware
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cpu")
    apple_chip: str = "auto"  # auto, m1, m1_pro, m1_max, m1_ultra, m2, m2_pro, etc.

    # Data loader optimizations (set based on chip)
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Feature Selection Configuration
    feature_selection_method: str = "none"  # none, stg_light, stg_aggressive, lassonet_standard, etc.
    stg_lambda: float = 0.1        # STG L0 regularization weight
    stg_sigma: float = 0.5         # STG gate distribution spread
    run_shap_validation: bool = False  # Run SHAP-Select post-training
    track_gradients: bool = False  # Track gradient/weight profiles

    # Model Preset (controls which features are included)
    model_preset: str = "model_b"  # model_a (all features) or model_b (curated)

    # User-selected features (subset of preset features)
    # If None or empty, all preset features are used
    selected_features: Optional[list] = None

    # Explainability Configuration (for lookalike/classification tasks)
    fit_explainability: bool = True  # Fit calibration + conformal prediction after training
    calibration_split: float = 0.5   # Fraction of validation data reserved for calibration
    conformal_alpha: float = 0.10    # Significance level (0.10 = 90% coverage)


@dataclass
class TrainingProgress:
    """Training progress update."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_smape: float
    val_rmse: float
    val_r2: float
    learning_rate: float
    elapsed_time: float
    status: str  # running, completed, error
    message: str = ""
    best_val_loss: float = float('inf')
    # Feature selection stats
    n_active_features: Optional[int] = None
    fs_reg_loss: float = 0.0


class TrainingJob:
    """
    Manages a single training job with progress tracking.
    Runs in a background thread, reports progress via queue.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.progress_queue: Queue = Queue()
        self.is_running = False
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.job_id = f"job_{int(time.time())}"
        self.final_metrics: Optional[Dict] = None

    def start(self):
        """Start training in background thread."""
        if self.is_running:
            return False

        self.is_running = True
        self.should_stop = False
        self.thread = threading.Thread(target=self._run_training, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Request training to stop."""
        self.should_stop = True

    def get_progress(self) -> Optional[TrainingProgress]:
        """Get next progress update if available."""
        try:
            return self.progress_queue.get_nowait()
        except:
            return None

    def _report_progress(self, progress: TrainingProgress):
        """Add progress update to queue."""
        self.progress_queue.put(progress)

    def _fit_explainability_pipeline(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        processor,
        pytorch_config,
        all_feature_names: list,
        output_dir: Path,
        device: torch.device,
    ) -> Dict:
        """
        Fit the explainability pipeline after training completes.

        This includes:
        1. Probability calibration (isotonic regression)
        2. Conformal prediction (MAPIE with APS method)
        3. Tier classifier initialization

        Args:
            model: Trained PyTorch model
            val_loader: Validation data loader
            processor: Data processor with encoders
            pytorch_config: Training configuration
            all_feature_names: List of feature names
            output_dir: Directory to save pipeline
            device: Device for inference

        Returns:
            Dict with calibration and conformal prediction statistics
        """
        import pandas as pd

        # Collect validation data for calibration
        model.eval()
        all_numeric = []
        all_categorical = []
        all_boolean = []
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for numeric, categorical, boolean, target in val_loader:
                numeric = numeric.to(device, non_blocking=True)
                categorical = categorical.to(device, non_blocking=True)
                boolean = boolean.to(device, non_blocking=True)

                predictions = model(numeric, categorical, boolean)
                proba = torch.sigmoid(predictions).cpu().numpy().ravel()

                all_numeric.append(numeric.cpu().numpy())
                all_categorical.append(categorical.cpu().numpy())
                all_boolean.append(boolean.cpu().numpy())
                all_targets.append(target.numpy().ravel())
                all_predictions.append(proba)

        # Concatenate all batches
        X_numeric = np.vstack(all_numeric)
        X_categorical = np.vstack(all_categorical)
        X_boolean = np.vstack(all_boolean)
        y_val = np.concatenate(all_targets)
        y_proba_val = np.concatenate(all_predictions)

        # Concatenate features for sklearn-style input
        X_val = np.hstack([X_numeric, X_categorical, X_boolean])

        # Split validation into calibration and holdout
        n_cal = int(len(X_val) * self.config.calibration_split)
        indices = np.random.permutation(len(X_val))
        cal_indices = indices[:n_cal]
        holdout_indices = indices[n_cal:]

        X_cal = X_val[cal_indices]
        y_cal = y_val[cal_indices]
        y_proba_cal = y_proba_val[cal_indices]

        X_holdout = X_val[holdout_indices]
        y_holdout = y_val[holdout_indices]

        # 1. Fit probability calibrator
        calibrator = ProbabilityCalibrator(method='isotonic')
        calibrator.fit(y_proba_cal, y_cal)

        # Evaluate calibration on holdout
        y_proba_holdout = y_proba_val[holdout_indices]
        y_calibrated_holdout = calibrator.calibrate(y_proba_holdout)
        calibration_ece = calibrator.get_expected_calibration_error(
            y_calibrated_holdout, y_holdout
        )

        # 2. Fit conformal predictor
        n_numeric = X_numeric.shape[1]
        n_categorical = X_categorical.shape[1]
        n_boolean = X_boolean.shape[1]

        conformal = ConformalClassifier(
            model=model,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_boolean=n_boolean,
            alpha=self.config.conformal_alpha,
            method='aps',
            device=str(device),
        )
        conformal.fit(X_cal, y_cal)

        # Evaluate coverage on holdout
        coverage_stats = conformal.evaluate_coverage(X_holdout, y_holdout)

        # 3. Initialize tier classifier
        tier_classifier = TierClassifier()

        # Save components
        explainability_dir = output_dir / "explainability"
        explainability_dir.mkdir(parents=True, exist_ok=True)

        calibrator.save(explainability_dir / "calibrator.pkl")
        conformal.save(explainability_dir / "conformal.pkl")

        # Save tier classifier config
        import pickle
        with open(explainability_dir / "tier_classifier.pkl", 'wb') as f:
            pickle.dump(tier_classifier.to_dict(), f)

        # Save metadata
        metadata = {
            'feature_names': all_feature_names,
            'n_numeric': n_numeric,
            'n_categorical': n_categorical,
            'n_boolean': n_boolean,
            'conformal_alpha': self.config.conformal_alpha,
            'calibration_split': self.config.calibration_split,
            'n_calibration_samples': n_cal,
        }
        with open(explainability_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        # Return statistics
        return {
            'calibration_method': 'isotonic',
            'calibration_ece': float(calibration_ece) if not np.isnan(calibration_ece) else None,
            'calibration_brier_before': calibrator.brier_score_before,
            'calibration_brier_after': calibrator.brier_score_after,
            'conformal_method': 'aps',
            'conformal_alpha': self.config.conformal_alpha,
            'conformal_coverage': coverage_stats['coverage'],
            'conformal_target_coverage': coverage_stats['target_coverage'],
            'conformal_avg_set_size': coverage_stats['avg_set_size'],
            'conformal_pct_uncertain': coverage_stats['pct_uncertain'],
            'n_calibration_samples': n_cal,
            'pipeline_path': str(explainability_dir),
        }

    def _prepare_tabular_data(
        self,
        data_loader: DataLoader,
        device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert PyTorch DataLoader to numpy arrays for sklearn-style models.

        Args:
            data_loader: PyTorch DataLoader with (numeric, categorical, boolean, target) batches
            device: Device to use (ignored for numpy conversion)

        Returns:
            Tuple of (X, y) numpy arrays where X is concatenated features
        """
        all_numeric = []
        all_categorical = []
        all_boolean = []
        all_targets = []

        for numeric, categorical, boolean, target in data_loader:
            all_numeric.append(numeric.numpy())
            all_categorical.append(categorical.numpy())
            all_boolean.append(boolean.numpy())
            all_targets.append(target.numpy())

        # Concatenate batches
        numeric_arr = np.vstack(all_numeric)
        categorical_arr = np.vstack(all_categorical)
        boolean_arr = np.vstack(all_boolean)
        targets_arr = np.vstack(all_targets).ravel()

        # Concatenate features: [numeric | categorical | boolean]
        X = np.hstack([numeric_arr, categorical_arr, boolean_arr])

        return X, targets_arr

    def _run_gradient_boosting_training(
        self,
        model_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        processor,
        pytorch_config,
        start_time: float,
    ) -> Dict:
        """
        Train a gradient boosting model (CatBoost or XGBoost).

        Args:
            model_type: "catboost" or "xgboost"
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            processor: Data processor with encoders
            pytorch_config: Configuration object
            start_time: Training start time for progress tracking

        Returns:
            Dict with training results and metrics
        """
        from sklearn.metrics import roc_auc_score

        # Build feature names list
        all_feature_names = (
            list(pytorch_config.numeric_features) +
            list(pytorch_config.categorical_features) +
            list(pytorch_config.boolean_features)
        )

        # Report progress: preparing data
        self._report_progress(TrainingProgress(
            epoch=0,
            total_epochs=1,  # Gradient boosting reports internally
            train_loss=0,
            val_loss=0,
            val_mae=0,
            val_smape=0,
            val_rmse=0,
            val_r2=0,
            learning_rate=self.config.learning_rate,
            elapsed_time=time.time() - start_time,
            status="running",
            message=f"Preparing data for {model_type.upper()}..."
        ))

        # Convert DataLoaders to numpy arrays
        X_train, y_train = self._prepare_tabular_data(train_loader)
        X_val, y_val = self._prepare_tabular_data(val_loader)
        X_test, y_test = self._prepare_tabular_data(test_loader)

        print(f"[Training] Data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # Calculate categorical feature indices
        n_numeric = processor.n_numeric_features
        n_categorical = len(processor.categorical_vocab_sizes)
        cat_feature_indices = list(range(n_numeric, n_numeric + n_categorical))

        # Report progress: creating model
        self._report_progress(TrainingProgress(
            epoch=0,
            total_epochs=1,
            train_loss=0,
            val_loss=0,
            val_mae=0,
            val_smape=0,
            val_rmse=0,
            val_r2=0,
            learning_rate=self.config.learning_rate,
            elapsed_time=time.time() - start_time,
            status="running",
            message=f"Creating {model_type.upper()} model..."
        ))

        # Create model
        if model_type == "catboost":
            model = CatBoostModel(
                task_type=self.config.task_type,
                cat_feature_indices=cat_feature_indices,
                feature_names=all_feature_names,
                iterations=self.config.epochs * 20,  # More iterations for GB
                learning_rate=self.config.learning_rate * 3,  # GB typically needs higher LR
                depth=6,
                early_stopping_rounds=50,
                verbose=100,
            )
        else:  # xgboost
            model = XGBoostModel(
                task_type=self.config.task_type,
                feature_names=all_feature_names,
                n_estimators=self.config.epochs * 20,
                learning_rate=self.config.learning_rate * 3,
                max_depth=6,
                early_stopping_rounds=50,
            )

        # Report progress: training
        self._report_progress(TrainingProgress(
            epoch=0,
            total_epochs=1,
            train_loss=0,
            val_loss=0,
            val_mae=0,
            val_smape=0,
            val_rmse=0,
            val_r2=0,
            learning_rate=self.config.learning_rate,
            elapsed_time=time.time() - start_time,
            status="running",
            message=f"Training {model_type.upper()} (this may take a few minutes)..."
        ))

        # Train model
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        predictions = model.predict(X_test)

        if self.config.task_type == "lookalike":
            # Classification metrics
            if hasattr(model, 'predict_proba'):
                predictions_prob = model.predict_proba(X_test)[:, 1]
            else:
                predictions_prob = predictions
            try:
                test_auc = float(roc_auc_score(y_test, predictions_prob))
            except ValueError:
                test_auc = 0.0
            test_accuracy = float(np.mean((predictions_prob >= 0.5) == y_test))
            test_mae, test_smape, test_rmse, test_r2 = 0.0, 0.0, 0.0, test_auc
        else:
            # Regression metrics - inverse transform predictions
            predictions_orig = processor.target_scaler.inverse_transform(predictions.reshape(-1, 1))
            targets_orig = processor.target_scaler.inverse_transform(y_test.reshape(-1, 1))
            test_errors = predictions_orig - targets_orig
            test_mae = float(np.mean(np.abs(test_errors)))
            test_rmse = float(np.sqrt(np.mean(test_errors ** 2)))
            denominator = (np.abs(targets_orig) + np.abs(predictions_orig)) / 2
            nonzero_mask = denominator > 0
            if nonzero_mask.any():
                test_smape = float(np.mean(np.abs(test_errors[nonzero_mask]) / denominator[nonzero_mask]) * 100)
            else:
                test_smape = 0.0
            ss_res = np.sum(test_errors ** 2)
            ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
            test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Save model
        output_dir = pytorch_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        import pickle
        model_path = output_dir / f"{model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[Training] Model saved to {model_path}")

        # Get feature importance (built-in to gradient boosting)
        feature_importance = model.get_feature_importance()

        # Save feature importance
        importance_path = output_dir / f"{model_type}_feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)

        # Compute SHAP values using TreeExplainer (much faster than KernelExplainer)
        self._report_progress(TrainingProgress(
            epoch=1,
            total_epochs=1,
            train_loss=0,
            val_loss=0,
            val_mae=test_mae if self.config.task_type != "lookalike" else 0.0,
            val_smape=test_smape if self.config.task_type != "lookalike" else 0.0,
            val_rmse=test_rmse if self.config.task_type != "lookalike" else 0.0,
            val_r2=float(test_r2),
            learning_rate=self.config.learning_rate,
            elapsed_time=time.time() - start_time,
            status="running",
            message="Computing SHAP feature importance..."
        ))

        shap_success = compute_shap_values_tree(
            model=model,
            X_test=X_test,
            feature_names=all_feature_names,
            output_dir=output_dir,
            n_explain=500,
            task_type=self.config.task_type,
            progress_callback=lambda msg: self._report_progress(TrainingProgress(
                epoch=1,
                total_epochs=1,
                train_loss=0,
                val_loss=0,
                val_mae=test_mae if self.config.task_type != "lookalike" else 0.0,
                val_smape=test_smape if self.config.task_type != "lookalike" else 0.0,
                val_rmse=test_rmse if self.config.task_type != "lookalike" else 0.0,
                val_r2=float(test_r2),
                learning_rate=self.config.learning_rate,
                elapsed_time=time.time() - start_time,
                status="running",
                message=msg
            ))
        )

        # Build results
        results = {
            "model_type": model_type,
            "task_type": self.config.task_type,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_smape": test_smape,
            "test_r2": test_r2,
            "best_iteration": model.best_iteration,
            "feature_importance": feature_importance,
            "model_path": str(model_path),
            "n_features": len(all_feature_names),
            "feature_names": all_feature_names,
            "shap_available": shap_success,  # Enable SHAP values link in UI
        }

        # Report completion
        self._report_progress(TrainingProgress(
            epoch=1,
            total_epochs=1,
            train_loss=0,
            val_loss=0,
            val_mae=test_mae if self.config.task_type != "lookalike" else 0.0,
            val_smape=test_smape if self.config.task_type != "lookalike" else 0.0,
            val_rmse=test_rmse if self.config.task_type != "lookalike" else 0.0,
            val_r2=float(test_r2),
            learning_rate=self.config.learning_rate,
            elapsed_time=time.time() - start_time,
            status="completed",
            message=f"{model_type.upper()} training complete! Test R²={test_r2:.4f}"
        ))

        return results

    def _run_training(self):
        """Main training loop - runs in background thread."""
        start_time = time.time()

        try:
            # Report starting
            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
                val_smape=0,
                val_rmse=0,
                val_r2=0,
                learning_rate=self.config.learning_rate,
                elapsed_time=0,
                status="running",
                message="Loading and processing data..."
            ))

            # Create PyTorch config from user config with Apple Silicon optimizations
            pytorch_config = Config()
            pytorch_config.target = self.config.target
            pytorch_config.epochs = self.config.epochs
            pytorch_config.batch_size = self.config.batch_size
            pytorch_config.learning_rate = self.config.learning_rate
            pytorch_config.weight_decay = self.config.weight_decay
            pytorch_config.dropout = self.config.dropout
            pytorch_config.hidden_dims = self.config.hidden_layers
            pytorch_config.embedding_dim = self.config.embedding_dim
            pytorch_config.early_stopping_patience = self.config.early_stopping_patience
            pytorch_config.scheduler_patience = self.config.scheduler_patience
            pytorch_config.device = self.config.device

            # Apply Apple Silicon optimizations to data loader config
            pytorch_config.num_workers = self.config.num_workers
            pytorch_config.pin_memory = self.config.pin_memory
            pytorch_config.prefetch_factor = self.config.prefetch_factor
            pytorch_config.task_type = self.config.task_type

            # Apply model preset (controls which features are included)
            if self.config.model_preset:
                pytorch_config.apply_model_preset(self.config.model_preset)

                # Apply user feature selection (filter to selected subset)
                if self.config.selected_features:
                    from site_scoring.config import filter_features_by_selection
                    filtered = filter_features_by_selection(
                        self.config.model_preset,
                        self.config.selected_features
                    )
                    pytorch_config.numeric_features = filtered["numeric"]
                    pytorch_config.categorical_features = filtered["categorical"]
                    pytorch_config.boolean_features = filtered["boolean"]
                    print(f"[Training] User feature selection applied: "
                          f"{len(self.config.selected_features)} features selected")

                print(f"[Training] Model preset: {self.config.model_preset} "
                      f"({len(pytorch_config.numeric_features)} numeric, "
                      f"{len(pytorch_config.categorical_features)} categorical, "
                      f"{len(pytorch_config.boolean_features)} boolean)")

            # Get chip name for display
            chip_name = APPLE_CHIP_SPECS.get(self.config.apple_chip, {})
            chip_display = f"Apple {self.config.apple_chip.upper().replace('_', ' ')}" if self.config.apple_chip != "auto" else "Apple Silicon"

            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
                val_smape=0,
                val_rmse=0,
                val_r2=0,
                learning_rate=self.config.learning_rate,
                elapsed_time=time.time() - start_time,
                status="running",
                message=f"Optimizing for {chip_display} ({self.config.num_workers} workers, batch={self.config.batch_size})..."
            ))

            # Load data with optimized settings
            train_loader, val_loader, test_loader, processor = create_data_loaders(pytorch_config)

            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
                val_smape=0,
                val_rmse=0,
                val_r2=0,
                learning_rate=self.config.learning_rate,
                elapsed_time=time.time() - start_time,
                status="running",
                message=f"Data loaded. Creating model on {self.config.device}..."
            ))

            # Branch based on model type
            if self.config.model_type in ("catboost", "xgboost"):
                # Use gradient boosting training path
                results = self._run_gradient_boosting_training(
                    model_type=self.config.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    processor=processor,
                    pytorch_config=pytorch_config,
                    start_time=start_time,
                )
                self.final_metrics = results
                self.is_running = False
                return  # Exit early - gradient boosting training is complete

            # Neural Network training path (original code)
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

            device = torch.device(self.config.device)
            model = model.to(device)

            # Initialize feature selection
            fs_trainer = None
            fs_config = None
            use_stg = False

            if self.config.feature_selection_method != "none":
                self._report_progress(TrainingProgress(
                    epoch=0,
                    total_epochs=self.config.epochs,
                    train_loss=0,
                    val_loss=0,
                    val_mae=0,
                    val_smape=0,
                    val_rmse=0,
                    val_r2=0,
                    learning_rate=self.config.learning_rate,
                    elapsed_time=time.time() - start_time,
                    status="running",
                    message=f"Initializing feature selection: {self.config.feature_selection_method}..."
                ))

                # Get preset or create custom config
                try:
                    fs_config = get_preset(self.config.feature_selection_method)
                except ValueError:
                    # Custom config with user parameters
                    fs_config = FeatureSelectionConfig(
                        method=FeatureSelectionMethod.STOCHASTIC_GATES,
                        stg_lambda=self.config.stg_lambda,
                        stg_sigma=self.config.stg_sigma,
                        run_shap_validation=self.config.run_shap_validation,
                        track_gradients=self.config.track_gradients,
                    )

                # Build feature names list
                all_feature_names = (
                    list(pytorch_config.numeric_features) +
                    list(pytorch_config.categorical_features) +
                    list(pytorch_config.boolean_features)
                )

                # Calculate total input dimension
                # Must match CategoricalEmbedding formula: max(min(embedding_dim, (vocab+1)//2), 4)
                cat_dim = sum(
                    max(min(pytorch_config.embedding_dim, (vocab + 1) // 2), 4)
                    for vocab in processor.categorical_vocab_sizes.values()
                )
                total_input_dim = cat_dim + processor.n_numeric_features + processor.n_boolean_features

                # Create feature selection model/trainer
                model, fs_trainer = create_feature_selection_model(
                    config=fs_config,
                    base_model=model,
                    n_numeric=processor.n_numeric_features,
                    n_boolean=processor.n_boolean_features,
                    categorical_vocab_sizes=processor.categorical_vocab_sizes,
                    embedding_dim=pytorch_config.embedding_dim,
                    hidden_dims=pytorch_config.hidden_dims,
                    dropout=pytorch_config.dropout,
                    feature_names=all_feature_names,
                    device=self.config.device,
                )

                # Check if using STG (requires modified forward pass)
                use_stg = (fs_config.method == FeatureSelectionMethod.STOCHASTIC_GATES and
                          fs_trainer is not None and fs_trainer.stg_gates is not None)

                if use_stg:
                    # Wrap model with STG
                    model = STGAugmentedModel(model, fs_trainer.stg_gates).to(device)

                print(f"[Training] Feature selection initialized: {fs_config.get_method_display_name()}")

            # Training setup - branch loss function by task type
            if self.config.task_type == "lookalike":
                # Classification: BCEWithLogitsLoss with class weighting for 90/10 imbalance
                pos_weight = torch.tensor([9.0], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                # Regression: Huber loss (robust to revenue outliers)
                criterion = nn.HuberLoss(delta=1.0)
            optimizer = AdamW(
                model.parameters(),
                lr=pytorch_config.learning_rate,
                weight_decay=pytorch_config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=pytorch_config.scheduler_patience,
            )

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            # Training loop
            for epoch in range(1, self.config.epochs + 1):
                if self.should_stop:
                    self._report_progress(TrainingProgress(
                        epoch=epoch,
                        total_epochs=self.config.epochs,
                        train_loss=0,
                        val_loss=0,
                        val_mae=0,
                        val_smape=0,
                        val_rmse=0,
                                val_r2=0,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        elapsed_time=time.time() - start_time,
                        status="stopped",
                        message="Training stopped by user"
                    ))
                    break

                # Train epoch
                model.train()
                total_train_loss = 0.0
                n_batches = 0

                total_fs_reg_loss = 0.0  # Feature selection regularization loss

                for numeric, categorical, boolean, target in train_loader:
                    numeric = numeric.to(device, non_blocking=True)
                    categorical = categorical.to(device, non_blocking=True)
                    boolean = boolean.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(numeric, categorical, boolean)
                    task_loss = criterion(predictions, target)

                    # Add feature selection regularization loss if using STG
                    if use_stg and hasattr(model, 'get_regularization_loss'):
                        fs_reg_loss = model.get_regularization_loss()
                        loss = task_loss + fs_reg_loss
                        total_fs_reg_loss += fs_reg_loss.item()
                    else:
                        loss = task_loss

                    loss.backward()

                    # Record gradients for gradient analyzer if enabled
                    if fs_trainer is not None and fs_trainer.gradient_analyzer is not None:
                        fs_trainer.gradient_analyzer.record_gradients()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += task_loss.item()
                    n_batches += 1

                train_loss = total_train_loss / n_batches
                fs_reg_loss_avg = total_fs_reg_loss / n_batches if n_batches > 0 else 0.0

                # Validate
                model.eval()
                total_val_loss = 0.0
                all_predictions = []
                all_targets = []

                with torch.no_grad():
                    for numeric, categorical, boolean, target in val_loader:
                        numeric = numeric.to(device, non_blocking=True)
                        categorical = categorical.to(device, non_blocking=True)
                        boolean = boolean.to(device, non_blocking=True)
                        target = target.to(device, non_blocking=True)

                        predictions = model(numeric, categorical, boolean)
                        loss = criterion(predictions, target)

                        total_val_loss += loss.item()
                        all_predictions.append(predictions.cpu())
                        all_targets.append(target.cpu())

                val_loss = total_val_loss / len(val_loader)

                # Calculate metrics
                predictions_np = torch.cat(all_predictions).numpy()
                targets_np = torch.cat(all_targets).numpy()

                if self.config.task_type == "lookalike":
                    # Classification metrics
                    from sklearn.metrics import roc_auc_score
                    predictions_prob = 1 / (1 + np.exp(-predictions_np.flatten()))  # sigmoid
                    targets_binary = targets_np.flatten()
                    try:
                        auc = float(roc_auc_score(targets_binary, predictions_prob))
                    except ValueError:
                        auc = 0.0  # Edge case: only one class in batch
                    accuracy = float(np.mean((predictions_prob >= 0.5) == targets_binary))
                    # Map to progress fields: val_r2 → AUC (frontend uses this for display)
                    mae, smape, rmse, r2 = 0.0, 0.0, 0.0, auc
                else:
                    # Regression metrics
                    predictions_orig = processor.target_scaler.inverse_transform(predictions_np)
                    targets_orig = processor.target_scaler.inverse_transform(targets_np)
                    errors = predictions_orig - targets_orig
                    mae = float(np.mean(np.abs(errors)))
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    denominator = (np.abs(targets_orig) + np.abs(predictions_orig)) / 2
                    nonzero_mask = denominator > 0
                    if nonzero_mask.any():
                        smape = float(np.mean(np.abs(errors[nonzero_mask]) / denominator[nonzero_mask]) * 100)
                    else:
                        smape = 0.0
                    ss_res = np.sum(errors ** 2)
                    ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Update scheduler
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]["lr"]

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                # Feature selection: call on_epoch_end
                n_active_features = None
                if fs_trainer is not None:
                    fs_stats = fs_trainer.on_epoch_end(epoch)
                    n_active_features = fs_stats.get('n_active_features')

                # Build progress message
                progress_msg = f"Epoch {epoch}/{self.config.epochs}"
                if n_active_features is not None:
                    progress_msg += f" | {n_active_features} features active"
                if fs_reg_loss_avg > 0:
                    progress_msg += f" | FS reg: {fs_reg_loss_avg:.4f}"

                # Report progress
                self._report_progress(TrainingProgress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mae=mae,
                    val_smape=smape,
                    val_rmse=rmse,
                    val_r2=float(r2),
                    learning_rate=current_lr,
                    elapsed_time=time.time() - start_time,
                    status="running",
                    message=progress_msg,
                    best_val_loss=best_val_loss,
                    n_active_features=n_active_features,
                    fs_reg_loss=fs_reg_loss_avg,
                ))

                # Early stopping
                if patience_counter >= pytorch_config.early_stopping_patience:
                    break

            # Restore best model and evaluate on test set
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # Test evaluation
            model.eval()
            total_test_loss = 0.0
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for numeric, categorical, boolean, target in test_loader:
                    numeric = numeric.to(device, non_blocking=True)
                    categorical = categorical.to(device, non_blocking=True)
                    boolean = boolean.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    predictions = model(numeric, categorical, boolean)
                    loss = criterion(predictions, target)

                    total_test_loss += loss.item()
                    all_predictions.append(predictions.cpu())
                    all_targets.append(target.cpu())

            test_loss = total_test_loss / len(test_loader)
            predictions_np = torch.cat(all_predictions).numpy()
            targets_np = torch.cat(all_targets).numpy()

            if self.config.task_type == "lookalike":
                # Classification test metrics
                from sklearn.metrics import roc_auc_score
                predictions_prob = 1 / (1 + np.exp(-predictions_np.flatten()))  # sigmoid
                targets_binary = targets_np.flatten()
                try:
                    test_auc = float(roc_auc_score(targets_binary, predictions_prob))
                except ValueError:
                    test_auc = 0.0
                test_accuracy = float(np.mean((predictions_prob >= 0.5) == targets_binary))
                test_mae, test_smape, test_rmse, test_r2 = 0.0, 0.0, 0.0, test_auc
            else:
                # Regression test metrics
                predictions_orig = processor.target_scaler.inverse_transform(predictions_np)
                targets_orig = processor.target_scaler.inverse_transform(targets_np)
                test_errors = predictions_orig - targets_orig
                test_mae = float(np.mean(np.abs(test_errors)))
                test_rmse = float(np.sqrt(np.mean(test_errors ** 2)))
                denominator = (np.abs(targets_orig) + np.abs(predictions_orig)) / 2
                nonzero_mask = denominator > 0
                if nonzero_mask.any():
                    test_smape = float(np.mean(np.abs(test_errors[nonzero_mask]) / denominator[nonzero_mask]) * 100)
                else:
                    test_smape = 0.0
                ss_res = np.sum(test_errors ** 2)
                ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
                test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Save model
            output_dir = pytorch_config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / "best_model.pt"
            if self.config.task_type == "lookalike":
                test_metrics_dict = {
                    "test_loss": test_loss,
                    "test_auc": test_auc,
                    "test_accuracy": test_accuracy,
                    "test_r2": test_auc,  # Frontend reads this field
                }
            else:
                test_metrics_dict = {
                    "test_loss": test_loss,
                    "test_mae": test_mae,
                    "test_smape": test_smape,
                    "test_rmse": test_rmse,
                    "test_r2": float(test_r2),
                }

            # Include feature selection summary if available
            fs_summary = None
            if fs_trainer is not None:
                fs_summary = fs_trainer.get_selection_summary()
                fs_trainer.save_results(output_dir)
                print(f"[Training] Feature selection: {fs_summary['n_selected']}/{fs_summary['n_total_features']} features selected")

            # Build feature info for export
            features_used = {
                "numeric": list(pytorch_config.numeric_features),
                "categorical": list(pytorch_config.categorical_features),
                "boolean": list(pytorch_config.boolean_features),
                "total": (len(pytorch_config.numeric_features) +
                          len(pytorch_config.categorical_features) +
                          len(pytorch_config.boolean_features)),
            }

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": pytorch_config,
                "test_metrics": test_metrics_dict,
                "feature_selection": fs_summary,
                "model_preset": self.config.model_preset,
                "selected_features": self.config.selected_features,
                "features_used": features_used,
            }, checkpoint_path)
            processor.save(output_dir / "preprocessor.pkl")

            # Compute SHAP values for model interpretability
            shap_available = False
            if not self.should_stop:
                # Build combined feature names list matching the order used in data loading
                all_feature_names = (
                    list(pytorch_config.numeric_features) +
                    list(pytorch_config.categorical_features) +
                    list(pytorch_config.boolean_features)
                )

                def shap_progress(msg):
                    self._report_progress(TrainingProgress(
                        epoch=self.config.epochs,
                        total_epochs=self.config.epochs,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_mae=test_mae if self.config.task_type != "lookalike" else 0.0,
                        val_smape=test_smape if self.config.task_type != "lookalike" else 0.0,
                        val_rmse=test_rmse if self.config.task_type != "lookalike" else 0.0,
                        val_r2=float(test_r2) if self.config.task_type != "lookalike" else float(test_auc),
                        learning_rate=current_lr,
                        elapsed_time=time.time() - start_time,
                        status="running",
                        message=msg,
                        best_val_loss=best_val_loss
                    ))

                shap_available = compute_shap_values(
                    model=model,
                    processor=processor,
                    test_loader=test_loader,
                    feature_names=all_feature_names,
                    output_dir=output_dir,
                    device=self.config.device,
                    task_type=self.config.task_type,
                    progress_callback=shap_progress,
                )

            # Fit explainability pipeline for lookalike (classification) tasks
            explainability_stats = None
            if (self.config.task_type == "lookalike" and
                self.config.fit_explainability and
                not self.should_stop):

                self._report_progress(TrainingProgress(
                    epoch=self.config.epochs,
                    total_epochs=self.config.epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mae=0.0,
                    val_smape=0.0,
                    val_rmse=0.0,
                    val_r2=float(test_auc),
                    learning_rate=current_lr,
                    elapsed_time=time.time() - start_time,
                    status="running",
                    message="Fitting calibration & conformal prediction...",
                    best_val_loss=best_val_loss
                ))

                try:
                    explainability_stats = self._fit_explainability_pipeline(
                        model=model,
                        val_loader=val_loader,
                        processor=processor,
                        pytorch_config=pytorch_config,
                        all_feature_names=all_feature_names,
                        output_dir=output_dir,
                        device=device,
                    )
                    print(f"[Training] Explainability: calibration ECE={explainability_stats.get('calibration_ece', 'N/A'):.4f}, "
                          f"coverage={explainability_stats.get('conformal_coverage', 'N/A'):.2%}")
                except Exception as e:
                    print(f"[Training] Warning: Explainability fitting failed: {e}")
                    traceback.print_exc()

            self.final_metrics = {
                **test_metrics_dict,
                "best_val_loss": float(best_val_loss),
                "model_path": str(checkpoint_path),
                "shap_available": shap_available,
                "feature_selection": fs_summary,
                "explainability": explainability_stats,
                "model_preset": self.config.model_preset,
                "selected_features": self.config.selected_features,
                "features_used": features_used,
            }

            # Report completion
            if self.config.task_type == "lookalike":
                completion_msg = f"Training complete! AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.2%}"
            else:
                completion_msg = f"Training complete! MAE: ${test_mae:,.0f}, SMAPE: {test_smape:.2f}%, RMSE: ${test_rmse:,.0f}, R²: {test_r2:.4f}"

            self._report_progress(TrainingProgress(
                epoch=self.config.epochs,
                total_epochs=self.config.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=test_mae,
                val_smape=test_smape,
                val_rmse=test_rmse,
                val_r2=float(test_r2),
                learning_rate=current_lr,
                elapsed_time=time.time() - start_time,
                status="completed",
                message=completion_msg,
                best_val_loss=best_val_loss
            ))

        except Exception as e:
            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
                val_smape=0,
                val_rmse=0,
                val_r2=0,
                learning_rate=self.config.learning_rate,
                elapsed_time=time.time() - start_time,
                status="error",
                message=f"Error: {str(e)}\n{traceback.format_exc()}"
            ))
        finally:
            self.is_running = False


# Global training job tracker
_current_job: Optional[TrainingJob] = None


def detect_apple_chip() -> Dict:
    """
    Detect the Apple Silicon chip model using system commands.
    Returns chip info including model, GPU cores, and memory.
    """
    import subprocess
    import re

    chip_info = {
        "detected_chip": "unknown",
        "chip_name": "Unknown",
        "gpu_cores": None,
        "total_memory": None,
    }

    try:
        # Get chip info from sysctl on macOS
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        cpu_brand = result.stdout.strip()

        # Get total memory
        mem_result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if mem_result.stdout.strip():
            mem_bytes = int(mem_result.stdout.strip())
            mem_gb = mem_bytes / (1024 ** 3)
            chip_info["total_memory"] = f"{int(mem_gb)} GB"

        # Parse Apple Silicon chip model
        if "Apple" in cpu_brand:
            chip_info["chip_name"] = cpu_brand

            # Match patterns like M1, M1 Pro, M1 Max, M1 Ultra, M2, M3, M4, etc.
            match = re.search(r'M(\d+)(?:\s+(Pro|Max|Ultra))?', cpu_brand, re.IGNORECASE)
            if match:
                generation = match.group(1)
                variant = match.group(2)

                if variant:
                    chip_id = f"m{generation}_{variant.lower()}"
                else:
                    chip_id = f"m{generation}"

                chip_info["detected_chip"] = chip_id

        # Try to get GPU core count from system_profiler
        gpu_result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        if gpu_result.stdout:
            import json
            gpu_data = json.loads(gpu_result.stdout)
            displays = gpu_data.get("SPDisplaysDataType", [])
            for display in displays:
                if "sppci_cores" in display:
                    chip_info["gpu_cores"] = display["sppci_cores"]
                    break

    except Exception as e:
        print(f"Warning: Could not detect Apple chip: {e}")

    return chip_info


# Apple Silicon chip specifications for optimization
APPLE_CHIP_SPECS = {
    "m1":       {"gpu_cores": 8, "max_batch": 4096, "tier": 1, "memory_bandwidth": 68.25},
    "m1_pro":   {"gpu_cores": 16, "max_batch": 8192, "tier": 2, "memory_bandwidth": 200},
    "m1_max":   {"gpu_cores": 32, "max_batch": 16384, "tier": 3, "memory_bandwidth": 400},
    "m1_ultra": {"gpu_cores": 64, "max_batch": 32768, "tier": 4, "memory_bandwidth": 800},
    "m2":       {"gpu_cores": 10, "max_batch": 4096, "tier": 1, "memory_bandwidth": 100},
    "m2_pro":   {"gpu_cores": 19, "max_batch": 8192, "tier": 2, "memory_bandwidth": 200},
    "m2_max":   {"gpu_cores": 38, "max_batch": 16384, "tier": 3, "memory_bandwidth": 400},
    "m2_ultra": {"gpu_cores": 76, "max_batch": 32768, "tier": 4, "memory_bandwidth": 800},
    "m3":       {"gpu_cores": 10, "max_batch": 4096, "tier": 1, "memory_bandwidth": 100},
    "m3_pro":   {"gpu_cores": 18, "max_batch": 8192, "tier": 2, "memory_bandwidth": 150},
    "m3_max":   {"gpu_cores": 40, "max_batch": 16384, "tier": 3, "memory_bandwidth": 400},
    "m4":       {"gpu_cores": 10, "max_batch": 8192, "tier": 2, "memory_bandwidth": 120},
    "m4_pro":   {"gpu_cores": 20, "max_batch": 16384, "tier": 3, "memory_bandwidth": 273},
    "m4_max":   {"gpu_cores": 40, "max_batch": 32768, "tier": 4, "memory_bandwidth": 546},
}


def get_optimized_training_params(chip_id: str, user_batch_size: int) -> Dict:
    """
    Get optimized training parameters based on the Apple Silicon chip.
    Returns adjusted batch size, number of workers, and other optimizations.
    """
    specs = APPLE_CHIP_SPECS.get(chip_id, {"gpu_cores": 8, "max_batch": 4096, "tier": 1})

    # Ensure batch size doesn't exceed chip's capability
    optimized_batch = min(user_batch_size, specs["max_batch"])

    # Number of data loader workers based on chip tier
    num_workers = min(specs["tier"] * 2, 8)

    # Pin memory for faster data transfer (beneficial for all Apple Silicon)
    pin_memory = True

    # Prefetch factor based on tier
    prefetch_factor = specs["tier"] + 1

    return {
        "batch_size": optimized_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "chip_tier": specs["tier"],
        "gpu_cores": specs["gpu_cores"],
    }


def get_system_info() -> Dict:
    """Get system information for training, including Apple Silicon detection."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "recommended_device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)

    if torch.backends.mps.is_available():
        info["mps_device"] = "Apple Silicon GPU (MPS)"

        # Detect specific Apple Silicon chip
        chip_info = detect_apple_chip()
        info.update(chip_info)

    return info


def start_training(config_dict: Dict) -> Tuple[bool, str]:
    """Start a new training job with Apple Silicon optimizations."""
    global _current_job

    if _current_job is not None and _current_job.is_running:
        return False, "A training job is already running"

    try:
        # Determine the Apple Silicon chip to use
        apple_chip = config_dict.get("apple_chip", "auto")
        if apple_chip == "auto":
            # Auto-detect the chip
            chip_info = detect_apple_chip()
            apple_chip = chip_info.get("detected_chip", "m1")
            if apple_chip == "unknown":
                apple_chip = "m1"  # Default fallback

        # Get chip-optimized parameters
        user_batch_size = int(config_dict.get("batch_size", 4096))
        optimized_params = get_optimized_training_params(apple_chip, user_batch_size)

        print(f"=== Apple Silicon Optimization ===")
        print(f"Selected Chip: {apple_chip}")
        print(f"GPU Cores: {optimized_params['gpu_cores']}")
        print(f"Chip Tier: {optimized_params['chip_tier']}")
        print(f"Optimized Batch Size: {optimized_params['batch_size']} (requested: {user_batch_size})")
        print(f"Data Loader Workers: {optimized_params['num_workers']}")
        print(f"Pin Memory: {optimized_params['pin_memory']}")
        print(f"Prefetch Factor: {optimized_params['prefetch_factor']}")

        config = TrainingConfig(
            model_type=config_dict.get("model_type", "neural_network"),
            task_type=config_dict.get("task_type", "regression"),
            target=config_dict.get("target", "avg_monthly_revenue"),
            epochs=int(config_dict.get("epochs", 50)),
            batch_size=optimized_params["batch_size"],
            learning_rate=float(config_dict.get("learning_rate", 1e-4)),
            weight_decay=float(config_dict.get("weight_decay", 1e-5)),
            dropout=float(config_dict.get("dropout", 0.2)),
            hidden_layers=config_dict.get("hidden_layers", [512, 256, 128, 64]),
            embedding_dim=int(config_dict.get("embedding_dim", 16)),
            early_stopping_patience=int(config_dict.get("early_stopping_patience", 10)),
            scheduler_patience=int(config_dict.get("scheduler_patience", 5)),
            device=config_dict.get("device", "mps" if torch.backends.mps.is_available() else "cpu"),
            apple_chip=apple_chip,
            num_workers=optimized_params["num_workers"],
            pin_memory=optimized_params["pin_memory"],
            prefetch_factor=optimized_params["prefetch_factor"],
            # Model preset
            model_preset=config_dict.get("model_preset", "model_b"),
            # User-selected features (subset of preset)
            selected_features=config_dict.get("selected_features"),
            # Feature selection parameters
            feature_selection_method=config_dict.get("feature_selection_method", "none"),
            stg_lambda=float(config_dict.get("stg_lambda", 0.1)),
            stg_sigma=float(config_dict.get("stg_sigma", 0.5)),
            run_shap_validation=config_dict.get("run_shap_validation", False),
            track_gradients=config_dict.get("track_gradients", False),
            # Explainability parameters
            fit_explainability=config_dict.get("fit_explainability", True),
            calibration_split=float(config_dict.get("calibration_split", 0.5)),
            conformal_alpha=float(config_dict.get("conformal_alpha", 0.10)),
        )

        # Log model preset
        print(f"=== Model Preset ===")
        print(f"Preset: {config.model_preset}")

        # Log explainability settings for lookalike tasks
        if config.task_type == "lookalike" and config.fit_explainability:
            print(f"=== Explainability Pipeline ===")
            print(f"Calibration: isotonic (split={config.calibration_split:.0%})")
            print(f"Conformal: APS (alpha={config.conformal_alpha}, coverage={1-config.conformal_alpha:.0%})")

        # Log feature selection settings if enabled
        if config.feature_selection_method != "none":
            print(f"=== Feature Selection ===")
            print(f"Method: {config.feature_selection_method}")
            if "stg" in config.feature_selection_method:
                print(f"STG Lambda: {config.stg_lambda}")
                print(f"STG Sigma: {config.stg_sigma}")
            if config.run_shap_validation:
                print(f"SHAP Validation: Enabled")
            if config.track_gradients:
                print(f"Gradient Tracking: Enabled")

        _current_job = TrainingJob(config)
        _current_job.start()

        return True, _current_job.job_id

    except Exception as e:
        traceback.print_exc()
        return False, f"Failed to start training: {str(e)}"


def stop_training() -> Tuple[bool, str]:
    """Stop the current training job."""
    global _current_job

    if _current_job is None or not _current_job.is_running:
        return False, "No training job is running"

    _current_job.stop()
    return True, "Training stop requested"


def get_training_status() -> Optional[Dict]:
    """Get status of current/last training job."""
    global _current_job

    if _current_job is None:
        return None

    progress = _current_job.get_progress()

    if progress is None:
        # Return last known state
        return {
            "job_id": _current_job.job_id,
            "is_running": _current_job.is_running,
            "final_metrics": _current_job.final_metrics,
        }

    return {
        "job_id": _current_job.job_id,
        "is_running": _current_job.is_running,
        "epoch": progress.epoch,
        "total_epochs": progress.total_epochs,
        "train_loss": progress.train_loss,
        "val_loss": progress.val_loss,
        "val_mae": progress.val_mae,
        "val_smape": progress.val_smape,
        "val_rmse": progress.val_rmse,
        "val_r2": progress.val_r2,
        "learning_rate": progress.learning_rate,
        "elapsed_time": progress.elapsed_time,
        "status": progress.status,
        "message": progress.message,
        "best_val_loss": progress.best_val_loss,
        "final_metrics": _current_job.final_metrics,
    }


def _sanitize_for_json(obj):
    """Convert Infinity/NaN values to None for JSON serialization."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    return obj


def stream_training_progress() -> Generator[str, None, None]:
    """Generator that yields SSE events for training progress."""
    global _current_job

    if _current_job is None:
        yield f"data: {json.dumps({'status': 'error', 'message': 'No training job'})}\n\n"
        return

    while _current_job.is_running or not _current_job.progress_queue.empty():
        progress = _current_job.get_progress()
        if progress:
            data = {
                "epoch": progress.epoch,
                "total_epochs": progress.total_epochs,
                "train_loss": progress.train_loss,
                "val_loss": progress.val_loss,
                "val_mae": progress.val_mae,
                "val_smape": progress.val_smape,
                "val_rmse": progress.val_rmse,
                "val_r2": progress.val_r2,
                "learning_rate": progress.learning_rate,
                "elapsed_time": progress.elapsed_time,
                "status": progress.status,
                "message": progress.message,
                "best_val_loss": progress.best_val_loss,
                # Feature selection stats
                "n_active_features": progress.n_active_features,
                "fs_reg_loss": progress.fs_reg_loss,
            }

            # Include final_metrics in the completion message so frontend gets it immediately
            if progress.status in ("completed", "error", "stopped"):
                if _current_job.final_metrics:
                    data["final_metrics"] = _current_job.final_metrics
                # Sanitize Infinity/NaN values before JSON serialization
                yield f"data: {json.dumps(_sanitize_for_json(data))}\n\n"
                break
            else:
                # Sanitize Infinity/NaN values before JSON serialization
                yield f"data: {json.dumps(_sanitize_for_json(data))}\n\n"
        else:
            time.sleep(0.5)

    yield f"data: {json.dumps({'status': 'stream_end'})}\n\n"


def load_explainability_components(output_dir: Path = None) -> Dict:
    """
    Load the explainability pipeline components from saved files.

    Returns a dict with:
    - calibrator: ProbabilityCalibrator for calibrating raw predictions
    - conformal: ConformalClassifier for prediction sets (requires model)
    - tier_classifier: TierClassifier for executive-friendly labels
    - metadata: Feature names, dimensions, etc.

    Note: To use conformal prediction, you need to also load the model
    and call conformal.sklearn_wrapper with the model.
    """
    import pickle

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    explainability_dir = output_dir / "explainability"

    if not explainability_dir.exists():
        return None

    components = {}

    # Load calibrator
    cal_path = explainability_dir / "calibrator.pkl"
    if cal_path.exists():
        components['calibrator'] = ProbabilityCalibrator.load(cal_path)

    # Load tier classifier
    tier_path = explainability_dir / "tier_classifier.pkl"
    if tier_path.exists():
        with open(tier_path, 'rb') as f:
            tier_data = pickle.load(f)
            components['tier_classifier'] = TierClassifier.from_dict(tier_data)

    # Load metadata
    meta_path = explainability_dir / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            components['metadata'] = pickle.load(f)

    # Conformal state (for loading with model)
    conf_path = explainability_dir / "conformal.pkl"
    if conf_path.exists():
        components['conformal_path'] = conf_path

    return components


def explain_prediction(
    raw_probability: float,
    output_dir: Path = None,
) -> Dict:
    """
    Explain a single prediction using the saved explainability pipeline.

    This is a lightweight function that doesn't require loading the full model.
    It takes a raw model probability and returns:
    - Calibrated probability
    - Tier classification with confidence statement
    - Historical accuracy for the tier

    Args:
        raw_probability: Raw sigmoid output from model (0-1)
        output_dir: Directory containing saved pipeline

    Returns:
        Dict with explanation data
    """
    components = load_explainability_components(output_dir)
    if components is None:
        return {"error": "Explainability pipeline not found"}

    calibrator = components.get('calibrator')
    tier_classifier = components.get('tier_classifier')

    if calibrator is None or tier_classifier is None:
        return {"error": "Missing calibrator or tier classifier"}

    # Calibrate the probability
    calibrated_prob = calibrator.calibrate(np.array([raw_probability]))[0]

    # Classify into tier
    tier_result = tier_classifier.classify(calibrated_prob)

    return {
        "raw_probability": raw_probability,
        "calibrated_probability": float(calibrated_prob),
        "tier": tier_result.tier,
        "tier_label": tier_result.label,
        "tier_action": tier_result.action,
        "confidence_statement": tier_result.confidence_statement,
        "historical_accuracy": tier_result.historical_accuracy,
        "color": tier_result.color,
    }
