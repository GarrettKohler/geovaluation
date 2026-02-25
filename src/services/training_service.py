"""
TRAINING FOR BINARY CLASSIFICATION FOR LOOKALIKE CALCULATION
Training service for model training with GPU acceleration.
Provides async training with progress tracking via Server-Sent Events.
Supports multiple concurrent experiments and job management.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Generator, List
from dataclasses import dataclass, field, asdict
import time
import json
import threading
from queue import Queue, Empty
import traceback
import uuid
import shutil

# Import from site_scoring module (at project root)
from site_scoring.config import Config, DEFAULT_OUTPUT_DIR
from site_scoring.model import (
    SiteScoringModel,
    ClusteringModel,
    XGBoostModel,
    create_model,
    XGBOOST_AVAILABLE,
)
from site_scoring.data_loader import DataProcessor, create_data_loaders
from src.services.shap_service import (
    compute_shap_values,
    compute_shap_values_tree,
    compute_cluster_shap_values,
    ShapCache,
)

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

    # Lookalike classifier percentile bounds
    lookalike_lower_percentile: int = 90  # Lower bound (inclusive), 1-99
    lookalike_upper_percentile: int = 100  # Upper bound (inclusive), 1-100

    # Clustering configuration (for clustering task type)
    n_clusters: int = 5  # Number of clusters to discover
    latent_dim: int = 32  # Dimension of autoencoder latent space
    cluster_probability_threshold: float = 0.5  # Min lookalike prob to include in clustering
    pretrain_epochs: int = 20  # Epochs for autoencoder pretraining
    clustering_epochs: int = 30  # Epochs for clustering refinement
    
    # Network filter: None = all networks, or "Gilbarco", "Speedway", "Wayne/Dover"
    network_filter: Optional[str] = None

    # Custom output directory for this experiment
    output_dir: Optional[Path] = None


@dataclass
class TrainingProgress:
    """Training progress update."""
    job_id: str
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
    status: str  # pending, running, completed, error, stopped
    message: str = ""
    best_val_loss: float = float('inf')
    # Feature selection stats
    n_active_features: Optional[int] = None
    fs_reg_loss: float = 0.0
    # Classification metrics (lookalike tasks)
    val_f1: float = 0.0
    val_logloss: float = 0.0
    # MAPE metric (regression only)
    val_mape: float = 0.0
    # Weight/bias distribution histograms (neural network only)
    weight_histograms: Optional[Dict] = None
    # Final metrics (only populated on completion, contains test set results)
    final_metrics: Optional[Dict] = None


MAX_EXPERIMENTS = 10


def _enforce_experiment_limit(experiments_dir: Path, max_count: int = MAX_EXPERIMENTS):
    """Delete oldest experiment directories to stay within the limit (FIFO)."""
    if not experiments_dir.exists():
        return
    dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if len(dirs) < max_count:
        return
    # Sort by modification time, oldest first
    dirs.sort(key=lambda d: d.stat().st_mtime)
    # Delete oldest until we have room for one new experiment
    to_remove = len(dirs) - max_count + 1
    for d in dirs[:to_remove]:
        print(f"Removing old experiment: {d.name}")
        shutil.rmtree(d, ignore_errors=True)


class TrainingJob:
    """
    Manages a single training job.
    """

    def __init__(self, job_id: str, config: TrainingConfig):
        self.job_id = job_id
        self.config = config
        # If output_dir is not set in config, create a job-specific one
        if self.config.output_dir is None:
            experiments_dir = DEFAULT_OUTPUT_DIR / "experiments"
            _enforce_experiment_limit(experiments_dir)
            self.output_dir = experiments_dir / self.job_id
        else:
            self.output_dir = self.config.output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = self.output_dir
        
        self.progress_queue: Queue = Queue()
        self.should_stop = False
        self.final_metrics: Optional[Dict] = None
        self.status = "pending"
        self.start_time = 0.0
        self.error_message = None

    def stop(self):
        """Request training to stop."""
        self.should_stop = True

    def _report_progress(self, progress: TrainingProgress):
        """Add progress update to queue."""
        self.status = progress.status
        self.progress_queue.put(progress)


class JobManager:
    """
    Manages multiple training jobs and a worker queue.
    """
    def __init__(self, max_workers: int = 1):
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: Queue = Queue()
        self.max_workers = max_workers
        self.workers: List[threading.Thread] = []
        self.history: List[Dict] = []
        self.broadcaster: List[Queue] = []  # List of queues to broadcast events to
        self.lock = threading.Lock()
        
        # Start workers
        for i in range(max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"Worker-{i}")
            t.start()
            self.workers.append(t)

    def submit_job(self, config: TrainingConfig) -> str:
        """Submit a job to the queue."""
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        job = TrainingJob(job_id, config)
        
        with self.lock:
            self.active_jobs[job_id] = job
            self.job_queue.put(job_id)
            
        # Broadcast pending status
        self._broadcast(TrainingProgress(
            job_id=job_id, epoch=0, total_epochs=config.epochs,
            train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
            learning_rate=config.learning_rate, elapsed_time=0,
            status="pending", message="Job queued"
        ))
        
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.active_jobs.get(job_id)

    def stop_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job:
            job.stop()
            return True
        return False

    def get_all_jobs(self) -> List[Dict]:
        """Get summary of all active and completed jobs."""
        jobs = []
        # Add active jobs
        for job_id, job in self.active_jobs.items():
            jobs.append({
                "job_id": job_id,
                "status": job.status,
                "model_type": job.config.model_type,
                "task_type": job.config.task_type,
                "target": job.config.target,
                "start_time": job.start_time,
                "metrics": job.final_metrics
            })
        # Add history (archived jobs) if we implement archiving
        return jobs

    def subscribe(self) -> Queue:
        """Subscribe to global progress events."""
        q = Queue()
        with self.lock:
            self.broadcaster.append(q)
        return q

    def unsubscribe(self, q: Queue):
        """Unsubscribe from global progress events."""
        with self.lock:
            if q in self.broadcaster:
                self.broadcaster.remove(q)

    def _broadcast(self, progress: TrainingProgress):
        """Broadcast progress to all subscribers."""
        with self.lock:
            for q in self.broadcaster:
                q.put(progress)

    def _worker_loop(self):
        """Worker thread loop to process jobs."""
        while True:
            try:
                job_id = self.job_queue.get()
                job = self.active_jobs.get(job_id)
                
                if job:
                    # Run training
                    self._run_training_wrapper(job)
                
                self.job_queue.task_done()
            except Exception as e:
                print(f"Worker error: {e}")
                traceback.print_exc()

    def _run_training_wrapper(self, job: TrainingJob):
        """Wrapper to run training and handle progress reporting."""
        job.start_time = time.time()
        
        # Helper to report progress for this job AND broadcast it
        def report(p: TrainingProgress):
            p.job_id = job.job_id
            job._report_progress(p)
            self._broadcast(p)

        try:
            # Delegate to the training logic
            # We reuse the existing _run_training logic but adapted for the JobManager
            run_training_logic(job, report)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            job.error_message = error_msg
            report(TrainingProgress(
                job_id=job.job_id, epoch=0, total_epochs=job.config.epochs,
                train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
                learning_rate=job.config.learning_rate, elapsed_time=time.time() - job.start_time,
                status="error", message=error_msg
            ))

# Global Job Manager
job_manager = JobManager(max_workers=1) # Set to 1 for M4 to avoid memory thrashing, or 2 if sufficient RAM


def _dataloaders_to_numpy(train_loader, val_loader, test_loader, processor, config):
    """
    Convert PyTorch DataLoaders to numpy arrays for tree-based models.

    DataLoaders yield: (numeric_tensor, categorical_tensor, boolean_tensor, target_tensor)
    Tree models need: X (all features concatenated), y (target)

    For regression tasks, targets are inverse-transformed back to original dollar scale
    because tree models train on raw values (unlike neural networks which use standardized targets).

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
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

    # For regression: inverse-transform targets to original $ scale
    # Tree models train on raw revenue, not standardized values
    if config.task_type != "lookalike" and processor.target_scaler is not None:
        y_train = processor.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val = processor.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_test = processor.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Build feature names list matching the concatenation order: numeric + categorical + boolean
    feature_names = (
        list(config.numeric_features) +
        list(config.categorical_features) +
        list(config.boolean_features)
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def _create_xgboost_progress_callback(report_fn, job, config, X_val, y_val, total_rounds, start_time, report_interval=10):
    """
    Factory that creates an XGBoost TrainingCallback for SSE progress reporting.

    Uses lazy import to avoid importing xgboost at module load time
    (XGBoost may not be available if libomp is missing).
    """
    import xgboost as xgb

    class XGBoostProgressCallback(xgb.callback.TrainingCallback):
        """Reports training progress to SSE stream every N rounds."""

        def __init__(self):
            super().__init__()
            self.report_fn = report_fn
            self.job = job
            self.config = config
            self.X_val = X_val
            self.y_val = y_val
            self.total_rounds = total_rounds
            self.start_time = start_time
            self.report_interval = report_interval
            self.best_val_rmse = float('inf')

        def after_iteration(self, model, epoch, evals_log):
            """Called by XGBoost after each boosting round."""
            if epoch % self.report_interval != 0 and epoch != self.total_rounds - 1:
                return False  # Don't stop training

            # Extract eval metric from evals_log
            val_rmse = 0.0
            if evals_log:
                for dataset_name, metrics in evals_log.items():
                    if 'rmse' in metrics:
                        val_rmse = float(metrics['rmse'][-1])

            # Predict on validation set for detailed metrics
            try:
                dval = xgb.DMatrix(self.X_val)
                preds = model.predict(dval)
                errors = preds - self.y_val
                mae = float(np.mean(np.abs(errors)))
                rmse = float(np.sqrt(np.mean(errors**2)))
                ss_res = np.sum(errors**2)
                ss_tot = np.sum((self.y_val - np.mean(self.y_val))**2)
                r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
                denominator = (np.abs(preds) + np.abs(self.y_val)) / 2
                denominator = np.where(denominator == 0, 1, denominator)
                smape = float(np.mean(np.abs(errors) / denominator) * 100)
            except Exception:
                mae, rmse, r2, smape = 0.0, val_rmse, 0.0, 0.0

            if rmse < self.best_val_rmse:
                self.best_val_rmse = rmse

            self.report_fn(TrainingProgress(
                job_id=self.job.job_id,
                epoch=epoch + 1,
                total_epochs=self.total_rounds,
                train_loss=val_rmse,
                val_loss=rmse,
                val_mae=mae,
                val_smape=smape,
                val_rmse=rmse,
                val_r2=r2,
                learning_rate=self.config.learning_rate,
                elapsed_time=float(time.time() - self.start_time),
                status="running",
                message=f"Round {epoch + 1}/{self.total_rounds}",
                best_val_loss=self.best_val_rmse,
            ))
            return False  # Don't stop training

    return XGBoostProgressCallback()


def _run_tree_training(job, config, pytorch_config, train_loader, val_loader, test_loader, processor, report_callback, start_time):
    """
    Training path for XGBoost models.

    Converts DataLoaders to numpy, trains with per-round progress callbacks,
    evaluates on held-out test set, computes SHAP TreeExplainer importance,
    and saves model artifacts.
    """
    import pickle

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=0, total_epochs=config.epochs,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message=f"Converting data for {config.model_type}..."
    ))

    # Step 1: Convert DataLoaders to numpy arrays
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = _dataloaders_to_numpy(
        train_loader, val_loader, test_loader, processor, pytorch_config
    )
    print(f"[{job.job_id}] Tree data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Step 2: Create model via factory
    n_estimators = config.epochs  # UI "epochs" maps to boosting rounds for tree models
    if n_estimators < 100:
        n_estimators = 500  # sensible default for tree models if user set low epochs

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

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=0, total_epochs=n_estimators,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message=f"Training {config.model_type} ({n_estimators} rounds)..."
    ))

    # Step 3: Fit with progress callbacks
    callbacks = []
    if config.model_type == "xgboost":
        callbacks.append(_create_xgboost_progress_callback(
            report_fn=report_callback,
            job=job,
            config=config,
            X_val=X_val,
            y_val=y_val,
            total_rounds=n_estimators,
            start_time=start_time,
            report_interval=10,
        ))
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, callbacks=callbacks)

    # Step 4: Test set evaluation
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=n_estimators, total_epochs=n_estimators,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message="Evaluating on test set..."
    ))

    test_preds = model.predict(X_test)
    test_mae, test_smape, test_rmse, test_r2, test_loss, test_mape, test_f1, test_logloss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if config.task_type == "lookalike":
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = test_preds
        try:
            test_r2 = float(roc_auc_score(y_test, probs))
        except Exception:
            test_r2 = 0.0
        try:
            binary_preds = (probs >= 0.5).astype(int)
            test_f1 = float(f1_score(y_test.astype(int), binary_preds, zero_division=0))
        except Exception:
            test_f1 = 0.0
        try:
            test_logloss = float(log_loss(y_test, probs))
        except Exception:
            test_logloss = 0.0
    else:
        errors = test_preds - y_test
        test_mae = float(np.mean(np.abs(errors)))
        test_rmse = float(np.sqrt(np.mean(errors**2)))
        test_loss = test_rmse  # Use RMSE as "loss" for tree models
        # SMAPE
        denominator = (np.abs(test_preds) + np.abs(y_test)) / 2
        denominator = np.where(denominator == 0, 1, denominator)
        test_smape = float(np.mean(np.abs(errors) / denominator) * 100)
        # MAPE (exclude zero actuals)
        nonzero_mask = np.abs(y_test) > 0
        test_mape = float(np.mean(np.abs(errors[nonzero_mask]) / np.abs(y_test[nonzero_mask])) * 100) if nonzero_mask.any() else 0.0
        # R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        test_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Step 5: SHAP feature importance (TreeExplainer — fast for tree models)
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=n_estimators, total_epochs=n_estimators,
        train_loss=test_rmse, val_loss=test_rmse,
        val_mae=test_mae, val_smape=test_smape, val_rmse=test_rmse, val_r2=test_r2,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message="Computing SHAP feature importance..."
    ))

    def shap_progress(msg: str):
        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=n_estimators, total_epochs=n_estimators,
            train_loss=test_rmse, val_loss=test_rmse,
            val_mae=test_mae, val_smape=test_smape, val_rmse=test_rmse, val_r2=test_r2,
            learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
            status="running", message=msg
        ))

    shap_success = compute_shap_values_tree(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        output_dir=job.output_dir,
        n_explain=500,
        task_type=config.task_type,
        progress_callback=shap_progress,
    )

    if shap_success:
        shap_cache = ShapCache(job.output_dir)
        shap_importance = shap_cache.get_feature_importance(top_n=20)
        if shap_importance:
            with open(job.output_dir / "shap_importance.json", "w") as f:
                json.dump(shap_importance, f, indent=2)

    # Step 6: Save model artifacts
    if config.model_type == "xgboost":
        # Clear training callbacks before saving (they contain unpicklable local classes)
        model.model.set_params(callbacks=None)
        model.model.save_model(str(job.output_dir / "best_model.json"))

    # Also pickle the wrapper for easy reloading
    with open(job.output_dir / "model_wrapper.pkl", "wb") as f:
        pickle.dump(model, f)

    processor.save(job.output_dir / "preprocessor.pkl")

    # Save model metadata
    with open(job.output_dir / "model_metadata.json", "w") as f:
        json.dump({
            "model_type": config.model_type,
            "task_type": config.task_type,
            "n_estimators": n_estimators,
            "best_iteration": model.best_iteration,
            "feature_names": feature_names,
            "test_metrics": {
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_smape": test_smape,
                "test_mape": test_mape,
                "test_f1": test_f1,
                "test_logloss": test_logloss,
            },
        }, f, indent=2)

    # Step 7: Export classification results (lookalike tasks only)
    if config.task_type == "lookalike":
        try:
            _export_classification_results(
                job=job,
                processor=processor,
                test_predictions=probs if 'probs' in dir() else test_preds,
                test_targets=y_test,
                test_roc_auc=test_r2,
                report_callback=report_callback,
            )
        except Exception as e:
            print(f"Warning: Classification export failed: {e}")

    # Step 8: Report completion
    job.final_metrics = {
        "test_loss": float(test_loss),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_smape": float(test_smape),
        "test_mape": float(test_mape),
        "test_f1": float(test_f1),
        "test_logloss": float(test_logloss),
        "val_loss": 0.0,
        "val_r2": 0.0,
        "val_mae": 0.0,
        "shap_available": shap_success,
        "experiment_dir": str(job.output_dir.name),
    }

    print(f"\n{'='*60}")
    print(f"[{config.model_type.upper()}] FINAL TEST METRICS")
    if config.task_type == "lookalike":
        print(f"  ROC-AUC:    {test_r2:.6f}")
        print(f"  Test F1:    {test_f1:.6f}")
        print(f"  Log Loss:   {test_logloss:.6f}")
    else:
        print(f"  Test R²:    {test_r2:.6f}")
        print(f"  Test MAE:   ${test_mae:,.2f}")
        print(f"  Test RMSE:  ${test_rmse:,.2f}")
        print(f"  Test SMAPE: {test_smape:.2f}%")
        print(f"  Test MAPE:  {test_mape:.2f}%")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"{'='*60}\n")

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=n_estimators, total_epochs=n_estimators,
        train_loss=0.0, val_loss=float(test_loss),
        val_mae=float(test_mae), val_smape=float(test_smape),
        val_rmse=float(test_rmse), val_r2=float(test_r2),
        learning_rate=config.learning_rate, elapsed_time=float(time.time() - start_time),
        status="completed", message="Training completed successfully",
        best_val_loss=float(test_rmse),
        val_mape=float(test_mape),
        val_f1=float(test_f1), val_logloss=float(test_logloss),
        final_metrics=job.final_metrics,
    ))


def _export_classification_results(job, processor, test_predictions, test_targets, test_roc_auc, report_callback=None):
    """
    Export classification results after a lookalike training run.

    Produces three CSV files in the experiment directory:
      1. training_sites.csv — all active sites used for training with labels
      2. test_predictions.csv — test split predictions with probabilities
      3. non_active_classification.csv — non-active sites scored by the model

    Args:
        job: TrainingJob with output_dir
        processor: DataProcessor with source_gtvids, source_revenues, and split indices
        test_predictions: numpy array of predicted probabilities for test set
        test_targets: numpy array of actual labels (0/1) for test set
        test_roc_auc: float, ROC-AUC score on test set
        report_callback: optional SSE progress callback
    """
    import csv

    experiment_dir = job.output_dir

    def _report(msg):
        if report_callback:
            report_callback(TrainingProgress(
                job_id=job.job_id, epoch=0, total_epochs=0,
                train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
                learning_rate=0, elapsed_time=0,
                status="running", message=msg
            ))

    # ── Export 1: Training Sites CSV ─────────────────────────────────────
    _report("Exporting classification results: training sites...")
    if processor.source_gtvids and processor.source_revenues:
        threshold = getattr(processor, 'top_performer_threshold', None)
        training_sites_path = experiment_dir / "training_sites.csv"
        with open(training_sites_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gtvid", "avg_monthly_revenue", "actual_label"])
            for gtvid, revenue in zip(processor.source_gtvids, processor.source_revenues):
                label = 1 if (threshold is not None and revenue >= threshold) else 0
                writer.writerow([gtvid, revenue, label])
        print(f"  Exported {len(processor.source_gtvids)} training sites → {training_sites_path.name}")

    # ── Export 2: Test Predictions CSV ───────────────────────────────────
    _report("Exporting classification results: test predictions...")
    if (processor.source_gtvids and
            hasattr(processor, 'test_indices') and
            processor.test_indices is not None):
        test_predictions_path = experiment_dir / "test_predictions.csv"
        test_idx = processor.test_indices
        with open(test_predictions_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gtvid", "predicted_probability", "actual_label", "model_roc_auc"])
            for i, idx in enumerate(test_idx):
                if i < len(test_predictions) and idx < len(processor.source_gtvids):
                    gtvid = processor.source_gtvids[idx]
                    prob = float(test_predictions[i])
                    label = int(test_targets[i])
                    writer.writerow([gtvid, prob, label, test_roc_auc])
        print(f"  Exported {len(test_idx)} test predictions → {test_predictions_path.name}")

    # ── Export 3: Non-Active Site Predictions ────────────────────────────
    _report("Exporting classification results: scoring non-active sites...")
    try:
        from site_scoring.predict import BatchPredictor
        from site_scoring.data_transform import get_all_sites_for_prediction
        import polars as pl_export

        all_sites_df = get_all_sites_for_prediction()

        # Filter to non-Active sites
        if 'status' in all_sites_df.columns:
            non_active_df = all_sites_df.filter(pl_export.col('status') != 'Active')
        else:
            print("  Warning: 'status' column not found, skipping non-active export")
            non_active_df = None

        if non_active_df is not None and len(non_active_df) > 0:
            predictor = BatchPredictor(experiment_dir)
            scores = predictor.predict(non_active_df)

            threshold = getattr(processor, 'top_performer_threshold', None)
            target_col = job.config.target if hasattr(job.config, 'target') else 'avg_monthly_revenue'

            non_active_path = experiment_dir / "non_active_classification.csv"
            with open(non_active_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gtvid", "status", "predicted_probability", "actual_label", "model_roc_auc"])
                gtvids_list = non_active_df['gtvid'].to_list()
                statuses_list = non_active_df['status'].to_list()
                revenues = non_active_df[target_col].fill_null(0).to_list() if target_col in non_active_df.columns else [0] * len(gtvids_list)
                for gtvid, status, revenue in zip(gtvids_list, statuses_list, revenues):
                    prob = scores.get(gtvid, 0.0)
                    label = 1 if (threshold is not None and revenue >= threshold) else 0
                    writer.writerow([gtvid, status, prob, label, test_roc_auc])
            print(f"  Exported {len(gtvids_list)} non-active sites → {non_active_path.name}")
    except Exception as e:
        print(f"  Warning: Non-active site export failed: {e}")

    _report("Classification exports complete.")


def _compute_weight_histograms(model, n_bins=30):
    """Compute histogram data for all linear layer weights and biases."""
    all_weights = []
    all_biases = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bn' in name or 'embedding' in name:
            continue
        data = param.detach().cpu().numpy().flatten()
        if 'weight' in name and param.ndim == 2:
            all_weights.append(data)
        elif 'bias' in name and param.ndim == 1:
            all_biases.append(data)
    result = {}
    if all_weights:
        w = np.concatenate(all_weights)
        counts, edges = np.histogram(w, bins=n_bins)
        result['weights'] = {
            'bin_edges': [float(x) for x in edges],
            'counts': [int(x) for x in counts],
        }
    if all_biases:
        b = np.concatenate(all_biases)
        counts, edges = np.histogram(b, bins=n_bins)
        result['biases'] = {
            'bin_edges': [float(x) for x in edges],
            'counts': [int(x) for x in counts],
        }
    return result if result else None


def _run_clustering_training(job, config, pytorch_config, train_loader, val_loader, test_loader, processor, report_callback, start_time):
    """
    Deep Embedded Clustering (DEC) training path.

    Two-phase training:
      Phase 1 (Pretrain): Train autoencoder on reconstruction loss (MSE)
      Phase 2 (Clustering): Initialize centroids via k-means, then refine
                             with KL-divergence + reconstruction loss

    After training, computes per-cluster SHAP feature importance.
    """
    device = torch.device(config.device)
    n_clusters = config.n_clusters
    pretrain_epochs = config.pretrain_epochs if hasattr(config, 'pretrain_epochs') else 20
    clustering_epochs = config.clustering_epochs if hasattr(config, 'clustering_epochs') else 30
    total_epochs = pretrain_epochs + clustering_epochs

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=0, total_epochs=total_epochs,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message=f"Creating DEC model ({n_clusters} clusters, latent_dim={pytorch_config.latent_dim})..."
    ))

    # Create clustering model
    model = ClusteringModel(
        n_numeric=processor.n_numeric_features,
        n_boolean=processor.n_boolean_features,
        categorical_vocab_sizes=processor.categorical_vocab_sizes,
        embedding_dim=pytorch_config.embedding_dim,
        latent_dim=pytorch_config.latent_dim,
        n_clusters=n_clusters,
        dropout=pytorch_config.dropout,
    ).to(device)

    criterion_recon = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_loss = float('inf')
    best_model_state = None

    # =========================================================================
    # Phase 1: Autoencoder Pretraining
    # =========================================================================
    for epoch in range(1, pretrain_epochs + 1):
        if job.should_stop:
            report_callback(TrainingProgress(
                job_id=job.job_id, epoch=epoch, total_epochs=total_epochs,
                train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
                learning_rate=optimizer.param_groups[0]["lr"], elapsed_time=time.time() - start_time,
                status="stopped", message="Training stopped by user"
            ))
            return

        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for numeric, categorical, boolean, _target in train_loader:
            numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
            optimizer.zero_grad(set_to_none=True)

            result = model(numeric, categorical, boolean)
            # Reconstruction target: the combined input features
            x_input = model._prepare_input(numeric, categorical, boolean)
            loss = criterion_recon(result['x_reconstructed'], x_input.detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        train_loss = total_train_loss / n_batches if n_batches > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for numeric, categorical, boolean, _target in val_loader:
                numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
                result = model(numeric, categorical, boolean)
                x_input = model._prepare_input(numeric, categorical, boolean)
                loss = criterion_recon(result['x_reconstructed'], x_input.detach())
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()

        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=epoch, total_epochs=total_epochs,
            train_loss=float(train_loss), val_loss=float(val_loss),
            val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_time=float(time.time() - start_time),
            status="running",
            message=f"Phase 1 (Pretrain): Epoch {epoch}/{pretrain_epochs} — Recon loss: {val_loss:.4f}",
            best_val_loss=float(best_loss),
            weight_histograms=_compute_weight_histograms(model),
        ))

    # Restore best pretrained model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # =========================================================================
    # Initialize centroids via k-means on latent representations
    # =========================================================================
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=pretrain_epochs, total_epochs=total_epochs,
        train_loss=0, val_loss=float(best_loss), val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        elapsed_time=float(time.time() - start_time),
        status="running", message=f"Initializing {n_clusters} cluster centroids (k-means)..."
    ))

    model.eval()
    all_z = []
    with torch.no_grad():
        for numeric, categorical, boolean, _target in train_loader:
            numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
            z = model.encode(numeric, categorical, boolean)
            all_z.append(z.cpu())
        for numeric, categorical, boolean, _target in val_loader:
            numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
            z = model.encode(numeric, categorical, boolean)
            all_z.append(z.cpu())

    z_all = torch.cat(all_z)
    initial_labels = model.initialize_centroids(z_all)
    print(f"[{job.job_id}] K-means init: {np.bincount(initial_labels)} samples per cluster")

    # =========================================================================
    # Phase 2: Clustering Refinement (KL-divergence + reconstruction)
    # =========================================================================
    optimizer_phase2 = AdamW(model.parameters(), lr=config.learning_rate * 0.1, weight_decay=config.weight_decay)
    scheduler_phase2 = ReduceLROnPlateau(optimizer_phase2, mode="min", factor=0.5, patience=5)
    recon_weight = 0.1  # Balance reconstruction vs clustering loss

    best_cluster_loss = float('inf')
    best_cluster_state = None

    for epoch in range(1, clustering_epochs + 1):
        if job.should_stop:
            report_callback(TrainingProgress(
                job_id=job.job_id, epoch=pretrain_epochs + epoch, total_epochs=total_epochs,
                train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
                learning_rate=optimizer_phase2.param_groups[0]["lr"], elapsed_time=time.time() - start_time,
                status="stopped", message="Training stopped by user"
            ))
            return

        model.train()
        total_train_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0

        for numeric, categorical, boolean, _target in train_loader:
            numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
            optimizer_phase2.zero_grad(set_to_none=True)

            result = model(numeric, categorical, boolean)
            x_input = model._prepare_input(numeric, categorical, boolean)

            # Reconstruction loss
            loss_recon = criterion_recon(result['x_reconstructed'], x_input.detach())

            # Clustering loss (KL-divergence)
            q = result['q']
            p = model.target_distribution(q).detach()
            loss_kl = model.clustering_loss(q, p)

            loss = loss_kl + recon_weight * loss_recon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_phase2.step()
            total_train_loss += loss.item()
            total_kl_loss += loss_kl.item()
            n_batches += 1

        train_loss = total_train_loss / n_batches if n_batches > 0 else 0
        kl_loss = total_kl_loss / n_batches if n_batches > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for numeric, categorical, boolean, _target in val_loader:
                numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
                result = model(numeric, categorical, boolean)
                x_input = model._prepare_input(numeric, categorical, boolean)
                loss_recon = criterion_recon(result['x_reconstructed'], x_input.detach())
                q = result['q']
                p = model.target_distribution(q).detach()
                loss_kl = model.clustering_loss(q, p)
                total_val_loss += (loss_kl + recon_weight * loss_recon).item()

        val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler_phase2.step(val_loss)

        if val_loss < best_cluster_loss:
            best_cluster_loss = val_loss
            best_cluster_state = model.state_dict().copy()

        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=pretrain_epochs + epoch, total_epochs=total_epochs,
            train_loss=float(train_loss), val_loss=float(val_loss),
            val_mae=float(kl_loss), val_smape=0, val_rmse=0, val_r2=0,
            learning_rate=float(optimizer_phase2.param_groups[0]["lr"]),
            elapsed_time=float(time.time() - start_time),
            status="running",
            message=f"Phase 2 (Clustering): Epoch {epoch}/{clustering_epochs} — KL: {kl_loss:.4f}",
            best_val_loss=float(best_cluster_loss),
            weight_histograms=_compute_weight_histograms(model),
        ))

    # Restore best model
    if best_cluster_state:
        model.load_state_dict(best_cluster_state)

    # =========================================================================
    # Final: Get cluster assignments and compute metrics
    # =========================================================================
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=total_epochs, total_epochs=total_epochs,
        train_loss=0, val_loss=float(best_cluster_loss), val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=0, elapsed_time=float(time.time() - start_time),
        status="running", message="Computing final cluster assignments..."
    ))

    model.eval()
    # Collect ALL data (train + val + test) for final cluster assignments
    all_z = []
    all_numeric, all_cat, all_bool = [], [], []
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            for numeric, categorical, boolean, _target in loader:
                numeric, categorical, boolean = numeric.to(device), categorical.to(device), boolean.to(device)
                z = model.encode(numeric, categorical, boolean)
                all_z.append(z.cpu())
                all_numeric.append(numeric.cpu())
                all_cat.append(categorical.cpu())
                all_bool.append(boolean.cpu())

    z_all = torch.cat(all_z)
    cluster_assignments = model.get_cluster_assignments(z_all).numpy()

    # Build concatenated feature matrix for SHAP
    numeric_all = torch.cat(all_numeric).numpy()
    cat_all = torch.cat(all_cat).numpy().astype(np.float64)
    bool_all = torch.cat(all_bool).numpy()
    X_all = np.concatenate([numeric_all, cat_all, bool_all], axis=1)

    feature_names = (
        list(pytorch_config.numeric_features) +
        list(pytorch_config.categorical_features) +
        list(pytorch_config.boolean_features)
    )

    # Cluster distribution
    cluster_sizes = np.bincount(cluster_assignments, minlength=n_clusters)
    print(f"\n{'='*60}")
    print(f"[DEC] FINAL CLUSTER ASSIGNMENTS ({n_clusters} clusters)")
    for i in range(n_clusters):
        print(f"  Cluster {i}: {cluster_sizes[i]:,} sites ({cluster_sizes[i]/len(cluster_assignments)*100:.1f}%)")
    print(f"{'='*60}\n")

    # Silhouette score
    from sklearn.metrics import silhouette_score
    try:
        sil_score = float(silhouette_score(z_all.numpy(), cluster_assignments))
    except Exception:
        sil_score = 0.0

    # Compute per-cluster SHAP feature importance
    def shap_progress(msg: str):
        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=total_epochs, total_epochs=total_epochs,
            train_loss=0, val_loss=float(best_cluster_loss), val_mae=0, val_smape=0, val_rmse=0, val_r2=float(sil_score),
            learning_rate=0, elapsed_time=float(time.time() - start_time),
            status="running", message=msg
        ))

    cluster_shap_results = compute_cluster_shap_values(
        model=model,
        cluster_assignments=cluster_assignments,
        X_data=X_all,
        feature_names=feature_names,
        n_numeric=processor.n_numeric_features,
        n_categorical=len(pytorch_config.categorical_features),
        output_dir=job.output_dir,
        n_explain=100,
        progress_callback=shap_progress,
    )

    # Save model and metadata
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": pytorch_config,
        "cluster_assignments": cluster_assignments,
        "cluster_sizes": cluster_sizes.tolist(),
        "silhouette_score": sil_score,
    }, job.output_dir / "best_model.pt")
    processor.save(job.output_dir / "preprocessor.pkl")

    with open(job.output_dir / "model_metadata.json", "w") as f:
        json.dump({
            "model_type": "clustering",
            "task_type": "clustering",
            "n_clusters": n_clusters,
            "latent_dim": pytorch_config.latent_dim,
            "pretrain_epochs": pretrain_epochs,
            "clustering_epochs": clustering_epochs,
            "silhouette_score": sil_score,
            "cluster_sizes": cluster_sizes.tolist(),
            "total_samples": len(cluster_assignments),
            "feature_names": feature_names,
            "shap_available": cluster_shap_results is not None,
        }, f, indent=2)

    # Report completion
    job.final_metrics = {
        "test_loss": float(best_cluster_loss),
        "test_r2": float(sil_score),  # Repurpose R² field for silhouette score
        "test_mae": 0.0,
        "test_rmse": 0.0,
        "test_smape": 0.0,
        "test_mape": 0.0,
        "val_loss": float(best_cluster_loss),
        "val_r2": float(sil_score),
        "val_mae": 0.0,
        "shap_available": cluster_shap_results is not None,
        "experiment_dir": str(job.output_dir.name),
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.tolist(),
        "silhouette_score": sil_score,
    }

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=total_epochs, total_epochs=total_epochs,
        train_loss=0.0, val_loss=float(best_cluster_loss),
        val_mae=0.0, val_smape=0.0, val_rmse=0.0, val_r2=float(sil_score),
        learning_rate=0.0, elapsed_time=float(time.time() - start_time),
        status="completed",
        message=f"Clustering complete: {n_clusters} clusters, silhouette={sil_score:.3f}",
        best_val_loss=float(best_cluster_loss),
        final_metrics=job.final_metrics,
    ))


def run_training_logic(job: TrainingJob, report_callback):
    """
    Refactored training logic that accepts a job object and a reporting callback.
    This contains the core logic previously in TrainingJob._run_training.
    """
    config = job.config
    start_time = job.start_time
    
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=0, total_epochs=config.epochs,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=0,
        status="running", message="Loading and processing data..."
    ))

    # Create PyTorch config from user config with Apple Silicon optimizations
    pytorch_config = Config()
    # Override output_dir with job specific dir
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

    # Apply Apple Silicon optimizations
    pytorch_config.num_workers = config.num_workers
    pytorch_config.pin_memory = config.pin_memory
    pytorch_config.prefetch_factor = config.prefetch_factor
    pytorch_config.task_type = config.task_type

    # Lookalike classifier percentile bounds
    pytorch_config.lookalike_lower_percentile = config.lookalike_lower_percentile
    pytorch_config.lookalike_upper_percentile = config.lookalike_upper_percentile

    # Network subset filter
    pytorch_config.network_filter = config.network_filter

    # Clustering parameters
    pytorch_config.n_clusters = config.n_clusters
    pytorch_config.latent_dim = config.latent_dim

    # Apply model preset
    if config.model_preset:
        pytorch_config.apply_model_preset(config.model_preset)

        # Apply user feature selection
        if config.selected_features:
            from site_scoring.config import filter_features_by_selection
            filtered = filter_features_by_selection(
                config.model_preset,
                config.selected_features
            )
            pytorch_config.numeric_features = filtered["numeric"]
            pytorch_config.categorical_features = filtered["categorical"]
            pytorch_config.boolean_features = filtered["boolean"]
            print(f"[{job.job_id}] User feature selection applied: {len(config.selected_features)} features selected")

    # Save config to job dir
    with open(job.output_dir / "config.json", "w") as f:
        # Convert dataclass to dict, handle Path objects
        cfg_dict = asdict(config)
        cfg_dict['output_dir'] = str(cfg_dict['output_dir'])
        # Include resolved feature lists so the config is self-documenting
        cfg_dict['training_features'] = {
            'numeric': pytorch_config.numeric_features,
            'categorical': pytorch_config.categorical_features,
            'boolean': pytorch_config.boolean_features,
        }
        json.dump(cfg_dict, f, indent=2)

    # Load data
    train_loader, val_loader, test_loader, processor = create_data_loaders(pytorch_config)

    # Dispatch: tree-based models (XGBoost) use a separate training path
    if config.model_type == "xgboost":
        _run_tree_training(
            job, config, pytorch_config, train_loader, val_loader,
            test_loader, processor, report_callback, start_time
        )
        return

    # Dispatch: clustering uses Deep Embedded Clustering (DEC) path
    if config.task_type == "clustering":
        _run_clustering_training(
            job, config, pytorch_config, train_loader, val_loader,
            test_loader, processor, report_callback, start_time
        )
        return

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=0, total_epochs=config.epochs,
        train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
        learning_rate=config.learning_rate, elapsed_time=time.time() - start_time,
        status="running", message=f"Data loaded. Creating model on {config.device}..."
    ))

    # Neural Network training
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

    # Initialize feature selection
    fs_trainer = None
    if config.feature_selection_method != "none":
        # ... Feature selection init code (simplified for brevity, assume similar logic) ...
        # For full implementation we would copy the logic from original file.
        # Given "Do Everything", I will replicate the logic.
        try:
            fs_config = get_preset(config.feature_selection_method)
        except ValueError:
            fs_config = FeatureSelectionConfig(
                method=FeatureSelectionMethod.STOCHASTIC_GATES,
                stg_lambda=config.stg_lambda,
                stg_sigma=config.stg_sigma,
                run_shap_validation=config.run_shap_validation,
                track_gradients=config.track_gradients,
            )
            
        all_feature_names = (
            list(pytorch_config.numeric_features) +
            list(pytorch_config.categorical_features) +
            list(pytorch_config.boolean_features)
        )
        
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
            device=config.device,
        )
        
        if (fs_config.method == FeatureSelectionMethod.STOCHASTIC_GATES and
            fs_trainer is not None and fs_trainer.stg_gates is not None):
             model = STGAugmentedModel(model, fs_trainer.stg_gates).to(torch.device(config.device))

    # Loss function
    device = torch.device(config.device)
    if config.task_type == "lookalike":
        pos_weight = torch.tensor([9.0], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.HuberLoss(delta=1.0)
        
    optimizer = AdamW(model.parameters(), lr=pytorch_config.learning_rate, weight_decay=pytorch_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=pytorch_config.scheduler_patience)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config.epochs + 1):
        if job.should_stop:
            report_callback(TrainingProgress(
                job_id=job.job_id, epoch=epoch, total_epochs=config.epochs,
                train_loss=0, val_loss=0, val_mae=0, val_smape=0, val_rmse=0, val_r2=0,
                learning_rate=optimizer.param_groups[0]["lr"], elapsed_time=time.time() - start_time,
                status="stopped", message="Training stopped by user"
            ))
            return

        # Train
        model.train()
        total_train_loss = 0.0
        n_batches = 0
        total_fs_reg_loss = 0.0
        
        for numeric, categorical, boolean, target in train_loader:
            numeric, categorical, boolean, target = numeric.to(device), categorical.to(device), boolean.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(numeric, categorical, boolean)
            task_loss = criterion(predictions, target)
            
            loss = task_loss
            if hasattr(model, 'get_regularization_loss'):
                fs_reg_loss = model.get_regularization_loss()
                loss += fs_reg_loss
                total_fs_reg_loss += fs_reg_loss.item()
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += task_loss.item()
            n_batches += 1
            
        train_loss = total_train_loss / n_batches if n_batches > 0 else 0
        fs_reg_loss_avg = total_fs_reg_loss / n_batches if n_batches > 0 else 0
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targs = []
        
        with torch.no_grad():
            for numeric, categorical, boolean, target in val_loader:
                numeric, categorical, boolean, target = numeric.to(device), categorical.to(device), boolean.to(device), target.to(device)
                preds = model(numeric, categorical, boolean)
                loss = criterion(preds, target)
                total_val_loss += loss.item()
                all_preds.append(preds.cpu())
                all_targs.append(target.cpu())
        
        val_loss = float(total_val_loss / len(val_loader))
        
        # Calculate metrics (simplified logic matching original)
        predictions_np = torch.cat(all_preds).numpy()
        targets_np = torch.cat(all_targs).numpy()
        
        # ... Metric calculation logic (reused) ...
        mae, smape, rmse, r2, f1, logloss, mape = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if config.task_type == "lookalike":
             from sklearn.metrics import roc_auc_score, f1_score, log_loss
             probs = 1 / (1 + np.exp(-predictions_np.flatten()))
             try:
                 r2 = float(roc_auc_score(targets_np.flatten(), probs))  # AUC
             except: pass
             try:
                 binary_preds = (probs >= 0.5).astype(int)
                 f1 = float(f1_score(targets_np.flatten().astype(int), binary_preds, zero_division=0))
             except: pass
             try:
                 logloss = float(log_loss(targets_np.flatten(), probs))
             except: pass
        else:
            # Regression metrics
            preds_orig = processor.target_scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()
            targs_orig = processor.target_scaler.inverse_transform(targets_np.reshape(-1, 1)).flatten()
            errors = preds_orig - targs_orig
            mae = float(np.mean(np.abs(errors)))
            # MAPE (exclude zero actuals)
            nonzero_mask = np.abs(targs_orig) > 0
            mape = float(np.mean(np.abs(errors[nonzero_mask]) / np.abs(targs_orig[nonzero_mask])) * 100) if nonzero_mask.any() else 0.0
            # R2
            ss_res = np.sum(errors**2)
            ss_tot = np.sum((targs_orig - np.mean(targs_orig))**2)
            r2 = float(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)

        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        n_active = None
        if fs_trainer:
             stats = fs_trainer.on_epoch_end(epoch)
             n_active = stats.get('n_active_features')

        weight_hists = _compute_weight_histograms(model)

        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=epoch, total_epochs=config.epochs,
            train_loss=float(train_loss), val_loss=float(val_loss),
            val_mae=float(mae), val_smape=float(smape), val_rmse=float(rmse), val_r2=float(r2),
            learning_rate=float(current_lr), elapsed_time=float(time.time() - start_time),
            status="running", message=f"Epoch {epoch}/{config.epochs}",
            best_val_loss=float(best_val_loss), n_active_features=n_active, fs_reg_loss=float(fs_reg_loss_avg),
            val_f1=float(f1), val_logloss=float(logloss),
            val_mape=float(mape), weight_histograms=weight_hists
        ))

        if patience_counter >= pytorch_config.early_stopping_patience:
            break

    # Save best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # ==========================================================================
    # CRITICAL: Evaluate on TEST SET for unbiased final metrics
    # The validation metrics during training are biased because they influenced
    # model selection (early stopping). Test set was never seen during training.
    # ==========================================================================
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=config.epochs, total_epochs=config.epochs,
        train_loss=float(train_loss), val_loss=float(best_val_loss),
        val_mae=0.0, val_smape=0.0, val_rmse=0.0, val_r2=0.0,
        learning_rate=float(current_lr), elapsed_time=float(time.time() - start_time),
        status="running", message="Evaluating on test set..."
    ))

    model.eval()
    test_preds = []
    test_targs = []
    total_test_loss = 0.0

    with torch.no_grad():
        for numeric, categorical, boolean, target in test_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            boolean = boolean.to(device)
            target = target.to(device)

            preds = model(numeric, categorical, boolean)
            loss = criterion(preds, target)
            total_test_loss += loss.item()
            test_preds.append(preds.cpu())
            test_targs.append(target.cpu())

    test_loss = float(total_test_loss / len(test_loader)) if len(test_loader) > 0 else 0.0
    test_predictions_np = torch.cat(test_preds).numpy()
    test_targets_np = torch.cat(test_targs).numpy()

    # Calculate test metrics
    test_mae, test_smape, test_rmse, test_r2, test_mape, test_f1, test_logloss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if config.task_type == "lookalike":
        from sklearn.metrics import roc_auc_score, f1_score, log_loss
        probs = 1 / (1 + np.exp(-test_predictions_np.flatten()))
        try:
            test_r2 = float(roc_auc_score(test_targets_np.flatten(), probs))
        except Exception:
            test_r2 = 0.0
        try:
            binary_preds = (probs >= 0.5).astype(int)
            test_f1 = float(f1_score(test_targets_np.flatten().astype(int), binary_preds, zero_division=0))
        except Exception:
            test_f1 = 0.0
        try:
            test_logloss = float(log_loss(test_targets_np.flatten(), probs))
        except Exception:
            test_logloss = 0.0
    else:
        # Regression: inverse transform to original scale for interpretable metrics
        test_preds_orig = processor.target_scaler.inverse_transform(
            test_predictions_np.reshape(-1, 1)
        ).flatten()
        test_targs_orig = processor.target_scaler.inverse_transform(
            test_targets_np.reshape(-1, 1)
        ).flatten()

        errors = test_preds_orig - test_targs_orig
        test_mae = float(np.mean(np.abs(errors)))
        test_rmse = float(np.sqrt(np.mean(errors**2)))

        # SMAPE: Symmetric Mean Absolute Percentage Error
        denominator = (np.abs(test_preds_orig) + np.abs(test_targs_orig)) / 2
        denominator = np.where(denominator == 0, 1, denominator)  # Avoid division by zero
        test_smape = float(np.mean(np.abs(errors) / denominator) * 100)

        # MAPE: Mean Absolute Percentage Error (exclude zero actuals)
        nonzero_mask = np.abs(test_targs_orig) > 0
        test_mape = float(np.mean(np.abs(errors[nonzero_mask]) / np.abs(test_targs_orig[nonzero_mask])) * 100) if nonzero_mask.any() else 0.0

        # R² (coefficient of determination)
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((test_targs_orig - np.mean(test_targs_orig))**2)
        test_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Save validation metrics before overwriting for comparison
    val_r2_final = r2  # Validation R² (regression) or ROC-AUC (classification)
    val_mae_final = mae
    val_f1_final = f1
    val_logloss_final = logloss

    # Use test metrics for final reporting (not validation metrics)
    mae = test_mae
    smape = test_smape
    rmse = test_rmse
    r2 = test_r2

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": pytorch_config,
        "final_metrics": {
            "test_loss": test_loss,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_smape": test_smape,
            "test_mape": test_mape,
            "test_f1": test_f1,
            "test_logloss": test_logloss,
            "val_loss": best_val_loss,
        }
    }, job.output_dir / "best_model.pt")
    processor.save(job.output_dir / "preprocessor.pkl")

    # Write model_metadata.json (consistent with XGBoost/Clustering paths)
    feature_names = (
        list(pytorch_config.numeric_features) +
        list(pytorch_config.categorical_features) +
        list(pytorch_config.boolean_features)
    )
    with open(job.output_dir / "model_metadata.json", "w") as f:
        json.dump({
            "model_type": "neural_network",
            "task_type": config.task_type,
            "epochs_trained": epoch,
            "feature_names": feature_names,
            "test_metrics": {
                "test_r2": test_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_smape": test_smape,
                "test_mape": test_mape,
                "test_f1": test_f1,
                "test_logloss": test_logloss,
            },
        }, f, indent=2)

    # Export classification results (lookalike tasks only)
    if config.task_type == "lookalike":
        try:
            # For NN: convert logits to probabilities for export
            nn_test_probs = 1 / (1 + np.exp(-test_predictions_np.flatten()))
            _export_classification_results(
                job=job,
                processor=processor,
                test_predictions=nn_test_probs,
                test_targets=test_targets_np.flatten(),
                test_roc_auc=test_r2,
                report_callback=report_callback,
            )
        except Exception as e:
            print(f"Warning: Classification export failed: {e}")

    # Compute SHAP feature importance
    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=config.epochs, total_epochs=config.epochs,
        train_loss=float(train_loss), val_loss=float(best_val_loss),
        val_mae=float(mae), val_smape=0.0, val_rmse=0.0, val_r2=float(r2),
        learning_rate=float(current_lr), elapsed_time=float(time.time() - start_time),
        status="running", message="Computing SHAP feature importance..."
    ))

    # Build feature names list for SHAP
    feature_names = (
        list(pytorch_config.numeric_features) +
        list(pytorch_config.categorical_features) +
        list(pytorch_config.boolean_features)
    )

    # Create progress callback for SHAP messages
    def shap_progress(msg: str):
        report_callback(TrainingProgress(
            job_id=job.job_id, epoch=config.epochs, total_epochs=config.epochs,
            train_loss=float(train_loss), val_loss=float(best_val_loss),
            val_mae=float(mae), val_smape=0.0, val_rmse=0.0, val_r2=float(r2),
            learning_rate=float(current_lr), elapsed_time=float(time.time() - start_time),
            status="running", message=msg
        ))

    # Run SHAP computation
    shap_success = compute_shap_values(
        model=model,
        processor=processor,
        test_loader=test_loader,
        feature_names=feature_names,
        output_dir=job.output_dir,
        device=config.device,
        n_background=100,
        n_explain=200,
        task_type=config.task_type,
        progress_callback=shap_progress,
    )

    if shap_success:
        # Load SHAP results for final metrics
        shap_cache = ShapCache(job.output_dir)
        shap_importance = shap_cache.get_feature_importance(top_n=20)
        if shap_importance:
            # Save SHAP summary to JSON for frontend
            with open(job.output_dir / "shap_importance.json", "w") as f:
                json.dump(shap_importance, f, indent=2)

    job.final_metrics = {
        # Test set metrics (unbiased final evaluation)
        "test_loss": float(test_loss),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_smape": float(test_smape),
        "test_mape": float(test_mape),
        "test_f1": float(test_f1),
        "test_logloss": float(test_logloss),
        # Validation metrics (for reference/comparison)
        "val_loss": float(best_val_loss),
        "val_r2": float(val_r2_final),  # Last epoch validation R² for comparison
        "val_mae": float(val_mae_final),
        # Metadata
        "shap_available": shap_success,
        "experiment_dir": str(job.output_dir.name),
    }

    # DEBUG: Print test vs validation metrics to verify they're different
    print(f"\n{'='*60}")
    print(f"[DEBUG] FINAL METRICS COMPARISON (val vs test)")
    if config.task_type == "lookalike":
        print(f"  Validation ROC-AUC:  {val_r2_final:.6f}")
        print(f"  Test ROC-AUC:        {test_r2:.6f}")
        print(f"  Difference:          {test_r2 - val_r2_final:+.6f}")
        print(f"  ---")
        print(f"  Validation F1:       {val_f1_final:.6f}")
        print(f"  Test F1:             {test_f1:.6f}")
        print(f"  ---")
        print(f"  Validation Log Loss: {val_logloss_final:.6f}")
        print(f"  Test Log Loss:       {test_logloss:.6f}")
        print(f"  Test Loss (BCE):     {test_loss:.6f}")
    else:
        print(f"  Validation R² (last epoch):  {val_r2_final:.6f}")
        print(f"  Test R² (held-out 15%):      {test_r2:.6f}")
        print(f"  Difference:                  {test_r2 - val_r2_final:+.6f}")
        print(f"  ---")
        print(f"  Validation MAE: ${val_mae_final:,.2f}")
        print(f"  Test MAE:       ${test_mae:,.2f}")
        print(f"  Test Loss:      {test_loss:.6f}")
    print(f"{'='*60}\n")

    report_callback(TrainingProgress(
        job_id=job.job_id, epoch=config.epochs, total_epochs=config.epochs,
        train_loss=0.0, val_loss=float(test_loss), val_mae=float(test_mae), val_smape=float(test_smape), val_rmse=float(test_rmse), val_r2=float(test_r2),
        learning_rate=float(current_lr), elapsed_time=float(time.time() - start_time),
        status="completed", message="Training completed successfully",
        best_val_loss=float(best_val_loss),
        val_mape=float(test_mape),
        val_f1=float(test_f1), val_logloss=float(test_logloss),
        final_metrics=job.final_metrics,  # Include test set metrics for frontend
    ))


# =============================================================================
# Public API Functions
# =============================================================================

def detect_apple_chip() -> Dict:
    """Detect the Apple Silicon chip model, GPU cores, and unified memory."""
    import subprocess
    import re
    chip_info = {"detected_chip": "unknown", "chip_name": "Unknown", "gpu_cores": None, "total_memory": None}

    # 1. Get chip name from sysctl and parse into lookup key
    try:
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and "Apple" in res.stdout:
            chip_info["chip_name"] = res.stdout.strip()
            # Parse "Apple M4 Pro" → "m4_pro", "Apple M2" → "m2"
            m = re.match(r"Apple (M\d+)(?:\s+(Pro|Max|Ultra))?", res.stdout.strip())
            if m:
                key = m.group(1).lower()  # "m4"
                if m.group(2):
                    key += "_" + m.group(2).lower()  # "m4_pro"
                chip_info["detected_chip"] = key
    except Exception:
        pass

    # 2. Get GPU core count from system_profiler
    try:
        res = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            m = re.search(r"Total Number of Cores:\s*(\d+)", res.stdout)
            if m:
                chip_info["gpu_cores"] = int(m.group(1))
    except Exception:
        pass

    # 3. Get total unified memory from sysctl
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            mem_bytes = int(res.stdout.strip())
            chip_info["total_memory"] = f"{mem_bytes // (1024 ** 3)} GB"
    except Exception:
        pass

    return chip_info

def get_system_info() -> Dict:
    info = {"pytorch_version": torch.__version__, "mps_available": torch.backends.mps.is_available()}
    if info["mps_available"]:
        info.update(detect_apple_chip())
    return info

def start_training(config_dict: Dict) -> Tuple[bool, str]:
    """Start a new training job."""
    try:
        # Construct TrainingConfig
        config = TrainingConfig(
            model_type=config_dict.get("model_type", "neural_network"),
            task_type=config_dict.get("task_type", "regression"),
            target=config_dict.get("target", "avg_monthly_revenue"),
            epochs=int(config_dict.get("epochs", 50)),
            # ... map other fields ...
            batch_size=int(config_dict.get("batch_size", 4096)),
            learning_rate=float(config_dict.get("learning_rate", 1e-4)),
            device=config_dict.get("device", "mps"),
            model_preset=config_dict.get("model_preset", "model_b"),
            selected_features=config_dict.get("selected_features"),
            network_filter=config_dict.get("network_filter"),
            # Clustering parameters
            n_clusters=int(config_dict.get("n_clusters", 5)),
            cluster_probability_threshold=float(config_dict.get("cluster_probability_threshold", 0.5)),
        )
        
        job_id = job_manager.submit_job(config)
        return True, job_id
    except Exception as e:
        return False, str(e)

def stop_training(job_id: Optional[str] = None) -> Tuple[bool, str]:
    """Stop a specific job or all jobs."""
    if job_id:
        if job_manager.stop_job(job_id):
            return True, f"Job {job_id} stopped"
        return False, "Job not found"
    else:
        # Stop all?
        return False, "Please specify job_id to stop"

def get_training_status(job_id: Optional[str] = None) -> Dict:
    """Get status of a specific job or summary of all."""
    if job_id:
        job = job_manager.get_job(job_id)
        if job:
            return {"status": job.status, "metrics": job.final_metrics, "job_id": job.job_id}
        return {"status": "not_found"}
    else:
        return {"jobs": job_manager.get_all_jobs()}

def _sanitize_for_json(obj):
    """Recursively replace Infinity/NaN with None for valid JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    return obj


# =============================================================================
# Experiment Catalog — Persistent, restart-safe experiment registry
# =============================================================================

_catalog_cache: Optional[List[Dict]] = None
_catalog_cache_mtime: float = 0.0


def _parse_experiment_folder(exp_dir: Path) -> Optional[Dict]:
    """Parse a single experiment folder into a catalog entry.

    Reads config.json for model/task/feature info, model_metadata.json for
    test metrics, and checks which artifact files exist. Falls back to
    reading metrics from best_model.pt for legacy NN experiments that
    lack model_metadata.json.
    """
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Skipping {exp_dir.name} — invalid config.json: {e}")
        return None

    job_id = exp_dir.name

    # Parse created_at from job_id: "job_{unix_timestamp}_{hex}"
    created_at = None
    parts = job_id.split("_")
    if len(parts) >= 2:
        try:
            from datetime import datetime, timezone
            ts = int(parts[1])
            created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            pass

    # Extract feature counts from training_features dict
    training_features = config.get("training_features", {})
    feature_count = {
        "numeric": len(training_features.get("numeric", [])),
        "categorical": len(training_features.get("categorical", [])),
        "boolean": len(training_features.get("boolean", [])),
    }

    # Load test metrics from model_metadata.json
    test_metrics = {}
    metadata_path = exp_dir / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            test_metrics = metadata.get("test_metrics", {})
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback for legacy NN experiments: read from best_model.pt checkpoint
    if not test_metrics and (exp_dir / "best_model.pt").exists() and not metadata_path.exists():
        try:
            import torch as _torch
            checkpoint = _torch.load(
                exp_dir / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            test_metrics = checkpoint.get("final_metrics", {})
        except Exception:
            pass

    # Check which artifacts exist
    artifact_names = [
        "config.json", "best_model.pt", "best_model.json",
        "model_wrapper.pkl", "preprocessor.pkl",
        "shap_cache.npz", "shap_importance.json",
        "training_sites.csv", "test_predictions.csv",
        "non_active_classification.csv", "model_metadata.json",
    ]
    artifacts = [name for name in artifact_names if (exp_dir / name).exists()]

    # Determine completeness
    has_model = (exp_dir / "best_model.pt").exists() or (exp_dir / "model_wrapper.pkl").exists()
    is_complete = (
        config_path.exists()
        and (exp_dir / "preprocessor.pkl").exists()
        and has_model
    )

    return _sanitize_for_json({
        "job_id": job_id,
        "created_at": created_at,
        "model_type": config.get("model_type", "unknown"),
        "task_type": config.get("task_type", "unknown"),
        "target": config.get("target", ""),
        "feature_count": feature_count,
        "training_features": training_features,
        "test_metrics": test_metrics,
        "network_filter": config.get("network_filter"),
        "lookalike_lower_percentile": config.get("lookalike_lower_percentile"),
        "lookalike_upper_percentile": config.get("lookalike_upper_percentile"),
        "is_complete": is_complete,
        "has_shap": (exp_dir / "shap_cache.npz").exists(),
        "has_predictions": (exp_dir / "test_predictions.csv").exists(),
        "artifacts": artifacts,
    })


def scan_experiment_folders() -> List[Dict]:
    """Scan disk for all experiment folders and return catalog.

    Results are cached and invalidated when the experiments directory
    modification time changes (i.e., when a new experiment is created
    or an old one is deleted).
    """
    global _catalog_cache, _catalog_cache_mtime

    experiments_dir = DEFAULT_OUTPUT_DIR / "experiments"
    if not experiments_dir.exists():
        return []

    current_mtime = experiments_dir.stat().st_mtime
    if _catalog_cache is not None and current_mtime == _catalog_cache_mtime:
        return _catalog_cache

    catalog = []
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("job_"):
            continue
        entry = _parse_experiment_folder(exp_dir)
        if entry:
            catalog.append(entry)

    # Sort newest first by created_at (or job_id as fallback)
    catalog.sort(key=lambda e: e.get("created_at") or e["job_id"], reverse=True)

    _catalog_cache = catalog
    _catalog_cache_mtime = current_mtime
    return catalog


def stream_training_progress() -> Generator[str, None, None]:
    """Stream progress events from the global broadcaster."""
    q = job_manager.subscribe()

    def json_serializer(obj):
        if isinstance(obj, (np.float32, np.float64)):
            v = float(obj)
            return None if (np.isinf(v) or np.isnan(v)) else v
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    try:
        while True:
            progress = q.get()
            data = _sanitize_for_json(asdict(progress))
            yield f"data: {json.dumps(data, default=json_serializer)}\n\n"
    except GeneratorExit:
        job_manager.unsubscribe(q)

def load_explainability_components(output_dir: Path) -> Dict:
    """
    Load explainability components from the output directory.

    Searches for calibrator, conformal predictor, and tier classifier in:
    1. The explainability subdirectory (legacy location)
    2. The output directory itself (experiment location)

    Returns:
        Dict with keys: 'calibrator', 'conformal', 'tier_classifier'
        Values may be None if components don't exist.
    """
    import pickle
    from site_scoring.explainability import ProbabilityCalibrator, TierClassifier

    components = {
        'calibrator': None,
        'conformal': None,
        'tier_classifier': None,
        'metadata': {},
    }

    # Check explainability subdirectory first (legacy location)
    explainability_dir = output_dir / "explainability"
    if not explainability_dir.exists():
        explainability_dir = output_dir  # Fall back to output_dir itself

    # Load calibrator
    calibrator_path = explainability_dir / "calibrator.pkl"
    if calibrator_path.exists():
        try:
            components['calibrator'] = ProbabilityCalibrator.load(calibrator_path)
        except Exception as e:
            print(f"Warning: Failed to load calibrator: {e}")

    # Load conformal predictor
    conformal_path = explainability_dir / "conformal.pkl"
    if conformal_path.exists():
        try:
            with open(conformal_path, 'rb') as f:
                components['conformal'] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load conformal predictor: {e}")

    # Load tier classifier
    tier_path = explainability_dir / "tier_classifier.pkl"
    if tier_path.exists():
        try:
            with open(tier_path, 'rb') as f:
                tier_data = pickle.load(f)
                if isinstance(tier_data, dict):
                    components['tier_classifier'] = TierClassifier.from_dict(tier_data)
                else:
                    components['tier_classifier'] = tier_data
        except Exception as e:
            print(f"Warning: Failed to load tier classifier: {e}")

    # Load metadata (feature names, dimensions, etc.)
    metadata_path = explainability_dir / "metadata.pkl"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                if isinstance(metadata, dict):
                    components['metadata'] = metadata
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")

    return components


def explain_prediction(probability: float, output_dir: Path) -> Dict:
    """
    Explain a single prediction with calibration and tier classification.

    Args:
        probability: Raw model probability (0-1)
        output_dir: Directory containing explainability components

    Returns:
        Dict with calibrated_probability, tier, tier_label, etc.
    """
    components = load_explainability_components(output_dir)

    calibrator = components.get('calibrator')
    tier_classifier = components.get('tier_classifier')

    result = {
        'raw_probability': probability,
        'calibrated_probability': probability,  # Default to raw if no calibrator
        'tier': None,
        'tier_label': None,
        'tier_action': None,
        'confidence_statement': None,
    }

    # Apply calibration if available
    if calibrator is not None:
        try:
            import numpy as np
            calibrated = calibrator.calibrate(np.array([probability]))[0]
            result['calibrated_probability'] = float(calibrated)
        except Exception as e:
            print(f"Warning: Calibration failed: {e}")

    # Apply tier classification if available
    if tier_classifier is not None:
        try:
            tier_result = tier_classifier.classify(result['calibrated_probability'])
            result['tier'] = tier_result.tier
            result['tier_label'] = tier_result.label
            result['tier_action'] = tier_result.action
            result['confidence_statement'] = tier_result.confidence_statement
            result['historical_accuracy'] = tier_result.historical_accuracy
            result['color'] = tier_result.color
        except Exception as e:
            print(f"Warning: Tier classification failed: {e}")

    return result