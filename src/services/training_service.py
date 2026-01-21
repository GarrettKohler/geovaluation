"""
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
from site_scoring.config import Config
from site_scoring.model import SiteScoringModel
from site_scoring.data_loader import DataProcessor, create_data_loaders


@dataclass
class TrainingConfig:
    """User-configurable training parameters."""
    # Model type
    model_type: str = "neural_network"  # neural_network, gradient_boosting, random_forest

    # Target variable
    target: str = "revenue"  # revenue, monthly_impressions, monthly_revenue_per_screen

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


@dataclass
class TrainingProgress:
    """Training progress update."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_r2: float
    learning_rate: float
    elapsed_time: float
    status: str  # running, completed, error
    message: str = ""
    best_val_loss: float = float('inf')


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

            # Get chip name for display
            chip_name = APPLE_CHIP_SPECS.get(self.config.apple_chip, {})
            chip_display = f"Apple {self.config.apple_chip.upper().replace('_', ' ')}" if self.config.apple_chip != "auto" else "Apple Silicon"

            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
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
                val_r2=0,
                learning_rate=self.config.learning_rate,
                elapsed_time=time.time() - start_time,
                status="running",
                message=f"Data loaded. Creating model on {self.config.device}..."
            ))

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

            # Training setup
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

                for numeric, categorical, boolean, target in train_loader:
                    numeric = numeric.to(device, non_blocking=True)
                    categorical = categorical.to(device, non_blocking=True)
                    boolean = boolean.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(numeric, categorical, boolean)
                    loss = criterion(predictions, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += loss.item()
                    n_batches += 1

                train_loss = total_train_loss / n_batches

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
                predictions_orig = processor.target_scaler.inverse_transform(predictions_np)
                targets_orig = processor.target_scaler.inverse_transform(targets_np)

                mae = np.mean(np.abs(predictions_orig - targets_orig))
                ss_res = np.sum((targets_orig - predictions_orig) ** 2)
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

                # Report progress
                self._report_progress(TrainingProgress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mae=float(mae),
                    val_r2=float(r2),
                    learning_rate=current_lr,
                    elapsed_time=time.time() - start_time,
                    status="running",
                    message=f"Epoch {epoch}/{self.config.epochs}",
                    best_val_loss=best_val_loss
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
            predictions_orig = processor.target_scaler.inverse_transform(predictions_np)
            targets_orig = processor.target_scaler.inverse_transform(targets_np)

            test_mae = np.mean(np.abs(predictions_orig - targets_orig))
            ss_res = np.sum((targets_orig - predictions_orig) ** 2)
            ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
            test_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Save model
            output_dir = pytorch_config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": pytorch_config,
                "test_metrics": {
                    "test_loss": test_loss,
                    "test_mae": float(test_mae),
                    "test_r2": float(test_r2),
                }
            }, checkpoint_path)
            processor.save(output_dir / "preprocessor.pkl")

            self.final_metrics = {
                "test_loss": float(test_loss),
                "test_mae": float(test_mae),
                "test_r2": float(test_r2),
                "best_val_loss": float(best_val_loss),
                "model_path": str(checkpoint_path),
            }

            # Report completion
            self._report_progress(TrainingProgress(
                epoch=self.config.epochs,
                total_epochs=self.config.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=float(test_mae),
                val_r2=float(test_r2),
                learning_rate=current_lr,
                elapsed_time=time.time() - start_time,
                status="completed",
                message=f"Training complete! Test MAE: ${test_mae:,.2f}, R²: {test_r2:.4f}",
                best_val_loss=best_val_loss
            ))

        except Exception as e:
            self._report_progress(TrainingProgress(
                epoch=0,
                total_epochs=self.config.epochs,
                train_loss=0,
                val_loss=0,
                val_mae=0,
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
            target=config_dict.get("target", "revenue"),
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
        )

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
        "val_r2": progress.val_r2,
        "learning_rate": progress.learning_rate,
        "elapsed_time": progress.elapsed_time,
        "status": progress.status,
        "message": progress.message,
        "best_val_loss": progress.best_val_loss,
        "final_metrics": _current_job.final_metrics,
    }


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
                "val_r2": progress.val_r2,
                "learning_rate": progress.learning_rate,
                "elapsed_time": progress.elapsed_time,
                "status": progress.status,
                "message": progress.message,
                "best_val_loss": progress.best_val_loss,
            }

            # Include final_metrics in the completion message so frontend gets it immediately
            if progress.status in ("completed", "error", "stopped"):
                if _current_job.final_metrics:
                    data["final_metrics"] = _current_job.final_metrics
                yield f"data: {json.dumps(data)}\n\n"
                break
            else:
                yield f"data: {json.dumps(data)}\n\n"
        else:
            time.sleep(0.5)

    yield f"data: {json.dumps({'status': 'stream_end'})}\n\n"
