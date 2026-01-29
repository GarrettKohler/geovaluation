"""
Training pipeline optimized for Apple M4 MPS.
Includes early stopping, learning rate scheduling, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time
from .config import Config
from .model import SiteScoringModel
from .data_loader import DataProcessor


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    train_loss: float
    val_loss: float
    val_mae: float
    val_r2: float
    epoch: int
    learning_rate: float


class Trainer:
    """
    Training loop optimized for M4 MPS backend.

    Features:
    - Automatic device placement (MPS/CPU)
    - Learning rate scheduling with plateau detection
    - Early stopping to prevent overfitting
    - Model checkpointing
    - Gradient clipping for stability
    """

    def __init__(
        self,
        model: SiteScoringModel,
        config: Config,
        processor: DataProcessor,
    ):
        self.model = model
        self.config = config
        self.processor = processor
        self.device = torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss function - Huber loss is robust to outliers
        self.criterion = nn.HuberLoss(delta=1.0)

        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=config.scheduler_patience,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history: list[TrainingMetrics] = []

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (numeric, categorical, boolean, target) in enumerate(train_loader):
            # Move to device
            numeric = numeric.to(self.device, non_blocking=True)
            categorical = categorical.to(self.device, non_blocking=True)
            boolean = boolean.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)  # More efficient
            predictions = self.model(numeric, categorical, boolean)
            loss = self.criterion(predictions, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Progress logging
            if batch_idx % self.config.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for numeric, categorical, boolean, target in val_loader:
            # Move to device
            numeric = numeric.to(self.device, non_blocking=True)
            categorical = categorical.to(self.device, non_blocking=True)
            boolean = boolean.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Forward pass
            predictions = self.model(numeric, categorical, boolean)
            loss = self.criterion(predictions, target)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())

        # Calculate metrics
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()

        # Inverse transform to original scale
        predictions_orig = self.processor.target_scaler.inverse_transform(predictions)
        targets_orig = self.processor.target_scaler.inverse_transform(targets)

        # MAE in original scale
        mae = np.mean(np.abs(predictions_orig - targets_orig))

        # R² score
        ss_res = np.sum((targets_orig - predictions_orig) ** 2)
        ss_tot = np.sum((targets_orig - np.mean(targets_orig)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return total_loss / len(val_loader), mae, r2

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> SiteScoringModel:
        """
        Full training loop with early stopping.

        Returns:
            Best model based on validation loss.
        """
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        best_model_state = None
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_loss, val_mae, val_r2 = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Store metrics
            metrics = TrainingMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=val_mae,
                val_r2=val_r2,
                epoch=epoch,
                learning_rate=current_lr,
            )
            self.training_history.append(metrics)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE: ${val_mae:,.2f} | "
                f"R²: {val_r2:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_loss)
                print(f"  → New best model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.model

    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.config.output_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, checkpoint_path)

        # Also save preprocessor
        self.processor.save(self.config.output_dir / "preprocessor.pkl")

    def load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["val_loss"]


def run_training(config: Optional[Config] = None) -> Tuple[SiteScoringModel, DataProcessor, Dict]:
    """
    Main training function.

    Returns:
        Tuple of (trained_model, processor, test_metrics)
    """
    from .data_loader import create_data_loaders

    if config is None:
        config = Config()

    print("=" * 60)
    print("Site Scoring Model Training")
    print("=" * 60)
    print(f"Target: {config.target}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")

    # Load data
    print("\nLoading and processing data...")
    train_loader, val_loader, test_loader, processor = create_data_loaders(config)

    # Create model with actual feature counts from data
    print("\nCreating model...")
    model = SiteScoringModel(
        n_numeric=processor.n_numeric_features,
        n_boolean=processor.n_boolean_features,
        categorical_vocab_sizes=processor.categorical_vocab_sizes,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_batch_norm=config.use_batch_norm,
    )
    print(f"Model architecture:\n{model}")

    # Train
    trainer = Trainer(model, config, processor)
    trained_model = trainer.train(train_loader, val_loader)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae, test_r2 = trainer.evaluate(test_loader)

    test_metrics = {
        "test_loss": test_loss,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"  R²: {test_r2:.4f}")

    return trained_model, processor, test_metrics


if __name__ == "__main__":
    run_training()
