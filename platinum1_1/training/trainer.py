"""Neural network training loop with early stopping and progress reporting.

Handles:
- AdamW optimiser with weight decay
- ReduceLROnPlateau scheduler
- Early stopping with patience
- Gradient clipping (max_norm=1.0)
- Best-model checkpoint (in-memory state dict)
- Proper test-set evaluation after restoring best model
- Regression (HuberLoss) and lookalike classification (BCEWithLogitsLoss)
"""

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from torch.utils.data import DataLoader

from .progress import TrainingProgress


class NNTrainer:
    """Trains PyTorch neural network models with early stopping.

    The training loop emits ``TrainingProgress`` objects via a callback so
    that callers (e.g. ``JobManager``) can forward them to SSE streams.

    Args:
        model: PyTorch ``nn.Module`` to train.
        device: Compute device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        learning_rate: Initial LR for AdamW.
        weight_decay: L2 penalty for AdamW.
        scheduler_patience: Epochs to wait before LR reduction.
        early_stopping_patience: Epochs without val-loss improvement
            before stopping.
        task_type: ``"regression"`` or ``"lookalike"``.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_patience: int = 5,
        early_stopping_patience: int = 10,
        task_type: str = "regression",
    ):
        self.model = model.to(device)
        self.device = device
        self.task_type = task_type

        # Loss function
        if task_type == "regression":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimiser and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=scheduler_patience, factor=0.5
        )

        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss: float = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.epochs_without_improvement: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        stop_event=None,
    ) -> dict:
        """Run the full training loop.

        Args:
            train_loader: Training ``DataLoader`` yielding
                ``(numeric, categorical, boolean, target)`` tuples.
            val_loader: Validation ``DataLoader`` (same format).
            epochs: Maximum number of epochs.
            progress_callback: Called after each epoch with a
                ``TrainingProgress`` snapshot.
            stop_event: ``threading.Event`` -- if set, stops training
                gracefully at the end of the current epoch.

        Returns:
            Dict with keys ``best_val_loss``, ``epochs_trained``,
            ``history``, ``model_state``, ``elapsed_seconds``.
        """
        start_time = time.time()
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        epoch = 0

        for epoch in range(1, epochs + 1):
            if stop_event and stop_event.is_set():
                break

            # Train one epoch
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            # Step scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Progress callback
            if progress_callback:
                progress = TrainingProgress(
                    epoch=epoch,
                    total_epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=self.best_val_loss,
                    learning_rate=current_lr,
                    elapsed_seconds=time.time() - start_time,
                    status="running",
                )
                progress_callback(progress)

            # Stop if no improvement for too long
            if self.epochs_without_improvement >= self.early_stopping_patience:
                break

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return {
            "best_val_loss": self.best_val_loss,
            "epochs_trained": epoch,
            "history": history,
            "model_state": self.best_model_state,
            "elapsed_seconds": time.time() - start_time,
        }

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate on the held-out test set.

        Must be called **after** training so that the model contains the
        best checkpoint weights.

        Args:
            test_loader: Test ``DataLoader`` (same format as train/val).

        Returns:
            Dict of evaluation metrics.  For regression: ``test_loss``,
            ``mae``, ``r2``, ``mape``.  For classification: ``test_loss``,
            ``accuracy``, ``f1``.
        """
        self.model.eval()
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for numeric, categorical, boolean, target in test_loader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                boolean = boolean.to(self.device)
                target = target.to(self.device)

                output = self.model(numeric, categorical, boolean)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                n_batches += 1
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()

        metrics: Dict[str, float] = {
            "test_loss": total_loss / max(n_batches, 1),
        }

        if self.task_type == "regression":
            metrics["mae"] = float(mean_absolute_error(targets, preds))
            metrics["r2"] = float(r2_score(targets, preds))
            # MAPE -- avoid division by zero
            nonzero_mask = np.abs(targets) > 1e-8
            if nonzero_mask.sum() > 0:
                metrics["mape"] = float(
                    np.mean(
                        np.abs(
                            (targets[nonzero_mask] - preds[nonzero_mask])
                            / targets[nonzero_mask]
                        )
                    )
                    * 100
                )
        else:
            # BCEWithLogitsLoss: logits > 0 => class 1
            binary_preds = (preds > 0).astype(int)
            binary_targets = targets.astype(int)
            metrics["accuracy"] = float(accuracy_score(binary_targets, binary_preds))
            metrics["f1"] = float(
                f1_score(binary_targets, binary_preds, zero_division=0)
            )

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch.  Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for numeric, categorical, boolean, target in loader:
            numeric = numeric.to(self.device)
            categorical = categorical.to(self.device)
            boolean = boolean.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(numeric, categorical, boolean)
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> float:
        """Run validation pass.  Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for numeric, categorical, boolean, target in loader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                boolean = boolean.to(self.device)
                target = target.to(self.device)

                output = self.model(numeric, categorical, boolean)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
