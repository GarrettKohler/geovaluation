"""XGBoost/tree model training pipeline.

Converts PyTorch DataLoaders to numpy arrays and trains an XGBoost model
with progress callback integration.  Supports both regression and
lookalike (binary classification) tasks.

Important: XGBoost 2.0+ removed the ``callbacks`` kwarg from the sklearn
``.fit()`` method.  Use ``model.set_params(callbacks=...)`` before calling
``.fit()``, and clear callbacks before pickling.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from torch.utils.data import DataLoader

from .progress import TrainingProgress


# ---------------------------------------------------------------------------
# DataLoader -> numpy conversion
# ---------------------------------------------------------------------------

def dataloaders_to_numpy(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Convert PyTorch DataLoaders to numpy arrays for tree models.

    Concatenates numeric + boolean features into a single feature matrix.
    Categorical features are excluded here because XGBoost's native
    categorical support expects them in a separate pipeline step
    (label-encoded columns).  In this codebase the booleans are already
    0/1-encoded so they concatenate cleanly with the numeric block.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``
    """

    def _extract(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_features: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for numeric, categorical, boolean, target in loader:
            # Concatenate numeric + boolean (skip categorical for tree models)
            feats = np.concatenate(
                [numeric.numpy(), boolean.numpy()], axis=1
            )
            all_features.append(feats)
            all_targets.append(target.numpy())

        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0).flatten()
        return X, y

    X_train, y_train = _extract(train_loader)
    X_val, y_val = _extract(val_loader)
    X_test, y_test = _extract(test_loader)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# XGBoost progress callback (2.0+ compatible)
# ---------------------------------------------------------------------------

class _XGBProgressCallback:
    """XGBoost callback that forwards training progress via a callable.

    Compatible with XGBoost >= 2.0 callback protocol.
    """

    def __init__(
        self,
        total_rounds: int,
        progress_callback: Optional[Callable[[TrainingProgress], None]],
        start_time: float,
        stop_event=None,
    ):
        self.total_rounds = total_rounds
        self.progress_callback = progress_callback
        self.start_time = start_time
        self.stop_event = stop_event
        self.best_val_loss = float("inf")

    def __call__(self, env):
        """Called by XGBoost after each boosting round.

        ``env`` is an ``xgboost.core.CallbackEnv`` with attributes:
        ``model``, ``cvfolds``, ``iteration``, ``begin_iteration``,
        ``end_iteration``, ``rank``, ``evaluation_result_list``.
        """
        if self.stop_event and self.stop_event.is_set():
            raise KeyboardInterrupt("Training stopped by user")

        iteration = env.iteration + 1
        eval_results = dict(env.evaluation_result_list) if env.evaluation_result_list else {}

        # Extract train and val metrics (XGBoost names them by eval set order)
        train_loss = eval_results.get("train-rmse", eval_results.get("train-logloss", 0.0))
        val_loss = eval_results.get("val-rmse", eval_results.get("val-logloss", 0.0))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        if self.progress_callback:
            progress = TrainingProgress(
                epoch=iteration,
                total_epochs=self.total_rounds,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                best_val_loss=float(self.best_val_loss),
                learning_rate=0.0,  # XGBoost LR is static
                elapsed_seconds=time.time() - self.start_time,
                status="running",
            )
            self.progress_callback(progress)


# ---------------------------------------------------------------------------
# Tree trainer
# ---------------------------------------------------------------------------

class TreeTrainer:
    """Trains XGBoost models with progress reporting.

    Wraps the XGBoost sklearn API and provides a unified interface
    consistent with ``NNTrainer``.

    Args:
        task_type: ``"regression"`` or ``"lookalike"``.
    """

    def __init__(self, task_type: str = "regression"):
        self.task_type = task_type

    def train(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        stop_event=None,
    ) -> dict:
        """Train an XGBoost model with progress reporting.

        Args:
            model: An XGBoost sklearn estimator (``XGBRegressor`` or
                ``XGBClassifier``).
            X_train: Training feature matrix.
            y_train: Training targets.
            X_val: Validation feature matrix.
            y_val: Validation targets.
            progress_callback: Called after each boosting round.
            stop_event: ``threading.Event`` for early stop.

        Returns:
            Dict with keys ``best_val_loss``, ``n_estimators_trained``,
            ``elapsed_seconds``.
        """
        start_time = time.time()

        # Build callback (XGBoost 2.0+ compatible)
        xgb_callback = _XGBProgressCallback(
            total_rounds=model.get_params().get("n_estimators", 100),
            progress_callback=progress_callback,
            start_time=start_time,
            stop_event=stop_event,
        )

        # Set callbacks via set_params (XGBoost 2.0+ pattern)
        model.set_params(callbacks=[xgb_callback])

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        except KeyboardInterrupt:
            # Graceful stop via stop_event
            pass

        elapsed = time.time() - start_time

        # Clear callbacks before any serialization
        model.set_params(callbacks=None)

        return {
            "best_val_loss": float(xgb_callback.best_val_loss),
            "n_estimators_trained": model.get_booster().num_boosted_rounds()
            if hasattr(model, "get_booster")
            else 0,
            "elapsed_seconds": elapsed,
        }

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Evaluate on the held-out test set.

        Args:
            model: Trained XGBoost estimator.
            X_test: Test feature matrix.
            y_test: Test targets.

        Returns:
            Dict of evaluation metrics matching ``NNTrainer.evaluate``
            output format.
        """
        preds = model.predict(X_test)
        metrics: Dict[str, float] = {}

        if self.task_type == "regression":
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["r2"] = float(r2_score(y_test, preds))
            # MAPE -- avoid division by zero
            nonzero_mask = np.abs(y_test) > 1e-8
            if nonzero_mask.sum() > 0:
                metrics["mape"] = float(
                    np.mean(
                        np.abs(
                            (y_test[nonzero_mask] - preds[nonzero_mask])
                            / y_test[nonzero_mask]
                        )
                    )
                    * 100
                )
        else:
            # Classification
            if hasattr(preds[0], "__len__"):
                # predict_proba output
                binary_preds = (preds[:, 1] > 0.5).astype(int)
            else:
                binary_preds = preds.astype(int)
            binary_targets = y_test.astype(int)
            metrics["accuracy"] = float(accuracy_score(binary_targets, binary_preds))
            metrics["f1"] = float(
                f1_score(binary_targets, binary_preds, zero_division=0)
            )

        return metrics
