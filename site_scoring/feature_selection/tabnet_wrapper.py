"""
TabNet Integration for Sparsemax Attention-based Feature Selection.

Wraps the pytorch_tabnet library for integration with the site scoring pipeline.

TabNet uses instance-wise feature selection through a sequential attention mechanism.
At each decision step, a sparsemax activation produces a sparse mask that selects
the most salient features for that specific input sample.

Reference:
"TabNet: Attentive Interpretable Tabular Learning" - AAAI 2021
https://cdn.aaai.org/ojs/16826/16826-13-20320-1-2-20210518.pdf
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings


class TabNetWrapper:
    """
    Wrapper for TabNet regression/classification with site scoring integration.

    TabNet provides:
    - Instance-wise feature selection via sparsemax attention
    - Global feature importance aggregated across samples
    - Interpretable per-sample feature masks

    Note: This replaces the standard MLP architecture entirely.

    Args:
        n_d: Width of decision prediction layer (default: 64)
        n_a: Width of attention embedding (default: 64)
        n_steps: Number of decision steps (default: 5)
        gamma: Coefficient for feature reuse penalty (default: 1.5)
        mask_type: 'sparsemax' or 'entmax' for feature masking
        task_type: 'regression' or 'classification'
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        mask_type: str = 'sparsemax',
        task_type: str = 'regression',
        learning_rate: float = 1e-3,
        patience: int = 20,
        max_epochs: int = 100,
        batch_size: int = 4096,
        virtual_batch_size: int = 256,
        device: str = 'auto',
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device = device

        self.model = None
        self.feature_names: Optional[List[str]] = None
        self._is_fitted = False

    def _check_tabnet_available(self):
        """Check if pytorch_tabnet is available."""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
            return True
        except ImportError:
            return False

    def create_model(self):
        """Create the TabNet model instance."""
        if not self._check_tabnet_available():
            raise ImportError(
                "pytorch_tabnet required. Install with: pip install pytorch_tabnet"
            )

        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

        # Choose model class based on task type
        ModelClass = TabNetRegressor if self.task_type == 'regression' else TabNetClassifier

        self.model = ModelClass(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            mask_type=self.mask_type,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=self.learning_rate),
            scheduler_params=dict(step_size=10, gamma=0.9),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            device_name=self.device if self.device != 'auto' else 'cpu',
            verbose=0,
        )

        return self.model

    def set_feature_names(self, names: List[str]):
        """Set feature names for interpretable reporting."""
        self.feature_names = names

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        eval_metric: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Fit the TabNet model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            eval_metric: Evaluation metric(s) for validation
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with training history and metrics.
        """
        if self.model is None:
            self.create_model()

        # Default evaluation metric
        if eval_metric is None:
            eval_metric = ['rmse'] if self.task_type == 'regression' else ['auc']

        # Prepare evaluation set
        eval_set = None
        eval_name = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            eval_name = ['val']

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=eval_set,
                eval_name=eval_name,
                eval_metric=eval_metric,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                virtual_batch_size=self.virtual_batch_size,
            )

        self._is_fitted = True

        # Collect training history
        history = {
            'best_epoch': self.model.best_epoch,
            'best_cost': self.model.best_cost,
        }

        if hasattr(self.model, 'history'):
            history['loss_history'] = self.model.history.get('loss', [])
            history['val_history'] = self.model.history.get('val_0_rmse', [])

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        """
        Get global feature importance scores.

        Returns aggregated importance across all samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.feature_importances_

    def get_feature_importance_dict(self) -> Dict[str, float]:
        """Get feature importance as a dictionary with names."""
        importance = self.get_feature_importances()

        if self.feature_names is not None:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get per-sample feature masks and aggregated masks.

        Args:
            X: Input features

        Returns:
            masks: Per-step attention masks, shape (n_samples, n_steps, n_features)
            aggregated: Aggregated importance per sample, shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.explain(X)

    def get_selected_features(self, threshold: float = 0.01) -> List[str]:
        """
        Get features with global importance above threshold.

        Args:
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names.
        """
        importance = self.get_feature_importances()
        selected_indices = np.where(importance > threshold)[0]

        if self.feature_names is not None:
            return [self.feature_names[i] for i in selected_indices]
        return [f"feature_{i}" for i in selected_indices]

    def get_eliminated_features(self, threshold: float = 0.01) -> List[str]:
        """
        Get features with global importance below threshold.
        """
        importance = self.get_feature_importances()
        eliminated_indices = np.where(importance <= threshold)[0]

        if self.feature_names is not None:
            return [self.feature_names[i] for i in eliminated_indices]
        return [f"feature_{i}" for i in eliminated_indices]

    def get_selection_summary(self, threshold: float = 0.01) -> Dict:
        """Get comprehensive summary of feature selection."""
        importance = self.get_feature_importances()

        selected = self.get_selected_features(threshold)
        eliminated = self.get_eliminated_features(threshold)

        # Sort by importance
        if self.feature_names is not None:
            sorted_features = sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            sorted_features = sorted(
                [(f"feature_{i}", imp) for i, imp in enumerate(importance)],
                key=lambda x: x[1],
                reverse=True
            )

        return {
            'n_total_features': len(importance),
            'n_selected': len(selected),
            'n_eliminated': len(eliminated),
            'selection_rate': len(selected) / len(importance),
            'selected_features': selected,
            'eliminated_features': eliminated,
            'importance_scores': self.get_feature_importance_dict(),
            'top_10_features': [
                {'name': name, 'importance': float(imp)}
                for name, imp in sorted_features[:10]
            ],
            'threshold': threshold,
        }

    def save(self, path: Path):
        """Save the model."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self.model.save_model(str(path))

    def load(self, path: Path):
        """Load a saved model."""
        if self.model is None:
            self.create_model()
        self.model.load_model(str(path))
        self._is_fitted = True


def create_tabnet_model(
    task_type: str = 'regression',
    **kwargs
) -> TabNetWrapper:
    """
    Factory function to create a TabNet model.

    Args:
        task_type: 'regression' or 'classification'
        **kwargs: Additional TabNet parameters

    Returns:
        Configured TabNetWrapper instance.
    """
    default_params = {
        'n_d': 64,
        'n_a': 64,
        'n_steps': 5,
        'gamma': 1.5,
        'mask_type': 'sparsemax',
        'learning_rate': 1e-3,
        'patience': 20,
        'max_epochs': 100,
        'batch_size': 4096,
    }

    # Override with provided kwargs
    default_params.update(kwargs)
    default_params['task_type'] = task_type

    return TabNetWrapper(**default_params)


def prepare_tabnet_data(
    numeric: torch.Tensor,
    categorical: torch.Tensor,
    boolean: torch.Tensor,
    cat_embedding: Any,
    numeric_bn: Optional[Any] = None,
) -> np.ndarray:
    """
    Prepare data for TabNet by concatenating processed features.

    TabNet works with flat numpy arrays, so we:
    1. Process categorical through embeddings
    2. Normalize numeric if batch norm provided
    3. Concatenate all features
    4. Convert to numpy

    Args:
        numeric: Numeric features tensor
        categorical: Categorical features tensor (indices)
        boolean: Boolean features tensor
        cat_embedding: CategoricalEmbedding module
        numeric_bn: Optional BatchNorm module

    Returns:
        Concatenated features as numpy array.
    """
    with torch.no_grad():
        # Process categorical through embeddings
        cat_embedded = cat_embedding(categorical)

        # Normalize numeric if available
        if numeric_bn is not None:
            numeric = numeric_bn(numeric)

        # Concatenate
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        return x.cpu().numpy()
