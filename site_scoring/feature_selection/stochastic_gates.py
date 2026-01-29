"""
Stochastic Gates (STG) for Differentiable Feature Selection.

Implements the method from:
"Feature Selection using Stochastic Gates" - ICML 2020
https://proceedings.mlr.press/v119/yamada20a.html

The key idea is to attach a learnable gate z_d to each input feature.
The gate is drawn from a continuous relaxation of the Bernoulli distribution
using a hard-sigmoid applied to a Gaussian, enabling gradient-based optimization
of a differentiable approximation to L0 regularization.

Mathematical formulation:
    z_d = clamp(0.5 + μ_d + ε_d, 0, 1)   where ε_d ~ N(0, σ²)
    Loss = L(f(z ⊙ x), y) + λ·Σ_d P(z_d > 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class StochasticGates(nn.Module):
    """
    Stochastic Gates layer for differentiable feature selection.

    Uses hard-sigmoid Gaussian relaxation of Bernoulli distribution to create
    learnable gates that can be trained via gradient descent while approximating
    L0 (feature count) regularization.

    Args:
        n_features: Number of input features to gate
        sigma: Standard deviation of the Gaussian noise (default: 0.5)
        reg_weight: Weight for L0 regularization term (lambda)
        init_mean: Initial mean for gate parameters (higher = more likely active)
    """

    def __init__(
        self,
        n_features: int,
        sigma: float = 0.5,
        reg_weight: float = 0.1,
        init_mean: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.sigma = sigma
        self.reg_weight = reg_weight

        # Learnable gate parameters (μ_d for each feature)
        # Initialize slightly positive so all features start active
        self.mu = nn.Parameter(torch.ones(n_features) * init_mean)

        # Track feature elimination history
        self.register_buffer('elimination_history', torch.zeros(n_features))
        self.register_buffer('epoch_active_counts', torch.zeros(0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with stochastic gating.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            gated_x: Input weighted by gate values (batch_size, n_features)
            reg_loss: L0 regularization term (scalar)
        """
        if self.training:
            # Sample from hard-sigmoid Gaussian during training
            epsilon = torch.randn_like(self.mu) * self.sigma
            z = torch.clamp(0.5 + self.mu + epsilon, 0, 1)
        else:
            # Deterministic at inference: use expected gate value
            z = torch.clamp(0.5 + self.mu, 0, 1)

        # Gate the input (element-wise multiplication with broadcasting)
        gated_x = x * z.unsqueeze(0)

        # L0 regularization: expected number of active gates
        # P(z > 0) ≈ sigmoid((0.5 + mu) / sigma * scaling_factor)
        gate_active_prob = self._compute_active_prob()
        reg_loss = self.reg_weight * gate_active_prob.sum()

        return gated_x, reg_loss

    def _compute_active_prob(self) -> torch.Tensor:
        """
        Compute probability each gate is active (z > 0).

        Uses sigmoid approximation to the Gaussian CDF for differentiability.
        P(z > 0) = P(0.5 + μ + ε > 0) = P(ε > -0.5 - μ) = Φ((0.5 + μ) / σ)
        """
        # Sigmoid approximation with scaling factor ~1.7 to match Gaussian CDF
        return torch.sigmoid((0.5 + self.mu) / self.sigma * 1.702)

    def get_gate_values(self, clamp: bool = True) -> torch.Tensor:
        """
        Get current gate values (expected values, not sampled).

        Args:
            clamp: If True, clamp to [0, 1]; if False, return raw μ values
        """
        if clamp:
            return torch.clamp(0.5 + self.mu, 0, 1).detach()
        return self.mu.detach()

    def get_feature_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary mask of selected features.

        Args:
            threshold: Probability threshold for considering a feature active
        """
        probs = self._compute_active_prob()
        return (probs > threshold).detach()

    def get_feature_importance(self) -> torch.Tensor:
        """
        Return feature importance scores (gate activation probabilities).
        Higher probability = more important feature.
        """
        return self._compute_active_prob().detach()

    def get_n_active_features(self, threshold: float = 0.5) -> int:
        """Return count of active features above threshold."""
        return self.get_feature_mask(threshold).sum().item()

    def record_epoch_stats(self):
        """Record statistics at end of epoch for tracking."""
        n_active = self.get_n_active_features()
        self.epoch_active_counts = torch.cat([
            self.epoch_active_counts,
            torch.tensor([n_active], device=self.mu.device, dtype=torch.float)
        ])

    def get_selected_feature_indices(self, threshold: float = 0.5) -> List[int]:
        """Return indices of features that pass the selection threshold."""
        mask = self.get_feature_mask(threshold)
        return torch.where(mask)[0].tolist()

    def get_eliminated_feature_indices(self, threshold: float = 0.5) -> List[int]:
        """Return indices of features that were eliminated."""
        mask = self.get_feature_mask(threshold)
        return torch.where(~mask)[0].tolist()

    def get_gate_probabilities(self) -> torch.Tensor:
        """
        Get gate activation probabilities.
        Alias for get_feature_importance for compatibility.
        """
        return self.get_feature_importance()

    def get_selected_feature_names(
        self,
        feature_names: List[str],
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Get names of selected features.

        Args:
            feature_names: List of feature names
            threshold: Probability threshold for selection

        Returns:
            List of selected feature names
        """
        indices = self.get_selected_feature_indices(threshold)
        return [feature_names[i] for i in indices if i < len(feature_names)]

    def get_regularization_loss(self) -> torch.Tensor:
        """Get the L0 regularization loss term."""
        gate_active_prob = self._compute_active_prob()
        return self.reg_weight * gate_active_prob.sum()


class STGWrapper(nn.Module):
    """
    Wrapper that adds Stochastic Gates to any neural network model.

    This wrapper intercepts the forward pass, applies stochastic gating
    to the concatenated features, and returns both the model output
    and the STG regularization loss.

    Args:
        base_model: The underlying neural network
        n_features: Total number of input features (after embedding)
        stg_sigma: Sigma parameter for STG
        stg_lambda: Regularization weight (lambda)
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_features: int,
        stg_sigma: float = 0.5,
        stg_lambda: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.stg = StochasticGates(
            n_features=n_features,
            sigma=stg_sigma,
            reg_weight=stg_lambda,
        )

        # Store feature names for reporting
        self.feature_names: Optional[List[str]] = None

    def set_feature_names(self, names: List[str]):
        """Set feature names for interpretable reporting."""
        self.feature_names = names

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with STG gating applied after feature processing.

        Returns:
            predictions: Model output
            stg_reg: STG regularization loss term
        """
        # Process categorical through embeddings
        cat_embedded = self.base_model.cat_embedding(categorical)

        # Normalize numeric features
        if self.base_model.numeric_bn is not None and self.base_model.n_numeric > 0:
            numeric = self.base_model.numeric_bn(numeric)

        # Concatenate all features
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        # Apply stochastic gates
        x_gated, stg_reg = self.stg(x)

        # Pass through MLP
        x = self.base_model.mlp(x_gated)
        predictions = self.base_model.output(x)

        return predictions, stg_reg

    def predict(self, numeric: torch.Tensor, categorical: torch.Tensor, boolean: torch.Tensor) -> torch.Tensor:
        """Prediction without returning regularization (for inference)."""
        pred, _ = self.forward(numeric, categorical, boolean)
        return pred

    def get_feature_importance_dict(self) -> dict:
        """Get feature importance as a dictionary with names."""
        importance = self.stg.get_feature_importance().cpu().numpy()

        if self.feature_names is not None:
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def get_selected_features(self, threshold: float = 0.5) -> List[str]:
        """Get names of selected features."""
        indices = self.stg.get_selected_feature_indices(threshold)

        if self.feature_names is not None:
            return [self.feature_names[i] for i in indices]
        return [f"feature_{i}" for i in indices]

    def get_eliminated_features(self, threshold: float = 0.5) -> List[str]:
        """Get names of eliminated features."""
        indices = self.stg.get_eliminated_feature_indices(threshold)

        if self.feature_names is not None:
            return [self.feature_names[i] for i in indices]
        return [f"feature_{i}" for i in indices]

    def get_selection_summary(self, threshold: float = 0.5) -> dict:
        """Get comprehensive summary of feature selection."""
        importance = self.stg.get_feature_importance().cpu().numpy()
        mask = self.stg.get_feature_mask(threshold).cpu().numpy()

        summary = {
            'n_total_features': self.stg.n_features,
            'n_selected': int(mask.sum()),
            'n_eliminated': int((~mask).sum()),
            'selection_rate': float(mask.mean()),
            'selected_features': self.get_selected_features(threshold),
            'eliminated_features': self.get_eliminated_features(threshold),
            'importance_scores': self.get_feature_importance_dict(),
        }

        # Top 10 most important features
        if self.feature_names is not None:
            sorted_features = sorted(
                zip(self.feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )
            summary['top_10_features'] = [
                {'name': name, 'importance': float(imp)}
                for name, imp in sorted_features[:10]
            ]

        return summary
