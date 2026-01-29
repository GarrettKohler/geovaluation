"""
LassoNet: Neural Networks with Hierarchical Feature Sparsity.

Implements the method from:
"LassoNet: A Neural Network with Feature Sparsity" - JMLR 2021
https://jmlr.org/papers/volume22/20-848/20-848.pdf

Key idea: Enforce a hierarchy constraint where a feature can only participate
in hidden layers if its linear (skip connection) representative is active.
This creates a direct connection between L1 sparsity on the skip layer and
complete feature elimination from the entire network.

Mathematical formulation:
    minimize θ,W  L(θ,W) + λ‖θ‖₁
    subject to   ‖W_j^(1)‖_∞ ≤ M|θ_j|,  j=1,…,d

When θ_j = 0, the constraint forces W_j = 0, completely removing feature j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import numpy as np


class HierProx:
    """
    Hierarchical Proximal Operator for LassoNet.

    Applies soft-thresholding with hierarchy constraints to enforce
    that when a feature's skip connection weight θ_j is zero,
    all corresponding first-layer weights W_j must also be zero.

    This is the key algorithmic component of LassoNet training.
    """

    @staticmethod
    def apply(
        theta: torch.Tensor,
        W: torch.Tensor,
        lam: float,
        M: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical proximal operator.

        Args:
            theta: Skip connection weights, shape (input_dim,)
            W: First hidden layer weights, shape (hidden_dim, input_dim)
            lam: L1 regularization strength (λ)
            M: Hierarchy coefficient controlling nonlinearity strength

        Returns:
            theta_new: Updated skip connection weights
            W_new: Updated first layer weights (constrained by hierarchy)
        """
        input_dim = theta.shape[0]
        hidden_dim = W.shape[0]

        theta_new = theta.clone()
        W_new = W.clone()

        for j in range(input_dim):
            # Get weights connecting feature j to all hidden units
            w_j = W[:, j].abs()  # Shape: (hidden_dim,)

            # Sort in descending order
            w_sorted, sort_indices = torch.sort(w_j, descending=True)

            # Find optimal m (number of active hidden units for feature j)
            best_m = 0
            best_threshold = lam

            for m in range(hidden_dim + 1):
                # Sum of top-m weight magnitudes
                sum_w = w_sorted[:m].sum() if m > 0 else 0.0

                # Compute threshold
                threshold = (lam + M * sum_w) / (1 + m * M ** 2)

                # Check if this m is valid
                upper_bound = w_sorted[m - 1].item() if m > 0 else float('inf')
                lower_bound = w_sorted[m].item() if m < hidden_dim else 0.0

                if lower_bound <= threshold <= upper_bound:
                    best_m = m
                    best_threshold = threshold
                    break

            # Apply soft-thresholding to theta_j
            theta_abs = torch.abs(theta[j])
            shrinkage = lam / (1 + best_m * M ** 2)

            if theta_abs > shrinkage:
                theta_new[j] = torch.sign(theta[j]) * (theta_abs - shrinkage)
            else:
                theta_new[j] = 0.0

            # Constrain W_j based on hierarchy: ‖W_j‖_∞ ≤ M|θ_j|
            max_w = M * torch.abs(theta_new[j])
            W_new[:, j] = torch.sign(W[:, j]) * torch.clamp(W[:, j].abs(), max=max_w)

        return theta_new, W_new


class LassoNetLayer(nn.Module):
    """
    LassoNet layer combining skip connection with hierarchical constraints.

    This layer implements:
    - A linear skip connection (θ) with L1 regularization
    - A standard linear layer (W) constrained by the hierarchy

    The forward pass computes: θ·x + W·x (skip + transform)
    The hierarchy constraint is applied during optimization via HierProx.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        M: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.M = M

        # Skip connection weights (θ)
        self.theta = nn.Parameter(torch.randn(input_dim) * 0.01)

        # First hidden layer (W)
        self.linear = nn.Linear(input_dim, hidden_dim, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            skip_out: Skip connection output, shape (batch_size, 1)
            hidden_out: Hidden layer output, shape (batch_size, hidden_dim)
        """
        # Skip connection: θ·x
        skip_out = (x * self.theta).sum(dim=1, keepdim=True)

        # Hidden layer: W·x + b
        hidden_out = self.linear(x)

        return skip_out, hidden_out

    def apply_hier_prox(self, lam: float):
        """Apply hierarchical proximal operator to enforce constraints."""
        with torch.no_grad():
            self.theta.data, self.linear.weight.data = HierProx.apply(
                self.theta.data,
                self.linear.weight.data.T,  # Transpose: (input, hidden) -> (hidden, input)
                lam=lam,
                M=self.M,
            )
            self.linear.weight.data = self.linear.weight.data  # Already updated

    def get_active_features(self, threshold: float = 1e-6) -> torch.Tensor:
        """Return binary mask of active features (|θ| > threshold)."""
        return (torch.abs(self.theta) > threshold).detach()

    def get_feature_importance(self) -> torch.Tensor:
        """Return feature importance based on |θ| values."""
        return torch.abs(self.theta).detach()

    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on theta for loss function."""
        return torch.abs(self.theta).sum()


class LassoNetModel(nn.Module):
    """
    Complete LassoNet model for site scoring.

    Architecture:
    1. Categorical embeddings
    2. Feature concatenation (embedded + numeric + boolean)
    3. LassoNet layer (skip connection + first hidden)
    4. Residual MLP blocks
    5. Output layer

    The skip connection output is added to the final prediction,
    creating the hierarchical constraint structure.
    """

    def __init__(
        self,
        n_numeric: int,
        n_boolean: int,
        categorical_vocab_sizes: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        M: float = 10.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]
        self.M = M
        self.n_numeric = n_numeric
        self.n_boolean = n_boolean

        # Categorical embeddings
        from ..model import CategoricalEmbedding
        self.cat_embedding = CategoricalEmbedding(categorical_vocab_sizes, embedding_dim)

        # Numeric normalization
        self.numeric_bn = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None

        # Total input dimension
        total_input_dim = self.cat_embedding.output_dim + n_numeric + n_boolean
        self.total_input_dim = total_input_dim

        # LassoNet layer (skip + first hidden)
        self.lassonet_layer = LassoNetLayer(
            input_dim=total_input_dim,
            hidden_dim=hidden_dims[0],
            M=M,
        )

        # Remaining MLP layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

        # Feature names for reporting
        self.feature_names: Optional[List[str]] = None

    def set_feature_names(self, names: List[str]):
        """Set feature names for interpretable reporting."""
        self.feature_names = names

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            predictions: Combined skip + nonlinear output
        """
        # Process categorical through embeddings
        cat_embedded = self.cat_embedding(categorical)

        # Normalize numeric features
        if self.numeric_bn is not None and self.n_numeric > 0:
            numeric = self.numeric_bn(numeric)

        # Concatenate all features
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        # LassoNet layer: skip connection + first hidden
        skip_out, hidden = self.lassonet_layer(x)

        # Apply ReLU to hidden
        hidden = F.relu(hidden)

        # Pass through remaining MLP layers
        for layer in self.mlp_layers:
            hidden = layer(hidden)

        # Final output: skip + nonlinear
        nonlinear_out = self.output(hidden)

        return skip_out + nonlinear_out

    def apply_hier_prox(self, lam: float):
        """Apply hierarchical proximal operator after gradient step."""
        self.lassonet_layer.apply_hier_prox(lam)

    def get_l1_penalty(self) -> torch.Tensor:
        """Get L1 penalty term for loss function."""
        return self.lassonet_layer.l1_penalty()

    def get_active_features(self, threshold: float = 1e-6) -> torch.Tensor:
        """Return mask of active features."""
        return self.lassonet_layer.get_active_features(threshold)

    def get_feature_importance(self) -> torch.Tensor:
        """Return feature importance scores."""
        return self.lassonet_layer.get_feature_importance()

    def get_n_active_features(self, threshold: float = 1e-6) -> int:
        """Return count of active features."""
        return self.get_active_features(threshold).sum().item()

    def get_selection_summary(self, threshold: float = 1e-6) -> dict:
        """Get comprehensive summary of feature selection."""
        importance = self.get_feature_importance().cpu().numpy()
        mask = self.get_active_features(threshold).cpu().numpy()

        selected_indices = np.where(mask)[0].tolist()
        eliminated_indices = np.where(~mask)[0].tolist()

        if self.feature_names is not None:
            selected = [self.feature_names[i] for i in selected_indices]
            eliminated = [self.feature_names[i] for i in eliminated_indices]
            importance_dict = {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        else:
            selected = [f"feature_{i}" for i in selected_indices]
            eliminated = [f"feature_{i}" for i in eliminated_indices]
            importance_dict = {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

        return {
            'n_total_features': self.total_input_dim,
            'n_selected': int(mask.sum()),
            'n_eliminated': int((~mask).sum()),
            'selection_rate': float(mask.mean()),
            'selected_features': selected,
            'eliminated_features': eliminated,
            'importance_scores': importance_dict,
        }


class LassoNetWrapper:
    """
    Wrapper for training LassoNet models with proper optimization.

    Implements the two-step training procedure:
    1. Standard gradient update
    2. Hierarchical proximal operator application

    Also supports training across a regularization path (λ path)
    for automatic feature selection.
    """

    def __init__(
        self,
        model: LassoNetModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.HuberLoss(delta=1.0)

    def train_step(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
        target: torch.Tensor,
        lam: float,
    ) -> dict:
        """
        Single training step with Hier-Prox.

        Args:
            numeric, categorical, boolean: Input features
            target: Target values
            lam: Current lambda (regularization strength)

        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        predictions = self.model(numeric, categorical, boolean)

        # Task loss + L1 penalty
        task_loss = self.criterion(predictions, target)
        l1_penalty = lam * self.model.get_l1_penalty()
        total_loss = task_loss + l1_penalty

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Apply hierarchical proximal operator
        self.model.apply_hier_prox(lam)

        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'l1_penalty': l1_penalty.item(),
            'n_active_features': self.model.get_n_active_features(),
        }

    @staticmethod
    def generate_lambda_path(
        n_lambdas: int = 50,
        lambda_min: float = 1e-4,
        lambda_max: float = 1.0,
    ) -> List[float]:
        """
        Generate logarithmic lambda path from dense to sparse.

        Returns lambdas in increasing order (start dense, become sparse).
        """
        return np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas).tolist()
