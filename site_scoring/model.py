"""
Site Scoring Neural Network Model.
Optimized tabular architecture with embeddings for Apple M4 MPS.

Model Choice Rationale:
-----------------------
For this tabular regression problem (1.47M rows, 94 features), we use an
**Embedding + Residual MLP** architecture because:

1. **Embeddings for Categoricals**: High-cardinality features (retailer, DMA, etc.)
   benefit from learned dense representations vs one-hot encoding.

2. **Residual Connections**: Enable training deeper networks without vanishing
   gradients, critical for capturing complex feature interactions.

3. **Batch Normalization**: Stabilizes training with large batch sizes on M4 GPU.

4. **MPS Optimization**: Architecture uses operations well-supported by Metal
   (linear, batchnorm, relu, embedding) avoiding unsupported operations.

Alternative approaches considered:
- TabNet: More interpretable but slower, attention overhead
- FT-Transformer: Better for <100k samples, more memory intensive
- XGBoost/LightGBM: Often better for tabular, but user requested PyTorch

This architecture offers a good balance of performance, training speed, and
M4 compatibility for this specific dataset.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .config import Config


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


class CategoricalEmbedding(nn.Module):
    """Embedding layer for categorical features with learned representations."""

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.feature_names = list(vocab_sizes.keys())

        for name, vocab_size in vocab_sizes.items():
            # Use embedding dimension based on cardinality
            dim = min(embedding_dim, (vocab_size + 1) // 2)
            dim = max(dim, 4)  # Minimum dimension
            self.embeddings[name] = nn.Embedding(vocab_size + 1, dim, padding_idx=0)

        # Calculate total output dimension
        self.output_dim = sum(
            self.embeddings[name].embedding_dim for name in self.feature_names
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_categorical_features)
        Returns:
            Tensor of shape (batch_size, total_embedding_dim)
        """
        embeddings = []
        for i, name in enumerate(self.feature_names):
            # Clamp indices to valid range
            idx = x[:, i].clamp(0, self.embeddings[name].num_embeddings - 1)
            embeddings.append(self.embeddings[name](idx))
        return torch.cat(embeddings, dim=1)


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

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization is more stable for deep networks
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Smaller initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            numeric: (batch, n_numeric) float tensor
            categorical: (batch, n_categorical) long tensor
            boolean: (batch, n_boolean) float tensor

        Returns:
            (batch, 1) predictions
        """
        # Process categorical through embeddings
        cat_embedded = self.cat_embedding(categorical)

        # Normalize numeric features
        if self.numeric_bn is not None and self.n_numeric > 0:
            numeric = self.numeric_bn(numeric)

        # Concatenate all features
        x = torch.cat([cat_embedded, numeric, boolean], dim=1)

        # Pass through MLP
        x = self.mlp(x)

        # Output
        return self.output(x)

    @classmethod
    def from_config(cls, config: Config, categorical_vocab_sizes: Dict[str, int]) -> "SiteScoringModel":
        """Create model from config."""
        # Count available features
        n_numeric = len(config.numeric_features)
        n_boolean = len(config.boolean_features)

        return cls(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
        )


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved predictions.
    Uses model averaging for more robust results.
    """

    def __init__(self, models: List[SiteScoringModel]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        predictions = [model(numeric, categorical, boolean) for model in self.models]
        return torch.stack(predictions).mean(dim=0)
