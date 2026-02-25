"""
Site Scoring Neural Network Model.
Optimized tabular architecture with embeddings for Apple Silicon MPS.

Architecture:
    Numeric (N)      -> BatchNorm1d ->
    Categorical (7)  -> CategoricalEmbedding ->
    Boolean (30+)    -> pass-through ->
        Concatenate all ->
        ResidualBlock chain [512 -> 256 -> 128 -> 64] ->
        Linear(64 -> 1) ->
        Sigmoid (if classification) or raw (regression)

Model Choice Rationale:
    For tabular regression/classification (26K-57K rows, 60+ features),
    we use an Embedding + Residual MLP because:

    1. Embeddings for categoricals: High-cardinality features (retailer, DMA)
       benefit from learned dense representations vs one-hot encoding.
    2. Residual connections: Enable training deeper networks without vanishing
       gradients, critical for capturing complex feature interactions.
    3. Batch normalization: Stabilizes training with large batch sizes on GPU.
    4. MPS optimization: Uses operations well-supported by Metal
       (linear, batchnorm, relu, embedding).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization.

    Two-layer linear path with BatchNorm + ReLU + Dropout, plus a skip
    connection with optional projection when dimensions change.
    """

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
    """
    Embedding layer for categorical features with learned representations.

    Each categorical feature gets its own embedding table with dimension
    based on vocabulary size: min(embedding_dim, (vocab_size + 1) // 2),
    with a floor of 4.
    """

    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.feature_names = list(vocab_sizes.keys())

        for name, vocab_size in vocab_sizes.items():
            dim = min(embedding_dim, (vocab_size + 1) // 2)
            dim = max(dim, 4)  # Minimum dimension
            self.embeddings[name] = nn.Embedding(
                vocab_size + 1, dim, padding_idx=0
            )

        # Total output dimension (sum of all per-feature embedding dims)
        self.output_dim = sum(
            self.embeddings[name].embedding_dim for name in self.feature_names
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, num_categorical_features).

        Returns:
            Tensor of shape (batch_size, total_embedding_dim).
        """
        embeddings = []
        for i, name in enumerate(self.feature_names):
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

    For classification (lookalike), a sigmoid activation is applied
    to the output. For regression, the raw linear output is returned.
    """

    def __init__(
        self,
        n_numeric: int,
        n_boolean: int,
        categorical_vocab_sizes: Dict[str, int],
        embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        task_type: str = "regression",
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256, 128, 64]
        self.task_type = task_type

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
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
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
            numeric: (batch, n_numeric) float tensor.
            categorical: (batch, n_categorical) long tensor.
            boolean: (batch, n_boolean) float tensor.

        Returns:
            (batch, 1) predictions. Sigmoid-activated for classification,
            raw for regression.
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
        out = self.output(x)

        if self.task_type == "lookalike":
            out = torch.sigmoid(out)

        return out

    @classmethod
    def from_config(
        cls,
        config: Any,
        categorical_vocab_sizes: Dict[str, int],
        n_numeric: int,
        n_boolean: int,
    ) -> "SiteScoringModel":
        """
        Create model from a configuration object.

        Args:
            config: Config with embedding_dim, hidden_dims, dropout,
                    use_batch_norm, and task_type attributes.
            categorical_vocab_sizes: Mapping of feature name to vocab size.
            n_numeric: Number of numeric features (from FeatureProcessor).
            n_boolean: Number of boolean features (from FeatureProcessor).

        Returns:
            Configured SiteScoringModel instance.
        """
        return cls(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=getattr(config, "embedding_dim", 16),
            hidden_dims=getattr(config, "hidden_dims", [512, 256, 128, 64]),
            dropout=getattr(config, "dropout", 0.2),
            use_batch_norm=getattr(config, "use_batch_norm", True),
            task_type=getattr(config, "task_type", "regression"),
        )
