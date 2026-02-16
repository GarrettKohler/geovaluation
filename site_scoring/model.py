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


class ClusteringModel(nn.Module):
    """
    Deep Embedded Clustering (DEC) model combining autoencoder with learnable centroids.

    Architecture:
    - Phase 1 (Pretrain): Autoencoder learns compact latent representation
    - Phase 2 (Clustering): Soft assignment using Student's t-distribution + KL-divergence loss

    The model learns to:
    1. Compress site features into a low-dimensional latent space
    2. Discover natural clusters among top-performing sites
    3. Identify distinguishing features for each cluster

    Reference: Xie et al. "Unsupervised Deep Embedding for Clustering Analysis" (ICML 2016)
    """

    def __init__(
        self,
        n_numeric: int,
        n_boolean: int,
        categorical_vocab_sizes: Dict[str, int],
        embedding_dim: int = 16,
        latent_dim: int = 32,
        n_clusters: int = 5,
        encoder_dims: List[int] = None,
        dropout: float = 0.2,
        alpha: float = 1.0,  # Student's t-distribution degrees of freedom
    ):
        """
        Initialize the clustering model.

        Args:
            n_numeric: Number of numeric features
            n_boolean: Number of boolean features
            categorical_vocab_sizes: Dict mapping categorical feature names to vocab sizes
            embedding_dim: Dimension for categorical embeddings
            latent_dim: Dimension of the latent space (z)
            n_clusters: Number of clusters to learn
            encoder_dims: Hidden layer dimensions for encoder (default: [512, 256, 128])
            dropout: Dropout rate
            alpha: Degrees of freedom for Student's t-distribution (default 1.0 = Cauchy)
        """
        super().__init__()
        encoder_dims = encoder_dims or [512, 256, 128]

        self.n_numeric = n_numeric
        self.n_boolean = n_boolean
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Categorical embeddings (same as SiteScoringModel)
        self.cat_embedding = CategoricalEmbedding(categorical_vocab_sizes, embedding_dim)

        # Numeric normalization
        self.numeric_bn = nn.BatchNorm1d(n_numeric) if n_numeric > 0 else None

        # Calculate input dimension
        total_input_dim = self.cat_embedding.output_dim + n_numeric + n_boolean
        self.input_dim = total_input_dim

        # Build encoder: input -> encoder_dims -> latent_dim
        encoder_layers = []
        in_dim = total_input_dim
        for out_dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: latent_dim -> reversed(encoder_dims) -> input
        decoder_dims = list(reversed(encoder_dims))
        decoder_layers = []
        in_dim = latent_dim
        for out_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, total_input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Cluster centroids in latent space (learnable parameters)
        # Initialized later with k-means
        self.cluster_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self._centroids_initialized = False

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

    def _prepare_input(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        """Combine all input features into a single tensor."""
        # Process categorical through embeddings
        cat_embedded = self.cat_embedding(categorical)

        # Normalize numeric features
        if self.numeric_bn is not None and self.n_numeric > 0:
            numeric = self.numeric_bn(numeric)

        # Concatenate all features
        return torch.cat([cat_embedded, numeric, boolean], dim=1)

    def encode(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode input features to latent space.

        Args:
            numeric: (batch, n_numeric) float tensor
            categorical: (batch, n_categorical) long tensor
            boolean: (batch, n_boolean) float tensor

        Returns:
            z: (batch, latent_dim) latent representation
        """
        x = self._prepare_input(numeric, categorical, boolean)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.

        Args:
            z: (batch, latent_dim) latent representation

        Returns:
            x_reconstructed: (batch, input_dim) reconstructed features
        """
        return self.decoder(z)

    def cluster_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft cluster assignments using Student's t-distribution.

        The soft assignment q_ij measures the probability that sample i
        belongs to cluster j, using the similarity between z_i and centroid μ_j:

        q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2) / Σ_k (1 + ||z_i - μ_k||² / α)^(-(α+1)/2)

        This is equivalent to the kernel used in t-SNE.

        Args:
            z: (batch, latent_dim) latent representations

        Returns:
            q: (batch, n_clusters) soft assignment probabilities
        """
        # Compute squared distances to all centroids
        # z: (batch, latent_dim), cluster_centers: (n_clusters, latent_dim)
        # ||z_i - μ_j||² = z² - 2*z*μ + μ²
        z_sq = (z ** 2).sum(dim=1, keepdim=True)  # (batch, 1)
        mu_sq = (self.cluster_centers ** 2).sum(dim=1)  # (n_clusters,)
        cross = torch.mm(z, self.cluster_centers.t())  # (batch, n_clusters)
        dist_sq = z_sq - 2 * cross + mu_sq  # (batch, n_clusters)

        # Student's t-distribution kernel
        q = (1.0 + dist_sq / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)  # Normalize to probabilities

        return q

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p for KL-divergence loss.

        The target distribution sharpens q to emphasize high-confidence assignments:
        p_ij = (q_ij² / f_j) / Σ_k (q_ik² / f_k)

        where f_j = Σ_i q_ij is the soft cluster frequency.

        This auxiliary target helps the model learn more discriminative clusters.

        Args:
            q: (batch, n_clusters) soft assignment probabilities

        Returns:
            p: (batch, n_clusters) target distribution
        """
        # f_j = soft cluster frequencies
        f = q.sum(dim=0)  # (n_clusters,)

        # Compute p = q² / f, then normalize
        p = (q ** 2) / f
        p = p / p.sum(dim=1, keepdim=True)

        return p

    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute KL-divergence clustering loss: KL(P || Q).

        This loss encourages the model to make confident cluster assignments
        by minimizing the divergence from the sharpened target distribution.

        Args:
            q: (batch, n_clusters) soft assignment probabilities
            p: (batch, n_clusters) target distribution (detached, no gradient)

        Returns:
            loss: scalar KL-divergence loss
        """
        # KL(P || Q) = Σ p * log(p / q)
        # Use log_softmax for numerical stability
        return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))) / q.size(0)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        boolean: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            numeric: (batch, n_numeric) float tensor
            categorical: (batch, n_categorical) long tensor
            boolean: (batch, n_boolean) float tensor

        Returns:
            Dict containing:
                - z: latent representation
                - x_reconstructed: decoder output
                - q: soft cluster assignments (if centroids initialized)
        """
        # Encode to latent space
        z = self.encode(numeric, categorical, boolean)

        # Decode for reconstruction
        x_reconstructed = self.decode(z)

        result = {
            'z': z,
            'x_reconstructed': x_reconstructed,
        }

        # Add cluster assignments if centroids are initialized
        if self._centroids_initialized:
            result['q'] = self.cluster_assignment(z)

        return result

    def initialize_centroids(self, z_all: torch.Tensor):
        """
        Initialize cluster centroids using k-means on latent representations.

        This should be called after pretraining the autoencoder but before
        the clustering refinement phase.

        Args:
            z_all: (n_samples, latent_dim) all latent representations
        """
        from sklearn.cluster import KMeans

        # Run k-means on CPU
        z_np = z_all.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        kmeans.fit(z_np)

        # Set centroids as model parameters
        with torch.no_grad():
            self.cluster_centers.copy_(
                torch.from_numpy(kmeans.cluster_centers_).to(self.cluster_centers.device)
            )
        self._centroids_initialized = True

        return kmeans.labels_

    def get_cluster_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get hard cluster assignments (argmax of soft assignments).

        Args:
            z: (batch, latent_dim) latent representations

        Returns:
            assignments: (batch,) cluster indices
        """
        q = self.cluster_assignment(z)
        return torch.argmax(q, dim=1)


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


# =============================================================================
# Gradient Boosting Models (XGBoost)
# =============================================================================

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import numpy as np


class XGBoostModel:
    """
    XGBoost wrapper for tabular regression/classification.

    Advantages:
    - Very fast training with histogram-based algorithm
    - Excellent performance on structured data
    - GPU acceleration support
    - Built-in feature importance
    """

    def __init__(
        self,
        task_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        early_stopping_rounds: int = 50,
        verbosity: int = 1,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize XGBoost model.

        Args:
            task_type: "regression" or "lookalike" (classification)
            feature_names: Names of all features (for interpretability)
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (eta)
            max_depth: Maximum tree depth
            early_stopping_rounds: Stop if no improvement for N rounds
            verbosity: Verbosity level (0=silent, 1=warnings, 2=info)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel threads (-1 for all cores)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.task_type = task_type
        self.feature_names = feature_names
        self.early_stopping_rounds = early_stopping_rounds
        self.is_fitted = False

        common_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'verbosity': verbosity,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'tree_method': 'hist',  # Fast histogram-based algorithm
            'enable_categorical': True,  # Native categorical support (XGBoost 2.0+)
        }

        if task_type == "regression":
            self.model = XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                **common_params
            )
        else:  # lookalike / classification
            self.model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                **common_params
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        progress_callback: Optional[callable] = None,
        callbacks: Optional[List] = None,
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets
            progress_callback: Optional callback for progress updates
            callbacks: Optional list of xgboost.callback.TrainingCallback instances

        Returns:
            self (fitted model)
        """
        fit_params = {}

        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = 100  # Print every 100 iterations

        # XGBoost 2.0+ removed callbacks from fit(); set via set_params instead
        if callbacks:
            self.model.set_params(callbacks=callbacks)

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task_type == "regression":
            raise ValueError("predict_proba not available for regression")
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (as native Python floats for JSON serialization)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        importance = self.model.feature_importances_

        # Convert numpy float32 to native Python float for JSON serialization
        if self.feature_names and len(self.feature_names) == len(importance):
            return {name: float(val) for name, val in zip(self.feature_names, importance)}
        return {i: float(val) for i, val in enumerate(importance)}

    @property
    def best_iteration(self) -> int:
        """Get the best iteration (with early stopping)."""
        return getattr(self.model, 'best_iteration', self.model.n_estimators) if self.is_fitted else 0


def create_model(
    model_type: str,
    task_type: str,
    n_numeric: int,
    n_boolean: int,
    categorical_vocab_sizes: Dict[str, int],
    feature_names: Optional[List[str]] = None,
    config: Optional[Config] = None,
    **kwargs
):
    """
    Factory function to create a model based on type.

    Args:
        model_type: "neural_network" or "xgboost"
        task_type: "regression" or "lookalike"
        n_numeric: Number of numeric features
        n_boolean: Number of boolean features
        categorical_vocab_sizes: Dict mapping categorical feature names to vocab sizes
        feature_names: List of all feature names (in order)
        config: Optional Config object for neural network settings
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Model instance (SiteScoringModel or XGBoostModel)
    """
    if model_type == "neural_network":
        if config is not None:
            return SiteScoringModel.from_config(config, categorical_vocab_sizes)
        return SiteScoringModel(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            **kwargs
        )

    elif model_type == "xgboost":
        return XGBoostModel(
            task_type=task_type,
            feature_names=feature_names,
            n_estimators=kwargs.get('epochs', 1000),
            learning_rate=kwargs.get('learning_rate', 0.03),
            max_depth=kwargs.get('max_depth', 6),
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 50),
        )

    elif model_type == "clustering":
        return ClusteringModel(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dim=kwargs.get('embedding_dim', 16),
            latent_dim=kwargs.get('latent_dim', 32),
            n_clusters=kwargs.get('n_clusters', 5),
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: neural_network, xgboost, clustering")
