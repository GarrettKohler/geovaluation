"""
Unit tests for the ClusteringModel (Deep Embedded Clustering).

Tests the autoencoder architecture, cluster assignment, and loss functions
without requiring trained models or actual data.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests for ClusteringModel architecture
# ---------------------------------------------------------------------------


class TestClusteringModelInit:
    """Tests for ClusteringModel initialization."""

    def test_model_creates_with_minimal_params(self):
        """Model initializes with minimal required parameters."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10, "cat2": 20},
        )

        assert model.n_numeric == 10
        assert model.n_boolean == 5
        assert model.latent_dim == 32  # default
        assert model.n_clusters == 5  # default

    def test_model_creates_with_custom_params(self):
        """Model initializes with custom parameters."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=15,
            n_boolean=8,
            categorical_vocab_sizes={"feature": 50},
            embedding_dim=32,
            latent_dim=64,
            n_clusters=10,
        )

        assert model.latent_dim == 64
        assert model.n_clusters == 10

    def test_encoder_is_sequential(self):
        """Encoder is a Sequential module."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10},
        )

        assert isinstance(model.encoder, torch.nn.Sequential)

    def test_decoder_is_sequential(self):
        """Decoder is a Sequential module."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10},
        )

        assert isinstance(model.decoder, torch.nn.Sequential)

    def test_cluster_centers_shape(self):
        """Cluster centers parameter has correct shape."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10},
            latent_dim=32,
            n_clusters=7,
        )

        assert model.cluster_centers.shape == (7, 32)

    def test_centroids_not_initialized_by_default(self):
        """Centroids initialized flag is False initially."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10},
        )

        assert model._centroids_initialized is False


# ---------------------------------------------------------------------------
# Tests for ClusteringModel forward pass
# ---------------------------------------------------------------------------


class TestClusteringModelForward:
    """Tests for ClusteringModel forward pass dimensions."""

    @pytest.fixture
    def model_and_data(self):
        """Create model and sample data for testing."""
        from site_scoring.model import ClusteringModel

        n_numeric = 10
        n_boolean = 5
        categorical_vocab_sizes = {"cat1": 10, "cat2": 20}
        latent_dim = 16
        batch_size = 8

        model = ClusteringModel(
            n_numeric=n_numeric,
            n_boolean=n_boolean,
            categorical_vocab_sizes=categorical_vocab_sizes,
            latent_dim=latent_dim,
            n_clusters=4,
        )

        # Create sample data
        numeric = torch.randn(batch_size, n_numeric)
        categorical = torch.randint(0, 10, (batch_size, len(categorical_vocab_sizes)))
        boolean = torch.randint(0, 2, (batch_size, n_boolean)).float()

        return model, numeric, categorical, boolean, batch_size, latent_dim

    def test_encode_output_shape(self, model_and_data):
        """Encode produces correct output shape."""
        model, numeric, categorical, boolean, batch_size, latent_dim = model_and_data

        z = model.encode(numeric, categorical, boolean)

        assert z.shape == (batch_size, latent_dim)

    def test_decode_output_shape(self, model_and_data):
        """Decode produces correct output shape matching input."""
        model, numeric, categorical, boolean, batch_size, latent_dim = model_and_data

        z = model.encode(numeric, categorical, boolean)
        x_reconstructed = model.decode(z)

        assert x_reconstructed.shape == (batch_size, model.input_dim)

    def test_forward_returns_dict(self, model_and_data):
        """Forward returns a dictionary."""
        model, numeric, categorical, boolean, _, _ = model_and_data

        output = model(numeric, categorical, boolean)

        assert isinstance(output, dict)
        assert 'z' in output
        assert 'x_reconstructed' in output

    def test_forward_no_q_without_centroids(self, model_and_data):
        """Forward doesn't include 'q' if centroids not initialized."""
        model, numeric, categorical, boolean, _, _ = model_and_data

        output = model(numeric, categorical, boolean)

        assert 'q' not in output

    def test_forward_includes_q_with_centroids(self, model_and_data):
        """Forward includes 'q' after centroid initialization."""
        model, numeric, categorical, boolean, batch_size, latent_dim = model_and_data

        # Initialize centroids manually
        model._centroids_initialized = True
        with torch.no_grad():
            model.cluster_centers.data = torch.randn(model.n_clusters, latent_dim)

        output = model(numeric, categorical, boolean)

        assert 'q' in output
        assert output['q'].shape == (batch_size, model.n_clusters)


# ---------------------------------------------------------------------------
# Tests for cluster assignment
# ---------------------------------------------------------------------------


class TestClusterAssignment:
    """Tests for soft cluster assignment."""

    @pytest.fixture
    def model_with_centroids(self):
        """Create model with initialized centroids."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=5,
            n_boolean=3,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=8,
            n_clusters=3,
        )

        # Initialize centroids
        model._centroids_initialized = True
        with torch.no_grad():
            model.cluster_centers.data = torch.tensor([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ])

        return model

    def test_assignment_sums_to_one(self, model_with_centroids):
        """Soft assignments sum to 1 for each sample."""
        model = model_with_centroids

        z = torch.randn(10, 8)
        q = model.cluster_assignment(z)

        # Check sums are close to 1
        sums = q.sum(dim=1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_assignment_probabilities_positive(self, model_with_centroids):
        """All assignment probabilities are positive."""
        model = model_with_centroids

        z = torch.randn(10, 8)
        q = model.cluster_assignment(z)

        assert (q >= 0).all()

    def test_assignment_prefers_nearest_centroid(self, model_with_centroids):
        """Assignment gives highest probability to nearest centroid."""
        model = model_with_centroids

        # Point very close to first centroid
        z = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        q = model.cluster_assignment(z)

        # First cluster should have highest probability
        assert q[0, 0] > q[0, 1]
        assert q[0, 0] > q[0, 2]


# ---------------------------------------------------------------------------
# Tests for target distribution and clustering loss
# ---------------------------------------------------------------------------


class TestClusteringLoss:
    """Tests for target distribution and KL-divergence loss."""

    @pytest.fixture
    def model(self):
        """Create a clustering model."""
        from site_scoring.model import ClusteringModel

        return ClusteringModel(
            n_numeric=5,
            n_boolean=3,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=8,
            n_clusters=3,
        )

    def test_target_distribution_shape(self, model):
        """Target distribution has same shape as input."""
        q = torch.softmax(torch.randn(10, 3), dim=1)
        p = model.target_distribution(q)

        assert p.shape == q.shape

    def test_target_distribution_sums_to_one(self, model):
        """Target distribution sums to 1 for each sample."""
        q = torch.softmax(torch.randn(10, 3), dim=1)
        p = model.target_distribution(q)

        sums = p.sum(dim=1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_target_distribution_sharpens(self, model):
        """Target distribution is sharper than input (higher max probability)."""
        q = torch.softmax(torch.randn(10, 3), dim=1)
        p = model.target_distribution(q)

        # Target should have higher max probabilities on average
        assert p.max(dim=1).values.mean() >= q.max(dim=1).values.mean()

    def test_clustering_loss_is_scalar(self, model):
        """Clustering loss returns a scalar."""
        q = torch.softmax(torch.randn(10, 3), dim=1)
        p = model.target_distribution(q).detach()

        loss = model.clustering_loss(q, p)

        assert loss.dim() == 0  # scalar

    def test_clustering_loss_is_non_negative(self, model):
        """KL-divergence is always non-negative."""
        q = torch.softmax(torch.randn(10, 3), dim=1)
        p = model.target_distribution(q).detach()

        loss = model.clustering_loss(q, p)

        assert loss >= 0


# ---------------------------------------------------------------------------
# Tests for centroid initialization
# ---------------------------------------------------------------------------


class TestCentroidInitialization:
    """Tests for k-means centroid initialization."""

    def test_initialize_centroids_sets_flag(self):
        """initialize_centroids sets _centroids_initialized to True."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=5,
            n_boolean=3,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=8,
            n_clusters=3,
        )

        z_all = torch.randn(100, 8)
        model.initialize_centroids(z_all)

        assert model._centroids_initialized is True

    def test_initialize_centroids_returns_labels(self):
        """initialize_centroids returns cluster labels."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=5,
            n_boolean=3,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=8,
            n_clusters=3,
        )

        z_all = torch.randn(100, 8)
        labels = model.initialize_centroids(z_all)

        assert len(labels) == 100
        assert set(labels).issubset({0, 1, 2})

    def test_get_cluster_assignments_matches_soft(self):
        """Hard assignments match argmax of soft assignments."""
        from site_scoring.model import ClusteringModel

        model = ClusteringModel(
            n_numeric=5,
            n_boolean=3,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=8,
            n_clusters=3,
        )

        z_all = torch.randn(50, 8)
        model.initialize_centroids(z_all)

        q = model.cluster_assignment(z_all)
        hard = model.get_cluster_assignments(z_all)

        expected = torch.argmax(q, dim=1)
        assert torch.equal(hard, expected)


# ---------------------------------------------------------------------------
# Tests for create_model factory with clustering
# ---------------------------------------------------------------------------


class TestCreateModelClustering:
    """Tests for create_model factory with clustering type."""

    def test_create_model_clustering_returns_clustering_model(self):
        """create_model with 'clustering' returns ClusteringModel."""
        from site_scoring.model import create_model, ClusteringModel

        model = create_model(
            model_type="clustering",
            task_type="clustering",
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat": 10},
        )

        assert isinstance(model, ClusteringModel)

    def test_create_model_clustering_custom_params(self):
        """create_model passes custom params to ClusteringModel."""
        from site_scoring.model import create_model

        model = create_model(
            model_type="clustering",
            task_type="clustering",
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat": 10},
            latent_dim=64,
            n_clusters=8,
        )

        assert model.latent_dim == 64
        assert model.n_clusters == 8


# ---------------------------------------------------------------------------
# Tests for clustering config
# ---------------------------------------------------------------------------


class TestClusteringConfig:
    """Tests for clustering configuration in Config."""

    def test_config_has_clustering_params(self):
        """Config dataclass has clustering parameters."""
        from site_scoring.config import Config

        config = Config()

        assert hasattr(config, 'n_clusters')
        assert hasattr(config, 'latent_dim')
        assert hasattr(config, 'pretrain_epochs')
        assert hasattr(config, 'clustering_epochs')
        assert hasattr(config, 'cluster_probability_threshold')

    def test_config_clustering_defaults(self):
        """Config has expected default values for clustering."""
        from site_scoring.config import Config

        config = Config()

        assert config.n_clusters == 5
        assert config.latent_dim == 32
        assert config.pretrain_epochs == 20
        assert config.clustering_epochs == 30
        assert config.cluster_probability_threshold == 0.5


# ---------------------------------------------------------------------------
# Tests for TrainingConfig clustering params
# ---------------------------------------------------------------------------


class TestTrainingConfigClustering:
    """Tests for TrainingConfig clustering parameters."""

    def test_training_config_has_clustering_params(self):
        """TrainingConfig has clustering parameters."""
        from src.services.training_service import TrainingConfig

        config = TrainingConfig()

        assert hasattr(config, 'n_clusters')
        assert hasattr(config, 'latent_dim')
        assert hasattr(config, 'cluster_probability_threshold')
        assert hasattr(config, 'pretrain_epochs')
        assert hasattr(config, 'clustering_epochs')

    def test_training_config_clustering_defaults(self):
        """TrainingConfig has expected clustering defaults."""
        from src.services.training_service import TrainingConfig

        config = TrainingConfig()

        assert config.n_clusters == 5
        assert config.latent_dim == 32
        assert config.cluster_probability_threshold == 0.5
        assert config.pretrain_epochs == 20
        assert config.clustering_epochs == 30
