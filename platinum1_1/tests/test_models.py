"""Tests for models/neural_network.py and models/factory.py."""

import torch
import pytest

from platinum1_1.models.neural_network import (
    SiteScoringModel,
    ResidualBlock,
    CategoricalEmbedding,
)
from platinum1_1.models.factory import create_model


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------

class TestResidualBlock:
    def test_same_dim(self):
        block = ResidualBlock(64, 64)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 64)

    def test_dim_change(self):
        block = ResidualBlock(64, 32)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 32)


# ---------------------------------------------------------------------------
# CategoricalEmbedding
# ---------------------------------------------------------------------------

class TestCategoricalEmbedding:
    def test_output_shape(self):
        vocab_sizes = {"network": 5, "program": 3}
        emb = CategoricalEmbedding(vocab_sizes, embedding_dim=16)

        x = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
        out = emb(x)
        assert out.shape == (2, emb.output_dim)

    def test_minimum_dim(self):
        """Embedding dim should be at least 4."""
        vocab_sizes = {"tiny": 2}
        emb = CategoricalEmbedding(vocab_sizes, embedding_dim=16)
        assert emb.embeddings["tiny"].embedding_dim >= 4


# ---------------------------------------------------------------------------
# SiteScoringModel
# ---------------------------------------------------------------------------

class TestSiteScoringModel:
    @pytest.fixture
    def model_regression(self):
        return SiteScoringModel(
            n_numeric=5,
            n_boolean=10,
            categorical_vocab_sizes={"network": 5, "program": 3},
            embedding_dim=8,
            hidden_dims=[32, 16],
            task_type="regression",
        )

    @pytest.fixture
    def model_classification(self):
        return SiteScoringModel(
            n_numeric=5,
            n_boolean=10,
            categorical_vocab_sizes={"network": 5, "program": 3},
            embedding_dim=8,
            hidden_dims=[32, 16],
            task_type="lookalike",
        )

    def test_regression_forward(self, model_regression):
        numeric = torch.randn(4, 5)
        categorical = torch.randint(0, 3, (4, 2))
        boolean = torch.randn(4, 10)

        out = model_regression(numeric, categorical, boolean)
        assert out.shape == (4, 1)

    def test_classification_forward(self, model_classification):
        numeric = torch.randn(4, 5)
        categorical = torch.randint(0, 3, (4, 2))
        boolean = torch.randn(4, 10)

        out = model_classification(numeric, categorical, boolean)
        assert out.shape == (4, 1)
        # Sigmoid output: values in [0, 1]
        assert out.min().item() >= 0
        assert out.max().item() <= 1

    def test_from_config(self):
        class Config:
            embedding_dim = 8
            hidden_dims = [32, 16]
            dropout = 0.1
            use_batch_norm = True
            task_type = "regression"

        model = SiteScoringModel.from_config(
            Config(),
            categorical_vocab_sizes={"network": 5},
            n_numeric=3,
            n_boolean=5,
        )
        assert isinstance(model, SiteScoringModel)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestModelFactory:
    def test_create_nn(self):
        class Config:
            model_type = "neural_network"
            task_type = "regression"
            embedding_dim = 8
            hidden_dims = [32, 16]
            dropout = 0.1
            use_batch_norm = True

        model = create_model(Config(), {"network": 5}, n_numeric=3, n_boolean=5)
        assert isinstance(model, SiteScoringModel)

    def test_create_unknown_raises(self):
        class Config:
            model_type = "unknown"
            task_type = "regression"

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(Config(), {}, 3, 5)
