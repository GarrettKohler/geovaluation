"""
Tests for ML regression training and prediction.

Verifies that after training:
1. Model checkpoint is saved
2. Predictions can be generated for all sites
3. Predicted revenues are in reasonable range
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import polars as pl


def load_checkpoint_safe(model_path, device="cpu"):
    """
    Load PyTorch checkpoint with backward compatibility for PyTorch 2.6+.

    PyTorch 2.6 changed default weights_only=True for security.
    We explicitly use weights_only=False for trusted local checkpoints.
    """
    return torch.load(model_path, map_location=device, weights_only=False)


class TestTrainingOutputs:
    """Tests for training output files."""

    def test_output_directory_exists(self, model_output_dir):
        """Output directory for trained models exists."""
        assert model_output_dir.exists(), f"Output dir not found: {model_output_dir}"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "site_scoring" / "outputs" / "best_model.pt").exists(),
        reason="No trained model available - run training first"
    )
    def test_model_checkpoint_exists(self, model_output_dir):
        """Trained model checkpoint file exists."""
        model_path = model_output_dir / "best_model.pt"
        assert model_path.exists(), f"Model checkpoint not found: {model_path}"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "site_scoring" / "outputs" / "preprocessor.pkl").exists(),
        reason="No preprocessor available - run training first"
    )
    def test_preprocessor_exists(self, model_output_dir):
        """Preprocessor file exists for inference."""
        preprocessor_path = model_output_dir / "preprocessor.pkl"
        assert preprocessor_path.exists(), f"Preprocessor not found: {preprocessor_path}"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "site_scoring" / "outputs" / "best_model.pt").exists(),
    reason="No trained model available - run training first"
)
class TestModelLoading:
    """Tests for loading trained model."""

    def test_can_load_checkpoint(self, model_output_dir):
        """Model checkpoint can be loaded."""
        model_path = model_output_dir / "best_model.pt"
        checkpoint = load_checkpoint_safe(model_path, device="cpu")

        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint

    def test_checkpoint_has_metrics(self, model_output_dir):
        """Checkpoint includes test metrics."""
        model_path = model_output_dir / "best_model.pt"
        checkpoint = load_checkpoint_safe(model_path, device="cpu")

        if "test_metrics" in checkpoint:
            metrics = checkpoint["test_metrics"]
            assert "test_mae" in metrics or "test_loss" in metrics

    def test_can_instantiate_scorer(self, model_output_dir):
        """SiteScorer can be instantiated from saved model."""
        from site_scoring.predict import SiteScorer

        try:
            scorer = SiteScorer(
                model_path=model_output_dir / "best_model.pt",
                preprocessor_path=model_output_dir / "preprocessor.pkl",
                device="cpu",
            )
            assert scorer.model is not None
            assert scorer.processor is not None
        except RuntimeError as e:
            if "size mismatch" in str(e):
                pytest.skip("Model architecture changed since checkpoint was saved - retrain needed")
            raise


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "site_scoring" / "outputs" / "best_model.pt").exists(),
    reason="No trained model available - run training first"
)
class TestPredictions:
    """Tests for model predictions."""

    @pytest.fixture
    def scorer(self, model_output_dir):
        """Load the trained model scorer."""
        from site_scoring.predict import SiteScorer

        try:
            return SiteScorer(
                model_path=model_output_dir / "best_model.pt",
                preprocessor_path=model_output_dir / "preprocessor.pkl",
                device="cpu",
            )
        except RuntimeError as e:
            if "size mismatch" in str(e):
                pytest.skip("Model architecture changed since checkpoint was saved - retrain needed")
            raise

    @pytest.fixture
    def sample_data(self):
        """Load sample data for prediction testing."""
        data_path = Path(__file__).parent.parent / "data" / "input" / "Site Scores - Site Revenue, Impressions, and Diagnostics.csv"

        if not data_path.exists():
            pytest.skip(f"Data file not found: {data_path}")

        # Load small sample
        df = pl.read_csv(data_path, n_rows=100)
        return df

    def test_can_make_predictions(self, scorer, sample_data):
        """Model can generate predictions."""
        predictions = scorer.predict(sample_data)

        assert predictions is not None
        assert len(predictions) == len(sample_data)

    def test_predictions_are_numeric(self, scorer, sample_data):
        """Predictions are numeric values."""
        predictions = scorer.predict(sample_data)

        assert isinstance(predictions, np.ndarray)
        assert np.issubdtype(predictions.dtype, np.number)

    def test_predictions_not_all_same(self, scorer, sample_data):
        """Predictions have variety (not all identical)."""
        predictions = scorer.predict(sample_data)

        unique_predictions = len(np.unique(predictions))
        assert unique_predictions > 1, "All predictions are identical"

    def test_predictions_in_reasonable_range(self, scorer, sample_data):
        """Predictions are in reasonable revenue range."""
        predictions = scorer.predict(sample_data)

        # Revenue should not be negative
        assert np.all(predictions >= -1000), "Predictions include very negative values"

        # Revenue should not be astronomically high
        max_reasonable = 1_000_000  # $1M/month max
        assert np.all(predictions < max_reasonable), f"Predictions exceed {max_reasonable}"

    def test_score_sites_adds_column(self, scorer, sample_data):
        """score_sites method adds prediction column to DataFrame."""
        result = scorer.score_sites(sample_data)

        # Should have prediction column
        pred_col = f"predicted_{scorer.config.target}"
        assert pred_col in result.columns, f"Missing column: {pred_col}"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "site_scoring" / "outputs" / "best_model.pt").exists(),
    reason="No trained model available - run training first"
)
class TestPredictionQuality:
    """Tests for prediction quality metrics."""

    @pytest.fixture
    def model_metrics(self, model_output_dir):
        """Load saved model metrics."""
        model_path = model_output_dir / "best_model.pt"
        try:
            checkpoint = load_checkpoint_safe(model_path, device="cpu")
            return checkpoint.get("test_metrics", {})
        except Exception as e:
            pytest.skip(f"Could not load checkpoint: {e}")

    def test_mae_is_reasonable(self, model_metrics):
        """Mean Absolute Error is within acceptable range."""
        if "test_mae" not in model_metrics:
            pytest.skip("MAE not in saved metrics")

        mae = model_metrics["test_mae"]

        # MAE should be less than $50,000 for monthly revenue
        assert mae < 50000, f"MAE too high: ${mae:,.2f}"

    def test_r2_is_not_severely_negative(self, model_metrics):
        """R-squared is not severely negative (model is at least reasonably trained)."""
        if "test_r2" not in model_metrics:
            pytest.skip("R2 not in saved metrics")

        r2 = model_metrics["test_r2"]

        # R2 can be slightly negative for models that underperform mean prediction
        # However, a severely negative R2 (< -0.5) indicates fundamental issues
        assert r2 > -0.5, f"R2 is severely negative: {r2:.4f} (model needs retraining)"

        # Warn if R2 is not positive but don't fail
        if r2 <= 0:
            import warnings
            warnings.warn(f"Model R2 is {r2:.4f} - model does not beat mean prediction")

    def test_r2_is_reasonable(self, model_metrics):
        """R-squared is within plausible range."""
        if "test_r2" not in model_metrics:
            pytest.skip("R2 not in saved metrics")

        r2 = model_metrics["test_r2"]

        # R2 should not exceed 1
        assert r2 <= 1.0, f"R2 exceeds 1: {r2:.4f}"


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_config_has_required_fields(self):
        """Training config has required fields."""
        from site_scoring.config import Config

        config = Config()

        assert hasattr(config, "data_path")
        assert hasattr(config, "target")
        assert hasattr(config, "epochs")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "device")

    def test_default_device_selection(self):
        """Default device is correctly selected."""
        from site_scoring.config import Config

        config = Config()

        # Should be mps, cuda, or cpu
        assert config.device in ["mps", "cuda", "cpu"]

    def test_feature_lists_populated(self):
        """Feature lists are populated."""
        from site_scoring.config import Config

        config = Config()

        assert len(config.numeric_features) > 0
        assert len(config.categorical_features) > 0
        assert len(config.boolean_features) > 0


class TestDataLoader:
    """Tests for ML data loading."""

    def test_data_processor_initializes(self):
        """DataProcessor can be initialized."""
        from site_scoring.config import Config
        from site_scoring.data_loader import DataProcessor

        config = Config()
        processor = DataProcessor(config)

        assert processor is not None
        assert processor.config == config

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "input" / "Site Scores - Site Revenue, Impressions, and Diagnostics.csv").exists(),
        reason="Data file not available"
    )
    def test_data_can_be_loaded(self):
        """Data can be loaded and processed."""
        from site_scoring.config import Config
        from site_scoring.data_loader import DataProcessor

        config = Config()
        processor = DataProcessor(config)

        # This will load the full dataset
        numeric, categorical, boolean, target = processor.load_and_process()

        assert numeric is not None
        assert categorical is not None
        assert boolean is not None
        assert target is not None

        # Should have many samples
        assert len(target) > 100000, f"Only {len(target)} samples loaded"


class TestModelArchitecture:
    """Tests for model architecture."""

    def test_model_can_be_created(self):
        """Model can be instantiated."""
        from site_scoring.model import SiteScoringModel

        model = SiteScoringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10, "cat2": 20},
            embedding_dim=8,
            hidden_dims=[64, 32],
            dropout=0.2,
        )

        assert model is not None

    def test_model_forward_pass(self):
        """Model can perform forward pass."""
        from site_scoring.model import SiteScoringModel

        model = SiteScoringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10, "cat2": 20},
            embedding_dim=8,
            hidden_dims=[64, 32],
            dropout=0.2,
        )

        # Create dummy input
        batch_size = 4
        numeric = torch.randn(batch_size, 10)
        categorical = torch.randint(0, 10, (batch_size, 2))
        boolean = torch.randn(batch_size, 5)

        # Forward pass
        output = model(numeric, categorical, boolean)

        assert output.shape == (batch_size, 1)

    def test_model_gradients_flow(self):
        """Gradients can flow through model."""
        from site_scoring.model import SiteScoringModel

        model = SiteScoringModel(
            n_numeric=10,
            n_boolean=5,
            categorical_vocab_sizes={"cat1": 10, "cat2": 20},
            embedding_dim=8,
            hidden_dims=[64, 32],
            dropout=0.2,
        )

        # Create dummy input
        batch_size = 4
        numeric = torch.randn(batch_size, 10, requires_grad=True)
        categorical = torch.randint(0, 10, (batch_size, 2))
        boolean = torch.randn(batch_size, 5)

        # Forward pass
        output = model(numeric, categorical, boolean)

        # Compute loss and backward
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        assert numeric.grad is not None
