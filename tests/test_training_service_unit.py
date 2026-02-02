"""
Unit tests for src/services/training_service.py.

Tests the training service functions including Apple Silicon detection,
parameter optimization, training configuration, and JSON sanitization.
These tests do NOT require GPU hardware or trained models.
"""

import pytest
import math
import json
from unittest.mock import patch, MagicMock
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Tests for _sanitize_for_json()
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    """Tests for _sanitize_for_json() helper."""

    def setup_method(self):
        from src.services.training_service import _sanitize_for_json
        self.sanitize = _sanitize_for_json

    def test_regular_float_unchanged(self):
        """Normal floats pass through."""
        assert self.sanitize(3.14) == 3.14

    def test_inf_becomes_none(self):
        """Infinity becomes None."""
        assert self.sanitize(float('inf')) is None

    def test_negative_inf_becomes_none(self):
        """-Infinity becomes None."""
        assert self.sanitize(float('-inf')) is None

    def test_nan_becomes_none(self):
        """NaN becomes None."""
        assert self.sanitize(float('nan')) is None

    def test_nested_dict_sanitized(self):
        """Nested dict values are sanitized."""
        data = {"a": {"b": float('inf')}}
        result = self.sanitize(data)
        assert result == {"a": {"b": None}}

    def test_list_sanitized(self):
        """List values are sanitized."""
        data = [1.0, float('nan'), 3.0]
        result = self.sanitize(data)
        assert result == [1.0, None, 3.0]

    def test_mixed_types_in_dict(self):
        """Mixed types in a progress-like dict are handled."""
        data = {
            "epoch": 5,
            "train_loss": 0.5,
            "val_loss": float('inf'),
            "val_smape": float('nan'),
            "status": "running",
            "best_val_loss": float('inf'),
        }
        result = self.sanitize(data)
        assert result["epoch"] == 5
        assert result["train_loss"] == 0.5
        assert result["val_loss"] is None
        assert result["val_smape"] is None
        assert result["status"] == "running"
        assert result["best_val_loss"] is None

    def test_result_is_json_serializable(self):
        """Sanitized output can be serialized to JSON."""
        data = {
            "loss": float('inf'),
            "metrics": [float('nan'), 1.0, float('-inf')],
            "nested": {"val": float('nan')},
        }
        result = self.sanitize(data)
        # Should not raise
        json_str = json.dumps(result)
        assert "Infinity" not in json_str
        assert "NaN" not in json_str

    def test_non_float_types_unchanged(self):
        """Strings, ints, booleans, None pass through."""
        assert self.sanitize("text") == "text"
        assert self.sanitize(42) == 42
        assert self.sanitize(True) is True
        assert self.sanitize(None) is None

    def test_zero_not_sanitized(self):
        """Zero is a valid float and not sanitized."""
        assert self.sanitize(0.0) == 0.0

    def test_empty_structures(self):
        """Empty dict and list pass through."""
        assert self.sanitize({}) == {}
        assert self.sanitize([]) == []


# ---------------------------------------------------------------------------
# Tests for get_optimized_training_params()
# ---------------------------------------------------------------------------


class TestGetOptimizedTrainingParams:
    """Tests for get_optimized_training_params() chip optimization."""

    def setup_method(self):
        from src.services.training_service import get_optimized_training_params
        self.get_params = get_optimized_training_params

    def test_m1_basic_params(self):
        """M1 base chip returns tier 1 parameters."""
        params = self.get_params("m1", 4096)
        assert params["batch_size"] == 4096
        assert params["chip_tier"] == 1
        assert params["gpu_cores"] == 8

    def test_m4_max_params(self):
        """M4 Max returns tier 4 parameters."""
        params = self.get_params("m4_max", 32768)
        assert params["batch_size"] == 32768
        assert params["chip_tier"] == 4
        assert params["gpu_cores"] == 40

    def test_batch_size_capped_by_chip(self):
        """Batch size is capped by chip's max capability."""
        # M1 max batch is 4096, requesting 16384 should cap it
        params = self.get_params("m1", 16384)
        assert params["batch_size"] == 4096

    def test_batch_size_not_increased(self):
        """Batch size below chip max is not increased."""
        params = self.get_params("m4_max", 1024)
        assert params["batch_size"] == 1024

    def test_unknown_chip_uses_defaults(self):
        """Unknown chip ID falls back to default specs."""
        params = self.get_params("unknown_chip", 4096)
        assert params["batch_size"] == 4096
        assert params["gpu_cores"] == 8  # Default fallback
        assert params["chip_tier"] == 1

    def test_workers_scale_with_tier(self):
        """Number of workers scales with chip tier."""
        m1_params = self.get_params("m1", 4096)        # tier 1
        m4_max_params = self.get_params("m4_max", 4096)  # tier 4

        assert m4_max_params["num_workers"] > m1_params["num_workers"]

    def test_workers_capped_at_eight(self):
        """Workers never exceed 8."""
        params = self.get_params("m4_max", 4096)  # tier 4
        assert params["num_workers"] <= 8

    def test_pin_memory_always_true(self):
        """Pin memory is True for all Apple Silicon chips."""
        for chip in ["m1", "m2_pro", "m3_max", "m4"]:
            params = self.get_params(chip, 4096)
            assert params["pin_memory"] is True

    def test_prefetch_scales_with_tier(self):
        """Prefetch factor increases with chip tier."""
        m1_params = self.get_params("m1", 4096)
        m4_max_params = self.get_params("m4_max", 4096)

        assert m4_max_params["prefetch_factor"] > m1_params["prefetch_factor"]

    def test_all_known_chips_valid(self):
        """All chips in APPLE_CHIP_SPECS return valid parameters."""
        from src.services.training_service import APPLE_CHIP_SPECS

        for chip_id in APPLE_CHIP_SPECS:
            params = self.get_params(chip_id, 4096)
            assert params["batch_size"] > 0
            assert params["num_workers"] > 0
            assert params["gpu_cores"] > 0
            assert params["chip_tier"] >= 1


# ---------------------------------------------------------------------------
# Tests for detect_apple_chip()
# ---------------------------------------------------------------------------


class TestDetectAppleChip:
    """Tests for detect_apple_chip() system detection."""

    def setup_method(self):
        from src.services.training_service import detect_apple_chip
        self.detect = detect_apple_chip

    @patch('subprocess.run')
    def test_detects_m4_pro(self, mock_run):
        """Correctly identifies M4 Pro chip."""
        # Mock sysctl call
        mock_cpu = MagicMock()
        mock_cpu.stdout = "Apple M4 Pro"
        mock_mem = MagicMock()
        mock_mem.stdout = "34359738368"  # 32 GB
        mock_gpu = MagicMock()
        mock_gpu.stdout = "{}"

        mock_run.side_effect = [mock_cpu, mock_mem, mock_gpu]

        result = self.detect()
        assert result["detected_chip"] == "m4_pro"
        assert "M4 Pro" in result["chip_name"]

    @patch('subprocess.run')
    def test_detects_m1_base(self, mock_run):
        """Correctly identifies base M1 chip."""
        mock_cpu = MagicMock()
        mock_cpu.stdout = "Apple M1"
        mock_mem = MagicMock()
        mock_mem.stdout = "17179869184"  # 16 GB
        mock_gpu = MagicMock()
        mock_gpu.stdout = "{}"

        mock_run.side_effect = [mock_cpu, mock_mem, mock_gpu]

        result = self.detect()
        assert result["detected_chip"] == "m1"

    @patch('subprocess.run')
    def test_detects_m2_ultra(self, mock_run):
        """Correctly identifies M2 Ultra chip."""
        mock_cpu = MagicMock()
        mock_cpu.stdout = "Apple M2 Ultra"
        mock_mem = MagicMock()
        mock_mem.stdout = "206158430208"  # 192 GB
        mock_gpu = MagicMock()
        mock_gpu.stdout = "{}"

        mock_run.side_effect = [mock_cpu, mock_mem, mock_gpu]

        result = self.detect()
        assert result["detected_chip"] == "m2_ultra"

    @patch('subprocess.run')
    def test_handles_non_apple_cpu(self, mock_run):
        """Returns unknown for non-Apple CPUs (Intel, etc.)."""
        mock_cpu = MagicMock()
        mock_cpu.stdout = "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"
        mock_mem = MagicMock()
        mock_mem.stdout = "34359738368"
        mock_gpu = MagicMock()
        mock_gpu.stdout = "{}"

        mock_run.side_effect = [mock_cpu, mock_mem, mock_gpu]

        result = self.detect()
        assert result["detected_chip"] == "unknown"

    @patch('subprocess.run')
    def test_handles_subprocess_failure(self, mock_run):
        """Returns defaults when subprocess calls fail."""
        mock_run.side_effect = Exception("sysctl not found")

        result = self.detect()
        assert result["detected_chip"] == "unknown"
        assert result["chip_name"] == "Unknown"

    @patch('subprocess.run')
    def test_parses_memory_correctly(self, mock_run):
        """Correctly converts memory bytes to GB string."""
        mock_cpu = MagicMock()
        mock_cpu.stdout = "Apple M3"
        mock_mem = MagicMock()
        mock_mem.stdout = "25769803776"  # 24 GB
        mock_gpu = MagicMock()
        mock_gpu.stdout = "{}"

        mock_run.side_effect = [mock_cpu, mock_mem, mock_gpu]

        result = self.detect()
        assert result["total_memory"] == "24 GB"


# ---------------------------------------------------------------------------
# Tests for TrainingConfig dataclass
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Tests for TrainingConfig defaults and behavior."""

    def test_default_device_selection(self):
        """Default device is mps if available, otherwise cpu."""
        from src.services.training_service import TrainingConfig
        import torch

        config = TrainingConfig()
        if torch.backends.mps.is_available():
            assert config.device == "mps"
        else:
            assert config.device == "cpu"

    def test_default_hidden_layers(self):
        """Default hidden layers are [512, 256, 128, 64]."""
        from src.services.training_service import TrainingConfig
        config = TrainingConfig()
        assert config.hidden_layers == [512, 256, 128, 64]

    def test_default_epochs(self):
        """Default epochs is 50."""
        from src.services.training_service import TrainingConfig
        config = TrainingConfig()
        assert config.epochs == 50

    def test_custom_config_overrides(self):
        """Custom values override defaults."""
        from src.services.training_service import TrainingConfig
        config = TrainingConfig(
            epochs=10,
            batch_size=512,
            learning_rate=0.01,
            device="cpu",
        )
        assert config.epochs == 10
        assert config.batch_size == 512
        assert config.learning_rate == 0.01
        assert config.device == "cpu"

    def test_default_early_stopping_patience(self):
        """Default early stopping patience is 10."""
        from src.services.training_service import TrainingConfig
        config = TrainingConfig()
        assert config.early_stopping_patience == 10

    def test_default_apple_chip_is_auto(self):
        """Default apple_chip is 'auto' for auto-detection."""
        from src.services.training_service import TrainingConfig
        config = TrainingConfig()
        assert config.apple_chip == "auto"


# ---------------------------------------------------------------------------
# Tests for TrainingProgress dataclass
# ---------------------------------------------------------------------------


class TestTrainingProgress:
    """Tests for TrainingProgress data container."""

    def test_creates_with_required_fields(self):
        """Can create TrainingProgress with all required fields."""
        from src.services.training_service import TrainingProgress

        progress = TrainingProgress(
            epoch=5,
            total_epochs=50,
            train_loss=0.5,
            val_loss=0.4,
            val_mae=500.0,
            val_smape=100.0,
            val_rmse=600.0,
            val_r2=0.85,
            learning_rate=0.001,
            elapsed_time=30.5,
            status="running",
        )
        assert progress.epoch == 5
        assert progress.status == "running"
        assert progress.best_val_loss == float('inf')  # Default

    def test_default_best_val_loss_is_inf(self):
        """Default best_val_loss is infinity."""
        from src.services.training_service import TrainingProgress

        progress = TrainingProgress(
            epoch=1, total_epochs=50, train_loss=0, val_loss=0,
            val_mae=0, val_smape=0, val_rmse=0, val_r2=0, learning_rate=0.001,
            elapsed_time=0, status="running"
        )
        assert progress.best_val_loss == float('inf')

    def test_message_defaults_to_empty(self):
        """Default message is empty string."""
        from src.services.training_service import TrainingProgress

        progress = TrainingProgress(
            epoch=1, total_epochs=50, train_loss=0, val_loss=0,
            val_mae=0, val_smape=0, val_rmse=0, val_r2=0, learning_rate=0.001,
            elapsed_time=0, status="running"
        )
        assert progress.message == ""


# ---------------------------------------------------------------------------
# Tests for start_training() / stop_training() / get_training_status()
# ---------------------------------------------------------------------------


class TestTrainingLifecycle:
    """Tests for training job lifecycle management."""

    def setup_method(self):
        """Reset global training job before each test."""
        import src.services.training_service as ts
        ts._current_job = None

    def test_stop_when_no_job_returns_false(self):
        """Stopping when no job exists returns failure."""
        from src.services.training_service import stop_training

        success, message = stop_training()
        assert success is False
        assert "No training job" in message

    def test_status_when_no_job_returns_none(self):
        """Status when no job exists returns None."""
        from src.services.training_service import get_training_status

        result = get_training_status()
        assert result is None

    def test_get_system_info_returns_dict(self):
        """get_system_info returns a dict with required keys."""
        from src.services.training_service import get_system_info

        info = get_system_info()
        assert isinstance(info, dict)
        assert "pytorch_version" in info
        assert "cuda_available" in info
        assert "mps_available" in info
        assert "recommended_device" in info

    def test_recommended_device_is_valid(self):
        """Recommended device is one of the known options."""
        from src.services.training_service import get_system_info

        info = get_system_info()
        assert info["recommended_device"] in ["mps", "cuda", "cpu"]


# ---------------------------------------------------------------------------
# Tests for APPLE_CHIP_SPECS constant
# ---------------------------------------------------------------------------


class TestAppleChipSpecs:
    """Tests for the APPLE_CHIP_SPECS lookup table."""

    def test_all_chips_have_required_keys(self):
        """Every chip spec has gpu_cores, max_batch, tier, memory_bandwidth."""
        from src.services.training_service import APPLE_CHIP_SPECS

        required_keys = {"gpu_cores", "max_batch", "tier", "memory_bandwidth"}
        for chip_id, specs in APPLE_CHIP_SPECS.items():
            for key in required_keys:
                assert key in specs, f"Chip {chip_id} missing key: {key}"

    def test_tiers_are_monotonic_within_generation(self):
        """Within a generation, Pro < Max < Ultra in tier."""
        from src.services.training_service import APPLE_CHIP_SPECS

        for gen in ["m1", "m2"]:
            base = APPLE_CHIP_SPECS.get(gen, {}).get("tier", 0)
            pro = APPLE_CHIP_SPECS.get(f"{gen}_pro", {}).get("tier", 0)
            max_ = APPLE_CHIP_SPECS.get(f"{gen}_max", {}).get("tier", 0)
            ultra = APPLE_CHIP_SPECS.get(f"{gen}_ultra", {}).get("tier", 0)

            if base and pro:
                assert pro >= base, f"{gen}_pro tier should be >= {gen} tier"
            if pro and max_:
                assert max_ >= pro, f"{gen}_max tier should be >= {gen}_pro tier"
            if max_ and ultra:
                assert ultra >= max_, f"{gen}_ultra tier should be >= {gen}_max tier"

    def test_gpu_cores_increase_with_tier(self):
        """GPU cores generally increase with tier within a generation."""
        from src.services.training_service import APPLE_CHIP_SPECS

        for gen in ["m1", "m2"]:
            base_cores = APPLE_CHIP_SPECS.get(gen, {}).get("gpu_cores", 0)
            pro_cores = APPLE_CHIP_SPECS.get(f"{gen}_pro", {}).get("gpu_cores", 0)
            max_cores = APPLE_CHIP_SPECS.get(f"{gen}_max", {}).get("gpu_cores", 0)

            if base_cores and pro_cores:
                assert pro_cores > base_cores
            if pro_cores and max_cores:
                assert max_cores > pro_cores

    def test_max_batch_increases_with_tier(self):
        """Max batch size increases with chip tier."""
        from src.services.training_service import APPLE_CHIP_SPECS

        for gen in ["m1", "m2"]:
            base_batch = APPLE_CHIP_SPECS.get(gen, {}).get("max_batch", 0)
            pro_batch = APPLE_CHIP_SPECS.get(f"{gen}_pro", {}).get("max_batch", 0)

            if base_batch and pro_batch:
                assert pro_batch > base_batch
