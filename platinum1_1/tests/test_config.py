"""Tests for config/settings.py and config/features.py."""

import pytest

from platinum1_1.config.settings import Settings, get_settings, _detect_device
from platinum1_1.config.features import (
    FeatureType,
    FeatureDefinition,
    FeatureRegistry,
    ModelConfig,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_settings_creates_instance(self):
        settings = Settings()
        assert settings.PROJECT_ROOT.exists()
        assert settings.MAX_EXPERIMENTS == 10
        assert settings.DEFAULT_BATCH_SIZE == 4096

    def test_settings_resolves_data_paths(self):
        settings = Settings()
        assert settings.DATA_INPUT_DIR.name == "input"
        assert settings.DATA_PROCESSED_DIR.name == "processed"
        assert "platinum" in str(settings.DATA_PLATINUM_DIR)

    def test_settings_device_detection(self):
        device = _detect_device()
        assert device in ("cpu", "mps", "cuda")

    def test_get_settings_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_output_dir_created(self):
        settings = Settings()
        assert settings.OUTPUT_DIR is not None


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------

class TestFeatureRegistry:
    def test_registry_has_features(self):
        """Registry should be populated at import time."""
        all_features = FeatureRegistry.all_features()
        assert len(all_features) > 0

    def test_registry_types(self):
        numeric = FeatureRegistry.get_by_type(FeatureType.NUMERIC)
        categorical = FeatureRegistry.get_by_type(FeatureType.CATEGORICAL)
        boolean = FeatureRegistry.get_by_type(FeatureType.BOOLEAN)

        assert len(numeric) > 0
        assert len(categorical) > 0
        assert len(boolean) > 0

    def test_no_kroger_features(self):
        """Kroger features should not exist in platinum1_1."""
        all_features = FeatureRegistry.all_features()
        for name in all_features:
            assert "kroger" not in name.lower(), f"Kroger feature found: {name}"

    def test_geospatial_features_present(self):
        """Distance features for McDonald's, Walmart, Target should exist."""
        numeric = FeatureRegistry.get_by_type(FeatureType.NUMERIC)
        assert "log_min_distance_to_mcdonalds_mi" in numeric
        assert "log_min_distance_to_walmart_mi" in numeric
        assert "log_min_distance_to_target_mi" in numeric

    def test_register_custom_feature(self):
        """Test that new features can be registered dynamically."""
        custom = FeatureDefinition(
            name="test_custom_feature",
            type=FeatureType.NUMERIC,
            description="Test feature",
            group="test",
        )
        FeatureRegistry.register(custom)
        assert FeatureRegistry.get("test_custom_feature") == custom

        # Cleanup
        del FeatureRegistry._features["test_custom_feature"]

    def test_summary_counts(self):
        summary = FeatureRegistry.summary()
        assert "numeric" in summary
        assert "categorical" in summary
        assert "boolean" in summary
        assert all(v > 0 for v in summary.values())

    def test_get_by_group(self):
        momentum = FeatureRegistry.get_by_group("momentum")
        assert len(momentum) > 0
        assert all("rs_" in f for f in momentum)

    def test_feature_definition_frozen(self):
        """FeatureDefinition should be immutable."""
        fd = FeatureDefinition("test", FeatureType.NUMERIC)
        with pytest.raises(AttributeError):
            fd.name = "changed"


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.target == "avg_monthly_revenue"
        assert config.task_type == "regression"
        assert config.model_type == "neural_network"
        assert len(config.hidden_dims) == 4

    def test_leakage_prevention(self):
        """Revenue features should be excluded when predicting revenue."""
        config = ModelConfig(target="avg_monthly_revenue")
        numeric = config.get_numeric_features()
        assert "log_total_revenue" not in numeric
        assert "avg_monthly_revenue" not in numeric

    def test_get_all_features(self):
        config = ModelConfig()
        all_features = config.get_all_features()
        assert len(all_features) > 50  # ~60 features expected
