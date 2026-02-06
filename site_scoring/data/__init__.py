"""
Data module for standardized data loading and processing.

This module provides:
- DataRegistry: Singleton for consistent data access across web/ML
- FeatureProcessor: Unified feature processing for train/inference
- FilterConfig: Configurable filtering rules

Usage:
    from site_scoring.data import DataRegistry, FeatureProcessor, FilterConfig

    # Get singleton registry
    registry = DataRegistry()

    # Load data with standard ML filters
    training_data = registry.get_training_data()

    # Process features for training
    processor = FeatureProcessor(config)
    tensors = processor.fit_transform(training_data)
"""

from .registry import DataRegistry, FilterConfig, DataPaths, get_registry
from .outputs.tensor import FeatureProcessor, TensorBundle, ProcessorState

__all__ = [
    "DataRegistry",
    "FilterConfig",
    "DataPaths",
    "get_registry",
    "FeatureProcessor",
    "TensorBundle",
    "ProcessorState",
]
