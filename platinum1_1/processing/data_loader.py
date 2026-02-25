"""
Fast data loading optimized for Apple Silicon with MPS backend.

Uses Polars for rapid data processing and PyTorch DataLoaders with
M4/MPS-specific optimizations (contiguous memory, pin_memory logic,
batch size capping).

The DataProcessor class from the original site_scoring/data_loader.py is
intentionally removed -- FeatureProcessor replaces it entirely.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .feature_processor import FeatureProcessor, ModelConfig

logger = logging.getLogger(__name__)


class SiteDataset(Dataset):
    """
    PyTorch Dataset for site scoring data.

    Stores pre-processed tensors (numeric, categorical, boolean, target)
    in contiguous memory layout for optimal M4/MPS throughput.
    """

    def __init__(
        self,
        numeric_tensor: torch.Tensor,
        categorical_tensor: torch.Tensor,
        boolean_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ):
        self.numeric = numeric_tensor
        self.categorical = categorical_tensor
        self.boolean = boolean_tensor
        self.target = target_tensor

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.numeric[idx],
            self.categorical[idx],
            self.boolean[idx],
            self.target[idx],
        )


def create_data_loaders(
    config: Any,
    df: pl.DataFrame,
    processor: Optional[FeatureProcessor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureProcessor]:
    """
    Create train/val/test DataLoaders optimized for M4 MPS.

    Accepts a Polars DataFrame directly (no file I/O) and processes it
    through FeatureProcessor to produce tensors.

    Args:
        config: Model configuration object satisfying the ModelConfig protocol.
                Must have at minimum: numeric_features, categorical_features,
                boolean_features, target, task_type, lookalike_lower_percentile,
                lookalike_upper_percentile. Also uses optional attributes:
                train_ratio (default 0.7), val_ratio (default 0.15),
                batch_size (default 4096), num_workers (default 4),
                pin_memory (default True), prefetch_factor (default 4),
                device (default "cpu").
        df: Polars DataFrame with all feature and target columns.
        processor: Optional pre-fitted FeatureProcessor. If None, a new one
                   is created and fit on the data.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, processor).
    """
    # Extract config values with sensible defaults
    train_ratio = getattr(config, "train_ratio", 0.7)
    val_ratio = getattr(config, "val_ratio", 0.15)
    batch_size = getattr(config, "batch_size", 4096)
    num_workers = getattr(config, "num_workers", 4)
    pin_memory = getattr(config, "pin_memory", True)
    prefetch_factor = getattr(config, "prefetch_factor", 4)
    device = getattr(config, "device", "cpu")

    # Process data through FeatureProcessor
    if processor is None:
        processor = FeatureProcessor(config)
        bundle = processor.fit_transform(df)
    else:
        bundle = processor.fit_transform(df) if not processor.is_fitted else processor.transform(df)
        # If transform was called, we need target -- re-fit if needed
        if bundle.target is None:
            # Processor was already fitted but we need targets for training
            # Re-create with fit_transform
            processor = FeatureProcessor(config)
            bundle = processor.fit_transform(df)

    numeric, categorical, boolean, target = bundle

    n_samples = len(target)
    logger.info("Total samples: %d", n_samples)

    # Create indices for splitting
    indices = torch.randperm(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    logger.info(
        "Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx)
    )

    # Create datasets
    train_dataset = SiteDataset(
        numeric[train_idx],
        categorical[train_idx],
        boolean[train_idx],
        target[train_idx],
    )
    val_dataset = SiteDataset(
        numeric[val_idx],
        categorical[val_idx],
        boolean[val_idx],
        target[val_idx],
    )
    test_dataset = SiteDataset(
        numeric[test_idx],
        categorical[test_idx],
        boolean[test_idx],
        target[test_idx],
    )

    # MPS works best with pin_memory=False
    effective_pin_memory = pin_memory and device == "cpu"

    # prefetch_factor only valid when num_workers > 0
    use_multiprocessing = num_workers > 0
    prefetch = prefetch_factor if use_multiprocessing else None

    # Cap batch_size to training set size to prevent zero-batch issue
    # with drop_last=True (happens when filtered subset < batch_size)
    effective_batch_size = min(batch_size, len(train_dataset))
    use_drop_last = len(train_dataset) > effective_batch_size
    if effective_batch_size < batch_size:
        logger.warning(
            "Batch size capped: %d -> %d (training set too small)",
            batch_size,
            effective_batch_size,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
        drop_last=use_drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
    )

    return train_loader, val_loader, test_loader, processor
