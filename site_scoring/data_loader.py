"""
Fast data loading optimized for Apple M4 with MPS backend.
Uses Polars for rapid CSV parsing and PyTorch for GPU acceleration.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from .config import Config


class SiteDataset(Dataset):
    """
    PyTorch Dataset for site scoring data.
    Optimized for M4 with contiguous memory layout.
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.numeric[idx],
            self.categorical[idx],
            self.boolean[idx],
            self.target[idx],
        )


class DataProcessor:
    """
    Fast data processor using Polars for CSV parsing.
    Handles encoding, scaling, and tensor conversion.
    """

    def __init__(self, config: Config):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.categorical_vocab_sizes: Dict[str, int] = {}
        self._fitted = False
        # Track actual feature counts after processing
        self.n_numeric_features = 0
        self.n_boolean_features = 0

    def load_and_process(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load data with Polars and convert to tensors.
        Supports both CSV and Parquet files.
        """
        print(f"Loading data from {self.config.data_path}...")

        # Detect file type and load appropriately
        data_path = str(self.config.data_path)
        if data_path.endswith('.parquet'):
            # Parquet is faster and more efficient
            df = pl.read_parquet(self.config.data_path)
        else:
            # CSV fallback with streaming
            df = pl.scan_csv(
                self.config.data_path,
                infer_schema_length=10000,
                null_values=["", "NA", "null", "Unknown"],
            ).collect(streaming=True)

        print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Filter to sites with sufficient history (more than 11 active months)
        # This ensures we train on sites with stable, representative performance data
        if 'active_months' in df.columns:
            original_count = len(df)
            df = df.filter(pl.col('active_months') > 11)
            filtered_count = len(df)
            print(f"Filtered to sites with >11 active months: {filtered_count:,} sites ({original_count - filtered_count:,} excluded)")

        # Process features
        numeric_data = self._process_numeric(df)
        categorical_data = self._process_categorical(df)
        boolean_data = self._process_boolean(df)
        target_data = self._process_target(df)

        self._fitted = True

        return numeric_data, categorical_data, boolean_data, target_data

    def _process_numeric(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract and scale numeric features."""
        # Get available numeric columns (only actual numeric types)
        available = []
        for c in self.config.numeric_features:
            if c in df.columns:
                dtype = df[c].dtype
                if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                    available.append(c)

        print(f"Processing {len(available)} numeric features...")
        self.n_numeric_features = len(available)

        # Process each column individually to handle different types
        processed_cols = []
        for col in available:
            col_data = df[col].cast(pl.Float64).fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
            processed_cols.append(col_data)

        numeric_array = np.column_stack(processed_cols) if processed_cols else np.zeros((len(df), 1), dtype=np.float32)
        if not processed_cols:
            self.n_numeric_features = 1

        # Clip extreme values (robust to outliers)
        for i in range(numeric_array.shape[1]):
            col_data = numeric_array[:, i]
            p1, p99 = np.percentile(col_data, [1, 99])
            numeric_array[:, i] = np.clip(col_data, p1, p99)

        # Fit and transform scaler
        numeric_scaled = self.scaler.fit_transform(numeric_array)

        # Final clip to prevent any extreme scaled values
        numeric_scaled = np.clip(numeric_scaled, -10, 10)

        # Convert to contiguous tensor for M4 optimization
        return torch.from_numpy(np.ascontiguousarray(numeric_scaled, dtype=np.float32))

    def _process_categorical(self, df: pl.DataFrame) -> torch.Tensor:
        """Encode categorical features as integers for embeddings."""
        available = [c for c in self.config.categorical_features if c in df.columns]
        print(f"Processing {len(available)} categorical features...")

        encoded_cols = []
        for col in available:
            # Get column and fill nulls
            col_data = df.select(col).fill_null("__MISSING__").to_series().to_list()

            # Fit label encoder
            le = LabelEncoder()
            encoded = le.fit_transform(col_data)
            self.label_encoders[col] = le
            self.categorical_vocab_sizes[col] = len(le.classes_)
            encoded_cols.append(encoded)

        # Stack into single array
        categorical_array = np.column_stack(encoded_cols).astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(categorical_array))

    def _process_boolean(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert boolean features to float tensor."""
        available = [c for c in self.config.boolean_features if c in df.columns]
        print(f"Processing {len(available)} boolean features...")
        self.n_boolean_features = len(available) if available else 1

        bool_cols = []
        for col in available:
            col_data = df.select(col).to_series()
            # Handle various boolean representations
            if col_data.dtype == pl.Boolean:
                values = col_data.fill_null(False).to_numpy().astype(np.float32)
            elif col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                # Pre-encoded integer booleans (0/1) from aggregated data
                values = col_data.fill_null(0).to_numpy().astype(np.float32)
            else:
                # String booleans
                values = (
                    col_data.fill_null("false")
                    .str.to_lowercase()
                    .is_in(["true", "1", "yes", "t"])
                    .to_numpy()
                    .astype(np.float32)
                )
            bool_cols.append(values)

        boolean_array = np.column_stack(bool_cols) if bool_cols else np.zeros((len(df), 1), dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(boolean_array, dtype=np.float32))

    def _process_target(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract and process target variable based on task type."""
        print(f"Processing target: {self.config.target}")

        # Get target and fill nulls with median
        target_col = df[self.config.target]
        median_val = target_col.median()
        if median_val is None:
            median_val = 0

        target_data = target_col.fill_null(median_val).fill_nan(median_val).to_numpy().astype(np.float32).reshape(-1, 1)

        # Clip extreme target values (1st and 99th percentile)
        p1, p99 = np.percentile(target_data, [1, 99])
        target_data = np.clip(target_data, p1, p99)

        if self.config.task_type == "lookalike":
            # Classification: binarize to top 10% performers
            threshold = float(np.percentile(target_data, 90))
            self.top_performer_threshold = threshold
            binary_labels = (target_data >= threshold).astype(np.float32)
            n_positive = int(binary_labels.sum())
            print(f"Classification: threshold=${threshold:,.0f}, {n_positive}/{len(binary_labels)} positive ({n_positive/len(binary_labels)*100:.1f}%)")
            self.target_scaler = None  # No inverse transform for binary labels
            return torch.from_numpy(np.ascontiguousarray(binary_labels, dtype=np.float32))
        else:
            # Regression: scale continuous target
            target_scaled = self.target_scaler.fit_transform(target_data)
            return torch.from_numpy(np.ascontiguousarray(target_scaled, dtype=np.float32))

    def save(self, path: Path):
        """Save fitted preprocessors."""
        with open(path, "wb") as f:
            pickle.dump({
                "label_encoders": self.label_encoders,
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
                "categorical_vocab_sizes": self.categorical_vocab_sizes,
            }, f)

    def load(self, path: Path):
        """Load fitted preprocessors."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.label_encoders = data["label_encoders"]
            self.scaler = data["scaler"]
            self.target_scaler = data["target_scaler"]
            self.categorical_vocab_sizes = data["categorical_vocab_sizes"]
            self._fitted = True


def create_data_loaders(
    config: Config,
    processor: Optional[DataProcessor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataProcessor]:
    """
    Create train/val/test DataLoaders optimized for M4 MPS.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, processor)
    """
    # Process data
    if processor is None:
        processor = DataProcessor(config)

    numeric, categorical, boolean, target = processor.load_and_process()

    # Get dataset size
    n_samples = len(target)
    print(f"Total samples: {n_samples:,}")

    # Create indices for splitting
    indices = torch.randperm(n_samples)

    # Calculate split sizes
    n_train = int(n_samples * config.train_ratio)
    n_val = int(n_samples * config.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")

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

    # Create DataLoaders with M4 optimization
    # Note: MPS works best with pin_memory=False
    pin_memory = config.pin_memory and config.device == "cpu"

    # prefetch_factor only valid when num_workers > 0
    use_multiprocessing = config.num_workers > 0
    prefetch = config.prefetch_factor if use_multiprocessing else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
        drop_last=True,  # For stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch,
        persistent_workers=use_multiprocessing,
    )

    return train_loader, val_loader, test_loader, processor
