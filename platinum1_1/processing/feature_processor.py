"""
FeatureProcessor: Single implementation for train AND inference.

This module provides a unified feature processing pipeline that ensures
consistency between training and inference. The same code processes
features in both contexts, eliminating train/inference skew.

Key improvements over the original site_scoring implementation:
- Clip thresholds stored during fit and reapplied during transform
  (fixes train/serve skew for extreme values)
- Uses Protocol for config duck-typing (no hard dependency on Config class)
- Serializable state includes clip thresholds for full reproducibility

Usage (Training):
    processor = FeatureProcessor(config)
    bundle = processor.fit_transform(train_df)
    processor.save("model_processor.pkl")

Usage (Inference):
    processor = FeatureProcessor.load("model_processor.pkl")
    bundle = processor.transform(new_df)
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config protocol -- duck-typed contract for the config parameter
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelConfig(Protocol):
    """
    Protocol defining the minimum interface a config object must satisfy.

    Any object with these attributes can be passed to FeatureProcessor.
    This avoids a hard import dependency on the Settings/Config class.
    """

    numeric_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]
    target: str
    task_type: str
    lookalike_lower_percentile: int
    lookalike_upper_percentile: int


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class TensorBundle(NamedTuple):
    """Container for processed tensors ready for model consumption."""

    numeric: torch.Tensor
    categorical: torch.Tensor
    boolean: torch.Tensor
    target: Optional[torch.Tensor] = None


@dataclass
class ProcessorState:
    """Serializable state for a fitted processor."""

    numeric_scaler: StandardScaler
    target_scaler: Optional[StandardScaler]
    label_encoders: Dict[str, LabelEncoder]
    categorical_vocab_sizes: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    # Lookalike thresholds (if applicable)
    top_performer_threshold: Optional[float] = None
    top_performer_upper_threshold: Optional[float] = None
    # Clip thresholds per numeric column index: {col_index: (p1, p99)}
    # Stored during fit and reapplied during transform to prevent train/serve skew.
    clip_thresholds: Optional[Dict[int, Tuple[float, float]]] = None


# ---------------------------------------------------------------------------
# FeatureProcessor
# ---------------------------------------------------------------------------

class FeatureProcessor:
    """
    Unified feature processor for training and inference.

    Handles:
    - Numeric features: null filling, outlier clipping (p1-p99), StandardScaler
    - Categorical features: LabelEncoder with unknown handling
    - Boolean features: type coercion (bool/int/string -> float)
    - Target: regression scaling or classification binarization

    The clip bug fix:
        The original implementation clipped outliers at p1-p99 during
        fit_transform but NOT during transform. This caused train/serve skew
        for extreme values. This version stores clip thresholds during fit
        and reapplies them during transform.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize processor.

        Args:
            config: Configuration object with feature lists and task_type.
                    Must satisfy the ModelConfig protocol (duck typing).
                    If None, must call load() before transform().
        """
        self.config = config
        self._fitted = False

        # Fitted state
        self._numeric_scaler = StandardScaler()
        self._target_scaler = StandardScaler()
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._categorical_vocab_sizes: Dict[str, int] = {}
        self._clip_thresholds: Dict[int, Tuple[float, float]] = {}

        # Column tracking
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._boolean_columns: List[str] = []

        # Lookalike thresholds
        self.top_performer_threshold: Optional[float] = None
        self.top_performer_upper_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Check if processor has been fitted."""
        return self._fitted

    @property
    def categorical_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for categorical features."""
        return self._categorical_vocab_sizes

    @property
    def n_numeric(self) -> int:
        """Number of numeric features after fitting."""
        return len(self._numeric_columns) if self._numeric_columns else 0

    @property
    def n_boolean(self) -> int:
        """Number of boolean features after fitting."""
        return len(self._boolean_columns) if self._boolean_columns else 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pl.DataFrame) -> TensorBundle:
        """
        Fit processor on training data and transform.

        Args:
            df: Training DataFrame (Polars).

        Returns:
            TensorBundle with processed tensors.
        """
        if self.config is None:
            raise ValueError(
                "Config required for fit_transform. Use load() for inference."
            )

        self._fitted = True

        numeric = self._process_numeric(df, fit=True)
        categorical = self._process_categorical(df, fit=True)
        boolean = self._process_boolean(df)
        target = self._process_target(df, fit=True)

        return TensorBundle(numeric, categorical, boolean, target)

    def transform(self, df: pl.DataFrame) -> TensorBundle:
        """
        Transform data using fitted processor.

        Args:
            df: DataFrame to transform (Polars).

        Returns:
            TensorBundle with processed tensors (target is None).
        """
        if not self._fitted:
            raise ValueError(
                "Processor not fitted. Call fit_transform() or load() first."
            )

        numeric = self._process_numeric(df, fit=False)
        categorical = self._process_categorical(df, fit=False)
        boolean = self._process_boolean(df)

        return TensorBundle(numeric, categorical, boolean, None)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save fitted processor state to disk."""
        if not self._fitted:
            raise ValueError("Cannot save unfitted processor")

        state = ProcessorState(
            numeric_scaler=self._numeric_scaler,
            target_scaler=self._target_scaler if hasattr(self, "_target_scaler") else None,
            label_encoders=self._label_encoders,
            categorical_vocab_sizes=self._categorical_vocab_sizes,
            numeric_columns=self._numeric_columns,
            categorical_columns=self._categorical_columns,
            boolean_columns=self._boolean_columns,
            top_performer_threshold=self.top_performer_threshold,
            top_performer_upper_threshold=self.top_performer_upper_threshold,
            clip_thresholds=self._clip_thresholds,
        )

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Saved processor state to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureProcessor":
        """Load fitted processor from file."""
        with open(path, "rb") as f:
            state: ProcessorState = pickle.load(f)

        processor = cls(config=None)
        processor._fitted = True
        processor._numeric_scaler = state.numeric_scaler
        processor._target_scaler = state.target_scaler
        processor._label_encoders = state.label_encoders
        processor._categorical_vocab_sizes = state.categorical_vocab_sizes
        processor._numeric_columns = state.numeric_columns
        processor._categorical_columns = state.categorical_columns
        processor._boolean_columns = state.boolean_columns
        processor.top_performer_threshold = state.top_performer_threshold
        processor.top_performer_upper_threshold = state.top_performer_upper_threshold

        # Restore clip thresholds (handles old pickles without the field)
        processor._clip_thresholds = getattr(state, "clip_thresholds", None) or {}

        logger.info("Loaded processor state from %s", path)
        return processor

    # ------------------------------------------------------------------
    # Target inverse transform
    # ------------------------------------------------------------------

    def inverse_transform_target(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale.

        Only applicable for regression tasks. For classification (lookalike),
        predictions are returned unchanged.

        Args:
            predictions: Model output array.

        Returns:
            Predictions in original target scale.
        """
        if self.config and self.config.task_type == "lookalike":
            return predictions

        # sklearn scalers need 2D arrays
        return self._target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()

    # ------------------------------------------------------------------
    # Feature Processing (Internal)
    # ------------------------------------------------------------------

    def _process_numeric(self, df: pl.DataFrame, fit: bool) -> torch.Tensor:
        """
        Process numeric features.

        Steps:
        1. Select configured numeric columns present in data
        2. Fill null/nan with 0
        3. If fitting: compute and store clip thresholds (p1-p99), fit scaler
        4. If transforming: apply stored clip thresholds
        5. Apply StandardScaler
        6. Clip scaled values to [-10, 10] for stability
        """
        if fit:
            self._numeric_columns = [
                col for col in self.config.numeric_features if col in df.columns
            ]

        if not self._numeric_columns:
            return torch.zeros((len(df), 1), dtype=torch.float32)

        # Extract and convert to numpy
        numeric_df = df.select(self._numeric_columns)
        data = numeric_df.to_numpy().astype(np.float64)

        # Fill null/nan with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            # Compute and store clip thresholds per column
            self._clip_thresholds = {}
            for i in range(data.shape[1]):
                col_data = data[:, i]
                p1, p99 = np.percentile(col_data, [1, 99])
                self._clip_thresholds[i] = (float(p1), float(p99))
                data[:, i] = np.clip(col_data, p1, p99)

            self._numeric_scaler.fit(data)
        else:
            # APPLY SAME CLIP THRESHOLDS during inference (the bug fix)
            if self._clip_thresholds:
                for i, (p1, p99) in self._clip_thresholds.items():
                    if i < data.shape[1]:
                        data[:, i] = np.clip(data[:, i], p1, p99)

        # Transform
        scaled = self._numeric_scaler.transform(data)

        # Clip extreme scaled values for numerical stability
        scaled = np.clip(scaled, -10, 10)

        return torch.from_numpy(
            np.ascontiguousarray(scaled, dtype=np.float32)
        )

    def _process_categorical(self, df: pl.DataFrame, fit: bool) -> torch.Tensor:
        """
        Process categorical features.

        Steps:
        1. Select configured categorical columns present in data
        2. Fill null with "__MISSING__"
        3. If fitting: fit LabelEncoders (includes __UNKNOWN__ sentinel)
        4. Transform to integer codes
        5. Map unseen categories to __UNKNOWN__ code
        """
        if fit:
            self._categorical_columns = [
                col for col in self.config.categorical_features if col in df.columns
            ]

        if not self._categorical_columns:
            return torch.zeros((len(df), 1), dtype=torch.int64)

        result_arrays = []

        for col in self._categorical_columns:
            col_data = df[col].fill_null("__MISSING__").cast(pl.Utf8).to_list()

            if fit:
                encoder = LabelEncoder()
                all_values = list(set(col_data)) + ["__UNKNOWN__"]
                encoder.fit(all_values)
                self._label_encoders[col] = encoder
                self._categorical_vocab_sizes[col] = len(encoder.classes_)

            encoder = self._label_encoders[col]
            encoded = []
            for val in col_data:
                if val in encoder.classes_:
                    encoded.append(encoder.transform([val])[0])
                else:
                    if "__UNKNOWN__" in encoder.classes_:
                        encoded.append(encoder.transform(["__UNKNOWN__"])[0])
                    else:
                        encoded.append(0)

            result_arrays.append(np.array(encoded, dtype=np.int64))

        stacked = np.column_stack(result_arrays)
        return torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.int64))

    def _process_boolean(self, df: pl.DataFrame) -> torch.Tensor:
        """
        Process boolean features.

        Handles multiple input types:
        - pl.Boolean: direct conversion
        - Integer (0/1): cast to float
        - String ("true"/"false", "yes"/"no"): parse and convert
        """
        if self._fitted and not self.config:
            bool_columns = self._boolean_columns
        else:
            self._boolean_columns = [
                col for col in self.config.boolean_features if col in df.columns
            ]
            bool_columns = self._boolean_columns

        if not bool_columns:
            return torch.zeros((len(df), 1), dtype=torch.float32)

        result_arrays = []

        for col in bool_columns:
            col_series = df[col]
            dtype = col_series.dtype

            if dtype == pl.Boolean:
                arr = col_series.fill_null(False).cast(pl.Float32).to_numpy()
            elif dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            ]:
                arr = col_series.fill_null(0).cast(pl.Float32).to_numpy()
            elif dtype == pl.Utf8:
                arr = (
                    col_series.str.to_lowercase()
                    .map_elements(
                        lambda x: 1.0 if x in ("true", "yes", "1") else 0.0,
                        return_dtype=pl.Float32,
                    )
                    .fill_null(0.0)
                    .to_numpy()
                )
            else:
                arr = col_series.fill_null(0).cast(pl.Float32).to_numpy()

            result_arrays.append(arr.astype(np.float32))

        stacked = np.column_stack(result_arrays)
        return torch.from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))

    def _process_target(self, df: pl.DataFrame, fit: bool) -> torch.Tensor:
        """
        Process target variable.

        For regression: StandardScaler normalization with p1-p99 clipping.
        For lookalike: binarize by configurable percentile thresholds.
        """
        target_col = self.config.target

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        target_series = df[target_col]
        median_val = target_series.median()
        if median_val is None:
            median_val = 0

        target_data = (
            target_series.fill_null(median_val)
            .fill_nan(median_val)
            .to_numpy()
            .astype(np.float32)
            .reshape(-1, 1)
        )

        # Clip extreme target values (p1-p99)
        p1, p99 = np.percentile(target_data, [1, 99])
        target_data = np.clip(target_data, p1, p99)

        if self.config.task_type == "lookalike":
            lower_pct = getattr(self.config, "lookalike_lower_percentile", 90)
            upper_pct = getattr(self.config, "lookalike_upper_percentile", 100)

            lower_threshold = float(np.percentile(target_data, lower_pct))
            upper_threshold = (
                float(np.percentile(target_data, upper_pct))
                if upper_pct < 100
                else float("inf")
            )

            self.top_performer_threshold = lower_threshold
            self.top_performer_upper_threshold = upper_threshold

            if upper_pct >= 100:
                binary_labels = (target_data >= lower_threshold).astype(np.float32)
            else:
                binary_labels = (
                    (target_data >= lower_threshold) & (target_data <= upper_threshold)
                ).astype(np.float32)

            n_positive = int(binary_labels.sum())
            pct_positive = n_positive / len(binary_labels) * 100
            logger.info(
                "Classification: p%d-p%d range | threshold: $%.0f | "
                "top performers: %d/%d (%.1f%%)",
                lower_pct,
                upper_pct,
                lower_threshold,
                n_positive,
                len(binary_labels),
                pct_positive,
            )

            return torch.from_numpy(
                np.ascontiguousarray(binary_labels, dtype=np.float32)
            )

        else:
            # Regression: scale target
            if fit:
                self._target_scaler.fit(target_data)

            scaled = self._target_scaler.transform(target_data)
            return torch.from_numpy(
                np.ascontiguousarray(scaled, dtype=np.float32)
            )
