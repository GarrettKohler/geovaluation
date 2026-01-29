"""
Inference module for site scoring predictions.
Load trained model and make predictions on new data.
"""

import torch
import polars as pl
import numpy as np
from pathlib import Path
from typing import Union, Optional
from .config import Config, DEFAULT_OUTPUT_DIR
from .model import SiteScoringModel
from .data_loader import DataProcessor


class SiteScorer:
    """
    Load trained model and make predictions.
    Optimized for M4 MPS inference.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        # Default paths (using portable paths from config)
        model_path = model_path or DEFAULT_OUTPUT_DIR / "best_model.pt"
        preprocessor_path = preprocessor_path or DEFAULT_OUTPUT_DIR / "preprocessor.pkl"

        # Device
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility with Config object)
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]

        # Load preprocessor
        self.processor = DataProcessor(self.config)
        self.processor.load(preprocessor_path)

        # Recreate model
        self.model = SiteScoringModel.from_config(
            self.config, self.processor.categorical_vocab_sizes
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Polars DataFrame with same columns as training data

        Returns:
            numpy array of predictions in original scale
        """
        # Process features
        numeric = self._process_numeric(df)
        categorical = self._process_categorical(df)
        boolean = self._process_boolean(df)

        # Move to device
        numeric = torch.from_numpy(numeric).to(self.device)
        categorical = torch.from_numpy(categorical).to(self.device)
        boolean = torch.from_numpy(boolean).to(self.device)

        # Predict
        predictions = self.model(numeric, categorical, boolean)

        # Inverse transform
        predictions_np = predictions.cpu().numpy()
        return self.processor.target_scaler.inverse_transform(predictions_np).flatten()

    def _process_numeric(self, df: pl.DataFrame) -> np.ndarray:
        """Process numeric features for inference."""
        available = [c for c in self.config.numeric_features if c in df.columns]
        numeric_df = df.select(available).fill_null(0).fill_nan(0)
        numeric_array = numeric_df.to_numpy().astype(np.float32)
        return self.processor.scaler.transform(numeric_array).astype(np.float32)

    def _process_categorical(self, df: pl.DataFrame) -> np.ndarray:
        """Process categorical features for inference."""
        available = [c for c in self.config.categorical_features if c in df.columns]
        encoded_cols = []

        for col in available:
            col_data = df.select(col).fill_null("__MISSING__").to_series().to_list()
            le = self.processor.label_encoders.get(col)
            if le is not None:
                # Handle unseen categories
                encoded = []
                for val in col_data:
                    if val in le.classes_:
                        encoded.append(le.transform([val])[0])
                    else:
                        encoded.append(0)  # Unknown category
                encoded_cols.append(encoded)
            else:
                encoded_cols.append([0] * len(df))

        return np.column_stack(encoded_cols).astype(np.int64)

    def _process_boolean(self, df: pl.DataFrame) -> np.ndarray:
        """Process boolean features for inference."""
        available = [c for c in self.config.boolean_features if c in df.columns]
        bool_cols = []

        for col in available:
            col_data = df.select(col).to_series()
            if col_data.dtype == pl.Boolean:
                values = col_data.fill_null(False).to_numpy().astype(np.float32)
            else:
                values = (
                    col_data.fill_null("false")
                    .str.to_lowercase()
                    .is_in(["true", "1", "yes", "t"])
                    .to_numpy()
                    .astype(np.float32)
                )
            bool_cols.append(values)

        if bool_cols:
            return np.column_stack(bool_cols).astype(np.float32)
        return np.zeros((len(df), 1), dtype=np.float32)

    def predict_from_csv(self, csv_path: Union[str, Path]) -> np.ndarray:
        """Load CSV and make predictions."""
        df = pl.read_csv(csv_path)
        return self.predict(df)

    def score_sites(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add predicted scores to DataFrame.

        Returns:
            DataFrame with added 'predicted_revenue' column
        """
        predictions = self.predict(df)
        return df.with_columns(pl.Series("predicted_" + self.config.target, predictions))


def batch_predict(
    input_path: Path,
    output_path: Path,
    model_path: Optional[Path] = None,
    batch_size: int = 10000,
):
    """
    Batch prediction for large files.
    Processes in chunks to manage memory.
    """
    scorer = SiteScorer(model_path=model_path)

    # Use lazy loading for large files
    df_lazy = pl.scan_csv(input_path)

    # Get total rows
    total_rows = df_lazy.select(pl.len()).collect().item()
    print(f"Processing {total_rows:,} rows in batches of {batch_size:,}")

    all_predictions = []
    for offset in range(0, total_rows, batch_size):
        chunk = df_lazy.slice(offset, batch_size).collect()
        predictions = scorer.predict(chunk)
        all_predictions.extend(predictions)
        print(f"  Processed {min(offset + batch_size, total_rows):,}/{total_rows:,}")

    # Save predictions
    result = pl.DataFrame({
        "predicted_" + scorer.config.target: all_predictions
    })
    result.write_csv(output_path)
    print(f"Predictions saved to {output_path}")

    return all_predictions
