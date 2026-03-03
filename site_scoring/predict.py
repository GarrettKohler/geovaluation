"""
Inference module for site scoring predictions.
Load trained model and make predictions on new data.
"""

import json
import pickle
import torch
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
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


class BatchPredictor:
    """
    Batch predictor that loads a trained experiment and scores all sites.
    Supports both Neural Network and XGBoost models, regression and classification.
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

        # Load config.json
        with open(self.experiment_dir / "config.json") as f:
            self.config = json.load(f)

        self.model_type = self.config["model_type"]
        self.task_type = self.config["task_type"]
        self.training_features = self.config["training_features"]

        # Load preprocessor
        with open(self.experiment_dir / "preprocessor.pkl", "rb") as f:
            preprocessor_data = pickle.load(f)

        self.scaler = preprocessor_data["scaler"]
        self.label_encoders = preprocessor_data["label_encoders"]
        self.target_scaler = preprocessor_data["target_scaler"]
        self.categorical_vocab_sizes = preprocessor_data["categorical_vocab_sizes"]

        # Load model
        if self.model_type == "xgboost":
            with open(self.experiment_dir / "model_wrapper.pkl", "rb") as f:
                self.model = pickle.load(f)
        else:
            # Neural network
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = torch.device(device)
            checkpoint = torch.load(
                self.experiment_dir / "best_model.pt",
                map_location=self.device,
                weights_only=False,
            )
            nn_config = checkpoint["config"]
            self.nn_model = SiteScoringModel.from_config(
                nn_config, self.categorical_vocab_sizes
            )
            self.nn_model.load_state_dict(checkpoint["model_state_dict"])
            self.nn_model.to(self.device)
            self.nn_model.eval()

    def predict(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Score all sites in the DataFrame.

        Args:
            df: Polars DataFrame with gtvid column and feature columns.

        Returns:
            Dict mapping gtvid -> predicted probability/score.
        """
        gtvids = df["gtvid"].to_list()

        # Process features
        numeric = self._process_numeric(df)
        categorical = self._process_categorical(df)
        boolean = self._process_boolean(df)

        if self.model_type == "xgboost":
            scores = self._predict_xgboost(numeric, categorical, boolean)
        else:
            scores = self._predict_nn(numeric, categorical, boolean)

        return {gtvid: float(score) for gtvid, score in zip(gtvids, scores)}

    def predict_with_metadata(
        self,
        df: pl.DataFrame,
        training_labels: Optional[Dict[str, int]] = None,
    ) -> pl.DataFrame:
        """Score sites and return DataFrame with predictions + full site metadata.

        Args:
            df: Polars DataFrame with gtvid column and feature columns.
            training_labels: Optional dict mapping gtvid -> actual_label (0/1).
                When provided (classification tasks), adds 'actual_label' and
                'category' (TRAINING/NON_ACTIVE) columns to the output.

        Returns:
            DataFrame with predictions, site metadata, rank, and percentile.
        """
        scores = self.predict(df)

        # Build scores DataFrame
        scores_df = pl.DataFrame({
            "gtvid": list(scores.keys()),
            "predicted_score": list(scores.values()),
        })

        # Load site metadata from the raw parquet (before geo IDs are dropped)
        metadata_df = self._load_site_metadata(df)

        # Join predictions with metadata
        result = scores_df.join(metadata_df, on="gtvid", how="left")

        # Add rank (1 = highest score) and percentile
        result = result.with_columns([
            pl.col("predicted_score").rank(descending=True).cast(pl.Int32).alias("rank"),
        ])
        n = len(result)
        result = result.with_columns([
            ((n - pl.col("rank")) / n * 100).round(1).alias("percentile"),
        ])

        # For classification: add actual labels and category
        if training_labels is not None:
            training_gtvids = set(training_labels.keys())
            result = result.with_columns([
                pl.col("gtvid").map_elements(
                    lambda g: training_labels.get(g), return_dtype=pl.Int32
                ).alias("actual_label"),
                pl.col("gtvid").map_elements(
                    lambda g: "TRAINING" if g in training_gtvids else "NON_ACTIVE",
                    return_dtype=pl.Utf8
                ).alias("category"),
            ])

        # Sort by rank
        result = result.sort("rank")

        return result

    @staticmethod
    def _load_site_metadata(prediction_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Load site metadata from the raw parquet for export enrichment.

        Falls back to columns available in the prediction DataFrame if the
        raw parquet is not accessible.
        """
        project_root = Path(__file__).parent.parent.resolve()
        parquet_path = project_root / "data" / "processed" / "site_aggregated_precleaned.parquet"

        metadata_cols = [
            "gtvid", "state", "county", "zip", "dma", "dma_rank",
            "network", "status", "retailer", "latitude", "longitude",
            "avg_monthly_revenue", "avg_daily_revenue",
        ]

        if parquet_path.exists():
            raw = pl.read_parquet(parquet_path)

            # Normalize the status column name (source CSV typo: statuis)
            if "statuis" in raw.columns and "status" not in raw.columns:
                raw = raw.rename({"statuis": "status"})

            available = [c for c in metadata_cols if c in raw.columns]
            return raw.select(available).unique(subset=["gtvid"])

        # Fallback: use whatever is available in the prediction DataFrame
        if prediction_df is None:
            return pl.DataFrame({"gtvid": []})

        fallback_cols = ["gtvid"]
        for col in ["network", "status", "statuis", "avg_monthly_revenue", "avg_daily_revenue"]:
            if col in prediction_df.columns:
                fallback_cols.append(col)

        result = prediction_df.select(fallback_cols).unique(subset=["gtvid"])
        if "statuis" in result.columns and "status" not in result.columns:
            result = result.rename({"statuis": "status"})
        return result

    def _process_numeric(self, df: pl.DataFrame) -> np.ndarray:
        """Process numeric features using the fitted scaler."""
        features = self.training_features["numeric"]
        available = [c for c in features if c in df.columns]
        if not available:
            return np.zeros((len(df), 1), dtype=np.float32)

        numeric_df = df.select(available).fill_null(0).fill_nan(0)
        numeric_array = numeric_df.to_numpy().astype(np.float32)
        return self.scaler.transform(numeric_array).astype(np.float32)

    def _process_categorical(self, df: pl.DataFrame) -> np.ndarray:
        """Process categorical features using the fitted label encoders."""
        features = self.training_features["categorical"]
        available = [c for c in features if c in df.columns]
        encoded_cols = []

        for col in available:
            col_data = df.select(col).fill_null("__MISSING__").to_series().to_list()
            le = self.label_encoders.get(col)
            if le is not None:
                encoded = []
                for val in col_data:
                    if val in le.classes_:
                        encoded.append(le.transform([val])[0])
                    else:
                        encoded.append(0)
                encoded_cols.append(encoded)
            else:
                encoded_cols.append([0] * len(df))

        return np.column_stack(encoded_cols).astype(np.int64)

    def _process_boolean(self, df: pl.DataFrame) -> np.ndarray:
        """Process boolean features."""
        features = self.training_features["boolean"]
        available = [c for c in features if c in df.columns]
        bool_cols = []

        for col in available:
            col_data = df.select(col).to_series()
            if col_data.dtype == pl.Boolean:
                values = col_data.fill_null(False).to_numpy().astype(np.float32)
            elif col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                values = col_data.fill_null(0).to_numpy().astype(np.float32)
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

    def _predict_xgboost(
        self, numeric: np.ndarray, categorical: np.ndarray, boolean: np.ndarray
    ) -> np.ndarray:
        """Run XGBoost prediction."""
        X = np.hstack([numeric, categorical.astype(np.float32), boolean])

        if self.task_type == "lookalike":
            return self.model.predict_proba(X)[:, 1]
        else:
            preds = self.model.predict(X)
            if self.target_scaler is not None:
                preds = self.target_scaler.inverse_transform(
                    preds.reshape(-1, 1)
                ).flatten()
            return preds

    @torch.no_grad()
    def _predict_nn(
        self, numeric: np.ndarray, categorical: np.ndarray, boolean: np.ndarray
    ) -> np.ndarray:
        """Run Neural Network prediction in batches."""
        batch_size = 4096
        n_samples = len(numeric)
        all_scores = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            num_t = torch.from_numpy(numeric[start:end]).to(self.device)
            cat_t = torch.from_numpy(categorical[start:end]).to(self.device)
            bool_t = torch.from_numpy(boolean[start:end]).to(self.device)

            output = self.nn_model(num_t, cat_t, bool_t)

            if self.task_type == "lookalike":
                scores = torch.sigmoid(output).cpu().numpy().flatten()
            else:
                scores = output.cpu().numpy()
                if self.target_scaler is not None:
                    scores = self.target_scaler.inverse_transform(scores).flatten()
                else:
                    scores = scores.flatten()

            all_scores.append(scores)

        return np.concatenate(all_scores)
