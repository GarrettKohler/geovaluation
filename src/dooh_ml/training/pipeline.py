"""Training pipeline orchestration with MLflow tracking."""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json
import pandas as pd

from ..config import Config, config as default_config
from ..data.loader import DataLoader
from ..data.preprocessing import TemporalSplitter, FeatureEngineer
from ..models.similarity import SimilarityModel
from ..models.causal import CausalModel
from ..models.classifier import ActivationClassifier
from ..utils.mlflow_utils import MLflowTracker


@dataclass
class TrainingResult:
    """Container for training results."""

    similarity_model: SimilarityModel
    causal_model: CausalModel
    classifier: ActivationClassifier
    metrics: Dict[str, Any]
    run_id: str


class TrainingPipeline:
    """End-to-end training pipeline for all three models.

    Handles:
    - Data loading and temporal splitting
    - Feature engineering
    - Model training with MLflow tracking
    - Evaluation and metric logging
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.loader = DataLoader(config)
        self.splitter = TemporalSplitter(gap_days=self.config.gap_days)
        self.engineer = FeatureEngineer(config)
        self.tracker = MLflowTracker(config)

    def run(
        self,
        train_end: str,
        validation_end: str,
        test_start: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TrainingResult:
        """Execute full training pipeline.

        Args:
            train_end: Last date for training data (YYYY-MM-DD)
            validation_end: Last date for validation data
            test_start: First date for test data
            experiment_name: MLflow experiment name

        Returns:
            TrainingResult with fitted models and metrics
        """
        run_id = self.tracker.start_run(
            experiment_name=experiment_name,
            run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        try:
            # Log parameters
            self.tracker.log_params({
                "train_end": train_end,
                "validation_end": validation_end,
                "test_start": test_start or "auto",
                "gap_days": self.config.gap_days,
            })

            # Load and split data
            print("Loading data...")
            df = self.loader.load_feature_snapshots()

            print("Splitting data...")
            splits = self.splitter.split(df, train_end, validation_end, test_start)

            self.tracker.log_params({
                "train_samples": len(splits.train),
                "val_samples": len(splits.validation),
                "test_samples": len(splits.test),
                "train_sites": len(splits.train_sites),
                "val_sites": len(splits.validation_sites),
                "test_sites": len(splits.test_sites),
            })

            # Prepare features
            print("Preparing features...")
            X_train, y_train, feature_names, cat_indices = self.engineer.prepare_for_training(
                splits.train
            )
            X_val, y_val, _, _ = self.engineer.prepare_for_training(splits.validation)
            X_test, y_test, _, _ = self.engineer.prepare_for_training(splits.test)

            # Get categorical feature names
            cat_features = [feature_names[i] for i in cat_indices]

            # Train models
            metrics = {}

            # 1. Similarity Model
            print("Training similarity model...")
            similarity_model = self._train_similarity(splits.train)
            metrics["similarity"] = {
                "n_high_performers": similarity_model.n_high_performers,
                "threshold": similarity_model.threshold,
            }

            # 2. Causal Model
            print("Training causal model...")
            causal_model = self._train_causal(splits.train)
            # Evaluate on test set
            test_effects = causal_model.effect(splits.test)
            metrics["causal"] = {
                "mean_effect": float(test_effects.mean()),
                "std_effect": float(test_effects.std()),
                "pct_positive": float((test_effects > 0).mean()),
            }

            # 3. Classifier
            print("Training classifier...")
            classifier = self._train_classifier(
                X_train, y_train, X_val, y_val, cat_features
            )
            classifier_metrics = classifier.evaluate(X_test, y_test)
            metrics["classifier"] = classifier_metrics

            # Log all metrics
            flat_metrics = {}
            for model_name, model_metrics in metrics.items():
                for metric_name, value in model_metrics.items():
                    flat_metrics[f"{model_name}_{metric_name}"] = value

            self.tracker.log_metrics(flat_metrics)

            # Save models
            print("Saving models...")
            self.tracker.log_model(similarity_model, "similarity_model")
            self.tracker.log_model(causal_model, "causal_model")
            self.tracker.log_model(classifier, "classifier")

            # Log feature importance
            importance_df = classifier.feature_importance()
            self.tracker.log_artifact(
                importance_df.to_csv(index=False),
                "feature_importance.csv",
            )

            print(f"Training complete. Run ID: {run_id}")

            return TrainingResult(
                similarity_model=similarity_model,
                causal_model=causal_model,
                classifier=classifier,
                metrics=metrics,
                run_id=run_id,
            )

        except Exception as e:
            self.tracker.log_params({"error": str(e)})
            raise
        finally:
            self.tracker.end_run()

    def _train_similarity(self, train_df: pd.DataFrame) -> SimilarityModel:
        """Train similarity model on active high performers."""
        active_sites = train_df[train_df.get("is_active", True) == True]

        model = SimilarityModel(
            k_neighbors=self.config.model.similarity_k_neighbors,
            high_performer_quantile=self.config.model.similarity_high_performer_quantile,
            config=self.config,
        )

        model.fit(active_sites)

        self.tracker.log_params({
            "similarity_k_neighbors": self.config.model.similarity_k_neighbors,
            "similarity_quantile": self.config.model.similarity_high_performer_quantile,
        })

        return model

    def _train_causal(self, train_df: pd.DataFrame) -> CausalModel:
        """Train causal model for treatment effects."""
        model = CausalModel(
            n_estimators=self.config.model.causal_n_estimators,
            min_samples_leaf=self.config.model.causal_min_samples_leaf,
            cv_folds=self.config.model.causal_cv_folds,
            config=self.config,
        )

        model.fit(train_df)

        self.tracker.log_params({
            "causal_n_estimators": self.config.model.causal_n_estimators,
            "causal_min_samples_leaf": self.config.model.causal_min_samples_leaf,
            "causal_cv_folds": self.config.model.causal_cv_folds,
        })

        return model

    def _train_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: list,
    ) -> ActivationClassifier:
        """Train activation success classifier."""
        model = ActivationClassifier(
            iterations=self.config.model.classifier_iterations,
            learning_rate=self.config.model.classifier_learning_rate,
            depth=self.config.model.classifier_depth,
            early_stopping_rounds=self.config.model.classifier_early_stopping_rounds,
            config=self.config,
        )

        model.fit(X_train, y_train, X_val, y_val, cat_features)

        self.tracker.log_params({
            "classifier_iterations": self.config.model.classifier_iterations,
            "classifier_learning_rate": self.config.model.classifier_learning_rate,
            "classifier_depth": self.config.model.classifier_depth,
        })

        return model

    def retrain_from_checkpoint(
        self,
        run_id: str,
        new_data_end: str,
    ) -> TrainingResult:
        """Retrain models with new data, starting from checkpoint.

        Args:
            run_id: Previous MLflow run ID to load models from
            new_data_end: End date for new training data
        """
        # Load previous models
        similarity_model = self.tracker.load_model(run_id, "similarity_model")
        causal_model = self.tracker.load_model(run_id, "causal_model")
        classifier = self.tracker.load_model(run_id, "classifier")

        # Load new data and retrain
        # This is a simplified implementation - production would do incremental training
        return self.run(
            train_end=new_data_end,
            validation_end=new_data_end,  # Would need proper dates
        )
