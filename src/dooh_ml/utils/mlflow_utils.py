"""MLflow tracking utilities for Azure ML integration."""

from typing import Optional, Dict, Any
import os
import tempfile
import pickle

from ..config import Config, config as default_config


class MLflowTracker:
    """MLflow tracking wrapper for Azure ML integration.

    Handles:
    - Experiment and run management
    - Parameter and metric logging
    - Model artifact storage
    - Azure ML workspace integration
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._run = None
        self._experiment = None

    def _ensure_mlflow(self):
        """Ensure MLflow is available and configured."""
        import mlflow

        if self.config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)

        return mlflow

    def start_run(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> str:
        """Start a new MLflow run.

        Args:
            experiment_name: Experiment name (default from config)
            run_name: Run name (optional)

        Returns:
            Run ID
        """
        mlflow = self._ensure_mlflow()

        experiment_name = experiment_name or self.config.mlflow.experiment_name
        mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_name=run_name)
        return self._run.info.run_id

    def end_run(self):
        """End the current MLflow run."""
        if self._run:
            import mlflow
            mlflow.end_run()
            self._run = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        import mlflow

        for key, value in params.items():
            # MLflow params must be strings
            mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        import mlflow

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)

    def log_model(self, model: Any, artifact_name: str):
        """Log a model artifact.

        Args:
            model: Model object (will be pickled)
            artifact_name: Name for the artifact
        """
        import mlflow

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{artifact_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, artifact_name)

    def log_artifact(self, content: str, filename: str):
        """Log a text artifact.

        Args:
            content: Text content to save
            filename: Filename for the artifact
        """
        import mlflow

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            mlflow.log_artifact(filepath)

    def load_model(self, run_id: str, artifact_name: str) -> Any:
        """Load a model from a previous run.

        Args:
            run_id: MLflow run ID
            artifact_name: Name of the model artifact

        Returns:
            Loaded model object
        """
        import mlflow

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{artifact_name}/{artifact_name}.pkl",
        )

        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics from a completed run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dict of metrics
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return run.data.metrics

    def register_model(
        self,
        run_id: str,
        artifact_name: str,
        model_name: str,
    ) -> str:
        """Register a model in the model registry.

        Args:
            run_id: MLflow run ID containing the model
            artifact_name: Artifact name
            model_name: Name for the registered model

        Returns:
            Model version
        """
        import mlflow

        model_uri = f"runs:/{run_id}/{artifact_name}"
        result = mlflow.register_model(model_uri, model_name)
        return result.version

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
    ):
        """Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

    def get_latest_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any:
        """Load the latest model from a given stage.

        Args:
            model_name: Registered model name
            stage: Stage to load from

        Returns:
            Loaded model
        """
        import mlflow

        model_uri = f"models:/{model_name}/{stage}"

        # Download and unpickle
        artifact_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

        # Find the pkl file
        for root, _, files in os.walk(artifact_path):
            for file in files:
                if file.endswith(".pkl"):
                    with open(os.path.join(root, file), "rb") as f:
                        return pickle.load(f)

        raise ValueError(f"No model found for {model_name} at stage {stage}")
