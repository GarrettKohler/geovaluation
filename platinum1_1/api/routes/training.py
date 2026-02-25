"""Training API routes.

Endpoints:
    POST /start        -- start a training job
    POST /stop         -- stop a running job
    GET  /status       -- get status of the latest or specified job
    GET  /stream       -- SSE stream for real-time progress
    GET  /system-info  -- device/hardware information
    GET  /features     -- available feature lists from the registry

All routes are prefixed with ``/api/training`` by the app router.
"""

import asyncio
import logging
import queue
from typing import Optional

import polars as pl
import torch
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...config.features import FeatureRegistry, FeatureType, ModelConfig
from ...training.job_manager import JobManager, TrainingJob
from ...training.experiment import ExperimentManager
from ...training.progress import TrainingProgress, sanitize_for_json
from ..dependencies import get_app_settings, get_experiment_manager, get_job_manager
from ..schemas import (
    FeatureListResponse,
    JobStatusResponse,
    StopRequest,
    SystemInfoResponse,
    TrainingRequest,
    TrainingResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /start
# ---------------------------------------------------------------------------

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    job_manager: JobManager = Depends(get_job_manager),
    experiment_manager: ExperimentManager = Depends(get_experiment_manager),
):
    """Start a new training job.

    The job runs on a background thread.  Use ``GET /stream?job_id=...``
    to consume real-time progress events.
    """
    settings = get_app_settings()

    # Build ModelConfig from the request
    model_config = ModelConfig(
        target=request.target,
        task_type=request.task_type.value,
        model_type=request.model_type.value,
        epochs=request.epochs,
        learning_rate=request.learning_rate,
        hidden_dims=request.hidden_dims,
        dropout=request.dropout,
        lookalike_lower_percentile=request.lookalike_lower_percentile,
        lookalike_upper_percentile=request.lookalike_upper_percentile,
        network_filter=request.network_filter,
    )

    # Serializable config for the job record
    config_dict = {
        "model_type": request.model_type.value,
        "task_type": request.task_type.value,
        "target": request.target,
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "batch_size": request.batch_size,
        "hidden_dims": request.hidden_dims,
        "dropout": request.dropout,
        "lookalike_lower_percentile": request.lookalike_lower_percentile,
        "lookalike_upper_percentile": request.lookalike_upper_percentile,
        "network_filter": request.network_filter,
        "device": settings.DEVICE,
    }

    def run_training(job: TrainingJob) -> dict:
        """Executed on the worker thread by JobManager."""
        # Create experiment folder
        exp_dir = experiment_manager.create_experiment(job.job_id)
        job.output_dir = exp_dir

        def progress_cb(progress: TrainingProgress):
            job.progress_queue.put(progress)

        # Notify start
        progress_cb(TrainingProgress(
            status="running",
            message=f"Training started: {request.model_type.value} / {request.task_type.value}",
        ))

        try:
            if request.model_type.value == "neural_network":
                result = _run_nn_training(
                    model_config, config_dict, settings,
                    progress_cb, job.stop_event, exp_dir,
                )
            else:
                result = _run_xgb_training(
                    model_config, config_dict, settings,
                    progress_cb, job.stop_event, exp_dir,
                )

            # Record completion
            experiment_manager.complete_experiment(
                job.job_id, result.get("test_metrics", {})
            )
            return result

        except Exception as exc:
            experiment_manager.fail_experiment(job.job_id, str(exc))
            raise

    job_id = job_manager.submit_job(config_dict, run_training)

    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training job submitted",
    )


# ---------------------------------------------------------------------------
# POST /stop
# ---------------------------------------------------------------------------

@router.post("/stop", response_model=TrainingResponse)
async def stop_training(
    request: StopRequest,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Stop a running training job."""
    stopped = job_manager.stop_job(request.job_id)
    if stopped:
        return TrainingResponse(
            job_id=request.job_id,
            status="stopped",
            message="Stop signal sent",
        )
    raise HTTPException(
        status_code=404,
        detail=f"No running job found with id {request.job_id}",
    )


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=JobStatusResponse)
async def get_status(
    job_id: Optional[str] = Query(None),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get the status of a training job.

    If ``job_id`` is omitted, returns the most recently submitted job.
    """
    if job_id:
        job = job_manager.get_job(job_id)
    else:
        # Return latest job
        jobs = job_manager.list_jobs()
        if not jobs:
            raise HTTPException(status_code=404, detail="No training jobs found")
        job = job_manager.get_job(jobs[-1]["job_id"])

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        metrics=sanitize_for_json(job.result.get("test_metrics")) if job.result else None,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /stream
# ---------------------------------------------------------------------------

@router.get("/stream")
async def stream_progress(
    job_id: Optional[str] = Query(None),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Server-Sent Events stream for real-time training progress.

    Streams ``TrainingProgress`` events until the job completes, fails,
    or is stopped.  The final event has ``status`` set to a terminal
    value.
    """
    if job_id:
        job = job_manager.get_job(job_id)
    else:
        job = job_manager.get_latest_running_job()
        if job is None:
            # Try the latest submitted job
            jobs = job_manager.list_jobs()
            if jobs:
                job = job_manager.get_job(jobs[-1]["job_id"])

    if job is None:
        raise HTTPException(status_code=404, detail="No job found to stream")

    async def event_generator():
        """Yield SSE events from the job's progress queue."""
        terminal_statuses = {"completed", "failed", "stopped"}

        while True:
            try:
                # Non-blocking get with asyncio-friendly polling
                progress: TrainingProgress = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: job.progress_queue.get(timeout=0.5)
                )
                yield progress.to_sse_event()

                if progress.status in terminal_statuses:
                    break
            except queue.Empty:
                # Send keepalive comment to prevent connection timeout
                yield ": keepalive\n\n"

                # Check if job is in a terminal state (no more events coming)
                if job.status in terminal_statuses:
                    # Drain any remaining events
                    while not job.progress_queue.empty():
                        try:
                            remaining = job.progress_queue.get_nowait()
                            yield remaining.to_sse_event()
                        except queue.Empty:
                            break
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /system-info
# ---------------------------------------------------------------------------

@router.get("/system-info", response_model=SystemInfoResponse)
async def system_info():
    """Return compute device information."""
    settings = get_app_settings()
    device_name = None

    if settings.DEVICE == "mps":
        device_name = "Apple Silicon (MPS)"
    elif settings.DEVICE == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    return SystemInfoResponse(
        device=settings.DEVICE,
        device_name=device_name,
        mps_available=torch.backends.mps.is_available(),
        cuda_available=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# GET /features
# ---------------------------------------------------------------------------

@router.get("/features", response_model=FeatureListResponse)
async def get_features():
    """Return the available feature lists from the FeatureRegistry."""
    numeric = FeatureRegistry.get_by_type(FeatureType.NUMERIC)
    categorical = FeatureRegistry.get_by_type(FeatureType.CATEGORICAL)
    boolean = FeatureRegistry.get_by_type(FeatureType.BOOLEAN)

    return FeatureListResponse(
        numeric=numeric,
        categorical=categorical,
        boolean=boolean,
        total=len(numeric) + len(categorical) + len(boolean),
    )


# ---------------------------------------------------------------------------
# Internal training orchestration helpers
# ---------------------------------------------------------------------------

def _run_nn_training(model_config, config_dict, settings, progress_cb, stop_event, exp_dir):
    """Orchestrate a neural network training run.

    Called on the worker thread. Pipeline:
    1. Load data via DataRegistry
    2. Build DataLoaders via FeatureProcessor
    3. Construct the NN model via factory
    4. Train with NNTrainer (early stopping, gradient clipping)
    5. Evaluate on held-out test set
    6. Save model checkpoint + processor state
    """
    import torch

    from ...data.registry import DataRegistry
    from ...models.factory import create_model
    from ...processing.data_loader import create_data_loaders
    from ...training.trainer import NNTrainer

    progress_cb(TrainingProgress(
        status="running",
        message="Loading training data...",
    ))

    # 1. Load data
    registry = DataRegistry()
    df = registry.get_training_data()

    # Apply network filter if specified
    network_filter = config_dict.get("network_filter")
    if network_filter and "network" in df.columns:
        df = df.filter(pl.col("network") == network_filter)

    progress_cb(TrainingProgress(
        status="running",
        message=f"Loaded {len(df):,} sites. Building data loaders...",
    ))

    # 2. Build DataLoaders (FeatureProcessor fit happens inside)
    # Attach feature lists from ModelConfig to a config-like object
    _cfg = _ConfigAdapter(model_config, config_dict, settings)
    train_loader, val_loader, test_loader, processor = create_data_loaders(_cfg, df)

    progress_cb(TrainingProgress(
        status="running",
        message="Building neural network model...",
    ))

    # 3. Create model
    model = create_model(
        config=_cfg,
        categorical_vocab_sizes=processor.categorical_vocab_sizes,
        n_numeric=processor.n_numeric,
        n_boolean=processor.n_boolean,
    )

    # 4. Train
    trainer = NNTrainer(
        model=model,
        device=settings.DEVICE,
        learning_rate=config_dict.get("learning_rate", 1e-4),
        weight_decay=1e-5,
        scheduler_patience=getattr(model_config, "scheduler_patience", 5),
        early_stopping_patience=getattr(model_config, "early_stopping_patience", 10),
        task_type=config_dict.get("task_type", "regression"),
    )

    train_result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config_dict.get("epochs", 50),
        progress_callback=progress_cb,
        stop_event=stop_event,
    )

    # 5. Evaluate on held-out test set (after best model restored)
    test_metrics = trainer.evaluate(test_loader)

    # 6. Save artifacts
    if exp_dir:
        torch.save(model.state_dict(), exp_dir / "model.pt")
        processor.save(exp_dir / "processor.pkl")

    progress_cb(TrainingProgress(
        status="completed",
        message="Training complete",
        metrics=test_metrics,
    ))

    return {
        "train_result": sanitize_for_json(train_result),
        "test_metrics": sanitize_for_json(test_metrics),
    }


def _run_xgb_training(model_config, config_dict, settings, progress_cb, stop_event, exp_dir):
    """Orchestrate an XGBoost training run.

    Called on the worker thread. Follows the same data pipeline as
    ``_run_nn_training`` but converts DataLoaders to numpy arrays
    for tree-based training.
    """
    from ...data.registry import DataRegistry
    from ...models.xgboost_model import XGBoostModel
    from ...processing.data_loader import create_data_loaders
    from ...training.tree_trainer import TreeTrainer, dataloaders_to_numpy

    progress_cb(TrainingProgress(
        status="running",
        message="Loading training data...",
    ))

    # 1. Load data
    registry = DataRegistry()
    df = registry.get_training_data()

    network_filter = config_dict.get("network_filter")
    if network_filter and "network" in df.columns:
        df = df.filter(pl.col("network") == network_filter)

    progress_cb(TrainingProgress(
        status="running",
        message=f"Loaded {len(df):,} sites. Building data loaders...",
    ))

    # 2. Build DataLoaders + convert to numpy
    _cfg = _ConfigAdapter(model_config, config_dict, settings)
    train_loader, val_loader, test_loader, processor = create_data_loaders(_cfg, df)
    X_train, y_train, X_val, y_val, X_test, y_test = dataloaders_to_numpy(
        train_loader, val_loader, test_loader
    )

    progress_cb(TrainingProgress(
        status="running",
        message="Training XGBoost model...",
    ))

    # 3. Create and train model
    task_type = config_dict.get("task_type", "regression")
    xgb_model = XGBoostModel(
        task_type=task_type,
        learning_rate=config_dict.get("learning_rate", 0.05),
    )

    tree_trainer = TreeTrainer(task_type=task_type)
    train_result = tree_trainer.train(
        model=xgb_model.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        progress_callback=progress_cb,
        stop_event=stop_event,
    )

    # 4. Evaluate on held-out test set
    test_metrics = tree_trainer.evaluate(xgb_model.model, X_test, y_test)

    # 5. Save artifacts
    if exp_dir:
        xgb_model.is_fitted = True
        xgb_model.save(exp_dir / "model.xgb")
        processor.save(exp_dir / "processor.pkl")

    progress_cb(TrainingProgress(
        status="completed",
        message="Training complete",
        metrics=test_metrics,
    ))

    return {
        "train_result": sanitize_for_json(train_result),
        "test_metrics": sanitize_for_json(test_metrics),
    }


class _ConfigAdapter:
    """Adapts ModelConfig + config_dict into the protocol expected by
    FeatureProcessor and create_data_loaders."""

    def __init__(self, model_config, config_dict, settings):
        self._mc = model_config
        self._cd = config_dict
        self._settings = settings

    def __getattr__(self, name):
        # Priority: model_config attributes > config_dict > settings
        if hasattr(self._mc, name):
            return getattr(self._mc, name)
        if name in self._cd:
            return self._cd[name]
        if hasattr(self._settings, name):
            return getattr(self._settings, name)
        raise AttributeError(f"_ConfigAdapter has no attribute '{name}'")

    @property
    def numeric_features(self):
        return self._mc.get_numeric_features()

    @property
    def categorical_features(self):
        return self._mc.get_categorical_features()

    @property
    def boolean_features(self):
        return self._mc.get_boolean_features()

    @property
    def device(self):
        return self._settings.DEVICE
