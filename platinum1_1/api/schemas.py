"""Pydantic request/response models for the Site Scoring API.

All request validation and response serialization goes through these
schemas.  Using ``str`` enums so that values are JSON-friendly without
extra conversion.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelType(str, Enum):
    NEURAL_NETWORK = "neural_network"
    XGBOOST = "xgboost"


class TaskType(str, Enum):
    REGRESSION = "regression"
    LOOKALIKE = "lookalike"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TrainingRequest(BaseModel):
    """Request body for ``POST /api/training/start``."""

    model_type: ModelType = ModelType.NEURAL_NETWORK
    task_type: TaskType = TaskType.REGRESSION
    target: str = "avg_monthly_revenue"

    # Training hyperparameters
    epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0, le=1)
    batch_size: int = Field(default=4096, ge=32, le=65536)

    # Neural network architecture
    hidden_dims: List[int] = Field(default=[512, 256, 128, 64])
    dropout: float = Field(default=0.2, ge=0, le=0.9)

    # Lookalike percentile bounds
    lookalike_lower_percentile: int = Field(default=90, ge=0, le=100)
    lookalike_upper_percentile: int = Field(default=100, ge=0, le=100)

    # Filters
    network_filter: Optional[str] = None

    # Optional feature overrides (None = use registry defaults)
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    boolean_features: Optional[List[str]] = None


class StopRequest(BaseModel):
    """Request body for ``POST /api/training/stop``."""

    job_id: str


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TrainingResponse(BaseModel):
    """Response for ``POST /api/training/start``."""

    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for ``GET /api/training/status``."""

    job_id: str
    status: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemInfoResponse(BaseModel):
    """Response for ``GET /api/training/system-info``."""

    device: str
    device_name: Optional[str] = None
    mps_available: bool
    cuda_available: bool


class FeatureListResponse(BaseModel):
    """Response for ``GET /api/training/features``."""

    numeric: List[str]
    categorical: List[str]
    boolean: List[str]
    total: int


class SiteResponse(BaseModel):
    """A single site in the sites listing."""

    id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    revenue_score: Optional[float] = None
    status: Optional[str] = None


class ExperimentResponse(BaseModel):
    """Response for experiment listing and detail."""

    job_id: str
    status: str
    created: Optional[str] = None
    completed: Optional[str] = None
    failed: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
