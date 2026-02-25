"""Experiment management API routes.

Endpoints:
    GET /experiments             -- list all experiments
    GET /experiments/{job_id}    -- experiment details

All routes are prefixed with ``/api`` by the app router.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ...training.experiment import ExperimentManager
from ...training.progress import sanitize_for_json
from ..dependencies import get_experiment_manager
from ..schemas import ExperimentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /experiments
# ---------------------------------------------------------------------------

@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    experiment_manager: ExperimentManager = Depends(get_experiment_manager),
):
    """List all experiments sorted by creation time."""
    experiments = experiment_manager.list_experiments()
    return [
        ExperimentResponse(
            job_id=exp.get("job_id", "unknown"),
            status=exp.get("status", "unknown"),
            created=exp.get("created"),
            completed=exp.get("completed"),
            failed=exp.get("failed"),
            metrics=sanitize_for_json(exp.get("metrics")),
            error=exp.get("error"),
        )
        for exp in experiments
    ]


# ---------------------------------------------------------------------------
# GET /experiments/{job_id}
# ---------------------------------------------------------------------------

@router.get("/experiments/{job_id}", response_model=ExperimentResponse)
async def get_experiment(
    job_id: str,
    experiment_manager: ExperimentManager = Depends(get_experiment_manager),
):
    """Get details for a single experiment."""
    exp = experiment_manager.get_experiment(job_id)
    if exp is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {job_id} not found",
        )

    return ExperimentResponse(
        job_id=exp.get("job_id", job_id),
        status=exp.get("status", "unknown"),
        created=exp.get("created"),
        completed=exp.get("completed"),
        failed=exp.get("failed"),
        metrics=sanitize_for_json(exp.get("metrics")),
        error=exp.get("error"),
    )
