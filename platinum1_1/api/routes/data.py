"""Data API routes.

Endpoints:
    GET /sites     -- all sites with coordinates and revenue scores
    GET /features  -- feature definitions from the registry

All routes are prefixed with ``/api`` by the app router.
"""

import logging
from typing import List

import numpy as np
import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from ...config.features import FeatureRegistry, FeatureType
from ...data.paths import DataPaths
from ..dependencies import get_data_paths
from ..schemas import FeatureListResponse, SiteResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /sites
# ---------------------------------------------------------------------------

@router.get("/sites", response_model=List[SiteResponse])
async def get_sites(
    data_paths: DataPaths = Depends(get_data_paths),
):
    """Return all sites with coordinates and revenue scores.

    Reads from the pre-processed training parquet using Polars.
    Falls back to a 404 if the file does not exist yet.
    """
    parquet_path = data_paths.training_parquet
    if not parquet_path.exists():
        logger.warning("Training parquet not found at %s", parquet_path)
        raise HTTPException(
            status_code=404,
            detail="Training data not found. Run the data pipeline first.",
        )

    try:
        # Read only needed columns (handle typo column name)
        df = pl.read_parquet(parquet_path)

        # Detect which columns are available
        id_col = "site_id" if "site_id" in df.columns else "gtvid"
        status_col = "status" if "status" in df.columns else "statuis"

        # Select available columns
        select_cols = [id_col]
        if "latitude" in df.columns:
            select_cols.append("latitude")
        if "longitude" in df.columns:
            select_cols.append("longitude")
        if "avg_monthly_revenue" in df.columns:
            select_cols.append("avg_monthly_revenue")
        if status_col in df.columns:
            select_cols.append(status_col)

        df = df.select(select_cols)

        sites = []
        for row in df.iter_rows(named=True):
            sites.append(SiteResponse(
                id=str(row.get(id_col, "")),
                latitude=_safe_float(row.get("latitude")),
                longitude=_safe_float(row.get("longitude")),
                revenue_score=_safe_float(row.get("avg_monthly_revenue")),
                status=row.get(status_col) if status_col != "statuis" else row.get("statuis"),
            ))

        return sites

    except Exception as exc:
        logger.error("Failed to load site data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /features
# ---------------------------------------------------------------------------

@router.get("/features", response_model=FeatureListResponse)
async def get_feature_definitions():
    """Return feature definitions grouped by type from the FeatureRegistry."""
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
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    """Convert a value to float, returning None for NaN/Inf/missing."""
    if val is None:
        return None
    try:
        import numpy as np

        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None
