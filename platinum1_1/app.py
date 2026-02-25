"""Platinum 1.1 Backend -- FastAPI application.

Entrypoint for the ML-powered site scoring API.  Run with::

    uvicorn platinum1_1.app:app --host 0.0.0.0 --port 8080 --reload

Or directly::

    python -m platinum1_1.app
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import data, experiments, training

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Site Scoring API",
    description="ML-powered site scoring and prediction platform",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "1.1.0"}


# ---------------------------------------------------------------------------
# Startup / shutdown hooks
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Log startup and validate data paths."""
    from .config.settings import get_settings
    from .data.paths import DataPaths

    settings = get_settings()
    paths = DataPaths(settings)
    validation = paths.validate()

    logger.info("Site Scoring API v1.1.0 starting")
    logger.info("Device: %s", settings.DEVICE)
    logger.info("Output dir: %s", settings.OUTPUT_DIR)
    logger.info("Data paths: %s", {k: "OK" if v else "MISSING" for k, v in validation.items()})


@app.on_event("shutdown")
async def on_shutdown():
    """Clean shutdown of background workers."""
    from .api.dependencies import get_job_manager

    try:
        job_manager = get_job_manager()
        job_manager.shutdown()
        logger.info("JobManager shut down")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "platinum1_1.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )
