"""Shared FastAPI dependencies (dependency injection singletons).

Each ``get_*`` function is cached so that repeated calls within the same
process return the same instance.  FastAPI's ``Depends()`` calls these
once per request, but the ``@lru_cache`` ensures the underlying objects
are singletons.

Note: ``DataRegistry`` is not yet implemented in platinum1_1.  The
import is forward-declared so that the dependency is ready when the
registry module lands.  Until then, routes that need raw data should
use ``DataPaths`` directly.
"""

from functools import lru_cache

from ..config.settings import get_settings, Settings
from ..data.paths import DataPaths
from ..training.experiment import ExperimentManager
from ..training.job_manager import JobManager


@lru_cache(maxsize=1)
def get_app_settings() -> Settings:
    """Return the singleton application settings."""
    return get_settings()


@lru_cache(maxsize=1)
def get_data_paths() -> DataPaths:
    """Return the singleton DataPaths resolver."""
    return DataPaths()


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    """Return the singleton JobManager (single worker thread)."""
    return JobManager(max_workers=1)


@lru_cache(maxsize=1)
def get_experiment_manager() -> ExperimentManager:
    """Return the singleton ExperimentManager.

    Uses ``Settings.OUTPUT_DIR`` as the base directory and
    ``Settings.MAX_EXPERIMENTS`` for the FIFO limit.
    """
    settings = get_app_settings()
    return ExperimentManager(
        base_dir=settings.OUTPUT_DIR,
        max_experiments=settings.MAX_EXPERIMENTS,
    )
