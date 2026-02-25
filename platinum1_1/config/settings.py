"""
Environment-based settings for the platinum1_1 backend.

Uses pydantic-settings for environment variable overrides and validation.
All paths are resolved relative to PROJECT_ROOT (the geospatial/ directory).

Override any setting via environment variables, e.g.:
    export PLATINUM_DEVICE=cpu
    export PLATINUM_MAX_EXPERIMENTS=20
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from pydantic_settings import BaseSettings


def _detect_device() -> str:
    """Auto-detect best available compute device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Settings(BaseSettings):
    """
    Application settings with sensible defaults.

    All paths default to locations relative to PROJECT_ROOT.
    Every field can be overridden via an environment variable prefixed
    with PLATINUM_ (e.g. PLATINUM_DEVICE=cpu).
    """

    model_config = {"env_prefix": "PLATINUM_"}

    # -------------------------------------------------------------------------
    # Path configuration
    # -------------------------------------------------------------------------

    # Root of the geospatial project (two levels up from config/settings.py)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    # Main data directory (contains the 927MB site_scores CSV)
    DATA_INPUT_DIR: Optional[Path] = None

    # Auxiliary geodata directory (platinum-specific CSVs)
    DATA_PLATINUM_DIR: Optional[Path] = None

    # Pre-processed parquets
    DATA_PROCESSED_DIR: Optional[Path] = None

    # Model outputs (experiments, checkpoints, SHAP plots)
    OUTPUT_DIR: Optional[Path] = None

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------

    DEVICE: str = ""
    MAX_EXPERIMENTS: int = 10
    DEFAULT_BATCH_SIZE: int = 4096
    NUM_WORKERS: int = 4

    # -------------------------------------------------------------------------
    # Resolve defaults that depend on PROJECT_ROOT
    # -------------------------------------------------------------------------

    def model_post_init(self, __context) -> None:
        """Resolve path defaults after all fields are set."""
        if self.DATA_INPUT_DIR is None:
            self.DATA_INPUT_DIR = self.PROJECT_ROOT / "data" / "input"

        if self.DATA_PLATINUM_DIR is None:
            self.DATA_PLATINUM_DIR = self.PROJECT_ROOT / "data" / "input" / "platinum"

        if self.DATA_PROCESSED_DIR is None:
            self.DATA_PROCESSED_DIR = self.PROJECT_ROOT / "data" / "processed"

        if self.OUTPUT_DIR is None:
            self.OUTPUT_DIR = self.PROJECT_ROOT / "platinum1_1" / "outputs"

        if self.DEVICE == "":
            self.DEVICE = _detect_device()

        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached so repeated calls return the same object without re-reading
    environment variables or re-resolving paths.
    """
    return Settings()
