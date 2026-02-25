"""
Data access package for the platinum1_1 backend.

Public API:
    DataPaths              -- centralized path resolution
    DataRegistry           -- singleton data loader with caching and filtering
    get_registry           -- convenience accessor for the singleton
    FilterConfig           -- filtering rules dataclass
    FILTER_NONE / FILTER_WEB / FILTER_ML  -- pre-built filter presets
"""

from .paths import DataPaths
from .registry import (
    DataRegistry,
    FilterConfig,
    FILTER_ML,
    FILTER_NONE,
    FILTER_WEB,
    get_registry,
)

__all__ = [
    "DataPaths",
    "DataRegistry",
    "FilterConfig",
    "FILTER_ML",
    "FILTER_NONE",
    "FILTER_WEB",
    "get_registry",
]
