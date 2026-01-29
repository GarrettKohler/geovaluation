"""
Site Scoring ML Module
Fast PyTorch-based site revenue/impression prediction optimized for Apple M4.
"""

from .config import Config
from .data_loader import SiteDataset, create_data_loaders
from .model import SiteScoringModel
from .train import Trainer

__version__ = "0.1.0"
__all__ = ["Config", "SiteDataset", "create_data_loaders", "SiteScoringModel", "Trainer"]
