"""Training orchestration package.

Public API::

    from platinum1_1.training import TrainingProgress, JobManager
    from platinum1_1.training.experiment import ExperimentManager
    from platinum1_1.training.trainer import NNTrainer
    from platinum1_1.training.tree_trainer import TreeTrainer
"""

from .job_manager import JobManager, TrainingJob
from .progress import TrainingProgress, sanitize_for_json

__all__ = [
    "JobManager",
    "TrainingJob",
    "TrainingProgress",
    "sanitize_for_json",
]
