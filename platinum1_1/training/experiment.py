"""Experiment lifecycle management with FIFO cleanup."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class ExperimentManager:
    """Manages experiment folders with automatic FIFO cleanup.

    Each experiment gets a timestamped folder under ``base_dir/experiments/``.
    When the folder count reaches ``max_experiments``, the oldest experiment
    is removed before creating a new one.

    Folder structure::

        experiments/
            20260215_143000_abc12345/
                metadata.json     # job metadata, status, final metrics
                model.pt          # saved model weights (written by trainer)
                processor.pkl     # fitted FeatureProcessor (written by pipeline)
    """

    def __init__(self, base_dir: Path, max_experiments: int = 10):
        self.base_dir = base_dir / "experiments"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_experiments = max_experiments

    def create_experiment(self, job_id: str) -> Path:
        """Create new experiment folder, cleaning oldest if at limit.

        Args:
            job_id: Unique identifier for the experiment (typically
                    ``YYYYMMDD_HHMMSS_<hex>``).

        Returns:
            Path to the newly created experiment directory.
        """
        self._cleanup_if_needed()
        exp_dir = self.base_dir / job_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "job_id": job_id,
            "created": datetime.now().isoformat(),
            "status": "running",
        }
        (exp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return exp_dir

    def complete_experiment(self, job_id: str, metrics: dict) -> None:
        """Mark an experiment as completed and persist final metrics.

        Args:
            job_id: The experiment to update.
            metrics: Dict of final evaluation metrics to record.
        """
        exp_dir = self.base_dir / job_id
        meta_path = exp_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "completed"
            meta["completed"] = datetime.now().isoformat()
            meta["metrics"] = metrics
            meta_path.write_text(json.dumps(meta, indent=2))

    def fail_experiment(self, job_id: str, error: str) -> None:
        """Mark an experiment as failed and record the error.

        Args:
            job_id: The experiment to update.
            error: Error message or traceback string.
        """
        exp_dir = self.base_dir / job_id
        meta_path = exp_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["status"] = "failed"
            meta["failed"] = datetime.now().isoformat()
            meta["error"] = error
            meta_path.write_text(json.dumps(meta, indent=2))

    def get_experiment(self, job_id: str) -> Optional[dict]:
        """Load metadata for a single experiment.

        Returns:
            Parsed metadata dict, or ``None`` if the experiment does not exist.
        """
        meta_path = self.base_dir / job_id / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return None

    def list_experiments(self) -> List[dict]:
        """List all experiments sorted by creation time (oldest first).

        Returns:
            List of metadata dicts, one per experiment folder.
        """
        experiments = []
        for exp_dir in sorted(self.base_dir.iterdir()):
            if exp_dir.is_dir():
                meta_path = exp_dir / "metadata.json"
                if meta_path.exists():
                    experiments.append(json.loads(meta_path.read_text()))
        return experiments

    def _cleanup_if_needed(self) -> None:
        """Remove oldest experiments if at or above the limit."""
        dirs = sorted(
            (d for d in self.base_dir.iterdir() if d.is_dir()),
            key=lambda p: p.stat().st_ctime,
        )
        while len(dirs) >= self.max_experiments:
            oldest = dirs.pop(0)
            shutil.rmtree(oldest)
