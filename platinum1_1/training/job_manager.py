"""Training job management with thread-based async execution.

The ``JobManager`` accepts training jobs, executes them on a background
thread (one at a time), and exposes a per-job ``queue.Queue`` for
progress events that the SSE endpoint can consume.

Architecture:
    submit_job(config, run_fn)
        -> creates TrainingJob with progress_queue + stop_event
        -> puts (job, run_fn) on _job_queue
        -> background worker picks it up, calls run_fn(job)
        -> run_fn pushes TrainingProgress to job.progress_queue
        -> SSE endpoint reads from job.progress_queue
"""

import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """A single training job with its communication channels.

    Attributes:
        job_id: Unique identifier (``YYYYMMDD_HHMMSS_<hex>``).
        config: Serializable training configuration dict.
        status: Current job status.
        created_at: ISO 8601 creation timestamp.
        progress_queue: Queue for ``TrainingProgress`` events (thread-safe).
        stop_event: Set this to request graceful cancellation.
        output_dir: Experiment folder path (set by orchestrator).
        result: Final result dict (set on completion).
        error: Error message (set on failure).
    """

    job_id: str
    config: dict
    status: str = "pending"  # pending | running | completed | failed | stopped
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    progress_queue: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    output_dir: Optional[Path] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class JobManager:
    """Manages training jobs with single-threaded background execution.

    Only one job runs at a time (``max_workers=1``).  Subsequent calls to
    ``submit_job`` queue the job until the current one finishes.

    Args:
        max_workers: Number of concurrent worker threads (default 1).
    """

    def __init__(self, max_workers: int = 1):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._job_queue: queue.Queue = queue.Queue()
        self._max_workers = max_workers
        self._running = True
        self._start_workers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_job(self, config: dict, run_fn: Callable) -> str:
        """Submit a training job for background execution.

        Args:
            config: Serializable training configuration.
            run_fn: Callable that receives a ``TrainingJob`` and executes
                the training pipeline.  Must push ``TrainingProgress``
                objects to ``job.progress_queue``.

        Returns:
            The ``job_id`` string.
        """
        job_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        job = TrainingJob(job_id=job_id, config=config)

        with self._lock:
            self._jobs[job_id] = job

        self._job_queue.put((job, run_fn))
        logger.info("Job %s submitted (status=pending)", job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Look up a job by ID.  Returns ``None`` if not found."""
        return self._jobs.get(job_id)

    def stop_job(self, job_id: str) -> bool:
        """Request graceful cancellation of a running job.

        Returns:
            ``True`` if the stop signal was sent, ``False`` otherwise.
        """
        job = self._jobs.get(job_id)
        if job and job.status == "running":
            job.stop_event.set()
            job.status = "stopped"
            logger.info("Job %s stop requested", job_id)
            return True
        return False

    def list_jobs(self) -> List[dict]:
        """Return a summary list of all known jobs."""
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "created_at": j.created_at,
            }
            for j in self._jobs.values()
        ]

    def get_latest_running_job(self) -> Optional[TrainingJob]:
        """Return the most recently created running job, or ``None``."""
        running = [
            j for j in self._jobs.values() if j.status == "running"
        ]
        if running:
            return max(running, key=lambda j: j.created_at)
        return None

    def shutdown(self) -> None:
        """Signal all workers to stop.  Non-blocking."""
        self._running = False
        logger.info("JobManager shutdown requested")

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        """Launch background worker thread(s)."""
        for i in range(self._max_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"training-worker-{i}",
                daemon=True,
            )
            t.start()

    def _worker_loop(self) -> None:
        """Background loop: pick jobs from queue and execute them."""
        while self._running:
            try:
                job, run_fn = self._job_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            job.status = "running"
            logger.info("Job %s started", job.job_id)

            try:
                result = run_fn(job)
                job.result = result
                if job.status != "stopped":
                    job.status = "completed"
                logger.info("Job %s completed", job.job_id)
            except Exception:
                tb = traceback.format_exc()
                job.error = tb
                job.status = "failed"
                logger.error("Job %s failed:\n%s", job.job_id, tb)
            finally:
                # Push a sentinel so SSE consumers know the stream is done
                from .progress import TrainingProgress

                final_progress = TrainingProgress(
                    status=job.status,
                    message=job.error or "Training finished",
                )
                job.progress_queue.put(final_progress)
