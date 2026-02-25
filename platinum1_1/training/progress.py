"""Training progress tracking and SSE event formatting.

Provides a ``sanitize_for_json`` utility that recursively replaces
``inf``, ``-inf``, and ``NaN`` with ``None`` so that ``json.dumps``
never emits invalid JSON tokens (RFC 8259 compliance).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace inf/NaN with None for JSON serialization.

    Handles:
    - Python ``float`` (inf, -inf, nan)
    - NumPy scalars (``np.floating``, ``np.integer``)
    - NumPy arrays (converted to nested lists first)
    - Dicts and lists/tuples (recursed)
    - Everything else passed through unchanged

    Returns:
        A JSON-safe copy of *obj*.
    """
    if isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Progress dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    """Snapshot of training state at a given epoch.

    Used by the trainer to communicate progress to the ``JobManager`` and
    ultimately to the browser via Server-Sent Events.
    """

    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float("inf")
    learning_rate: float = 0.0
    elapsed_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "running"  # running | completed | failed | stopped
    message: str = ""

    def to_sse_event(self, event_type: str = "progress") -> str:
        """Format as a Server-Sent Event string.

        The output follows the SSE specification::

            event: progress
            data: {"epoch": 1, ...}

        An extra trailing newline terminates the event block so the browser
        ``EventSource`` recognises it as complete.

        Args:
            event_type: SSE event name (default ``"progress"``).

        Returns:
            SSE-formatted string ready to be written to a streaming response.
        """
        data = sanitize_for_json(
            {
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "best_val_loss": self.best_val_loss,
                "learning_rate": self.learning_rate,
                "elapsed_seconds": self.elapsed_seconds,
                "metrics": self.metrics,
                "status": self.status,
                "message": self.message,
            }
        )
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def to_dict(self) -> dict:
        """Return a JSON-safe dict representation."""
        return sanitize_for_json(
            {
                "epoch": self.epoch,
                "total_epochs": self.total_epochs,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "best_val_loss": self.best_val_loss,
                "learning_rate": self.learning_rate,
                "elapsed_seconds": self.elapsed_seconds,
                "metrics": self.metrics,
                "status": self.status,
                "message": self.message,
            }
        )
