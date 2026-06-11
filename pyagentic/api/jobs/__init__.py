"""
Durable async job execution for the PyAgentic API layer.

Every agent run submitted through the job system becomes a durable record
(status + gapless sequenced update log) in a pluggable :class:`JobStore`,
executed by a pluggable :class:`ExecutionBackend`. Clients submit a job and
poll or stream its updates later, surviving reconnects and restarts.
"""

from pyagentic.api.jobs._models import (
    BackendHealth,
    JobRecord,
    JobStatus,
    JobUpdate,
    TERMINAL_EVENTS,
    TERMINAL_STATUSES,
)
from pyagentic.api.jobs._orchestrator import JobOrchestrator
from pyagentic.api.jobs.store import JobStore, SQLiteJobStore, build_store
from pyagentic.api.jobs.backends import ExecutionBackend, build_backend

__all__ = [
    "BackendHealth",
    "ExecutionBackend",
    "JobOrchestrator",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "JobUpdate",
    "SQLiteJobStore",
    "TERMINAL_EVENTS",
    "TERMINAL_STATUSES",
    "build_backend",
    "build_store",
]
