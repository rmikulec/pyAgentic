"""
Data models for the async job system: job records, update log entries,
backend health, and shared helpers.
"""

import json
import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states of a job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)

# SSE event names that end a job's update stream.
TERMINAL_EVENTS = frozenset({"agent_response", "job_failed", "job_cancelled"})


class JobRecord(BaseModel):
    """Durable record of a single agent run.

    Attributes:
        job_id (str): Unique job identifier.
        session_id (Optional[str]): Session the job belongs to, if any.
        status (JobStatus): Current lifecycle state.
        request (dict): The submitted agent input kwargs.
        result_json (Optional[str]): Final agent response as a raw JSON
            string, set when the job succeeds.
        error (Optional[str]): Failure description, set when the job fails.
        created_at (float): Unix timestamp of submission.
        started_at (Optional[float]): Unix timestamp when execution began.
        finished_at (Optional[float]): Unix timestamp of terminal transition.
    """

    job_id: str
    session_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    request: dict = Field(default_factory=dict)
    result_json: Optional[str] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def result(self) -> Optional[Any]:
        """Parse and return the final result payload.

        Returns:
            Optional[Any]: The parsed result JSON, or None if not terminal.
        """
        if self.result_json is None:
            return None
        return json.loads(self.result_json)

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal status."""
        return self.status in TERMINAL_STATUSES


class JobUpdate(BaseModel):
    """One entry in a job's append-only update log.

    Attributes:
        job_id (str): Job the update belongs to.
        seq (int): Monotonic per-job sequence number, assigned once at
            production time by the run's owner.
        event (str): SSE event name (``llm_response``, ``tool_response``,
            ``agent_response``, ``job_failed``, ``job_cancelled``).
        payload_json (str): The exact JSON string emitted on the wire.
        created_at (float): Unix timestamp the update was persisted.
    """

    job_id: str
    seq: int
    event: str
    payload_json: str
    created_at: float = Field(default_factory=time.time)

    @property
    def data(self) -> Any:
        """Parse and return the update payload."""
        return json.loads(self.payload_json)

    @property
    def is_terminal(self) -> bool:
        """Whether this update ends the job's stream."""
        return self.event in TERMINAL_EVENTS


class RunOutcome(BaseModel):
    """What a backend hands back when it completes a run locally.

    Attributes:
        result_json (Optional[str]): The final AgentResponse as raw JSON.
    """

    result_json: Optional[str] = None


class BackendHealth(BaseModel):
    """Health snapshot reported by an execution backend.

    Attributes:
        ok (bool): Whether the backend considers itself healthy.
        detail (dict): Backend-specific diagnostics (concurrency counts,
            last poll timestamps, etc.).
    """

    ok: bool = True
    detail: dict = Field(default_factory=dict)


def _build_prompt(request: dict) -> str:
    """Collapse agent request kwargs into the single prompt string step() accepts.

    Mirrors the historical /chat/stream behavior: a single non-None field is
    stringified directly; multiple fields are serialized to JSON to preserve
    structure.
    """
    non_none = {k: v for k, v in request.items() if v is not None}
    if len(non_none) == 1:
        return str(next(iter(non_none.values())))
    return json.dumps(non_none, default=str)
