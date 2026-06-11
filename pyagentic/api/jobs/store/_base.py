"""
Abstract base class for durable job stores.

A JobStore is the rendezvous point of the job system: every execution
backend reports progress by appending sequenced updates here, and the SSE
streaming layer only ever tails the store (plus an optional live channel).
"""

from abc import ABC, abstractmethod
from typing import Optional

from pyagentic.api.jobs._models import (
    JobRecord,
    JobStatus,
    JobUpdate,
)


class JobStore(ABC):
    """Durable storage for job records and their append-only update logs.

    Implementations must make :meth:`append_update` idempotent on
    ``(job_id, seq)`` so that at-least-once transports (e.g. SQS) can feed
    the store safely.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the store for use (create schema, open connections).

        Must be safe to call more than once.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the store."""

    @abstractmethod
    async def create_job(self, job: JobRecord) -> None:
        """Persist a new job record.

        Args:
            job (JobRecord): The record to persist.
        """

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a job record by ID.

        Args:
            job_id (str): The job identifier.

        Returns:
            Optional[JobRecord]: The record, or None if unknown.
        """

    @abstractmethod
    async def list_jobs(
        self,
        *,
        status: Optional[JobStatus] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List job records, newest first, with optional filters.

        Args:
            status (Optional[JobStatus]): Filter by status.
            session_id (Optional[str]): Filter by session.
            limit (int): Maximum number of records returned.
            offset (int): Number of records to skip.

        Returns:
            list[JobRecord]: Matching records ordered by created_at descending.
        """

    @abstractmethod
    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
    ) -> None:
        """Update a job's status and optional terminal fields.

        Args:
            job_id (str): The job identifier.
            status (JobStatus): New status.
            result_json (Optional[str]): Final result JSON when succeeding.
            error (Optional[str]): Failure description when failing.
            started_at (Optional[float]): Execution start timestamp.
            finished_at (Optional[float]): Terminal transition timestamp.
        """

    @abstractmethod
    async def append_update(
        self, job_id: str, seq: int, event: str, payload_json: str
    ) -> bool:
        """Append one update to a job's log, idempotently on ``(job_id, seq)``.

        Args:
            job_id (str): The job identifier.
            seq (int): Monotonic per-job sequence number.
            event (str): SSE event name.
            payload_json (str): Exact JSON payload string.

        Returns:
            bool: True if newly inserted, False if a duplicate was ignored.
        """

    @abstractmethod
    async def load_updates(
        self, job_id: str, since: int = -1, limit: Optional[int] = None
    ) -> list[JobUpdate]:
        """Load a job's updates with seq greater than the given cursor.

        Args:
            job_id (str): The job identifier.
            since (int): Exclusive seq cursor; -1 loads from the beginning.
            limit (Optional[int]): Maximum number of updates returned.

        Returns:
            list[JobUpdate]: Updates ordered by seq ascending.
        """

    @abstractmethod
    async def max_seq(self, job_id: str) -> int:
        """Return the highest persisted seq for a job, or -1 if none.

        Args:
            job_id (str): The job identifier.

        Returns:
            int: Highest seq, or -1 when the job has no updates.
        """

    @abstractmethod
    async def jobs_by_status(self, status: JobStatus) -> list[JobRecord]:
        """Return all jobs currently in the given status.

        Args:
            status (JobStatus): Status to match.

        Returns:
            list[JobRecord]: Matching records.
        """

    @abstractmethod
    async def purge_expired(self, ttl_seconds: int) -> int:
        """Delete terminal jobs that finished more than ttl_seconds ago.

        Args:
            ttl_seconds (int): Retention period in seconds.

        Returns:
            int: Number of job records deleted.
        """
