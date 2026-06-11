"""
Abstract base class for execution backends.

A backend decides where and how a job's agent run executes (in the serve
process, in a local Docker container, on AWS Lambda, ...). Backends report
progress through the ``emit`` callable handed to :meth:`submit`, which
persists each update to the JobStore and notifies live subscribers — the
streaming layer never needs to know which backend ran the job.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

from pyagentic.api.jobs._models import (
    BackendHealth,
    JobRecord,
    RunOutcome,
)

# emit(event_name, payload_json) -> assigned seq
EmitFn = Callable[[str, str], Awaitable[int]]


class ExecutionBackend(ABC):
    """Pluggable executor for agent runs."""

    @abstractmethod
    async def submit(
        self,
        job: JobRecord,
        emit: EmitFn,
        session_id: Optional[str] = None,
    ) -> Optional[RunOutcome]:
        """Execute a job, reporting every update through ``emit``.

        For local backends this coroutine drives the run to completion and
        returns only when the job is terminal.

        Args:
            job (JobRecord): The job to execute.
            emit (EmitFn): Async callable persisting one update and
                notifying subscribers; returns the assigned seq.
            session_id (Optional[str]): The session this job is bound to, if
                any. The backend resolves the live agent for it.

        Returns:
            Optional[RunOutcome]: The result when the run completed locally.

        Raises:
            Exception: Any error raised marks the job failed.
        """

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a running job.

        Args:
            job_id (str): The job identifier.

        Returns:
            bool: True if the run was cancelled, False if cancellation is
            unsupported or advisory for this backend.
        """

    @abstractmethod
    async def health(self) -> BackendHealth:
        """Report backend health and diagnostics.

        Returns:
            BackendHealth: Current health snapshot.
        """

    async def shutdown(self) -> None:
        """Release backend resources. Called on app shutdown."""
        return None


class LocalExecutionBackend(ExecutionBackend):
    """Base for backends that run jobs as tracked asyncio tasks in-process.

    Owns the in-flight task registry and the submit/cancel mechanics;
    subclasses implement :meth:`_execute` (drive one run to completion,
    emitting every update) and :meth:`health`. Remote fire-and-forget
    backends (e.g. AWS Lambda) do not fit this shape and subclass
    :class:`ExecutionBackend` directly.
    """

    def __init__(self) -> None:
        """Initialize the in-flight task registry."""
        self._tasks: dict[str, asyncio.Task] = {}

    async def submit(
        self,
        job: JobRecord,
        emit: EmitFn,
        session_id: Optional[str] = None,
    ) -> Optional[RunOutcome]:
        """Run the job to completion as a tracked, cancellable task.

        Args:
            job (JobRecord): The job to execute.
            emit (EmitFn): Update sink (persists + wakes subscribers).
            session_id (Optional[str]): The session this job is bound to, if any.

        Returns:
            Optional[RunOutcome]: Result JSON when complete.

        Raises:
            asyncio.CancelledError: If the job is cancelled mid-run.
            Exception: Any execution error marks the job failed.
        """
        task = asyncio.create_task(self._execute(job, emit, session_id))
        self._tasks[job.job_id] = task
        try:
            return await task
        finally:
            self._tasks.pop(job.job_id, None)

    @abstractmethod
    async def _execute(
        self,
        job: JobRecord,
        emit: EmitFn,
        session_id: Optional[str],
    ) -> RunOutcome:
        """Drive one job to completion, emitting each update."""

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job's task.

        Args:
            job_id (str): The job identifier.

        Returns:
            bool: True if a running task was cancelled, False otherwise.
        """
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False
