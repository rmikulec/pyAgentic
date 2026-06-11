"""
Job orchestrator: the coordination layer between the HTTP routes, the
durable JobStore, and the pluggable ExecutionBackend.

Responsibilities:
  - admission control (one coarse in-flight cap for the whole server process)
  - per-session FIFO serialization (at most one running job per session)
  - seq authority for locally-executed jobs (monotonic, gapless, assigned
    once at production time, before fan-out to store + live subscribers)
  - ingest path for out-of-band updates (idempotent on (job_id, seq))
  - validating that a session-bound job's session still exists before it runs
  - startup recovery, TTL cleanup, and cancellation
"""

import asyncio
import json
import time
import uuid
import logging
from typing import TYPE_CHECKING, Optional

from pyagentic.api._config import JobsConfig
from pyagentic.api.jobs._models import (
    JobRecord,
    JobStatus,
)
from pyagentic.api.jobs.backends._base import ExecutionBackend
from pyagentic.api.jobs.store._base import JobStore

if TYPE_CHECKING:
    from pyagentic.api._sessions import SessionManager

logger = logging.getLogger(__name__)


class JobOrchestrator:
    """Coordinates job submission, execution, streaming, and recovery."""

    def __init__(
        self,
        store: JobStore,
        backend: ExecutionBackend,
        config: JobsConfig,
        sessions: "SessionManager",
    ) -> None:
        """Create the orchestrator.

        Args:
            store (JobStore): Durable job record + update log store.
            backend (ExecutionBackend): Executor for agent runs.
            config (JobsConfig): The ``[jobs]`` configuration (admission cap,
                TTL, cleanup interval).
            sessions (SessionManager): Live session registry, used to validate
                session-bound submissions.
        """
        self.store = store
        self.backend = backend
        self._jobs_cfg = config
        self._sessions = sessions
        self._admission = asyncio.Semaphore(config.admission_cap)

        self._session_queues: dict[str, asyncio.Queue] = {}
        self._session_workers: dict[str, asyncio.Task] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}

        self._next_seq: dict[str, int] = {}
        self._subscribers: dict[str, list[asyncio.Event]] = {}
        self._done_events: dict[str, asyncio.Event] = {}
        self._cancel_requested: set[str] = set()
        self._started = False
        self._start_lock = asyncio.Lock()

    # ---- lifecycle ----

    async def ensure_started(self) -> None:
        """Initialize the store exactly once. Safe to call repeatedly."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self.store.initialize()
            self._started = True

    async def recover(self) -> None:
        """Reconcile job records after a restart.

        In-flight ``running`` jobs are marked failed (execution is
        ephemeral); ``queued`` jobs are re-enqueued through normal routing.
        """
        await self.ensure_started()
        for job in await self.store.jobs_by_status(JobStatus.RUNNING):
            seq = await self.store.max_seq(job.job_id) + 1
            error = "orphaned by server restart"
            await self.store.append_update(
                job.job_id,
                seq,
                "job_failed",
                json.dumps({"job_id": job.job_id, "error": error}),
            )
            await self.store.update_status(
                job.job_id,
                JobStatus.FAILED,
                error=error,
                finished_at=time.time(),
            )
        for job in await self.store.jobs_by_status(JobStatus.QUEUED):
            self._next_seq[job.job_id] = await self.store.max_seq(job.job_id) + 1
            self._done_events[job.job_id] = asyncio.Event()
            self._dispatch(job)

    async def purge_loop(self) -> None:
        """Periodically delete expired terminal job records. Runs forever."""
        jobs_cfg = self._jobs_cfg
        while True:
            await asyncio.sleep(jobs_cfg.cleanup_interval_seconds)
            try:
                purged = await self.store.purge_expired(jobs_cfg.ttl_seconds)
                if purged:
                    logger.info("Purged %d expired job records", purged)
            except Exception:  # noqa: BLE001 - keep the loop alive
                logger.exception("Job TTL cleanup failed")

    async def shutdown(self) -> None:
        """Cancel workers and in-flight tasks, then close the backend."""
        tasks = [*self._session_workers.values(), *self._job_tasks.values()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._session_workers.clear()
        self._job_tasks.clear()
        await self.backend.shutdown()

    # ---- submission & execution ----

    async def submit(
        self, request: dict, session_id: Optional[str] = None
    ) -> JobRecord:
        """Create a durable job record and route it for execution.

        Session-bound jobs are enqueued FIFO behind other jobs of the same
        session; sessionless jobs start immediately (subject to the
        admission cap). Never blocks on capacity — over-cap jobs simply
        remain ``queued``.

        Args:
            request (dict): Agent input kwargs.
            session_id (Optional[str]): Session to bind the job to.

        Returns:
            JobRecord: The persisted record in its initial queued state.
        """
        await self.ensure_started()
        job = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            session_id=session_id,
            status=JobStatus.QUEUED,
            request=request,
        )
        await self.store.create_job(job)
        self._next_seq[job.job_id] = 0
        self._done_events[job.job_id] = asyncio.Event()
        self._dispatch(job)
        return job

    def _dispatch(self, job: JobRecord) -> None:
        """Route a job to its session queue or a dedicated task."""
        if job.session_id is None:
            task = asyncio.create_task(self._run_one(job))
            self._job_tasks[job.job_id] = task
            task.add_done_callback(
                lambda _t, jid=job.job_id: self._job_tasks.pop(jid, None)
            )
        else:
            queue = self._session_queues.get(job.session_id)
            if queue is None:
                queue = asyncio.Queue()
                self._session_queues[job.session_id] = queue
                self._session_workers[job.session_id] = asyncio.create_task(
                    self._session_worker(job.session_id, queue)
                )
            queue.put_nowait(job)

    async def _session_worker(self, session_id: str, queue: asyncio.Queue) -> None:
        """Drain one session's job queue strictly in FIFO order."""
        while True:
            job = await queue.get()
            try:
                await self._run_one(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - _run_one records failures
                logger.exception("Session worker error for job %s", job.job_id)
            finally:
                queue.task_done()

    async def _run_one(self, job: JobRecord) -> None:
        """Execute a single job through the backend, recording the outcome."""
        if job.job_id in self._cancel_requested:
            return
        async with self._admission:
            if job.job_id in self._cancel_requested:
                return
            emit = self.emit_for(job)

            # Validate the session still exists (the live agent lives in the
            # in-memory SessionManager; the backend resolves it at run time).
            if job.session_id is not None and not self._sessions.exists(job.session_id):
                error = f"Session not found: {job.session_id}"
                await emit(
                    "job_failed",
                    json.dumps({"job_id": job.job_id, "error": error}),
                )
                await self._finalize(job.job_id, JobStatus.FAILED, error=error)
                return

            await self.store.update_status(
                job.job_id, JobStatus.RUNNING, started_at=time.time()
            )
            try:
                outcome = await self.backend.submit(
                    job, emit, session_id=job.session_id
                )
            except asyncio.CancelledError:
                if job.job_id in self._cancel_requested:
                    # Job-level cancellation: record it, keep the worker alive.
                    await emit(
                        "job_cancelled",
                        json.dumps({"job_id": job.job_id, "status": "cancelled"}),
                    )
                    await self._finalize(job.job_id, JobStatus.CANCELLED)
                    return
                raise
            except Exception as exc:  # noqa: BLE001 - all failures are recorded
                logger.exception("Job %s failed", job.job_id)
                await emit(
                    "job_failed",
                    json.dumps({"job_id": job.job_id, "error": str(exc)}),
                )
                await self._finalize(job.job_id, JobStatus.FAILED, error=str(exc))
                return

            if outcome is not None:
                await self._finalize(
                    job.job_id, JobStatus.SUCCEEDED, result_json=outcome.result_json
                )
                return

            # outcome None: completion will be reported out-of-band via ingest().
            # Hold the per-session FIFO slot until the terminal update lands.
            if job.session_id is not None:
                done = self._done_events.get(job.job_id)
                if done is not None:
                    await done.wait()

    async def _finalize(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write a terminal status and release in-memory job state."""
        await self.store.update_status(
            job_id,
            status,
            result_json=result_json,
            error=error,
            finished_at=time.time(),
        )
        event = self._done_events.get(job_id)
        if event is not None:
            event.set()
        self._next_seq.pop(job_id, None)
        self._cancel_requested.discard(job_id)

    # ---- update fan-out ----

    def emit_for(self, job: JobRecord):
        """Build the seq-assigning update sink for a locally-executed job.

        The returned coroutine function persists the update to the store and
        then wakes live subscribers, who re-read the store. seq is assigned
        here — once, at production time — so the store's log is the single
        authoritative copy of every update.

        Args:
            job (JobRecord): The job the sink belongs to.

        Returns:
            Callable: ``async emit(event, payload_json) -> int`` returning
            the assigned seq.
        """
        job_id = job.job_id

        async def emit(event: str, payload_json: str) -> int:
            seq = self._next_seq.get(job_id, 0)
            self._next_seq[job_id] = seq + 1
            await self.store.append_update(job_id, seq, event, payload_json)
            self._notify(job_id)
            return seq

        return emit

    async def ingest(
        self,
        job_id: str,
        seq: int,
        event: str,
        payload_json: str,
        *,
        terminal_status: Optional[JobStatus] = None,
        result_json: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Ingest an out-of-band update (e.g. an advisory cancellation).

        Idempotent on ``(job_id, seq)``: redelivered updates are dropped and
        subscribers are only notified on first insert. Terminal updates for
        jobs that are already terminal do not overwrite the existing status.

        Args:
            job_id (str): The job identifier.
            seq (int): Producer-assigned sequence number.
            event (str): SSE event name.
            payload_json (str): Exact JSON payload string.
            terminal_status (Optional[JobStatus]): Terminal status carried
                by this update, if any.
            result_json (Optional[str]): Final result JSON for succeeded.
            error (Optional[str]): Failure description for failed.

        Returns:
            bool: True if the update was newly inserted.
        """
        await self.ensure_started()
        inserted = await self.store.append_update(job_id, seq, event, payload_json)
        if inserted:
            self._notify(job_id)
        if terminal_status is not None:
            record = await self.store.get_job(job_id)
            if record is not None and not record.is_terminal:
                await self._finalize(
                    job_id, terminal_status, result_json=result_json, error=error
                )
        return inserted

    def subscribe(self, job_id: str) -> asyncio.Event:
        """Register a live wakeup signal for a job.

        Args:
            job_id (str): The job identifier.

        Returns:
            asyncio.Event: Set whenever a new update for the job lands in
            the store. The subscriber clears it and re-reads the store —
            the store is the only read path for update payloads.
        """
        event = asyncio.Event()
        self._subscribers.setdefault(job_id, []).append(event)
        return event

    def unsubscribe(self, job_id: str, event: asyncio.Event) -> None:
        """Remove a previously registered subscriber.

        Args:
            job_id (str): The job identifier.
            event (asyncio.Event): The event returned by :meth:`subscribe`.
        """
        subscribers = self._subscribers.get(job_id)
        if subscribers is None:
            return
        try:
            subscribers.remove(event)
        except ValueError:
            pass
        if not subscribers:
            self._subscribers.pop(job_id, None)

    def _notify(self, job_id: str) -> None:
        """Wake all live subscribers of a job after a store append."""
        for event in self._subscribers.get(job_id, []):
            event.set()

    # ---- cancellation & waiting ----

    async def cancel(self, job_id: str) -> JobRecord:
        """Cancel a queued or running job.

        Args:
            job_id (str): The job identifier.

        Returns:
            JobRecord: The record after cancellation.

        Raises:
            KeyError: If the job is unknown.
            ValueError: If the job is already terminal.
        """
        await self.ensure_started()
        record = await self.store.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.is_terminal:
            raise ValueError(f"Job {job_id} is already {record.status.value}.")

        self._cancel_requested.add(job_id)
        if record.status == JobStatus.QUEUED:
            emit = self.emit_for(record)
            await emit(
                "job_cancelled",
                json.dumps({"job_id": job_id, "status": "cancelled"}),
            )
            await self._finalize(job_id, JobStatus.CANCELLED)
        else:
            cancelled = await self.backend.cancel(job_id)
            if not cancelled:
                # Advisory cancel (e.g. a fired Lambda cannot be stopped):
                # record the intent; ingest() will not overwrite the
                # terminal status if a late remote terminal arrives.
                await self.ingest(
                    job_id,
                    await self.store.max_seq(job_id) + 1,
                    "job_cancelled",
                    json.dumps({"job_id": job_id, "status": "cancelled"}),
                    terminal_status=JobStatus.CANCELLED,
                )
        return await self.store.get_job(job_id)

    async def wait(self, job_id: str) -> JobRecord:
        """Block until a job reaches a terminal status.

        Args:
            job_id (str): The job identifier.

        Returns:
            JobRecord: The terminal record.

        Raises:
            KeyError: If the job is unknown.
        """
        event = self._done_events.get(job_id)
        if event is not None:
            await event.wait()
            self._done_events.pop(job_id, None)
        else:
            # No in-memory event (e.g. submitted before a restart): poll.
            while True:
                record = await self.store.get_job(job_id)
                if record is None:
                    raise KeyError(job_id)
                if record.is_terminal:
                    break
                await asyncio.sleep(0.25)
        record = await self.store.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        return record
