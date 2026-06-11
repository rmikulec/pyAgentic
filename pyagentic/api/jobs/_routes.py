"""
HTTP routes for the job system: the canonical /jobs resource and the SSE
streaming responder implementing replay-from-cursor + live tail. The store
is the responder's only read path; live activity just wakes it up.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import create_model

from pyagentic._base._agent._agent import BaseAgent
from pyagentic.api._sessions import SessionManager
from pyagentic.api.jobs._models import (
    JobListResponse,
    JobRecord,
    JobSnapshot,
    JobStatus,
    JobSubmitResponse,
    JobUpdatesResponse,
    TERMINAL_EVENTS,
)
from pyagentic.api.jobs._orchestrator import JobOrchestrator

logger = logging.getLogger(__name__)

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

# How long the live tail waits on the notifier before re-checking the store
# for a terminal record (covers edge cases where no notification arrives).
_TAIL_POLL_SECONDS = 15.0


def _snapshot(record: JobRecord, include_result: bool = True) -> dict:
    """Build the API representation of a job record."""
    body = {
        "job_id": record.job_id,
        "session_id": record.session_id,
        "status": record.status.value,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
    }
    if include_result:
        body["result"] = record.result() if record.is_terminal else None
    return body


def _format_frame(seq: int, event: str, payload_json: str) -> str:
    """Format one SSE frame carrying the update's seq as its event id."""
    return f"id: {seq}\nevent: {event}\ndata: {payload_json}\n\n"


async def job_event_stream(orchestrator: JobOrchestrator, job_id: str, cursor: int):
    """Yield a job's updates as SSE frames: replay from cursor, then tail.

    The store is the single read path. Each pass emits the contiguous
    prefix of persisted updates past the high-water mark (``last_emitted``),
    then waits on the orchestrator's wakeup signal for more. A hole in the
    log (out-of-order arrival over an at-least-once transport) simply pauses
    emission until the missing update lands; duplicates never reach the
    responder because the store is idempotent on (job_id, seq). The stream
    closes after emitting a terminal event.

    Args:
        orchestrator (JobOrchestrator): The orchestrator owning the job.
        job_id (str): The job identifier.
        cursor (int): Exclusive seq to resume after; -1 for the beginning.

    Yields:
        str: SSE-formatted frames (``id:``/``event:``/``data:`` lines).
    """
    last_emitted = cursor
    # Subscribe BEFORE the first read so nothing falls between replay and tail.
    wakeup = orchestrator.subscribe(job_id)
    try:
        closing = False
        while True:
            wakeup.clear()
            updates = await orchestrator.store.load_updates(job_id, since=last_emitted)
            for update in updates:
                if update.seq != last_emitted + 1:
                    break  # hole: wait for the missing seq to land
                yield _format_frame(update.seq, update.event, update.payload_json)
                last_emitted = update.seq
                if update.event in TERMINAL_EVENTS:
                    return
            if closing:
                return
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=_TAIL_POLL_SECONDS)
            except asyncio.TimeoutError:
                # No live traffic: if the record went terminal without a
                # notification reaching us, emit one final pass and close.
                record = await orchestrator.store.get_job(job_id)
                if record is None or record.is_terminal:
                    closing = True
    finally:
        orchestrator.unsubscribe(job_id, wakeup)


def _resolve_cursor(request: Request, since: Optional[int]) -> int:
    """Resolve the stream cursor from Last-Event-ID, ?since=, or -1."""
    header = request.headers.get("last-event-id")
    if header is not None:
        try:
            return int(header)
        except ValueError:
            pass
    if since is not None:
        return since
    return -1


def build_jobs_router(
    orchestrator: JobOrchestrator,
    agent_class: type[BaseAgent],
    sessions: SessionManager,
) -> APIRouter:
    """Build the canonical /jobs router for an agent app.

    Args:
        orchestrator (JobOrchestrator): The app's job orchestrator.
        agent_class (type[BaseAgent]): The served agent class (provides the
            metaclass-generated request model for submissions).
        sessions (SessionManager): Live session registry for validating
            session-bound submissions.

    Returns:
        APIRouter: Router exposing submit/get/list/updates/stream/cancel.
    """
    router = APIRouter()
    RequestModel = agent_class.__request_model__
    ResponseModel = agent_class.__response_model__

    SubmitModel = create_model(
        f"{agent_class.__name__}JobSubmitRequest",
        input=(RequestModel, ...),
        session_id=(Optional[str], None),
    )

    # Narrow the loosely-typed JobSnapshot.result to this agent's response model
    # so the get/cancel endpoints document the exact result shape.
    SnapshotModel = create_model(
        f"{agent_class.__name__}JobSnapshot",
        __base__=JobSnapshot,
        result=(Optional[ResponseModel], None),
    )

    @router.post("/jobs", status_code=202, response_model=JobSubmitResponse)
    async def submit_job(req: SubmitModel) -> dict:  # type: ignore[valid-type]
        """Submit an agent run as an async job."""
        await orchestrator.ensure_started()
        if req.session_id is not None and not sessions.exists(req.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        request = {
            field: getattr(req.input, field) for field in type(req.input).model_fields
        }
        job = await orchestrator.submit(request, session_id=req.session_id)
        return {"job_id": job.job_id, "status": job.status.value}

    @router.get("/jobs", response_model=JobListResponse)
    async def list_jobs(
        status: Optional[JobStatus] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """List jobs, newest first, with optional status/session filters."""
        await orchestrator.ensure_started()
        records = await orchestrator.store.list_jobs(
            status=status, session_id=session_id, limit=limit, offset=offset
        )
        return {"jobs": [_snapshot(r, include_result=False) for r in records]}

    @router.get("/jobs/{job_id}", response_model=SnapshotModel)
    async def get_job(job_id: str) -> dict:
        """Return a job's status and, when terminal, its result."""
        await orchestrator.ensure_started()
        record = await orchestrator.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _snapshot(record)

    @router.get("/jobs/{job_id}/updates", response_model=JobUpdatesResponse)
    async def get_updates(job_id: str, since: int = -1) -> dict:
        """Return a job's update log past the given seq cursor.

        Note: for jobs whose updates arrive over an at-least-once transport
        (e.g. the AWS SQS drain), this raw view may transiently expose seq
        holes; the /stream endpoint enforces contiguity.
        """
        await orchestrator.ensure_started()
        record = await orchestrator.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        updates = await orchestrator.store.load_updates(job_id, since=since)
        return {
            "job_id": job_id,
            "status": record.status.value,
            "updates": [
                {"seq": u.seq, "event": u.event, "data": u.data} for u in updates
            ],
        }

    @router.get(
        "/jobs/{job_id}/stream",
        responses={
            200: {
                "description": (
                    "SSE stream of job updates. Each frame is "
                    "`id: <seq>\\nevent: <type>\\ndata: <json>`. Reconnect "
                    "with the Last-Event-ID header (or ?since=<seq>) to "
                    "resume after the last received seq."
                ),
            }
        },
    )
    async def stream_job(
        job_id: str, request: Request, since: Optional[int] = None
    ) -> StreamingResponse:
        """Stream a job's updates: replay from cursor, then live tail."""
        await orchestrator.ensure_started()
        record = await orchestrator.store.get_job(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        cursor = _resolve_cursor(request, since)
        return StreamingResponse(
            job_event_stream(orchestrator, job_id, cursor),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @router.post("/jobs/{job_id}/cancel", response_model=SnapshotModel)
    async def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job."""
        await orchestrator.ensure_started()
        try:
            record = await orchestrator.cancel(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return _snapshot(record)

    return router
