import asyncio
import json

import pytest

from pyagentic import BaseAgent
from pyagentic.api._config import JobsConfig
from pyagentic.api._sessions import SessionManager
from pyagentic.api.jobs import (
    JobOrchestrator,
    JobRecord,
    JobStatus,
    SQLiteJobStore,
)
from pyagentic.api.jobs.backends._in_process import InProcessBackend

_MODEL = "_mock::test-model"


class _OrchTestAgent(BaseAgent):
    __system_message__ = "Orchestrator test agent"
    __input_template__ = ""


def _build(tmp_path, **jobs_overrides):
    """Build an orchestrator over an isolated SQLite store + in-memory sessions."""
    path = str(tmp_path / "jobs.db")
    store = SQLiteJobStore(path)
    sessions = SessionManager(_OrchTestAgent, default_model=_MODEL)
    backend = InProcessBackend(_OrchTestAgent, sessions, default_model=_MODEL)
    config = JobsConfig(store=path, **jobs_overrides)
    return JobOrchestrator(store, backend, config, sessions), sessions


@pytest.mark.asyncio
async def test_submit_succeeds_with_gapless_seq(tmp_path):
    """Test a sessionless job runs to success with a gapless update log."""
    orchestrator, _ = _build(tmp_path)
    job = await orchestrator.submit({"user_input": "hello"})
    record = await orchestrator.wait(job.job_id)

    assert record.status == JobStatus.SUCCEEDED
    assert record.result_json is not None
    assert "user said" in record.result_json

    updates = await orchestrator.store.load_updates(job.job_id)
    assert [u.seq for u in updates] == list(range(len(updates)))
    assert updates[-1].event == "agent_response"


@pytest.mark.asyncio
async def test_session_bound_job_uses_session_agent(tmp_path):
    """Test a session-bound job runs against the session's live agent instance."""
    orchestrator, sessions = _build(tmp_path)
    await orchestrator.ensure_started()
    sid = sessions.create()
    job = await orchestrator.submit({"user_input": "hi"}, session_id=sid)
    record = await orchestrator.wait(job.job_id)
    assert record.status == JobStatus.SUCCEEDED
    assert record.session_id == sid


@pytest.mark.asyncio
async def test_session_jobs_run_fifo(tmp_path):
    """Test two jobs on one session run strictly one after the other."""
    orchestrator, sessions = _build(tmp_path)
    await orchestrator.ensure_started()
    sid = sessions.create()
    job1 = await orchestrator.submit({"user_input": "first"}, session_id=sid)
    job2 = await orchestrator.submit({"user_input": "second"}, session_id=sid)
    rec1 = await orchestrator.wait(job1.job_id)
    rec2 = await orchestrator.wait(job2.job_id)
    assert rec1.status == JobStatus.SUCCEEDED
    assert rec2.status == JobStatus.SUCCEEDED
    # FIFO: the second job started only after the first finished.
    assert rec2.started_at >= rec1.finished_at


@pytest.mark.asyncio
async def test_unknown_session_job_fails(tmp_path):
    """Test a job naming a missing session is durably marked failed."""
    orchestrator, _ = _build(tmp_path)
    job = await orchestrator.submit({"user_input": "x"}, session_id="ghost")
    record = await orchestrator.wait(job.job_id)
    assert record.status == JobStatus.FAILED
    updates = await orchestrator.store.load_updates(job.job_id)
    assert updates[-1].event == "job_failed"


@pytest.mark.asyncio
async def test_admission_cap_queues_excess(tmp_path):
    """Test jobs beyond the admission cap stay queued until a slot frees."""
    orchestrator, _ = _build(tmp_path, admission_cap=1)
    release = asyncio.Event()

    real_submit = orchestrator.backend.submit

    async def slow_submit(job, emit, session_id=None):
        await release.wait()
        return await real_submit(job, emit, session_id=session_id)

    orchestrator.backend.submit = slow_submit

    job1 = await orchestrator.submit({"user_input": "a"})
    job2 = await orchestrator.submit({"user_input": "b"})
    await asyncio.sleep(0.05)

    rec1 = await orchestrator.store.get_job(job1.job_id)
    rec2 = await orchestrator.store.get_job(job2.job_id)
    assert rec1.status == JobStatus.RUNNING
    assert rec2.status == JobStatus.QUEUED

    release.set()
    assert (await orchestrator.wait(job1.job_id)).status == JobStatus.SUCCEEDED
    assert (await orchestrator.wait(job2.job_id)).status == JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_queued_job(tmp_path):
    """Test cancelling a queued job records a terminal cancelled update."""
    orchestrator, _ = _build(tmp_path, admission_cap=1)
    block = asyncio.Event()

    real_submit = orchestrator.backend.submit

    async def blocking_submit(job, emit, session_id=None):
        await block.wait()
        return await real_submit(job, emit, session_id=session_id)

    orchestrator.backend.submit = blocking_submit

    job1 = await orchestrator.submit({"user_input": "a"})
    job2 = await orchestrator.submit({"user_input": "b"})
    await asyncio.sleep(0.05)

    record = await orchestrator.cancel(job2.job_id)
    assert record.status == JobStatus.CANCELLED
    updates = await orchestrator.store.load_updates(job2.job_id)
    assert updates[-1].event == "job_cancelled"

    block.set()
    assert (await orchestrator.wait(job1.job_id)).status == JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_cancel_running_job(tmp_path):
    """Test cancelling a running job cancels its backend task."""
    orchestrator, _ = _build(tmp_path)
    started = asyncio.Event()

    real_execute = orchestrator.backend._execute

    async def hanging_execute(job, emit, session_id=None):
        started.set()
        await asyncio.sleep(60)
        return await real_execute(job, emit, session_id)

    orchestrator.backend._execute = hanging_execute

    job = await orchestrator.submit({"user_input": "x"})
    await asyncio.wait_for(started.wait(), timeout=2)
    record = await orchestrator.cancel(job.job_id)
    final = await orchestrator.wait(job.job_id)
    assert final.status == JobStatus.CANCELLED
    updates = await orchestrator.store.load_updates(job.job_id)
    assert updates[-1].event == "job_cancelled"
    assert record.status in (JobStatus.RUNNING, JobStatus.CANCELLED)


@pytest.mark.asyncio
async def test_cancel_terminal_job_raises(tmp_path):
    """Test cancelling a terminal job raises ValueError."""
    orchestrator, _ = _build(tmp_path)
    job = await orchestrator.submit({"user_input": "x"})
    await orchestrator.wait(job.job_id)
    with pytest.raises(ValueError):
        await orchestrator.cancel(job.job_id)


@pytest.mark.asyncio
async def test_cancel_unknown_job_raises(tmp_path):
    """Test cancelling an unknown job raises KeyError."""
    orchestrator, _ = _build(tmp_path)
    await orchestrator.ensure_started()
    with pytest.raises(KeyError):
        await orchestrator.cancel("nope")


@pytest.mark.asyncio
async def test_recover_marks_running_failed_and_requeues_queued(tmp_path):
    """Test recover() fails orphaned running jobs and re-runs queued ones."""
    orchestrator, _ = _build(tmp_path)
    await orchestrator.ensure_started()

    # Pre-seed records as a crashed process would have left them.
    orphan = JobRecord(job_id="orphan", status=JobStatus.RUNNING, request={})
    await orchestrator.store.create_job(orphan)
    await orchestrator.store.append_update("orphan", 0, "llm_response", "{}")
    queued = JobRecord(
        job_id="queued1", status=JobStatus.QUEUED, request={"user_input": "go"}
    )
    await orchestrator.store.create_job(queued)

    await orchestrator.recover()

    orphan_rec = await orchestrator.store.get_job("orphan")
    assert orphan_rec.status == JobStatus.FAILED
    assert "restart" in orphan_rec.error
    orphan_updates = await orchestrator.store.load_updates("orphan")
    assert orphan_updates[-1].event == "job_failed"
    assert orphan_updates[-1].seq == 1  # appended after the existing seq 0

    requeued = await orchestrator.wait("queued1")
    assert requeued.status == JobStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_ingest_idempotent_and_notifies(tmp_path):
    """Test ingest() deduplicates on seq and only wakes subscribers on insert."""
    orchestrator, _ = _build(tmp_path)
    await orchestrator.ensure_started()
    await orchestrator.store.create_job(JobRecord(job_id="remote", request={}))

    wakeup = orchestrator.subscribe("remote")
    assert await orchestrator.ingest("remote", 0, "llm_response", "{}") is True
    assert wakeup.is_set()
    wakeup.clear()
    assert await orchestrator.ingest("remote", 0, "llm_response", "{}") is False
    assert not wakeup.is_set()


@pytest.mark.asyncio
async def test_ingest_terminal_does_not_overwrite_terminal(tmp_path):
    """Test a late terminal cannot overwrite an existing terminal."""
    orchestrator, _ = _build(tmp_path)
    await orchestrator.ensure_started()
    await orchestrator.store.create_job(JobRecord(job_id="r2", request={}))
    await orchestrator.store.update_status("r2", JobStatus.CANCELLED)

    await orchestrator.ingest(
        "r2",
        0,
        "agent_response",
        json.dumps({"final_output": "late"}),
        terminal_status=JobStatus.SUCCEEDED,
        result_json="{}",
    )
    record = await orchestrator.store.get_job("r2")
    assert record.status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_purge_loop_purges(tmp_path):
    """Test the TTL loop purges expired terminal jobs."""
    orchestrator, _ = _build(tmp_path, ttl=1, cleanup_interval_seconds=1)
    await orchestrator.ensure_started()
    old = JobRecord(job_id="old", status=JobStatus.SUCCEEDED, request={})
    await orchestrator.store.create_job(old)
    await orchestrator.store.update_status("old", JobStatus.SUCCEEDED, finished_at=0.0)

    task = asyncio.create_task(orchestrator.purge_loop())
    try:
        for _ in range(40):
            await asyncio.sleep(0.05)
            if await orchestrator.store.get_job("old") is None:
                break
        assert await orchestrator.store.get_job("old") is None
    finally:
        task.cancel()
