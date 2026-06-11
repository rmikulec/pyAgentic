import asyncio
import time

import pytest
import pytest_asyncio

from pyagentic.api.jobs import JobRecord, JobStatus, SQLiteJobStore, build_store


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create an initialized SQLite store in a temp directory."""
    store = SQLiteJobStore(str(tmp_path / "jobs.db"))
    await store.initialize()
    yield store
    await store.close()


def _job(job_id="job1", **kwargs) -> JobRecord:
    return JobRecord(job_id=job_id, request={"user_input": "hi"}, **kwargs)


@pytest.mark.asyncio
async def test_initialize_creates_parent_dirs(tmp_path):
    """Test initialize() creates missing parent directories."""
    store = SQLiteJobStore(str(tmp_path / "nested" / "dir" / "jobs.db"))
    await store.initialize()
    assert (tmp_path / "nested" / "dir" / "jobs.db").exists()
    await store.close()


@pytest.mark.asyncio
async def test_initialize_idempotent(store):
    """Test initialize() can be called repeatedly."""
    await store.initialize()
    await store.initialize()


@pytest.mark.asyncio
async def test_create_and_get_job(store):
    """Test a created job round-trips through get_job."""
    job = _job(session_id="sess1")
    await store.create_job(job)
    fetched = await store.get_job("job1")
    assert fetched is not None
    assert fetched.job_id == "job1"
    assert fetched.session_id == "sess1"
    assert fetched.status == JobStatus.QUEUED
    assert fetched.request == {"user_input": "hi"}


@pytest.mark.asyncio
async def test_get_job_unknown(store):
    """Test get_job returns None for unknown IDs."""
    assert await store.get_job("nope") is None


@pytest.mark.asyncio
async def test_list_jobs_filters(store):
    """Test list_jobs filters by status and session."""
    await store.create_job(_job("a", session_id="s1"))
    await store.create_job(_job("b", session_id="s2"))
    await store.create_job(_job("c", session_id="s1"))
    await store.update_status("a", JobStatus.SUCCEEDED)

    assert {j.job_id for j in await store.list_jobs()} == {"a", "b", "c"}
    assert {j.job_id for j in await store.list_jobs(session_id="s1")} == {"a", "c"}
    assert {j.job_id for j in await store.list_jobs(status=JobStatus.QUEUED)} == {
        "b",
        "c",
    }
    assert {
        j.job_id
        for j in await store.list_jobs(status=JobStatus.QUEUED, session_id="s1")
    } == {"c"}


@pytest.mark.asyncio
async def test_list_jobs_limit_offset(store):
    """Test list_jobs respects limit and offset."""
    for i in range(5):
        await store.create_job(
            JobRecord(job_id=f"j{i}", request={}, created_at=time.time() + i)
        )
    page = await store.list_jobs(limit=2)
    assert len(page) == 2
    # Newest first
    assert page[0].job_id == "j4"
    page2 = await store.list_jobs(limit=2, offset=2)
    assert page2[0].job_id == "j2"


@pytest.mark.asyncio
async def test_update_status_terminal_fields(store):
    """Test update_status persists result, error, and timestamps."""
    await store.create_job(_job())
    await store.update_status(
        "job1",
        JobStatus.SUCCEEDED,
        result_json='{"final_output": "done"}',
        started_at=1.0,
        finished_at=2.0,
    )
    job = await store.get_job("job1")
    assert job.status == JobStatus.SUCCEEDED
    assert job.result() == {"final_output": "done"}
    assert job.started_at == 1.0
    assert job.finished_at == 2.0
    assert job.is_terminal


@pytest.mark.asyncio
async def test_append_update_idempotent(store):
    """Test append_update ignores duplicate (job_id, seq) pairs."""
    await store.create_job(_job())
    assert await store.append_update("job1", 0, "llm_response", "{}") is True
    assert await store.append_update("job1", 0, "llm_response", "{}") is False
    updates = await store.load_updates("job1")
    assert len(updates) == 1


@pytest.mark.asyncio
async def test_load_updates_since_cursor(store):
    """Test load_updates honors the exclusive since cursor."""
    await store.create_job(_job())
    for seq in range(4):
        await store.append_update("job1", seq, "llm_response", f'{{"n": {seq}}}')
    updates = await store.load_updates("job1", since=1)
    assert [u.seq for u in updates] == [2, 3]
    assert updates[0].data == {"n": 2}


@pytest.mark.asyncio
async def test_max_seq(store):
    """Test max_seq returns -1 when empty and the head otherwise."""
    await store.create_job(_job())
    assert await store.max_seq("job1") == -1
    await store.append_update("job1", 0, "e", "{}")
    await store.append_update("job1", 5, "e", "{}")
    assert await store.max_seq("job1") == 5


@pytest.mark.asyncio
async def test_jobs_by_status(store):
    """Test jobs_by_status returns matching records."""
    await store.create_job(_job("a"))
    await store.create_job(_job("b"))
    await store.update_status("a", JobStatus.RUNNING)
    running = await store.jobs_by_status(JobStatus.RUNNING)
    assert [j.job_id for j in running] == ["a"]


@pytest.mark.asyncio
async def test_purge_expired_cascades(store):
    """Test purge_expired deletes old terminal jobs and their updates."""
    await store.create_job(_job("old"))
    await store.append_update("old", 0, "e", "{}")
    await store.update_status("old", JobStatus.SUCCEEDED, finished_at=time.time() - 100)
    await store.create_job(_job("fresh"))
    await store.update_status("fresh", JobStatus.SUCCEEDED, finished_at=time.time())
    await store.create_job(_job("running"))
    await store.update_status("running", JobStatus.RUNNING)

    purged = await store.purge_expired(ttl_seconds=50)
    assert purged == 1
    assert await store.get_job("old") is None
    assert await store.load_updates("old") == []
    assert await store.get_job("fresh") is not None
    assert await store.get_job("running") is not None


@pytest.mark.asyncio
async def test_concurrent_appends(store):
    """Test concurrent appends are serialized safely."""
    await store.create_job(_job())
    await asyncio.gather(
        *(store.append_update("job1", seq, "e", "{}") for seq in range(20))
    )
    updates = await store.load_updates("job1")
    assert [u.seq for u in updates] == list(range(20))


def test_build_store_returns_sqlite(tmp_path):
    """Test build_store returns a SQLiteJobStore for a given path."""
    assert isinstance(build_store(str(tmp_path / "jobs.db")), SQLiteJobStore)


def test_build_store_memory():
    """Test build_store accepts the in-memory sentinel path."""
    assert isinstance(build_store(":memory:"), SQLiteJobStore)
