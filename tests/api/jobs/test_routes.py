import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from pyagentic import BaseAgent
from pyagentic.api import AgentsConfig, create_app
from pyagentic.api._config import JobsConfig
from pyagentic.api._sessions import SessionManager
from pyagentic.api.jobs import (
    JobOrchestrator,
    JobRecord,
    JobStatus,
    SQLiteJobStore,
)
from pyagentic.api.jobs._routes import job_event_stream
from pyagentic.api.jobs.backends._in_process import InProcessBackend

_MODEL = "_mock::test-model"


class _JobsTestAgent(BaseAgent):
    __system_message__ = "Jobs route test agent"
    __input_template__ = ""


def _make_app(tmp_path):
    config = AgentsConfig(
        app={"name": "jobs-test"},
        jobs={"enabled": True, "store": str(tmp_path / "jobs.db")},
    )
    return create_app(_JobsTestAgent, config=config, model=_MODEL)


@pytest.fixture
def client(tmp_path):
    """TestClient with an isolated SQLite store per test.

    Uses the context manager form so the lifespan runs and a single event
    loop persists across requests (background job tasks need to outlive the
    submitting request).
    """
    with TestClient(_make_app(tmp_path)) as test_client:
        yield test_client


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE frames into dicts with id/event/data keys."""
    frames = []
    current: dict = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                frames.append(current)
                current = {}
            continue
        key, _, value = line.partition(": ")
        current[key] = value
    if current:
        frames.append(current)
    return frames


def _wait_terminal(client, job_id, timeout=5.0) -> dict:
    """Poll GET /jobs/{id} until the job is terminal."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"Job {job_id} not terminal within {timeout}s")


def test_submit_job(client):
    """Test POST /jobs returns 202 with a job id."""
    resp = client.post("/jobs", json={"input": {"user_input": "hello"}})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert isinstance(body["job_id"], str)


def test_submit_job_unknown_session(client):
    """Test POST /jobs with a missing session returns 404."""
    resp = client.post(
        "/jobs", json={"input": {"user_input": "x"}, "session_id": "ghost"}
    )
    assert resp.status_code == 404


def test_get_job_lifecycle(client):
    """Test a submitted job reaches succeeded with a result."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    body = _wait_terminal(client, job_id)
    assert body["status"] == "succeeded"
    assert body["result"]["final_output"] == "user said hi"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None


def test_get_job_not_found(client):
    """Test GET /jobs/{id} returns 404 for unknown jobs."""
    assert client.get("/jobs/nope").status_code == 404


def test_list_jobs_and_filters(client):
    """Test GET /jobs lists jobs and filters by status."""
    sid = client.post("/sessions").json()["session_id"]
    j1 = client.post("/jobs", json={"input": {"user_input": "a"}}).json()["job_id"]
    j2 = client.post(
        "/jobs", json={"input": {"user_input": "b"}, "session_id": sid}
    ).json()["job_id"]
    _wait_terminal(client, j1)
    _wait_terminal(client, j2)

    all_ids = {j["job_id"] for j in client.get("/jobs").json()["jobs"]}
    assert {j1, j2} <= all_ids

    by_session = client.get(f"/jobs?session_id={sid}").json()["jobs"]
    assert [j["job_id"] for j in by_session] == [j2]

    succeeded = client.get("/jobs?status=succeeded").json()["jobs"]
    assert {j["job_id"] for j in succeeded} >= {j1, j2}


def test_session_jobs_view(client):
    """Test GET /jobs?session_id= returns only that session's jobs."""
    sid = client.post("/sessions").json()["session_id"]
    client.post("/jobs", json={"input": {"user_input": "a"}})
    j2 = client.post(
        "/jobs", json={"input": {"user_input": "b"}, "session_id": sid}
    ).json()["job_id"]
    _wait_terminal(client, j2)
    body = client.get(f"/jobs?session_id={sid}").json()
    assert [j["job_id"] for j in body["jobs"]] == [j2]


def test_updates_endpoint_with_cursor(client):
    """Test GET /jobs/{id}/updates returns the log and slices on since."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    _wait_terminal(client, job_id)

    body = client.get(f"/jobs/{job_id}/updates").json()
    seqs = [u["seq"] for u in body["updates"]]
    assert seqs == list(range(len(seqs)))
    assert body["updates"][-1]["event"] == "agent_response"

    sliced = client.get(f"/jobs/{job_id}/updates?since=0").json()
    assert [u["seq"] for u in sliced["updates"]] == seqs[1:]


def test_updates_not_found(client):
    """Test GET /jobs/{id}/updates returns 404 for unknown jobs."""
    assert client.get("/jobs/nope/updates").status_code == 404


def test_stream_replays_terminal_job(client):
    """Test /stream replays the full log with id: lines and closes."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    _wait_terminal(client, job_id)

    with client.stream("GET", f"/jobs/{job_id}/stream") as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    frames = _parse_sse(text)
    assert [int(f["id"]) for f in frames] == list(range(len(frames)))
    assert frames[-1]["event"] == "agent_response"


def test_stream_live_tail(client):
    """Test /stream attached at submit time receives the full run live."""
    job_id = client.post("/jobs", json={"input": {"user_input": "live"}}).json()[
        "job_id"
    ]
    with client.stream("GET", f"/jobs/{job_id}/stream") as resp:
        text = "".join(resp.iter_text())
    frames = _parse_sse(text)
    assert frames[-1]["event"] == "agent_response"
    assert "user said live" in frames[-1]["data"]


def test_stream_reconnect_with_last_event_id(client):
    """Test reconnecting with Last-Event-ID replays only the tail."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    _wait_terminal(client, job_id)

    with client.stream("GET", f"/jobs/{job_id}/stream") as resp:
        full = _parse_sse("".join(resp.iter_text()))
    assert len(full) >= 2

    with client.stream(
        "GET", f"/jobs/{job_id}/stream", headers={"Last-Event-ID": "0"}
    ) as resp:
        tail = _parse_sse("".join(resp.iter_text()))
    assert [int(f["id"]) for f in tail] == [int(f["id"]) for f in full][1:]


def test_stream_since_param(client):
    """Test ?since= behaves like Last-Event-ID."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    _wait_terminal(client, job_id)
    with client.stream("GET", f"/jobs/{job_id}/stream?since=0") as resp:
        frames = _parse_sse("".join(resp.iter_text()))
    assert all(int(f["id"]) > 0 for f in frames)


def test_stream_not_found(client):
    """Test /stream returns 404 for unknown jobs."""
    assert client.get("/jobs/nope/stream").status_code == 404


@pytest.mark.asyncio
async def test_stream_orders_out_of_order_ingest(tmp_path):
    """Test the responder holds at a seq hole and emits in order once filled."""
    path = str(tmp_path / "jobs.db")
    store = SQLiteJobStore(path)
    sessions = SessionManager(_JobsTestAgent, default_model=_MODEL)
    backend = InProcessBackend(_JobsTestAgent, sessions, default_model=_MODEL)
    orchestrator = JobOrchestrator(store, backend, JobsConfig(store=path), sessions)
    await orchestrator.ensure_started()
    await store.create_job(JobRecord(job_id="ooo", request={}))

    frames: list[str] = []

    async def consume():
        async for frame in job_event_stream(orchestrator, "ooo", -1):
            frames.append(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)

    # seq 1 lands before seq 0: nothing may be emitted yet.
    await orchestrator.ingest("ooo", 1, "tool_response", "{}")
    await asyncio.sleep(0.05)
    assert frames == []

    # The hole fills: 0 then 1 stream out, in order.
    await orchestrator.ingest("ooo", 0, "llm_response", "{}")
    await asyncio.sleep(0.05)
    assert [f.splitlines()[0] for f in frames] == ["id: 0", "id: 1"]

    # Terminal closes the stream.
    await orchestrator.ingest(
        "ooo",
        2,
        "agent_response",
        json.dumps({"final_output": "done"}),
        terminal_status=JobStatus.SUCCEEDED,
        result_json="{}",
    )
    await asyncio.wait_for(task, timeout=2)
    assert [f.splitlines()[0] for f in frames] == ["id: 0", "id: 1", "id: 2"]
    assert "event: agent_response" in frames[-1]


def test_cancel_terminal_conflict(client):
    """Test cancelling a finished job returns 409."""
    job_id = client.post("/jobs", json={"input": {"user_input": "hi"}}).json()["job_id"]
    _wait_terminal(client, job_id)
    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 409


def test_cancel_not_found(client):
    """Test cancelling an unknown job returns 404."""
    assert client.post("/jobs/nope/cancel").status_code == 404
