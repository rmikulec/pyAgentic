"""End-to-end: construct model over HTTP + Depends[T] injection (sessions & jobs)."""

import time

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pyagentic import BaseAgent, State, Link, Depends, tool
from pyagentic.api import AgentsConfig, create_app

_MODEL = "_mock::test-model"


class _Topic(BaseModel):
    name: str


class _Database:
    def __init__(self, dsn: str = "memory") -> None:
        self.dsn = dsn
        self.calls = 0


class _ResearchAgent(BaseAgent):
    __system_message__ = "research"
    __description__ = "researches a topic"
    __input_template__ = ""

    topic: State[_Topic]
    db: Depends[_Database]

    @tool("Look up the configured database dsn")
    def dsn(self) -> str:
        self.db.calls += 1
        return self.db.dsn


class _Orchestrator(BaseAgent):
    __system_message__ = "orchestrate"
    __input_template__ = ""

    researcher: Link[_ResearchAgent]


def test_create_router_fails_fast_without_dependency():
    with pytest.raises(ValueError, match="unsatisfied dependencies"):
        create_app(_Orchestrator, model=_MODEL)


def test_session_create_requires_nested_state_and_injects_dependency():
    db = _Database("postgres://x")
    app = create_app(_Orchestrator, model=_MODEL, dependencies=[db])
    client = TestClient(app)

    # Schema advertises the construct model and the dependency.
    info = client.get("/").json()
    assert info["dependencies"] == [] and info["linked_agents"] == ["researcher"]
    schema = client.get("/schema").json()
    assert "construct" in schema
    assert "db" not in schema["construct"]["properties"]

    # Missing the required nested researcher state -> 422.
    assert client.post("/sessions", json={}).status_code == 422
    assert client.post("/sessions", json={"researcher": {}}).status_code == 422

    # Full construct payload -> 201, dependency injected into the linked agent.
    r = client.post("/sessions", json={"researcher": {"topic": {"name": "attention"}}})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    state = client.get(f"/sessions/{sid}/state").json()
    assert state is not None


def _make_jobs_app(tmp_path, db):
    config = AgentsConfig(
        app={"name": "dep-jobs"},
        jobs={"enabled": True, "store": str(tmp_path / "jobs.db")},
    )
    return create_app(_Orchestrator, config=config, model=_MODEL, dependencies=[db])


def _wait_terminal(client, job_id, timeout=5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise TimeoutError(f"Job {job_id} not terminal within {timeout}s")


def test_sessionless_job_builds_from_construct_payload(tmp_path):
    db = _Database("sqlite://jobs")
    with TestClient(_make_jobs_app(tmp_path, db)) as client:
        resp = client.post(
            "/jobs",
            json={
                "input": {"user_input": "go"},
                "construct": {"researcher": {"topic": {"name": "kv-cache"}}},
            },
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        body = _wait_terminal(client, job_id)
        assert body["status"] == "succeeded"
