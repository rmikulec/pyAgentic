import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pyagentic import BaseAgent, State, spec, tool
from pyagentic.api import create_app


class NoteModel(BaseModel):
    text: str = ""


class _AppTestAgent(BaseAgent):
    __system_message__ = "Test agent for app tests"
    __input_template__ = ""

    notes: State[NoteModel] = spec.State(default_factory=lambda: NoteModel(text=""))

    @tool("Echo back the input")
    def echo(self, message: str) -> str:
        return f"echo: {message}"


class _WriterAgent(BaseAgent):
    __system_message__ = "A second agent"
    __input_template__ = ""


@pytest.fixture
def client():
    """A single-agent app, mounted at root."""
    app = create_app(_AppTestAgent, name="test-app", version="0.1.0", model="_mock::test-model")
    return TestClient(app)


# ---- single agent (mounted at root) ----


def test_agent_info(client):
    """GET / returns agent info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "test-app"
    assert data["agent_class"] == "_AppTestAgent"
    assert "echo" in data["tools"]
    assert "notes" in data["state_fields"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_schema(client):
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("request", "response", "stream_event", "state"):
        assert key in data


def test_session_lifecycle(client):
    sid = client.post("/sessions").json()["session_id"]
    assert sid in client.get("/sessions").json()["sessions"]
    assert "notes" in client.get(f"/sessions/{sid}/state").json()
    assert client.delete(f"/sessions/{sid}").json()["deleted"] == sid


def test_not_found(client):
    assert client.delete("/sessions/nope").status_code == 404
    assert client.get("/sessions/nope/state").status_code == 404
    assert client.post("/sessions/nope/chat", json={"user_input": "hi"}).status_code == 404


# ---- multiple agents (mounted under prefixes) ----


@pytest.fixture
def multi_client():
    app = create_app([_AppTestAgent, _WriterAgent], name="multi", model="_mock::test-model")
    return TestClient(app)


def test_multi_index(multi_client):
    """The top-level index lists mounted agents and their prefixes."""
    resp = multi_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "multi"
    prefixes = {a["prefix"] for a in data["agents"]}
    assert prefixes == {"/app-test", "/writer"}


def test_multi_top_health(multi_client):
    assert multi_client.get("/health").json() == {"status": "ok"}


def test_multi_agent_routes(multi_client):
    """Each agent is reachable under its derived prefix."""
    assert multi_client.get("/app-test/health").json() == {"status": "ok"}
    assert multi_client.get("/writer/health").json() == {"status": "ok"}

    sid = multi_client.post("/app-test/sessions").json()["session_id"]
    # Session belongs to that agent only, not the other.
    assert sid in multi_client.get("/app-test/sessions").json()["sessions"]
    assert sid not in multi_client.get("/writer/sessions").json()["sessions"]


def test_explicit_prefixes():
    """A {prefix: agent} dict mounts agents at the given paths."""
    app = create_app(
        {"/a": _AppTestAgent, "/b": _WriterAgent},
        name="explicit",
        model="_mock::test-model",
    )
    client = TestClient(app)
    assert client.get("/a/health").json() == {"status": "ok"}
    assert client.get("/b/health").json() == {"status": "ok"}


def test_prefix_collision_raises():
    """Two agents deriving the same prefix is an error."""
    with pytest.raises(ValueError, match="map to prefix"):
        create_app([_AppTestAgent, _AppTestAgent], model="_mock::test-model")


# ---- jobs opt-in ----


def test_jobs_disabled_by_default():
    """Without jobs=True there is no /jobs route."""
    app = create_app(_AppTestAgent, model="_mock::test-model")
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/jobs" not in paths


def test_jobs_enabled_single_agent():
    """jobs=True mounts /jobs at the root for a single agent."""
    app = create_app(_AppTestAgent, model="_mock::test-model", jobs=True)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/jobs" in paths
    assert "/jobs/{job_id}/stream" in paths


def test_jobs_enabled_multi_agent_prefixed():
    """jobs=True mounts a /jobs API under each agent's prefix."""
    app = create_app([_AppTestAgent, _WriterAgent], model="_mock::test-model", jobs=True)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/app-test/jobs" in paths
    assert "/writer/jobs" in paths


# ---- sessions opt-out ----


def test_sessions_disabled():
    """sessions=False omits the session routes from the app."""
    app = create_app(_AppTestAgent, model="_mock::test-model", sessions=False)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/sessions" not in paths
    assert "/schema" in paths  # info routes remain


def test_sessions_off_jobs_on():
    """An async-only app: no /sessions, but /jobs is served."""
    app = create_app(
        _AppTestAgent, model="_mock::test-model", sessions=False, jobs=True
    )
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/sessions" not in paths
    assert "/jobs" in paths
