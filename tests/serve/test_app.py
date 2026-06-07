import pytest
from pydantic import BaseModel
from fastapi.testclient import TestClient

from pyagentic import BaseAgent, tool, State, spec
from pyagentic.serve._manifest import Manifest
from pyagentic.serve._app import create_app


class NoteModel(BaseModel):
    text: str = ""


class _AppTestAgent(BaseAgent):
    __system_message__ = "Test agent for app tests"
    __input_template__ = ""

    notes: State[NoteModel] = spec.State(default_factory=lambda: NoteModel(text=""))

    @tool("Echo back the input")
    def echo(self, message: str) -> str:
        return f"echo: {message}"


def _make_manifest() -> Manifest:
    return Manifest(
        project={"name": "test-app", "version": "0.1.0", "description": "Test app"},
        agent={"entry": "test:Agent", "model": "_mock::test-model"},
    )


@pytest.fixture
def client():
    """Create a FastAPI TestClient with the test agent."""
    manifest = _make_manifest()
    app = create_app(_AppTestAgent, manifest)
    return TestClient(app)


def test_agent_info(client):
    """Test GET / returns agent info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "test-app"
    assert data["version"] == "0.1.0"
    assert data["agent_class"] == "_AppTestAgent"
    assert "echo" in data["tools"]
    assert "notes" in data["state_fields"]
    assert isinstance(data["linked_agents"], list)


def test_health(client):
    """Test GET /health returns ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_schema(client):
    """Test GET /schema returns JSON schemas for all models."""
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "request" in data
    assert "response" in data
    assert "stream_event" in data
    assert "state" in data
    # Each should be a valid JSON Schema object
    for key in ("request", "response", "stream_event", "state"):
        assert isinstance(data[key], dict)
        assert "properties" in data[key] or "$defs" in data[key] or "anyOf" in data[key]


def test_create_session(client):
    """Test POST /sessions creates a session."""
    resp = client.post("/sessions", json={"model": "_mock::test-model", "api_key": "key"})
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert isinstance(data["session_id"], str)


def test_create_session_no_body(client):
    """Test POST /sessions works with no body (uses manifest defaults)."""
    resp = client.post("/sessions")
    assert resp.status_code == 201
    assert "session_id" in resp.json()


def test_list_sessions(client):
    """Test GET /sessions lists active sessions."""
    client.post("/sessions", json={"model": "_mock::test-model", "api_key": "key"})
    client.post("/sessions", json={"model": "_mock::test-model", "api_key": "key"})
    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert len(resp.json()["sessions"]) == 2


def test_delete_session(client):
    """Test DELETE /sessions/{id} removes the session."""
    create_resp = client.post("/sessions", json={"model": "_mock::test-model", "api_key": "key"})
    sid = create_resp.json()["session_id"]
    del_resp = client.delete(f"/sessions/{sid}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] == sid

    # Session should no longer exist
    list_resp = client.get("/sessions")
    assert sid not in list_resp.json()["sessions"]


def test_delete_session_not_found(client):
    """Test DELETE /sessions/{id} returns 404 for non-existent session."""
    resp = client.delete("/sessions/doesnotexist")
    assert resp.status_code == 404


def test_get_state(client):
    """Test GET /sessions/{id}/state returns the agent state."""
    create_resp = client.post("/sessions", json={"model": "_mock::test-model", "api_key": "key"})
    sid = create_resp.json()["session_id"]

    resp = client.get(f"/sessions/{sid}/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "notes" in data


def test_get_state_not_found(client):
    """Test GET /sessions/{id}/state returns 404 for non-existent session."""
    resp = client.get("/sessions/doesnotexist/state")
    assert resp.status_code == 404


def test_chat_not_found(client):
    """Test POST /sessions/{id}/chat returns 404 for non-existent session."""
    resp = client.post(
        "/sessions/doesnotexist/chat",
        json={"user_input": "hello"},
    )
    assert resp.status_code == 404


def test_chat_stream_not_found(client):
    """Test POST /sessions/{id}/chat/stream returns 404 for non-existent session."""
    resp = client.post(
        "/sessions/doesnotexist/chat/stream",
        json={"user_input": "hello"},
    )
    assert resp.status_code == 404
