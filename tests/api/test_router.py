import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pyagentic import BaseAgent, State, spec, tool
from pyagentic.api import create_router
from pyagentic.api._sessions import SessionManager


class NoteModel(BaseModel):
    text: str = ""


class _RouterTestAgent(BaseAgent):
    __system_message__ = "Test agent for router tests"
    __input_template__ = ""

    notes: State[NoteModel] = spec.State(default_factory=lambda: NoteModel(text=""))

    @tool("Echo back the input")
    def echo(self, message: str) -> str:
        return f"echo: {message}"


PREFIX = "/agent"


@pytest.fixture
def client():
    """A host app with the agent router mounted under a prefix."""
    app = FastAPI()
    app.include_router(
        create_router(_RouterTestAgent, name="bot", model="_mock::test-model"),
        prefix=PREFIX,
    )
    return TestClient(app)


def test_create_router_returns_apirouter():
    router = create_router(_RouterTestAgent, model="_mock::test-model")
    assert isinstance(router, APIRouter)
    assert isinstance(router.sessions, SessionManager)


def test_routes_mounted_under_prefix(client):
    assert client.get(f"{PREFIX}/health").json() == {"status": "ok"}
    # Without the prefix, the routes don't exist.
    assert client.get("/health").status_code == 404


def test_agent_info(client):
    data = client.get(f"{PREFIX}/").json()
    assert data["name"] == "bot"
    assert data["agent_class"] == "_RouterTestAgent"
    assert "echo" in data["tools"]
    assert "notes" in data["state_fields"]


def test_session_lifecycle(client):
    sid = client.post(f"{PREFIX}/sessions").json()["session_id"]
    assert sid in client.get(f"{PREFIX}/sessions").json()["sessions"]
    assert "notes" in client.get(f"{PREFIX}/sessions/{sid}/state").json()
    assert client.delete(f"{PREFIX}/sessions/{sid}").json()["deleted"] == sid


def test_not_found(client):
    assert client.delete(f"{PREFIX}/sessions/nope").status_code == 404
    assert client.get(f"{PREFIX}/sessions/nope/state").status_code == 404


def test_two_routers_isolated_sessions():
    """Two routers on one app keep independent session stores."""
    app = FastAPI()
    app.include_router(create_router(_RouterTestAgent, model="_mock::test-model"), prefix="/a")
    app.include_router(create_router(_RouterTestAgent, model="_mock::test-model"), prefix="/b")
    client = TestClient(app)

    sid = client.post("/a/sessions").json()["session_id"]
    assert sid in client.get("/a/sessions").json()["sessions"]
    assert sid not in client.get("/b/sessions").json()["sessions"]


def test_routes_are_named():
    """Each route gets a name derived from the agent name, for url reversal."""
    router = create_router(_RouterTestAgent, name="my-agent", model="_mock::test-model")
    names = {route.name for route in router.routes}
    assert {"my-agent_info", "my-agent_chat", "my-agent_get_state"} <= names


def test_tags_applied():
    """The tags arg is attached to every route on the router."""
    router = create_router(
        _RouterTestAgent, model="_mock::test-model", tags=["bots"]
    )
    assert all("bots" in route.tags for route in router.routes)


def test_named_routes_unique_across_mounts():
    """Distinct names keep route names unique when mounted on one app."""
    app = FastAPI()
    app.include_router(
        create_router(_RouterTestAgent, name="a", model="_mock::test-model"), prefix="/a"
    )
    app.include_router(
        create_router(_RouterTestAgent, name="b", model="_mock::test-model"), prefix="/b"
    )
    # url_path_for resolves each agent's chat route by its unique name.
    assert app.url_path_for("a_chat", session_id="x") == "/a/sessions/x/chat"
    assert app.url_path_for("b_chat", session_id="x") == "/b/sessions/x/chat"


def test_sessions_disabled_omits_session_routes():
    """sessions=False drops the /sessions routes but keeps info/health/schema."""
    app = FastAPI()
    app.include_router(
        create_router(_RouterTestAgent, model="_mock::test-model", sessions=False),
        prefix=PREFIX,
    )
    client = TestClient(app)
    assert client.get(f"{PREFIX}/health").json() == {"status": "ok"}
    assert client.post(f"{PREFIX}/sessions").status_code == 404
    assert client.get(f"{PREFIX}/sessions").status_code == 404


def test_jobs_router_and_orchestrator_exposed():
    """jobs=True mounts /jobs and exposes the orchestrator on the router."""
    router = create_router(_RouterTestAgent, model="_mock::test-model", jobs=True)
    assert router.orchestrator is not None
    paths = {getattr(r, "path", None) for r in router.routes}
    assert "/jobs" in paths
    # Without jobs, no orchestrator.
    plain = create_router(_RouterTestAgent, model="_mock::test-model")
    assert plain.orchestrator is None


def test_sessions_off_jobs_on():
    """sessions=False with jobs=True serves an async-only agent."""
    router = create_router(
        _RouterTestAgent, model="_mock::test-model", sessions=False, jobs=True
    )
    paths = {getattr(r, "path", None) for r in router.routes}
    assert "/jobs" in paths
    assert "/sessions" not in paths
