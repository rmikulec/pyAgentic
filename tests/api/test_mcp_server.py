import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pyagentic import BaseAgent, State, spec, tool
from pyagentic.api import create_app, create_router, mount_mcp
from pyagentic.api._sessions import SessionManager

pytest.importorskip("fastmcp")


class NoteModel(BaseModel):
    text: str = ""


class _McpTestAgent(BaseAgent):
    __system_message__ = "Test agent for mcp tests"
    __input_template__ = ""

    notes: State[NoteModel] = spec.State(default_factory=lambda: NoteModel(text=""))

    @tool("Echo back the input")
    def echo(self, message: str) -> str:
        return f"echo: {message}"


def test_mount_mcp_creates_sessions_when_none():
    """mount_mcp creates and returns a SessionManager if none is given."""
    app = FastAPI()
    sessions = mount_mcp(app, _McpTestAgent, model="_mock::test-model")
    assert isinstance(sessions, SessionManager)


def test_mount_mcp_reuses_given_sessions():
    """mount_mcp returns the same SessionManager it was given."""
    app = FastAPI()
    router = create_router(_McpTestAgent, model="_mock::test-model")
    returned = mount_mcp(app, _McpTestAgent, sessions=router.sessions)
    assert returned is router.sessions


def test_mount_mcp_mounts_at_default_path():
    """The MCP ASGI app is mounted at /mcp by default."""
    app = FastAPI()
    mount_mcp(app, _McpTestAgent, model="_mock::test-model")
    mounted = [r.path for r in app.routes if getattr(r, "path", None) == "/mcp"]
    assert "/mcp" in mounted


def test_mount_mcp_custom_path_alongside_router():
    """mount_mcp accepts a custom path and coexists with an agent router."""
    app = FastAPI()
    router = create_router(_McpTestAgent, model="_mock::test-model")
    app.include_router(router, prefix="/agent")
    mount_mcp(app, _McpTestAgent, sessions=router.sessions, path="/agent/mcp")

    client = TestClient(app)
    assert client.get("/agent/health").status_code == 200

    mounted = [r.path for r in app.routes if getattr(r, "path", None) == "/agent/mcp"]
    assert "/agent/mcp" in mounted


def test_mounted_mcp_serves_requests_without_lifespan_wiring():
    """The mounted (stateless) MCP endpoint serves requests with no manual lifespan setup."""
    app = FastAPI()
    mount_mcp(app, _McpTestAgent, model="_mock::test-model")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    with TestClient(app) as client:
        init = client.post(
            "/mcp/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.0.0"},
                },
            },
        )
    assert init.status_code == 200
    assert "result" in init.text


def test_create_app_with_mcp():
    """create_app(mcp=True) mounts an MCP endpoint per agent at <prefix>/mcp."""
    app = create_app(_McpTestAgent, name="mcp-app", model="_mock::test-model", mcp=True)
    mounted = [r.path for r in app.routes if getattr(r, "path", None) == "/mcp"]
    assert "/mcp" in mounted
