"""
MCP server-side mounting for PyAgentic agents.

Exposes an agent as an MCP server so external MCP clients (Claude Desktop,
Cursor, other agents) can interact with it via the Model Context Protocol.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI
    from pyagentic._base._agent._agent import BaseAgent
    from pyagentic.serve._manifest import Manifest
    from pyagentic.serve._sessions import SessionManager


def mount_mcp(
    app: "FastAPI",
    agent_class: type["BaseAgent"],
    manifest: "Manifest",
    sessions: Optional["SessionManager"] = None,
    path: str = "/mcp",
) -> "SessionManager":
    """Mount an MCP endpoint for an agent on an existing FastAPI application.

    Creates a ``FastMCP`` server with ``ask``, ``create_session``, and ``chat``
    tools, then mounts it on the FastAPI app. Use this to expose an agent over
    the Model Context Protocol from your own app::

        app = FastAPI()
        router = create_router(MyAgent, manifest)
        app.include_router(router, prefix="/agent")
        # Share the router's sessions so HTTP and MCP see the same conversations
        mount_mcp(app, MyAgent, manifest, sessions=router.sessions, path="/agent/mcp")

    Args:
        app (FastAPI): The FastAPI application to mount on.
        agent_class (type[BaseAgent]): The agent class being served.
        manifest (Manifest): Parsed pyagentic.toml manifest.
        sessions (Optional[SessionManager]): Session manager to use for agent
            instances. If ``None``, a new one is created — pass an existing
            manager (e.g. ``router.sessions``) to share sessions with HTTP routes.
        path (str): The path to mount the MCP ASGI app at. Defaults to ``/mcp``.

    Note:
        This wraps the host app's lifespan so the MCP server's session manager
        initializes on startup. Call it during app setup, before the app starts
        serving.

    Returns:
        SessionManager: The session manager backing the MCP tools (the one
            passed in, or the newly created one).

    Raises:
        ImportError: If the ``fastmcp`` extra is not installed.
    """
    if sessions is None:
        from pyagentic.serve._sessions import SessionManager

        sessions = SessionManager(agent_class, manifest)

    try:
        from fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "fastmcp is required for MCP server support. "
            "Install it with: pip install pyagentic-core[mcp]"
        )

    mcp = FastMCP(
        name=manifest.project.name,
        version=manifest.project.version,
    )

    @mcp.tool(description="Send a message and get a complete agent response.")
    async def ask(message: str) -> str:
        """Send a one-shot message to the agent and return the final output."""
        session_id = sessions.create()
        agent = sessions.get(session_id)
        try:
            response = await agent.run(message)
            return response.final_output or ""
        finally:
            sessions.delete(session_id)

    @mcp.tool(description="Create a new agent session for multi-turn conversation.")
    async def create_session() -> str:
        """Create a new session and return the session ID."""
        session_id = sessions.create()
        return json.dumps({"session_id": session_id})

    @mcp.tool(description="Send a message to an existing agent session.")
    async def chat(session_id: str, message: str) -> str:
        """Send a message within an existing session and return the response."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            return json.dumps({"error": f"Session {session_id} not found"})

        response = await agent.run(message)
        return response.final_output or ""

    # Build a stateless MCP ASGI app: each request is self-contained, so we
    # don't depend on persistent MCP transport sessions (multi-turn agent state
    # is handled by our own SessionManager). The inner path is "/" so the
    # endpoint lives exactly at the mount point.
    mcp_app = mcp.http_app(path="/", stateless_http=True)

    # FastMCP's session manager initializes inside the ASGI app's lifespan, and
    # a mounted sub-app's lifespan is NOT run by the host app automatically.
    # Wrap the host app's existing lifespan so the MCP app's lifespan runs too.
    prev_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def _lifespan_with_mcp(host_app):
        async with mcp_app.lifespan(host_app):
            async with prev_lifespan(host_app):
                yield

    app.router.lifespan_context = _lifespan_with_mcp
    app.mount(path, mcp_app)
    return sessions
