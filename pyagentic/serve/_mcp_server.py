"""
MCP server-side mounting for PyAgentic agents.

Exposes an agent as an MCP server so external MCP clients (Claude Desktop,
Cursor, other agents) can interact with it via the Model Context Protocol.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from pyagentic._base._agent._agent import BaseAgent
    from pyagentic.serve._manifest import Manifest
    from pyagentic.serve._sessions import SessionManager


def _mount_mcp(
    app: "FastAPI",
    agent_class: type["BaseAgent"],
    sessions: "SessionManager",
    manifest: "Manifest",
) -> None:
    """Mount an MCP endpoint on an existing FastAPI application.

    Creates a ``FastMCP`` server with ``ask``, ``create_session``, and
    ``chat`` tools, then mounts it at ``/mcp`` on the FastAPI app.

    Args:
        app (FastAPI): The FastAPI application to mount on.
        agent_class (type[BaseAgent]): The agent class being served.
        sessions (SessionManager): The session manager for agent instances.
        manifest (Manifest): Parsed pyagentic.toml manifest.
    """
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

    app.mount("/mcp", mcp.get_asgi_app())
