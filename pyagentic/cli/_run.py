"""
`pyagentic run` command — start the agent as a FastAPI server or interactive REPL.
"""

import asyncio
from typing import Optional

import typer

from typing import TYPE_CHECKING

from pyagentic.serve._manifest import load_manifest
from pyagentic.serve._discovery import load_agent_class

if TYPE_CHECKING:
    from pyagentic.serve._manifest import Manifest


def run(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Server bind host."),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Server bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)."),
    repl: bool = typer.Option(
        False, "--repl", help="Start an interactive REPL instead of server."
    ),
    mcp: bool = typer.Option(
        False, "--mcp", help="Mount an MCP server endpoint at /mcp."
    ),
    mcp_stdio: bool = typer.Option(
        False, "--mcp-stdio", help="Run as a standalone MCP stdio server."
    ),
) -> None:
    """Start the agent as an HTTP server, interactive REPL, or MCP server.

    Loads the manifest and agent class, then delegates to either the
    FastAPI/uvicorn server, an interactive REPL loop, or a standalone
    MCP stdio server.

    Args:
        host (Optional[str]): Server bind host. Falls back to the manifest
            value.
        port (Optional[int]): Server bind port. Falls back to the manifest
            value.
        reload (bool): Enable uvicorn auto-reload for development.
        repl (bool): If True, start an interactive REPL instead of the HTTP
            server.
        mcp (bool): If True, mount an MCP server endpoint at /mcp on the
            HTTP server.
        mcp_stdio (bool): If True, run as a standalone MCP stdio server
            (for Claude Desktop / Cursor integration).
    """
    manifest = load_manifest()
    agent_class = load_agent_class(manifest.agent.entry)

    server_host = host or manifest.server.host
    server_port = port or manifest.server.port

    if mcp_stdio:
        _run_mcp_stdio(agent_class, manifest)
    elif repl:
        _run_repl(agent_class, manifest)
    else:
        _run_server(agent_class, manifest, server_host, server_port, reload, mcp=mcp)


def _run_server(
    agent_class: type, manifest: "Manifest", host: str, port: int, reload: bool, mcp: bool = False
) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    from pyagentic.serve._app import create_app

    app = create_app(agent_class, manifest, mcp=mcp)
    typer.echo(f"Serving {manifest.project.name} v{manifest.project.version}")
    typer.echo(f"Agent: {agent_class.__name__}")
    typer.echo(f"Listening on http://{host}:{port}")

    if reload:
        # For reload mode, point uvicorn at the import string
        # and let it handle module reloading
        uvicorn.run(
            "pyagentic.serve._app:create_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
    else:
        uvicorn.run(app, host=host, port=port)


def _run_repl(agent_class: type, manifest: "Manifest") -> None:
    """Run an interactive REPL session with the agent."""
    typer.echo(f"PyAgentic REPL — {manifest.project.name} v{manifest.project.version}")
    typer.echo(f"Agent: {agent_class.__name__}")
    typer.echo("Type 'exit' or 'quit' to stop.\n")

    agent = agent_class(model=manifest.agent.model)

    async def _repl_loop():
        """Read user input in a loop and print agent responses."""
        while True:
            try:
                user_input = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.strip().lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            try:
                response = await agent.run(user_input)
                print(f"\n{response.final_output}\n")
            except Exception as e:
                print(f"\nError: {e}\n")

    asyncio.run(_repl_loop())


def _run_mcp_stdio(agent_class: type, manifest: "Manifest") -> None:
    """Run the agent as a standalone MCP stdio server."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        typer.echo("Error: fastmcp is required. Install with: pip install pyagentic-core[mcp]")
        raise typer.Exit(1)

    import json

    mcp = FastMCP(
        name=manifest.project.name,
        version=manifest.project.version,
    )

    # Store sessions in a simple dict for stdio mode
    _sessions: dict[str, object] = {}
    _counter = 0

    def _create_agent():
        return agent_class(model=manifest.agent.model)

    @mcp.tool(description="Send a message and get a complete agent response.")
    async def ask(message: str) -> str:
        """Send a one-shot message to the agent."""
        agent = _create_agent()
        response = await agent.run(message)
        return response.final_output or ""

    @mcp.tool(description="Create a new agent session for multi-turn conversation.")
    async def create_session() -> str:
        """Create a new session and return the session ID."""
        nonlocal _counter
        _counter += 1
        session_id = f"stdio-{_counter}"
        _sessions[session_id] = _create_agent()
        return json.dumps({"session_id": session_id})

    @mcp.tool(description="Send a message to an existing agent session.")
    async def chat(session_id: str, message: str) -> str:
        """Send a message within an existing session."""
        agent = _sessions.get(session_id)
        if agent is None:
            return json.dumps({"error": f"Session {session_id} not found"})
        response = await agent.run(message)
        return response.final_output or ""

    typer.echo(f"Starting MCP stdio server for {manifest.project.name}")
    typer.echo(f"Agent: {agent_class.__name__}")
    mcp.run(transport="stdio")
