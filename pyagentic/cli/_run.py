"""
`pyagentic run` command — start the agent as a FastAPI server or interactive REPL.
"""

import asyncio
from typing import Optional

import typer

from pyagentic.serve._manifest import load_manifest
from pyagentic.serve._discovery import load_agent_class


def run(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Server bind host."),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Server bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)."),
    repl: bool = typer.Option(
        False, "--repl", help="Start an interactive REPL instead of server."
    ),
) -> None:
    """Start the agent as an HTTP server or interactive REPL.

    Loads the manifest and agent class, then delegates to either the
    FastAPI/uvicorn server or an interactive REPL loop.

    Args:
        host (Optional[str]): Server bind host. Falls back to the manifest
            value.
        port (Optional[int]): Server bind port. Falls back to the manifest
            value.
        reload (bool): Enable uvicorn auto-reload for development.
        repl (bool): If True, start an interactive REPL instead of the HTTP
            server.
    """
    manifest = load_manifest()
    agent_class = load_agent_class(manifest.agent.entry)

    server_host = host or manifest.server.host
    server_port = port or manifest.server.port

    if repl:
        _run_repl(agent_class, manifest)
    else:
        _run_server(agent_class, manifest, server_host, server_port, reload)


def _run_server(agent_class, manifest, host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    from pyagentic.serve._app import create_app

    app = create_app(agent_class, manifest)
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


def _run_repl(agent_class, manifest) -> None:
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
