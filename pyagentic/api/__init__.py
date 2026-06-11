"""
PyAgentic API: turn agents into a FastAPI app or router, and generate a
Dockerfile to deploy them.

  - ``create_app(agents)``    — standalone FastAPI app for one or several agents
  - ``create_router(agent)``  — an APIRouter to mount onto your own app
  - ``mount_mcp(app, agent)`` — expose an agent over the Model Context Protocol
  - ``generate_dockerfile()`` — render a Dockerfile from ``agents.toml``

Configuration lives in ``agents.toml`` beside your ``pyproject.toml`` (see
:class:`AgentsConfig`).
"""

from pyagentic.api._app import create_app, create_router
from pyagentic.api._config import (
    AgentsConfig,
    AppConfig,
    DeployConfig,
    JobsConfig,
    load_config,
)
from pyagentic.api._docker import generate_dockerfile, write_dockerfile
from pyagentic.api._mcp_server import mount_mcp
from pyagentic.api._sessions import SessionManager

__all__ = [
    "create_app",
    "create_router",
    "mount_mcp",
    "generate_dockerfile",
    "write_dockerfile",
    "AgentsConfig",
    "AppConfig",
    "DeployConfig",
    "JobsConfig",
    "load_config",
    "SessionManager",
]
