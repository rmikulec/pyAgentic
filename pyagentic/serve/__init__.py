"""
PyAgentic Serve: Server and deployment infrastructure for PyAgentic agents.

Provides FastAPI-based serving, session management, manifest parsing,
Docker build tooling, and client-side container management for deploying
and interacting with agents as HTTP services.
"""

from pyagentic.serve._agent_ref import AgentRef
from pyagentic.serve._app import create_app, create_router
from pyagentic.serve._client_session import Session
from pyagentic.serve._container import DockerContainer
from pyagentic.serve._discovery import load_agent_class
from pyagentic.serve._exceptions import (
    AgentAPIError,
    ContainerNotRunningError,
    ContainerStartError,
    ImageNotFoundError,
)
from pyagentic.serve._manifest import Manifest, load_manifest
from pyagentic.serve._mcp_server import mount_mcp

__all__ = [
    "AgentRef",
    "DockerContainer",
    "create_app",
    "create_router",
    "mount_mcp",
    "Manifest",
    "Session",
    "load_agent_class",
    "load_manifest",
    "AgentAPIError",
    "ContainerNotRunningError",
    "ContainerStartError",
    "ImageNotFoundError",
]
