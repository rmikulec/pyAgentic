"""
PyAgentic Serve: Server and deployment infrastructure for PyAgentic agents.

Provides FastAPI-based serving, session management, manifest parsing,
Docker build tooling, and client-side container management for deploying
and interacting with agents as HTTP services.
"""

from pyagentic.serve._manifest import Manifest, load_manifest
from pyagentic.serve._discovery import load_agent_class
from pyagentic.serve._agent_ref import AgentRef
from pyagentic.serve._client_session import Session
from pyagentic.serve._exceptions import (
    AgentAPIError,
    ContainerNotRunningError,
    ContainerStartError,
    ImageNotFoundError,
)

__all__ = [
    "Manifest",
    "load_manifest",
    "load_agent_class",
    "AgentRef",
    "Session",
    "AgentAPIError",
    "ContainerNotRunningError",
    "ContainerStartError",
    "ImageNotFoundError",
]
