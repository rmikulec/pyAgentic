"""
Pydantic models for parsing and validating pyagentic.toml manifest files.
"""

import re
import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator



class ProjectConfig(BaseModel):
    """Project metadata from the ``[project]`` section of ``pyagentic.toml``.

    Attributes:
        name (str): Project name.
        version (str): Semantic version string.
        description (str): Short project description.
    """

    name: str
    version: str = "0.1.0"
    description: str = ""


class AgentConfig(BaseModel):
    """Agent configuration from the ``[agent]`` section of ``pyagentic.toml``.

    Attributes:
        entry (str): ``module:ClassName`` entry point for the agent.
        model (str): Default LLM model in ``provider::model`` format.
    """

    entry: str = Field(
        ...,
        description="module:ClassName entry point for the agent",
    )
    model: str = Field(
        default="openai::gpt-4o",
        description="Default LLM model in provider::model format",
    )

    @field_validator("entry")
    @classmethod
    def _validate_entry(cls, v: str) -> str:
        """Ensure entry follows ``module:ClassName`` format."""
        if ":" not in v or not v.split(":", 1)[1].strip():
            raise ValueError(
                f"Invalid entry '{v}': must be in 'module:ClassName' format "
                f"(e.g. 'my_agent:MyAgent')."
            )
        return v


class ServerConfig(BaseModel):
    """HTTP server settings from the ``[server]`` section of ``pyagentic.toml``.

    Attributes:
        host (str): Bind address for the server.
        port (int): Port number for the server.
    """

    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid TCP range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}.")
        return v


class BuildConfig(BaseModel):
    """Docker build settings from the ``[build]`` section of ``pyagentic.toml``.

    Attributes:
        python_version (str): Python version for the Docker base image.
        dependencies (list[str]): Additional pip packages to install in the
            image.
    """

    python_version: str = "3.13"
    dependencies: list[str] = Field(default_factory=list)

    @field_validator("python_version")
    @classmethod
    def _validate_python_version(cls, v: str) -> str:
        """Ensure python_version matches ``3.x`` or ``3.x.y`` pattern."""
        if not re.match(r"^3\.\d+(\.\d+)?$", v):
            raise ValueError(
                f"Invalid python_version '{v}': must match '3.x' or '3.x.y' "
                f"(e.g. '3.12', '3.13.1')."
            )
        return v


class EnvConfig(BaseModel):
    """Environment variable requirements from the ``[env]`` section of ``pyagentic.toml``.

    Attributes:
        required (list[str]): Environment variable names that must be set at
            runtime.
    """

    required: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    """Top-level model representing the full ``pyagentic.toml`` manifest.

    Attributes:
        project (ProjectConfig): Project metadata.
        agent (AgentConfig): Agent entry point and model configuration.
        server (ServerConfig): HTTP server host/port settings.
        build (BuildConfig): Docker build configuration.
        env (EnvConfig): Required environment variables.
    """

    project: ProjectConfig
    agent: AgentConfig
    server: ServerConfig = Field(default_factory=ServerConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)


def load_manifest(path: Optional[Path] = None) -> Manifest:
    """Load and parse a pyagentic.toml manifest file.

    Args:
        path (Optional[Path]): Path to the manifest file. Defaults to
            ./pyagentic.toml.

    Returns:
        Manifest: Parsed Manifest instance.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
    if path is None:
        path = Path.cwd() / "pyagentic.toml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Manifest(**data)
