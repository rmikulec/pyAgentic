"""
Pydantic models for parsing and validating pyagentic.toml manifest files.
"""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


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


class ServerConfig(BaseModel):
    """HTTP server settings from the ``[server]`` section of ``pyagentic.toml``.

    Attributes:
        host (str): Bind address for the server.
        port (int): Port number for the server.
    """

    host: str = "0.0.0.0"
    port: int = 8000


class BuildConfig(BaseModel):
    """Docker build settings from the ``[build]`` section of ``pyagentic.toml``.

    Attributes:
        python_version (str): Python version for the Docker base image.
        dependencies (list[str]): Additional pip packages to install in the
            image.
    """

    python_version: str = "3.13"
    dependencies: list[str] = Field(default_factory=list)


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
