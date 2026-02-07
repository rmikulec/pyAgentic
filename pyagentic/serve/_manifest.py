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
    name: str
    version: str = "0.1.0"
    description: str = ""


class AgentConfig(BaseModel):
    entry: str = Field(
        ...,
        description="module:ClassName entry point for the agent",
    )
    model: str = Field(
        default="openai::gpt-4o",
        description="Default LLM model in provider::model format",
    )


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class BuildConfig(BaseModel):
    python_version: str = "3.13"
    dependencies: list[str] = Field(default_factory=list)


class EnvConfig(BaseModel):
    required: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    project: ProjectConfig
    agent: AgentConfig
    server: ServerConfig = Field(default_factory=ServerConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)


def load_manifest(path: Optional[Path] = None) -> Manifest:
    """Load and parse a pyagentic.toml manifest file.

    Args:
        path: Path to the manifest file. Defaults to ./pyagentic.toml.

    Returns:
        Parsed Manifest instance.

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
