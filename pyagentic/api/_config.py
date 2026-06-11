"""
Pydantic models for parsing ``agents.toml`` — the config file that sits beside a
project's ``pyproject.toml`` and describes the agent app and how to deploy it.

The file has two sections::

    [app]
    name = "my-agents"
    version = "0.1.0"
    model = "openai::gpt-4o"   # default model for agents that don't override

    [deploy]
    target = "main:app"        # ASGI app the container serves
    python_version = "3.13"
    dependencies = []
    port = 8000
    env = ["OPENAI_API_KEY"]

``[app]`` configures the running app (used by ``create_app``); ``[deploy]`` is
read only by ``generate_dockerfile``.
"""

import re
import tomllib
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_FILENAME = "agents.toml"

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_duration(value: Union[str, int]) -> int:
    """Parse a duration (``"30s"``/``"15m"``/``"24h"``/``"7d"`` or raw seconds) to seconds."""
    if isinstance(value, int):
        return value
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    unit = text[-1:]
    if unit in _DURATION_UNITS and text[:-1].isdigit():
        return int(text[:-1]) * _DURATION_UNITS[unit]
    raise ValueError(
        f"Invalid duration '{value}': use seconds or a '<n><unit>' string "
        f"(e.g. '30s', '15m', '24h', '7d')."
    )


class AppConfig(BaseModel):
    """The ``[app]`` section of ``agents.toml``.

    Attributes:
        name (str): App name, used as the FastAPI title and default image name.
        version (str): Semantic version string.
        description (str): Short description.
        model (Optional[str]): Default LLM model (``provider::model``) for agents
            that don't specify their own. ``None`` falls back to each agent's
            own default.
    """

    name: str = "agents"
    version: str = "0.1.0"
    description: str = ""
    model: Optional[str] = None


class DeployConfig(BaseModel):
    """The ``[deploy]`` section of ``agents.toml`` (read by the Dockerfile generator).

    Attributes:
        target (str): The ASGI app uvicorn serves, in ``module:attr`` form
            (e.g. ``main:app``).
        python_version (str): Python version for the Docker base image.
        dependencies (list[str]): Extra pip packages to install in the image.
        port (int): Port the container exposes and uvicorn binds.
        env (list[str]): Environment variable names required at runtime.
    """

    target: str = "main:app"
    python_version: str = "3.13"
    dependencies: list[str] = Field(default_factory=list)
    port: int = 8000
    env: list[str] = Field(default_factory=list)

    @field_validator("python_version")
    @classmethod
    def _validate_python_version(cls, v: str) -> str:
        """Ensure python_version matches ``3.x`` or ``3.x.y``."""
        if not re.match(r"^3\.\d+(\.\d+)?$", v):
            raise ValueError(
                f"Invalid python_version '{v}': must match '3.x' or '3.x.y' "
                f"(e.g. '3.12', '3.13.1')."
            )
        return v

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid TCP range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}.")
        return v

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        """Ensure target follows ``module:attr`` format."""
        if ":" not in v or not v.split(":", 1)[1].strip():
            raise ValueError(
                f"Invalid target '{v}': must be in 'module:attr' format "
                f"(e.g. 'main:app')."
            )
        return v


class JobsConfig(BaseModel):
    """The ``[jobs]`` section of ``agents.toml`` (durable async job system).

    When enabled, each agent gets a ``/jobs`` API: runs are submitted as durable
    records and streamed/polled later, surviving client timeouts and reconnects.

    Attributes:
        enabled (bool): Whether to mount the job system on the app.
        store (str): SQLite database path for durable job records. Use
            ``":memory:"`` for an ephemeral in-process store.
        admission_cap (int): Max jobs in-flight through the process at once;
            jobs beyond the cap remain ``queued``.
        max_concurrency (int): Max concurrent agent runs in the in-process backend.
        ttl (Union[str, int]): Retention for terminal job records — a duration
            string (``"24h"``) or raw seconds.
        cleanup_interval_seconds (int): How often the TTL cleanup loop runs.
    """

    enabled: bool = False
    store: str = ".pyagentic/jobs.db"
    admission_cap: int = 16
    max_concurrency: int = 8
    ttl: Union[str, int] = "24h"
    cleanup_interval_seconds: int = 300

    @property
    def ttl_seconds(self) -> int:
        """The TTL resolved to seconds."""
        return _parse_duration(self.ttl)


class AgentsConfig(BaseModel):
    """Top-level model representing the full ``agents.toml`` file.

    Attributes:
        app (AppConfig): Running-app configuration.
        deploy (DeployConfig): Docker build/deploy configuration.
        jobs (JobsConfig): Durable async job system configuration.
    """

    app: AppConfig = Field(default_factory=AppConfig)
    deploy: DeployConfig = Field(default_factory=DeployConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)


def load_config(path: Optional[Path] = None) -> AgentsConfig:
    """Load and parse an ``agents.toml`` config file.

    When ``path`` is not given, this looks for ``./agents.toml`` and returns a
    default :class:`AgentsConfig` if it's absent (so apps work with no config
    file). When ``path`` is given explicitly, a missing file is an error.

    Args:
        path (Optional[Path]): Path to the config file. Defaults to
            ``./agents.toml``.

    Returns:
        AgentsConfig: The parsed config (or defaults if the default file is
            absent).

    Raises:
        FileNotFoundError: If ``path`` is given explicitly but does not exist.
    """
    explicit = path is not None
    path = Path(path) if explicit else Path.cwd() / DEFAULT_CONFIG_FILENAME

    if not path.exists():
        if explicit:
            raise FileNotFoundError(f"Config not found: {path}")
        return AgentsConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return AgentsConfig(**data)
