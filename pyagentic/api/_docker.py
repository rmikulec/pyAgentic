"""
Dockerfile generation for deploying a PyAgentic agent app.

This only *generates* the Dockerfile — building and pushing images is left to
the user's existing Docker tooling. The generated image installs the project
and runs the app with uvicorn against the ``target`` declared in ``agents.toml``.
"""

from pathlib import Path
from typing import Optional

from pyagentic.api._config import AgentsConfig, load_config


def generate_dockerfile(config: Optional[AgentsConfig] = None) -> str:
    """Generate a Dockerfile string from an ``agents.toml`` config.

    The Dockerfile installs ``pyagentic-core[api]`` plus any extra
    ``[deploy].dependencies``, copies the project in, and runs uvicorn against
    ``[deploy].target`` (e.g. ``main:app``).

    Args:
        config (Optional[AgentsConfig]): Parsed config. If ``None``,
            ``./agents.toml`` is loaded (or defaults if absent).

    Returns:
        str: Dockerfile contents.
    """
    if config is None:
        config = load_config()
    deploy = config.deploy

    lines = [
        f"FROM python:{deploy.python_version}-slim",
        "WORKDIR /app",
        "RUN pip install uv",
        "",
        'RUN uv pip install --system "pyagentic-core[api]"',
    ]

    if deploy.dependencies:
        deps_str = " ".join(f'"{dep}"' for dep in deploy.dependencies)
        lines.append(f"RUN uv pip install --system {deps_str}")

    lines += [
        "",
        "COPY . .",
        f"EXPOSE {deploy.port}",
        'CMD ["uvicorn", "'
        + deploy.target
        + '", "--host", "0.0.0.0", "--port", "'
        + str(deploy.port)
        + '"]',
        "",
    ]

    return "\n".join(lines)


def write_dockerfile(
    config: Optional[AgentsConfig] = None,
    path: Path = Path("Dockerfile"),
) -> Path:
    """Generate a Dockerfile and write it to disk.

    Args:
        config (Optional[AgentsConfig]): Parsed config. If ``None``,
            ``./agents.toml`` is loaded (or defaults if absent).
        path (Path): Where to write the Dockerfile. Defaults to ``./Dockerfile``.

    Returns:
        Path: The path the Dockerfile was written to.
    """
    path = Path(path)
    path.write_text(generate_dockerfile(config))
    return path
