"""
`pyagentic init` command — scaffold a new agent project.
"""

import re
from pathlib import Path
from typing import Optional

import typer

from pyagentic.cli._templates import (
    MINIMAL_AGENT,
    MINIMAL_TOML,
    FULL_AGENT,
    FULL_TOML,
    REQUIREMENTS_TXT,
    ENV_EXAMPLE,
    GITIGNORE,
)


def _to_module_name(project_name: str) -> str:
    """Convert a project name like 'my-agent' to a valid Python module name 'my_agent'."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", project_name).strip("_")


def _to_class_name(project_name: str) -> str:
    """Convert 'my-agent' to 'MyAgent'."""
    parts = re.split(r"[-_ ]+", project_name)
    return "".join(p.capitalize() for p in parts if p)


def init(
    project_name: Optional[str] = typer.Argument(
        None,
        help="Name of the project to create. Defaults to current directory name.",
    ),
    template: str = typer.Option(
        "minimal",
        "--template",
        "-t",
        help="Project template to use.",
        show_choices=True,
    ),
) -> None:
    """Scaffold a new PyAgentic agent project."""
    if template not in ("minimal", "full"):
        typer.echo(f"Unknown template: {template!r}. Choose 'minimal' or 'full'.", err=True)
        raise typer.Exit(1)

    if project_name is None:
        project_name = Path.cwd().name

    project_dir = Path(project_name).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Use just the directory name for module/class derivation
    short_name = project_dir.name
    module_name = _to_module_name(short_name)
    class_name = _to_class_name(short_name)

    fmt = dict(
        project_name=short_name,
        module_name=module_name,
        class_name=class_name,
    )

    # Select template
    if template == "full":
        agent_src = FULL_AGENT.format(**fmt)
        toml_src = FULL_TOML.format(**fmt)
    else:
        agent_src = MINIMAL_AGENT.format(**fmt)
        toml_src = MINIMAL_TOML.format(**fmt)

    # Write agent module
    (project_dir / f"{module_name}.py").write_text(agent_src)

    # Write manifest
    (project_dir / "pyagentic.toml").write_text(toml_src)

    # Write requirements.txt
    (project_dir / "requirements.txt").write_text(REQUIREMENTS_TXT)

    # Write .env.example
    env_lines = "\n".join(f"# {var}=" for var in ["OPENAI_API_KEY"])
    if template == "full":
        env_lines = "\n".join(f"# {var}=" for var in ["OPENAI_API_KEY"])
    (project_dir / ".env.example").write_text(ENV_EXAMPLE.format(env_lines=env_lines))

    # Write .gitignore
    (project_dir / ".gitignore").write_text(GITIGNORE)

    typer.echo(f"Created project '{short_name}' in {project_dir}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  cd {short_name}")
    typer.echo("  uv pip install pyagentic-core[deploy]")
    typer.echo("  # Edit your agent, then:")
    typer.echo("  pyagentic run")
