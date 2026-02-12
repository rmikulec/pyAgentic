"""
`pyagentic build` command — build a Docker image for the agent.
"""

from pathlib import Path
from typing import Optional

import typer

from pyagentic.serve._manifest import load_manifest
from pyagentic.serve._docker import build_image


def build(
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Docker image tag. Defaults to '<name>:<version>' from manifest.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Build without Docker cache.",
    ),
) -> None:
    """Build a Docker image for the agent project.

    Reads the ``pyagentic.toml`` manifest from the current directory and
    delegates to :func:`pyagentic.serve._docker.build_image`.

    Args:
        tag (Optional[str]): Docker image tag. Defaults to
            ``'<name>:<version>'`` from the manifest.
        no_cache (bool): If True, pass ``--no-cache`` to ``docker build``.

    Raises:
        typer.Exit: If the manifest is missing, Docker is not installed,
            or the build fails.
    """
    manifest = load_manifest()
    project_dir = Path.cwd()

    typer.echo(f"Building image for {manifest.project.name} v{manifest.project.version}...")

    try:
        image_tag = build_image(manifest, project_dir, tag=tag, no_cache=no_cache)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    typer.echo(f"Image built: {image_tag}")
