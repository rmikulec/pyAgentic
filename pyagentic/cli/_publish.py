"""
`pyagentic publish` command — push a Docker image to a registry.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer

from pyagentic.serve._manifest import load_manifest
from pyagentic.serve._docker import build_image


def publish(
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        "-r",
        help="Target registry (e.g. 'ghcr.io/myorg'). Defaults to Docker Hub.",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Image tag override.",
    ),
) -> None:
    """Build (if needed) and push the agent image to a Docker registry.

    If the image does not already exist locally it will be built first.
    The image is tagged for the target registry and pushed via
    ``docker push``.

    Args:
        registry (Optional[str]): Target registry prefix
            (e.g. ``'ghcr.io/myorg'``). Defaults to Docker Hub when not
            provided.
        tag (Optional[str]): Image tag override. Defaults to
            ``'<name>:<version>'`` from the manifest.

    Raises:
        typer.Exit: If Docker is not installed, the build fails, tagging
            fails, or the push fails.
    """
    if shutil.which("docker") is None:
        typer.echo("Docker is not installed or not on PATH.", err=True)
        raise typer.Exit(1)

    manifest = load_manifest()
    project_dir = Path.cwd()

    local_tag = tag or f"{manifest.project.name}:{manifest.project.version}"

    # Check if image already exists locally
    check = subprocess.run(
        ["docker", "image", "inspect", local_tag],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        typer.echo(f"Image {local_tag} not found locally. Building...")
        try:
            build_image(manifest, project_dir, tag=local_tag)
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)

    # Construct the remote tag
    if registry:
        registry = registry.rstrip("/")
        remote_tag = f"{registry}/{local_tag}"
    else:
        remote_tag = local_tag

    # Tag for the remote registry
    if remote_tag != local_tag:
        result = subprocess.run(
            ["docker", "tag", local_tag, remote_tag],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            typer.echo(f"Failed to tag image:\n{result.stderr}", err=True)
            raise typer.Exit(1)

    # Push
    typer.echo(f"Pushing {remote_tag}...")
    result = subprocess.run(
        ["docker", "push", remote_tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        typer.echo(f"Push failed:\n{result.stderr}", err=True)
        typer.echo("Make sure you are logged in: docker login", err=True)
        raise typer.Exit(1)

    typer.echo(f"Published: {remote_tag}")
