"""
Dockerfile generation and Docker build orchestration.

The generated Dockerfile is written to a temp build context that is cleaned
up after the build. The user's project directory is never modified.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from pyagentic.serve._manifest import Manifest


def generate_dockerfile(manifest: Manifest) -> str:
    """Generate a Dockerfile string from a manifest.

    Args:
        manifest (Manifest): Parsed pyagentic.toml.

    Returns:
        str: Dockerfile contents as a string.
    """
    python_version = manifest.build.python_version
    port = manifest.server.port
    deps = manifest.build.dependencies

    lines = [
        f"FROM python:{python_version}-slim",
        "WORKDIR /app",
        "RUN pip install uv",
        "",
        "COPY requirements.txt .",
        "RUN uv pip install --system -r requirements.txt",
        "RUN uv pip install --system pyagentic-core[deploy]",
        "",
    ]

    if deps:
        deps_str = " ".join(f'"{d}"' for d in deps)
        lines.append(f"RUN uv pip install --system {deps_str}")
        lines.append("")

    lines += [
        "COPY . .",
        f"EXPOSE {port}",
        'CMD ["python", "-m", "pyagentic.cli", "run", "--host", "0.0.0.0"]',
        "",
    ]

    return "\n".join(lines)


def build_image(
    manifest: Manifest,
    project_dir: Path,
    tag: str | None = None,
    no_cache: bool = False,
) -> str:
    """Build a Docker image for the agent project.

    Creates a temporary build context containing the project files and a
    generated Dockerfile, runs `docker build`, then cleans up.

    Args:
        manifest (Manifest): Parsed pyagentic.toml.
        project_dir (Path): Path to the project directory (where
            pyagentic.toml lives).
        tag (str | None): Docker image tag. Defaults to
            ``'<name>:<version>'`` from manifest.
        no_cache (bool): Pass ``--no-cache`` to docker build.

    Returns:
        str: The image tag that was built.

    Raises:
        RuntimeError: If the docker build fails.
        FileNotFoundError: If docker is not installed.
    """
    if shutil.which("docker") is None:
        raise FileNotFoundError(
            "Docker is not installed or not on PATH. " "Install Docker to use `pyagentic build`."
        )

    if tag is None:
        tag = f"{manifest.project.name}:{manifest.project.version}"

    # Build in a temp directory so we never touch the user's project
    with tempfile.TemporaryDirectory(prefix="pyagentic-build-") as tmp:
        tmp_path = Path(tmp)

        # Copy project files into the build context
        for item in project_dir.iterdir():
            # Skip hidden files, __pycache__, and .env
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            dest = tmp_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        # Generate Dockerfile
        dockerfile_content = generate_dockerfile(manifest)
        (tmp_path / "Dockerfile").write_text(dockerfile_content)

        # Ensure requirements.txt exists
        req_file = tmp_path / "requirements.txt"
        if not req_file.exists():
            req_file.write_text("pyagentic-core[deploy]\n")

        # Run docker build
        cmd = ["docker", "build", "-t", tag]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(str(tmp_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")

    return tag
