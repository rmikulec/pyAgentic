"""Docker container lifecycle management.

``DockerContainer`` handles starting, stopping, health-checking, and
log retrieval for a single Docker container. It provides both sync
and async APIs.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

import httpx

from pyagentic.serve._exceptions import (
    ContainerStartError,
    ImageNotFoundError,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_PORT = 8000
_DEFAULT_STARTUP_TIMEOUT = 30.0


class DockerContainer:
    """Manages a single Docker container lifecycle.

    Args:
        image (str): Docker image name (e.g. ``"my-agent:latest"``).
        env (Optional[dict[str, str]]): Environment variables to pass
            to the container.
        forward_env (Optional[list[str]]): Names of host environment
            variables to forward into the container.
        container_port (int): The port the agent listens on *inside* the
            container. Defaults to ``8000``.
        startup_timeout (float): Seconds to wait for the container to
            become healthy. Defaults to ``30``.
        auto_pull (bool): If ``True`` (default), pull the image from a
            registry when it is not available locally.
        docker_path (Optional[str]): Path to the ``docker`` binary.
            Resolved via ``shutil.which`` if not provided.

    Raises:
        FileNotFoundError: If Docker is not installed.
    """

    def __init__(
        self,
        image: str,
        *,
        env: Optional[dict[str, str]] = None,
        forward_env: Optional[list[str]] = None,
        container_port: int = _DEFAULT_CONTAINER_PORT,
        startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT,
        auto_pull: bool = True,
        docker_path: Optional[str] = None,
    ) -> None:
        self.image = image
        self._env = dict(env) if env else {}
        self._container_port = container_port
        self._startup_timeout = startup_timeout
        self._auto_pull = auto_pull
        self._docker = docker_path or shutil.which("docker")
        if self._docker is None:
            raise FileNotFoundError(
                "Docker is not installed or not on PATH."
            )

        # Resolve forwarded env vars from the host
        for key in forward_env or []:
            val = os.environ.get(key)
            if val is not None:
                self._env[key] = val

        # Container state
        self._container_id: Optional[str] = None
        self._host_port: Optional[int] = None

    @property
    def host_port(self) -> Optional[int]:
        """The ephemeral host port mapped to the container port."""
        return self._host_port

    @property
    def container_id(self) -> Optional[str]:
        """The Docker container ID (truncated to 12 chars)."""
        return self._container_id

    @property
    def is_running(self) -> bool:
        """Whether the container is currently tracked as running."""
        return self._container_id is not None

    # ---- sync lifecycle ----

    def sync_ensure_image(self) -> None:
        """Verify the image exists locally; pull if missing and auto_pull is on."""
        result = subprocess.run(
            [self._docker, "image", "inspect", self.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return

        if not self._auto_pull:
            raise ImageNotFoundError(
                self.image,
                "Image not found locally and auto_pull is disabled.",
            )
        self.sync_pull_image()

    def sync_pull_image(self) -> None:
        """Pull the image from a registry."""
        logger.info("Pulling image %s ...", self.image)
        result = subprocess.run(
            [self._docker, "pull", self.image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ImageNotFoundError(
                self.image,
                f"docker pull failed: {result.stderr.decode().strip()}",
            )

    def sync_start(self) -> None:
        """Ensure image, run the container, wait for healthy."""
        self.sync_ensure_image()

        cmd = [
            self._docker, "run", "-d",
            "-p", f"0:{self._container_port}",
        ]
        for key, val in self._env.items():
            cmd.extend(["-e", f"{key}={val}"])
        cmd.append(self.image)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ContainerStartError(
                self.image,
                f"docker run failed: {result.stderr.decode().strip()}",
            )

        self._container_id = result.stdout.decode().strip()[:12]
        self._sync_resolve_host_port()
        self._sync_wait_for_healthy()
        logger.info(
            "Container %s ready on port %s",
            self._container_id, self._host_port,
        )

    def sync_stop(self) -> None:
        """Stop and remove the container."""
        if self._container_id is None:
            return
        logger.info("Stopping container %s ...", self._container_id)
        stop_result = subprocess.run(
            [self._docker, "stop", "-t", "5", self._container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if stop_result.returncode != 0:
            logger.warning(
                "docker stop failed for %s: %s",
                self._container_id,
                stop_result.stderr.decode().strip(),
            )
        rm_result = subprocess.run(
            [self._docker, "rm", "-f", self._container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if rm_result.returncode != 0:
            logger.warning(
                "docker rm failed for %s: %s",
                self._container_id,
                rm_result.stderr.decode().strip(),
            )
        self._container_id = None
        self._host_port = None

    def sync_logs(self, tail: int = 50) -> str:
        """Fetch recent logs from the container.

        Args:
            tail (int): Number of log lines to retrieve.

        Returns:
            str: Container log output.
        """
        if self._container_id is None:
            return "(no container)"
        result = subprocess.run(
            [self._docker, "logs", "--tail", str(tail), self._container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.decode()

    def _sync_resolve_host_port(self) -> None:
        """Resolve the ephemeral host port assigned by Docker."""
        result = subprocess.run(
            [self._docker, "port", self._container_id, str(self._container_port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ContainerStartError(
                self.image,
                f"docker port failed: {result.stderr.decode().strip()}",
            )
        mapping = result.stdout.decode().strip().splitlines()[0]
        self._host_port = int(mapping.rsplit(":", 1)[-1])

    def _sync_wait_for_healthy(self) -> None:
        """Poll GET /health until the container is ready."""
        url = f"http://localhost:{self._host_port}/health"
        deadline = time.monotonic() + self._startup_timeout
        delay = 0.25

        while time.monotonic() < deadline:
            try:
                with httpx.Client() as client:
                    resp = client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, OSError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

        logs = self.sync_logs()
        self.sync_stop()
        raise ContainerStartError(
            self.image,
            f"Health check timed out after {self._startup_timeout}s. "
            f"Container logs:\n{logs}",
        )

    # ---- async lifecycle ----

    async def async_ensure_image(self) -> None:
        """Verify the image exists locally; pull if missing and auto_pull is on."""
        proc = await asyncio.create_subprocess_exec(
            self._docker, "image", "inspect", self.image,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            return

        if not self._auto_pull:
            raise ImageNotFoundError(
                self.image,
                "Image not found locally and auto_pull is disabled.",
            )
        await self.async_pull_image()

    async def async_pull_image(self) -> None:
        """Pull the image from a registry."""
        logger.info("Pulling image %s ...", self.image)
        proc = await asyncio.create_subprocess_exec(
            self._docker, "pull", self.image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ImageNotFoundError(
                self.image,
                f"docker pull failed: {stderr.decode().strip()}",
            )

    async def async_start(self) -> None:
        """Ensure image, run the container, wait for healthy."""
        await self.async_ensure_image()

        cmd = [
            self._docker, "run", "-d",
            "-p", f"0:{self._container_port}",
        ]
        for key, val in self._env.items():
            cmd.extend(["-e", f"{key}={val}"])
        cmd.append(self.image)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ContainerStartError(
                self.image,
                f"docker run failed: {stderr.decode().strip()}",
            )

        self._container_id = stdout.decode().strip()[:12]
        await self._async_resolve_host_port()
        await self._async_wait_for_healthy()
        logger.info(
            "Container %s ready on port %s",
            self._container_id, self._host_port,
        )

    async def async_stop(self) -> None:
        """Stop and remove the container."""
        if self._container_id is None:
            return
        logger.info("Stopping container %s ...", self._container_id)
        stop = await asyncio.create_subprocess_exec(
            self._docker, "stop", "-t", "5", self._container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stop_stderr = await stop.communicate()
        if stop.returncode != 0:
            logger.warning(
                "docker stop failed for %s: %s",
                self._container_id,
                stop_stderr.decode().strip(),
            )
        rm = await asyncio.create_subprocess_exec(
            self._docker, "rm", "-f", self._container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, rm_stderr = await rm.communicate()
        if rm.returncode != 0:
            logger.warning(
                "docker rm failed for %s: %s",
                self._container_id,
                rm_stderr.decode().strip(),
            )
        self._container_id = None
        self._host_port = None

    async def async_logs(self, tail: int = 50) -> str:
        """Fetch recent logs from the container.

        Args:
            tail (int): Number of log lines to retrieve.

        Returns:
            str: Container log output.
        """
        if self._container_id is None:
            return "(no container)"
        proc = await asyncio.create_subprocess_exec(
            self._docker, "logs", "--tail", str(tail), self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def _async_resolve_host_port(self) -> None:
        """Resolve the ephemeral host port assigned by Docker."""
        proc = await asyncio.create_subprocess_exec(
            self._docker, "port", self._container_id, str(self._container_port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise ContainerStartError(
                self.image,
                f"docker port failed: {stderr.decode().strip()}",
            )
        mapping = stdout.decode().strip().splitlines()[0]
        self._host_port = int(mapping.rsplit(":", 1)[-1])

    async def _async_wait_for_healthy(self) -> None:
        """Poll GET /health until the container is ready."""
        url = f"http://localhost:{self._host_port}/health"
        deadline = asyncio.get_event_loop().time() + self._startup_timeout
        delay = 0.25

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, OSError):
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

        logs = await self.async_logs()
        await self.async_stop()
        raise ContainerStartError(
            self.image,
            f"Health check timed out after {self._startup_timeout}s. "
            f"Container logs:\n{logs}",
        )
