"""Client-side Docker agent manager.

``AgentRef`` manages the lifecycle of a containerized PyAgentic agent:
start, health-check, session creation, and teardown — all driven by
an async context manager so the container is automatically cleaned up.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Generator, Optional

import httpx

from pyagentic.serve._client_session import Session
from pyagentic.serve._exceptions import (
    AgentAPIError,
    ContainerNotRunningError,
    ContainerStartError,
    ImageNotFoundError,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_PORT = 8000
_DEFAULT_STARTUP_TIMEOUT = 30.0


class _AgentRefSessionContext:
    """Reference-counted async context manager.

    The backing Docker container stays alive while at least one session is
    open.  When the last session exits, the container is stopped and removed.
    """

    def __init__(self, ref: "AgentRef", model: Optional[str], api_key: Optional[str]) -> None:
        """Initialize the session context.

        Args:
            ref (AgentRef): The parent AgentRef that owns the container.
            model (Optional[str]): LLM model override for the session.
            api_key (Optional[str]): API key for the session.
        """
        self._ref = ref
        self._model = model
        self._api_key = api_key
        self._session: Optional[Session] = None

    async def __aenter__(self) -> Session:
        """Start the container (if needed) and open a Session."""
        async with self._ref._lock:
            if self._ref._refcount == 0:
                await self._ref._start_container()
            self._ref._refcount += 1

        base_url = f"http://localhost:{self._ref._host_port}"
        self._session = Session(
            base_url,
            model=self._model,
            api_key=self._api_key,
        )
        return await self._session.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the session and stop the container when no sessions remain."""
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
            self._session = None

        async with self._ref._lock:
            self._ref._refcount -= 1
            if self._ref._refcount == 0:
                await self._ref._stop_container()


class AgentRef:
    """Manages a Docker container running a PyAgentic agent and provides
    ``Session`` objects for communicating with it.

    Args:
        image (str): Docker image name (e.g. ``"my-agent:latest"``).
        env (Optional[dict[str, str]]): Explicit environment variables to
            pass to the container.
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

    Example::

        ref = AgentRef("my-agent:latest", forward_env=["OPENAI_API_KEY"])
        async with ref.session() as s:
            resp = await s.chat(user_input="Hi!")
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
        self._refcount: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    def session(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> _AgentRefSessionContext:
        """Return an async context manager that yields a ``Session``.

        The backing container is started on the first ``session()`` entry
        and stopped when the last session exits.

        Args:
            model (Optional[str]): LLM model override for this session.
            api_key (Optional[str]): API key for this session.

        Returns:
            _AgentRefSessionContext: An async context manager yielding a
                :class:`Session`.
        """
        return _AgentRefSessionContext(self, model=model, api_key=api_key)

    # ---- sync context manager ----

    def __enter__(self) -> "AgentRef":
        """Start the container, create an HTTP client and server-side session.

        Returns:
            AgentRef: ``self``, ready for :meth:`run`, :meth:`step`, and
                :attr:`state` calls.
        """
        self._sync_start_container()
        self._sync_client = httpx.Client(
            base_url=f"http://localhost:{self._host_port}",
        )
        self._sync_session_id = self._sync_create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Delete the server-side session, close the client, stop the container."""
        try:
            if self._sync_session_id is not None:
                self._sync_delete_session()
        finally:
            self._sync_session_id = None
            if hasattr(self, "_sync_client") and self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None
            self._sync_stop_container()

    # ---- public sync API ----

    def run(self, **kwargs) -> dict:
        """Send a chat message and receive the complete agent response.

        Args:
            **kwargs: Request fields forwarded as JSON body to the agent's
                chat endpoint (e.g. ``user_input="Hello"``).

        Returns:
            dict: The full agent response as a dictionary.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._sync_session_id}/chat"
        resp = self._sync_client.post(endpoint, json=kwargs)
        Session._raise_for_status(resp, f"POST {endpoint}")
        return resp.json()

    def step(self, **kwargs) -> Generator[dict, None, None]:
        """Send a chat message and receive SSE events as a sync generator.

        Each yielded dict has ``"event"`` and ``"data"`` keys parsed from
        the SSE stream.

        Args:
            **kwargs: Request fields forwarded as JSON body to the agent's
                stream endpoint.

        Yields:
            dict: An SSE event with ``"event"`` (str) and ``"data"`` (dict)
                keys.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._sync_session_id}/chat/stream"
        with self._sync_client.stream("POST", endpoint, json=kwargs) as resp:
            if resp.status_code >= 400:
                resp.read()
                Session._raise_for_status(resp, f"POST {endpoint}")
            yield from self._parse_sse_sync(resp)

    @property
    def state(self) -> dict:
        """Return the current agent state for the active session.

        Returns:
            dict: The agent state as a dictionary.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._sync_session_id}/state"
        resp = self._sync_client.get(endpoint)
        Session._raise_for_status(resp, f"GET {endpoint}")
        return resp.json()

    # ---- sync container lifecycle (internal) ----

    def _sync_ensure_image(self) -> None:
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
        self._sync_pull_image()

    def _sync_pull_image(self) -> None:
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

    def _sync_start_container(self) -> None:
        """Ensure image, run the container, wait for healthy."""
        self._sync_ensure_image()

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
        self._sync_get_host_port()
        self._sync_wait_for_healthy()
        logger.info(
            "Container %s ready on port %s",
            self._container_id, self._host_port,
        )

    def _sync_get_host_port(self) -> None:
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
            except (httpx.ConnectError, httpx.ReadError, OSError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

        logs = self._sync_container_logs()
        self._sync_stop_container()
        raise ContainerStartError(
            self.image,
            f"Health check timed out after {self._startup_timeout}s. "
            f"Container logs:\n{logs}",
        )

    def _sync_container_logs(self) -> str:
        """Fetch recent logs from the container."""
        if self._container_id is None:
            return "(no container)"
        result = subprocess.run(
            [self._docker, "logs", "--tail", "50", self._container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.decode()

    def _sync_stop_container(self) -> None:
        """Stop and remove the container."""
        if self._container_id is None:
            return
        logger.info("Stopping container %s ...", self._container_id)
        subprocess.run(
            [self._docker, "stop", "-t", "5", self._container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [self._docker, "rm", "-f", self._container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._container_id = None
        self._host_port = None

    # ---- sync session helpers (internal) ----

    def _sync_create_session(self) -> str:
        """Create a server-side session via POST /sessions."""
        resp = self._sync_client.post("/sessions")
        Session._raise_for_status(resp, "POST /sessions")
        return resp.json()["session_id"]

    def _sync_delete_session(self) -> None:
        """Delete the server-side session via DELETE /sessions/{id}."""
        endpoint = f"/sessions/{self._sync_session_id}"
        resp = self._sync_client.delete(endpoint)
        Session._raise_for_status(resp, f"DELETE {endpoint}")

    # ---- sync SSE parsing ----

    @staticmethod
    def _parse_sse_sync(resp: httpx.Response) -> Generator[dict, None, None]:
        """Parse SSE lines from an httpx sync streaming response.

        Args:
            resp (httpx.Response): A streaming response to parse SSE from.

        Yields:
            dict: An SSE event with ``"event"`` (str) and ``"data"`` (dict)
                keys.
        """
        event_type: Optional[str] = None
        for line in resp.iter_lines():
            line = line.strip()
            if not line:
                event_type = None
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    data = raw
                yield {"event": event_type or "message", "data": data}

    # ---- async container lifecycle (internal) ----

    async def _ensure_image(self) -> None:
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
        await self._pull_image()

    async def _pull_image(self) -> None:
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

    async def _start_container(self) -> None:
        """Ensure image, run the container, wait for healthy."""
        await self._ensure_image()

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
        await self._get_host_port()
        await self._wait_for_healthy()
        logger.info(
            "Container %s ready on port %s",
            self._container_id, self._host_port,
        )

    async def _get_host_port(self) -> None:
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
        # Output is like "0.0.0.0:32768" or "[::]:32768"
        mapping = stdout.decode().strip().splitlines()[0]
        self._host_port = int(mapping.rsplit(":", 1)[-1])

    async def _wait_for_healthy(self) -> None:
        """Poll GET /health until the container is ready."""
        import httpx

        url = f"http://localhost:{self._host_port}/health"
        deadline = asyncio.get_event_loop().time() + self._startup_timeout
        delay = 0.25

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code == 200:
                        return
            except (httpx.ConnectError, httpx.ReadError, OSError):
                pass
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

        # Timed out — collect container logs for diagnostics
        logs = await self._container_logs()
        await self._stop_container()
        raise ContainerStartError(
            self.image,
            f"Health check timed out after {self._startup_timeout}s. "
            f"Container logs:\n{logs}",
        )

    async def _container_logs(self) -> str:
        """Fetch recent logs from the container."""
        if self._container_id is None:
            return "(no container)"
        proc = await asyncio.create_subprocess_exec(
            self._docker, "logs", "--tail", "50", self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def _stop_container(self) -> None:
        """Stop and remove the container."""
        if self._container_id is None:
            return
        logger.info("Stopping container %s ...", self._container_id)
        stop = await asyncio.create_subprocess_exec(
            self._docker, "stop", "-t", "5", self._container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await stop.communicate()
        rm = await asyncio.create_subprocess_exec(
            self._docker, "rm", "-f", self._container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await rm.communicate()
        self._container_id = None
        self._host_port = None
