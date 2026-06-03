"""Client-side Docker agent manager.

``AgentRef`` manages the lifecycle of a containerized PyAgentic agent:
start, health-check, session creation, and teardown — all driven by
context managers so the container is automatically cleaned up.

Container lifecycle is delegated to :class:`DockerContainer`; SSE
parsing is handled by :mod:`_sse`.
"""

import asyncio
import logging
from typing import Generator, Optional

import httpx

from pyagentic.serve._client_session import Session
from pyagentic.serve._container import DockerContainer
from pyagentic.serve._sse import parse_sse_sync

logger = logging.getLogger(__name__)


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
                await self._ref._container.async_start()
            self._ref._refcount += 1

        base_url = f"http://localhost:{self._ref._container.host_port}"
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
                await self._ref._container.async_stop()


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
        container_port: int = 8000,
        startup_timeout: float = 30.0,
        auto_pull: bool = True,
        docker_path: Optional[str] = None,
    ) -> None:
        self._container = DockerContainer(
            image,
            env=env,
            forward_env=forward_env,
            container_port=container_port,
            startup_timeout=startup_timeout,
            auto_pull=auto_pull,
            docker_path=docker_path,
        )

        # Async session reference counting
        self._refcount: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def image(self) -> str:
        """The Docker image name."""
        return self._container.image

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
        self._container.sync_start()
        try:
            self._sync_client = httpx.Client(
                base_url=f"http://localhost:{self._container.host_port}",
            )
            self._sync_session_id = self._sync_create_session()
        except Exception:
            self._container.sync_stop()
            raise
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
            self._container.sync_stop()

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
            yield from parse_sse_sync(resp)

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
