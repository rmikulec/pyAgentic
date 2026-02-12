"""Async HTTP session client for communicating with a containerized PyAgentic agent."""

from typing import AsyncGenerator, Optional

import httpx

from pyagentic.serve._exceptions import AgentAPIError


class Session:
    """Async context manager that wraps an ``httpx.AsyncClient`` and manages
    a server-side agent session over HTTP.

    On enter, creates a session via ``POST /sessions``.
    On exit, deletes it via ``DELETE /sessions/{id}``.

    Args:
        base_url (str): Base URL of the running agent container
            (e.g. ``http://localhost:12345``).
        model (Optional[str]): LLM model override forwarded to the agent.
        api_key (Optional[str]): API key forwarded to the agent.
        http_client (Optional[httpx.AsyncClient]): Pre-configured HTTP client.
            If provided, the caller is responsible for closing it.

    Raises:
        AgentAPIError: If the server returns a non-success status on any call.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._external_client = http_client is not None
        self._client = http_client or httpx.AsyncClient(base_url=self._base_url)
        self._session_id: Optional[str] = None

    async def __aenter__(self) -> "Session":
        """Create a server-side session and return self."""
        body: dict = {}
        if self._model is not None:
            body["model"] = self._model
        if self._api_key is not None:
            body["api_key"] = self._api_key

        resp = await self._client.post("/sessions", json=body or None)
        self._raise_for_status(resp, "POST /sessions")
        self._session_id = resp.json()["session_id"]
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Delete the server-side session and close the HTTP client."""
        if self._session_id is not None:
            try:
                resp = await self._client.delete(f"/sessions/{self._session_id}")
                self._raise_for_status(resp, f"DELETE /sessions/{self._session_id}")
            finally:
                self._session_id = None
        if not self._external_client:
            await self._client.aclose()

    # ---- agent API methods ----

    async def chat(self, **kwargs) -> dict:
        """Send a chat message and receive the complete agent response.

        Args:
            **kwargs: Request fields forwarded as JSON body to the agent's
                chat endpoint (e.g. ``user_input="Hello"``).

        Returns:
            dict: The full agent response as a dictionary.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._session_id}/chat"
        resp = await self._client.post(endpoint, json=kwargs)
        self._raise_for_status(resp, f"POST {endpoint}")
        return resp.json()

    async def stream(self, **kwargs) -> AsyncGenerator[dict, None]:
        """Send a chat message and receive SSE events as an async generator.

        Each yielded dict has ``"event"`` and ``"data"`` keys parsed from
        the SSE stream.

        Args:
            **kwargs: Request fields forwarded as JSON body to the agent's
                stream endpoint.

        Yields:
            dict: An SSE event with ``"event"`` (str) and ``"data"`` (dict) keys.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._session_id}/chat/stream"
        async with self._client.stream("POST", endpoint, json=kwargs) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_for_status(resp, f"POST {endpoint}")
            async for event in self._parse_sse(resp):
                yield event

    async def state(self) -> dict:
        """Get the current agent state for this session.

        Returns:
            dict: The agent state as a dictionary.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        endpoint = f"/sessions/{self._session_id}/state"
        resp = await self._client.get(endpoint)
        self._raise_for_status(resp, f"GET {endpoint}")
        return resp.json()

    async def info(self) -> dict:
        """Get agent metadata from the root endpoint.

        Returns:
            dict: Agent metadata including name, version, tools, etc.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        resp = await self._client.get("/")
        self._raise_for_status(resp, "GET /")
        return resp.json()

    async def health(self) -> dict:
        """Check the agent container health.

        Returns:
            dict: Health status (e.g. ``{"status": "ok"}``).

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        resp = await self._client.get("/health")
        self._raise_for_status(resp, "GET /health")
        return resp.json()

    async def schema(self) -> dict:
        """Get JSON schemas for the agent's request, response, stream event,
        and state models.

        Returns:
            dict: A dictionary of JSON schema objects keyed by model type.

        Raises:
            AgentAPIError: If the server returns a non-success status.
        """
        resp = await self._client.get("/schema")
        self._raise_for_status(resp, "GET /schema")
        return resp.json()

    # ---- helpers ----

    @staticmethod
    def _raise_for_status(resp: httpx.Response, endpoint: str) -> None:
        """Raise AgentAPIError if the response indicates failure."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise AgentAPIError(resp.status_code, detail, endpoint)

    @staticmethod
    async def _parse_sse(resp: httpx.Response) -> AsyncGenerator[dict, None]:
        """Parse SSE lines from an httpx streaming response."""
        import json

        event_type: Optional[str] = None
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                # Blank line resets for next event
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
