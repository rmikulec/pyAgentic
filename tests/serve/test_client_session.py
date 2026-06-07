import json

import pytest
import httpx

from pyagentic.serve._client_session import Session
from pyagentic.serve._exceptions import AgentAPIError


# ---- helpers ----


def _json_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Build an httpx.Response with JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("GET", "http://test"),
    )


def _error_response(status_code: int, detail: str) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json={"detail": detail},
        request=httpx.Request("GET", "http://test"),
    )


class MockTransport(httpx.MockTransport):
    """Convenience transport that dispatches by (method, path)."""

    def __init__(self, routes: dict):
        self._routes = routes

        async def handler(request: httpx.Request) -> httpx.Response:
            key = (request.method, request.url.path)
            if key in self._routes:
                return self._routes[key](request)
            return httpx.Response(404, json={"detail": "not found"}, request=request)

        super().__init__(handler)


# ---- fixtures ----


SESSION_ID = "abc123"


def _default_routes():
    """Routes that simulate a basic agent server."""
    return {
        ("POST", "/sessions"): lambda r: httpx.Response(
            201,
            json={"session_id": SESSION_ID},
            request=r,
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200,
            json={"deleted": SESSION_ID},
            request=r,
        ),
        ("POST", f"/sessions/{SESSION_ID}/chat"): lambda r: httpx.Response(
            200,
            json={"final_output": "hello back"},
            request=r,
        ),
        ("GET", f"/sessions/{SESSION_ID}/state"): lambda r: httpx.Response(
            200,
            json={"notes": {"text": ""}},
            request=r,
        ),
        ("GET", "/"): lambda r: httpx.Response(
            200,
            json={"name": "test", "version": "0.1.0"},
            request=r,
        ),
        ("GET", "/health"): lambda r: httpx.Response(
            200,
            json={"status": "ok"},
            request=r,
        ),
        ("GET", "/schema"): lambda r: httpx.Response(
            200,
            json={"request": {}, "response": {}, "stream_event": {}, "state": {}},
            request=r,
        ),
    }


@pytest.fixture
def mock_client():
    """An httpx.AsyncClient backed by mock routes."""
    transport = MockTransport(_default_routes())
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---- tests ----


@pytest.mark.asyncio
async def test_session_creates_and_deletes(mock_client):
    """Test that entering/exiting the session context creates/deletes a server session."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        assert s._session_id == SESSION_ID
    # After exit, session_id should be cleared
    assert session._session_id is None


@pytest.mark.asyncio
async def test_session_creates_with_model_and_api_key(mock_client):
    """Test that model and api_key are forwarded in the create request."""
    session = Session(
        "http://test",
        model="openai::gpt-4o",
        api_key="sk-test",
        http_client=mock_client,
    )
    async with session as s:
        assert s._session_id == SESSION_ID


@pytest.mark.asyncio
async def test_chat(mock_client):
    """Test Session.chat() returns the agent response."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        result = await s.chat(user_input="hi")
    assert result == {"final_output": "hello back"}


@pytest.mark.asyncio
async def test_state(mock_client):
    """Test Session.state() returns the agent state."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        result = await s.state()
    assert "notes" in result


@pytest.mark.asyncio
async def test_info(mock_client):
    """Test Session.info() returns agent metadata."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        result = await s.info()
    assert result["name"] == "test"


@pytest.mark.asyncio
async def test_health(mock_client):
    """Test Session.health() returns health status."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        result = await s.health()
    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_schema(mock_client):
    """Test Session.schema() returns model schemas."""
    session = Session("http://test", http_client=mock_client)
    async with session as s:
        result = await s.schema()
    assert "request" in result
    assert "response" in result


@pytest.mark.asyncio
async def test_agent_api_error_on_failure():
    """Test that a non-success response raises AgentAPIError."""
    routes = {
        ("POST", "/sessions"): lambda r: httpx.Response(
            201, json={"session_id": SESSION_ID}, request=r,
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200, json={"deleted": SESSION_ID}, request=r,
        ),
        ("POST", f"/sessions/{SESSION_ID}/chat"): lambda r: httpx.Response(
            500, json={"detail": "internal error"}, request=r,
        ),
    }
    transport = MockTransport(routes)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    session = Session("http://test", http_client=client)
    async with session as s:
        with pytest.raises(AgentAPIError) as exc_info:
            await s.chat(user_input="crash")
        assert exc_info.value.status_code == 500
        assert "internal error" in exc_info.value.detail


@pytest.mark.asyncio
async def test_stream():
    """Test Session.stream() parses SSE events."""
    sse_body = (
        "event: llm_response\n"
        'data: {"text": "hello"}\n'
        "\n"
        "event: agent_response\n"
        'data: {"final_output": "done"}\n'
        "\n"
    )

    routes = {
        ("POST", "/sessions"): lambda r: httpx.Response(
            201, json={"session_id": SESSION_ID}, request=r,
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200, json={"deleted": SESSION_ID}, request=r,
        ),
        ("POST", f"/sessions/{SESSION_ID}/chat/stream"): lambda r: httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
            request=r,
        ),
    }
    transport = MockTransport(routes)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    session = Session("http://test", http_client=client)
    events = []
    async with session as s:
        async for event in s.stream(user_input="stream me"):
            events.append(event)

    assert len(events) == 2
    assert events[0]["event"] == "llm_response"
    assert events[0]["data"] == {"text": "hello"}
    assert events[1]["event"] == "agent_response"
    assert events[1]["data"] == {"final_output": "done"}


@pytest.mark.asyncio
async def test_parse_sse_non_json_data():
    """Test that non-JSON data values are returned as raw strings."""
    sse_body = "event: log\ndata: plain text\n\n"
    routes = {
        ("POST", "/sessions"): lambda r: httpx.Response(
            201, json={"session_id": SESSION_ID}, request=r,
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200, json={"deleted": SESSION_ID}, request=r,
        ),
        ("POST", f"/sessions/{SESSION_ID}/chat/stream"): lambda r: httpx.Response(
            200,
            content=sse_body.encode(),
            headers={"content-type": "text/event-stream"},
            request=r,
        ),
    }
    transport = MockTransport(routes)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")

    session = Session("http://test", http_client=client)
    events = []
    async with session as s:
        async for event in s.stream(user_input="hi"):
            events.append(event)

    assert len(events) == 1
    assert events[0]["data"] == "plain text"
