import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import httpx

from pyagentic.serve._agent_ref import AgentRef, _AgentRefSessionContext
from pyagentic.serve._container import DockerContainer
from pyagentic.serve._exceptions import (
    AgentAPIError,
    ContainerStartError,
    ImageNotFoundError,
)


# ---- helpers ----


def _make_process(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _mock_subprocess_exec(*commands):
    """Return an async factory that yields mock processes for each command."""
    call_count = 0
    results = list(commands)

    async def factory(*args, **kwargs):
        nonlocal call_count
        if call_count < len(results):
            proc = results[call_count]
            call_count += 1
            return proc
        return _make_process()

    return factory


SESSION_ID = "sess123"


def _mock_http_routes():
    """Build mock httpx transport routes for session lifecycle + health."""
    return {
        ("GET", "/health"): lambda r: httpx.Response(
            200, json={"status": "ok"}, request=r,
        ),
        ("POST", "/sessions"): lambda r: httpx.Response(
            201, json={"session_id": SESSION_ID}, request=r,
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200, json={"deleted": SESSION_ID}, request=r,
        ),
    }


# ---- tests ----


def test_agent_ref_no_docker(monkeypatch):
    """Test that AgentRef raises FileNotFoundError when docker is not found."""
    monkeypatch.setattr("shutil.which", lambda x: None)
    with pytest.raises(FileNotFoundError, match="Docker is not installed"):
        AgentRef("test:latest")


def test_agent_ref_env_forwarding(monkeypatch):
    """Test that forward_env reads from host os.environ."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    monkeypatch.setenv("MY_KEY", "my_value")
    monkeypatch.setenv("MISSING_KEY", "")

    ref = AgentRef(
        "test:latest",
        env={"EXTRA": "val"},
        forward_env=["MY_KEY", "NONEXISTENT"],
        docker_path="/usr/bin/docker",
    )
    assert ref._container._env["EXTRA"] == "val"
    assert ref._container._env["MY_KEY"] == "my_value"
    assert "NONEXISTENT" not in ref._container._env


def test_agent_ref_explicit_env(monkeypatch):
    """Test that explicit env dict is stored."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("img:v1", env={"A": "1", "B": "2"}, docker_path="/usr/bin/docker")
    assert ref._container._env == {"A": "1", "B": "2"}


def test_session_returns_context_manager(monkeypatch):
    """Test that session() returns a _AgentRefSessionContext."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")
    ctx = ref.session(model="openai::gpt-4o", api_key="sk-test")
    assert isinstance(ctx, _AgentRefSessionContext)


@pytest.mark.asyncio
async def test_ensure_image_exists_locally(monkeypatch):
    """Test async_ensure_image passes when docker image inspect succeeds."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    inspect_proc = _make_process(returncode=0)
    with patch("asyncio.create_subprocess_exec", return_value=inspect_proc):
        await ref._container.async_ensure_image()


@pytest.mark.asyncio
async def test_ensure_image_pulls_when_missing(monkeypatch):
    """Test async_ensure_image pulls when inspect fails and auto_pull is on."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", auto_pull=True, docker_path="/usr/bin/docker")

    inspect_proc = _make_process(returncode=1, stderr=b"not found")
    pull_proc = _make_process(returncode=0)

    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return inspect_proc
        return pull_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        await ref._container.async_ensure_image()

    assert call_count == 2


@pytest.mark.asyncio
async def test_ensure_image_raises_when_no_auto_pull(monkeypatch):
    """Test async_ensure_image raises ImageNotFoundError when auto_pull is off."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", auto_pull=False, docker_path="/usr/bin/docker")

    inspect_proc = _make_process(returncode=1, stderr=b"not found")
    with patch("asyncio.create_subprocess_exec", return_value=inspect_proc):
        with pytest.raises(ImageNotFoundError, match="auto_pull is disabled"):
            await ref._container.async_ensure_image()


@pytest.mark.asyncio
async def test_pull_image_failure(monkeypatch):
    """Test async_pull_image raises ImageNotFoundError on docker pull failure."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    pull_proc = _make_process(returncode=1, stderr=b"manifest unknown")
    with patch("asyncio.create_subprocess_exec", return_value=pull_proc):
        with pytest.raises(ImageNotFoundError, match="manifest unknown"):
            await ref._container.async_pull_image()


@pytest.mark.asyncio
async def test_start_container_failure(monkeypatch):
    """Test async_start raises ContainerStartError on docker run failure."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    inspect_ok = _make_process(returncode=0)
    run_fail = _make_process(returncode=1, stderr=b"port conflict")

    call_count = 0

    async def mock_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return inspect_ok
        return run_fail

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        with pytest.raises(ContainerStartError, match="port conflict"):
            await ref._container.async_start()


@pytest.mark.asyncio
async def test_get_host_port(monkeypatch):
    """Test _async_resolve_host_port parses docker port output."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")
    ref._container._container_id = "abc123"

    port_proc = _make_process(returncode=0, stdout=b"0.0.0.0:32768\n")
    with patch("asyncio.create_subprocess_exec", return_value=port_proc):
        await ref._container._async_resolve_host_port()
    assert ref._container.host_port == 32768


@pytest.mark.asyncio
async def test_get_host_port_ipv6(monkeypatch):
    """Test _async_resolve_host_port parses IPv6 docker port output."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")
    ref._container._container_id = "abc123"

    port_proc = _make_process(returncode=0, stdout=b"[::]:45000\n")
    with patch("asyncio.create_subprocess_exec", return_value=port_proc):
        await ref._container._async_resolve_host_port()
    assert ref._container.host_port == 45000


@pytest.mark.asyncio
async def test_stop_container(monkeypatch):
    """Test async_stop stops and removes the container."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")
    ref._container._container_id = "abc123"
    ref._container._host_port = 12345

    with patch("asyncio.create_subprocess_exec", return_value=_make_process()):
        await ref._container.async_stop()

    assert ref._container.container_id is None
    assert ref._container.host_port is None


@pytest.mark.asyncio
async def test_stop_container_noop_when_no_container(monkeypatch):
    """Test async_stop is a no-op when no container is running."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")
    ref._container._container_id = None

    # Should not raise or call any subprocess
    await ref._container.async_stop()


@pytest.mark.asyncio
async def test_refcount_lifecycle(monkeypatch):
    """Test that the container starts on first session and stops on last exit."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    start_called = False
    stop_called = False

    async def mock_start(self_):
        nonlocal start_called
        start_called = True
        self_._host_port = 12345

    async def mock_stop(self_):
        nonlocal stop_called
        stop_called = True

    with patch.object(DockerContainer, "async_start", mock_start), \
         patch.object(DockerContainer, "async_stop", mock_stop):

        from pyagentic.serve._client_session import Session

        # Mock the Session context manager
        mock_session = AsyncMock(spec=Session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("pyagentic.serve._agent_ref.Session", return_value=mock_session):
            ctx = ref.session()
            async with ctx:
                assert start_called
                assert ref._refcount == 1
                assert not stop_called

    assert stop_called
    assert ref._refcount == 0


# ---- sync API helpers ----


def _make_subprocess_result(returncode=0, stdout=b"", stderr=b""):
    """Create a mock subprocess.CompletedProcess."""
    return MagicMock(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _sync_subprocess_sequence(*results):
    """Return a side_effect callable that yields results in order."""
    call_count = 0
    result_list = list(results)

    def factory(*args, **kwargs):
        nonlocal call_count
        if call_count < len(result_list):
            r = result_list[call_count]
            call_count += 1
            return r
        return _make_subprocess_result()

    return factory


def _sync_mock_client(routes, base_url="http://localhost:12345"):
    """Build a sync httpx.Client with a mock transport and base_url."""
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.raw_path.decode())
        if key in routes:
            return routes[key](request)
        return httpx.Response(404, json={"detail": "Not found"})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


def _sync_http_routes():
    """Standard sync route map for session lifecycle + health + state."""
    return {
        ("GET", "/health"): lambda r: httpx.Response(
            200, json={"status": "ok"},
        ),
        ("POST", "/sessions"): lambda r: httpx.Response(
            201, json={"session_id": SESSION_ID},
        ),
        ("DELETE", f"/sessions/{SESSION_ID}"): lambda r: httpx.Response(
            200, json={"deleted": SESSION_ID},
        ),
    }


# ---- sync API tests ----


def test_sync_context_manager(monkeypatch):
    """Test that __enter__ starts container + session and __exit__ cleans up."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    start_called = False
    stop_called = False

    def mock_start(self_):
        nonlocal start_called
        start_called = True
        self_._host_port = 12345

    def mock_stop(self_):
        nonlocal stop_called
        stop_called = True

    routes = _sync_http_routes()
    client = _sync_mock_client(routes)

    with patch.object(DockerContainer, "sync_start", mock_start), \
         patch.object(DockerContainer, "sync_stop", mock_stop), \
         patch("httpx.Client", return_value=client):
        with ref as r:
            assert r is ref
            assert start_called
            assert ref._sync_session_id == SESSION_ID

    assert stop_called


def test_sync_run(monkeypatch):
    """Test that run() posts to /sessions/{id}/chat and returns the response dict."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    chat_response = {"output": "Hello back!", "tool_calls": []}
    routes = _sync_http_routes()
    routes[("POST", f"/sessions/{SESSION_ID}/chat")] = lambda r: httpx.Response(
        200, json=chat_response,
    )
    client = _sync_mock_client(routes)

    with patch.object(DockerContainer, "sync_start", lambda s: setattr(s, "_host_port", 12345)), \
         patch.object(DockerContainer, "sync_stop", lambda s: None), \
         patch("httpx.Client", return_value=client):
        with ref:
            result = ref.run(user_input="Hello!")
            assert result == chat_response


def test_sync_step(monkeypatch):
    """Test that step() yields parsed SSE event dicts."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    sse_body = (
        "event: llm_response\n"
        f"data: {json.dumps({'event': 'llm_response', 'data': {'text': 'hi'}})}\n"
        "\n"
        "event: agent_response\n"
        f"data: {json.dumps({'event': 'agent_response', 'data': {'output': 'done'}})}\n"
        "\n"
    )

    routes = _sync_http_routes()
    routes[("POST", f"/sessions/{SESSION_ID}/chat/stream")] = lambda r: httpx.Response(
        200,
        content=sse_body.encode(),
        headers={"content-type": "text/event-stream"},
    )
    client = _sync_mock_client(routes)

    with patch.object(DockerContainer, "sync_start", lambda s: setattr(s, "_host_port", 12345)), \
         patch.object(DockerContainer, "sync_stop", lambda s: None), \
         patch("httpx.Client", return_value=client):
        with ref:
            events = list(ref.step(user_input="More"))
            assert len(events) == 2
            assert events[0]["event"] == "llm_response"
            assert events[1]["event"] == "agent_response"


def test_sync_state_property(monkeypatch):
    """Test that .state returns the agent state dict."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    state_data = {"mood": "happy", "turn_count": 3}
    routes = _sync_http_routes()
    routes[("GET", f"/sessions/{SESSION_ID}/state")] = lambda r: httpx.Response(
        200, json=state_data,
    )
    client = _sync_mock_client(routes)

    with patch.object(DockerContainer, "sync_start", lambda s: setattr(s, "_host_port", 12345)), \
         patch.object(DockerContainer, "sync_stop", lambda s: None), \
         patch("httpx.Client", return_value=client):
        with ref:
            assert ref.state == state_data


def test_sync_run_error(monkeypatch):
    """Test that run() raises AgentAPIError on non-success status."""
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/docker")
    ref = AgentRef("test:latest", docker_path="/usr/bin/docker")

    routes = _sync_http_routes()
    routes[("POST", f"/sessions/{SESSION_ID}/chat")] = lambda r: httpx.Response(
        404, json={"detail": "Session not found"},
    )
    client = _sync_mock_client(routes)

    with patch.object(DockerContainer, "sync_start", lambda s: setattr(s, "_host_port", 12345)), \
         patch.object(DockerContainer, "sync_stop", lambda s: None), \
         patch("httpx.Client", return_value=client):
        with ref:
            with pytest.raises(AgentAPIError, match="404"):
                ref.run(user_input="Hello!")
