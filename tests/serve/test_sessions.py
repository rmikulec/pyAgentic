import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, tool
from pyagentic.serve._manifest import Manifest
from pyagentic.serve._sessions import SessionManager


class _SessionTestAgent(BaseAgent):
    __system_message__ = "Test agent for session tests"
    __input_template__ = ""


def _make_manifest() -> Manifest:
    return Manifest(
        project={"name": "test", "version": "0.1.0"},
        agent={"entry": "test:TestAgent", "model": "_mock::test-model"},
    )


def test_create_session():
    """Test creating a new session returns a session ID."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    sid = sm.create(model="_mock::test-model", api_key="key")
    assert isinstance(sid, str)
    assert len(sid) == 12


def test_get_session():
    """Test getting a session returns the agent instance."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    sid = sm.create(model="_mock::test-model", api_key="key")
    agent = sm.get(sid)
    assert isinstance(agent, _SessionTestAgent)


def test_get_nonexistent_session():
    """Test getting a non-existent session raises KeyError."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    with pytest.raises(KeyError, match="Session not found"):
        sm.get("doesnotexist")


def test_delete_session():
    """Test deleting a session removes it."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    sid = sm.create(model="_mock::test-model", api_key="key")
    sm.delete(sid)
    with pytest.raises(KeyError):
        sm.get(sid)


def test_delete_nonexistent_session():
    """Test deleting a non-existent session raises KeyError."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    with pytest.raises(KeyError, match="Session not found"):
        sm.delete("doesnotexist")


def test_list_sessions():
    """Test listing sessions returns all active session IDs."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    sessions = sm.list_sessions()
    assert s1 in sessions
    assert s2 in sessions
    assert len(sessions) == 2


def test_sessions_are_independent():
    """Test that each session gets its own independent agent instance."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    agent1 = sm.get(s1)
    agent2 = sm.get(s2)
    assert agent1 is not agent2


def test_create_session_uses_manifest_model():
    """Test that create() falls back to the manifest model when none is provided."""
    manifest = Manifest(
        project={"name": "test", "version": "0.1.0"},
        agent={"entry": "test:TestAgent", "model": "_mock::test-model"},
    )
    sm = SessionManager(_SessionTestAgent, manifest)
    sid = sm.create(api_key="key")
    agent = sm.get(sid)
    assert isinstance(agent, _SessionTestAgent)


def test_list_sessions_after_delete():
    """Test that deleted sessions are removed from the list."""
    sm = SessionManager(_SessionTestAgent, _make_manifest())
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    sm.delete(s1)
    sessions = sm.list_sessions()
    assert s1 not in sessions
    assert s2 in sessions
    assert len(sessions) == 1
