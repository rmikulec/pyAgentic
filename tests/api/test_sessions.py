import pytest

from pyagentic import BaseAgent
from pyagentic.api._sessions import SessionManager


class _SessionTestAgent(BaseAgent):
    __system_message__ = "Test agent for session tests"
    __input_template__ = ""


def _make_manager() -> SessionManager:
    return SessionManager(_SessionTestAgent, default_model="_mock::test-model")


def test_create_session():
    """Test creating a new session returns a session ID."""
    sm = _make_manager()
    sid = sm.create(model="_mock::test-model", api_key="key")
    assert isinstance(sid, str)
    assert len(sid) == 12


def test_get_session():
    """Test getting a session returns the agent instance."""
    sm = _make_manager()
    sid = sm.create(model="_mock::test-model", api_key="key")
    agent = sm.get(sid)
    assert isinstance(agent, _SessionTestAgent)


def test_get_nonexistent_session():
    """Test getting a non-existent session raises KeyError."""
    sm = _make_manager()
    with pytest.raises(KeyError, match="Session not found"):
        sm.get("doesnotexist")


def test_delete_session():
    """Test deleting a session removes it."""
    sm = _make_manager()
    sid = sm.create(model="_mock::test-model", api_key="key")
    sm.delete(sid)
    with pytest.raises(KeyError):
        sm.get(sid)


def test_delete_nonexistent_session():
    """Test deleting a non-existent session raises KeyError."""
    sm = _make_manager()
    with pytest.raises(KeyError, match="Session not found"):
        sm.delete("doesnotexist")


def test_list_sessions():
    """Test listing sessions returns all active session IDs."""
    sm = _make_manager()
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    sessions = sm.list_sessions()
    assert s1 in sessions
    assert s2 in sessions
    assert len(sessions) == 2


def test_sessions_are_independent():
    """Test that each session gets its own independent agent instance."""
    sm = _make_manager()
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    assert sm.get(s1) is not sm.get(s2)


def test_create_session_uses_default_model():
    """Test that create() falls back to the manager's default_model."""
    sm = SessionManager(_SessionTestAgent, default_model="_mock::test-model")
    sid = sm.create(api_key="key")
    agent = sm.get(sid)
    assert isinstance(agent, _SessionTestAgent)
    assert agent.model == "_mock::test-model"


def test_list_sessions_after_delete():
    """Test that deleted sessions are removed from the list."""
    sm = _make_manager()
    s1 = sm.create(model="_mock::test-model", api_key="key")
    s2 = sm.create(model="_mock::test-model", api_key="key")
    sm.delete(s1)
    sessions = sm.list_sessions()
    assert s1 not in sessions
    assert s2 in sessions
    assert len(sessions) == 1
