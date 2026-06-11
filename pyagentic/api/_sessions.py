"""
In-memory session manager. Each session gets its own agent instance with
independent state and conversation history.
"""

import uuid
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyagentic._base._agent._agent import BaseAgent
    from pyagentic.llm._provider import LLMProvider


class SessionManager:
    """Manages agent sessions keyed by session ID.

    Each session holds its own agent instance, so state and conversation
    history are isolated per session.
    """

    def __init__(
        self,
        agent_class: "type[BaseAgent]",
        default_model: Optional[str] = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            agent_class (type[BaseAgent]): The agent class to instantiate for
                each session.
            default_model (Optional[str]): Model string used when a session is
                created without one. ``None`` falls back to the agent class's
                own default model.
        """
        self._agent_class = agent_class
        self._default_model = default_model
        self._sessions: dict[str, "BaseAgent"] = {}

    def create(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional["LLMProvider"] = None,
    ) -> str:
        """Create a new session with a fresh agent instance.

        Args:
            model (Optional[str]): LLM model string (e.g. 'openai::gpt-4o').
                Falls back to the manager's ``default_model``.
            api_key (Optional[str]): API key for the model provider.
            provider (Optional[LLMProvider]): Pre-configured LLMProvider
                instance. Overrides model/api_key.

        Returns:
            str: The new session ID.
        """
        session_id = uuid.uuid4().hex[:12]

        if provider is not None:
            agent = self._agent_class(provider=provider)
        else:
            kwargs: dict = {"api_key": api_key}
            resolved_model = model or self._default_model
            if resolved_model is not None:
                kwargs["model"] = resolved_model
            agent = self._agent_class(**kwargs)

        self._sessions[session_id] = agent
        return session_id

    def get(self, session_id: str) -> "BaseAgent":
        """Get the agent instance for a session.

        Args:
            session_id (str): The session identifier.

        Returns:
            BaseAgent: The agent instance bound to the session.

        Raises:
            KeyError: If the session does not exist.
        """
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"Session not found: {session_id}")

    def exists(self, session_id: str) -> bool:
        """Return whether a session exists.

        Args:
            session_id (str): The session identifier.

        Returns:
            bool: True if the session is active.
        """
        return session_id in self._sessions

    def delete(self, session_id: str) -> None:
        """Delete a session and its agent instance.

        Args:
            session_id (str): The session identifier.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        """Return all active session IDs.

        Returns:
            list[str]: List of session ID strings.
        """
        return list(self._sessions.keys())
