"""
In-memory session manager. Each session gets its own agent instance with
independent state and conversation history.
"""

import uuid
from typing import TYPE_CHECKING, Optional

from pyagentic.serve._manifest import Manifest

if TYPE_CHECKING:
    from pyagentic._base._agent._agent import BaseAgent
    from pyagentic.llm._provider import LLMProvider


class SessionManager:
    """Manages agent sessions keyed by session ID."""

    def __init__(self, agent_class: "type[BaseAgent]", manifest: Manifest) -> None:
        self._agent_class = agent_class
        self._manifest = manifest
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
            model: LLM model string (e.g. 'openai::gpt-4o'). Falls back to
                the manifest default.
            api_key: API key for the model provider.
            provider: Pre-configured LLMProvider instance. Overrides model/api_key.

        Returns:
            The new session ID.
        """
        session_id = uuid.uuid4().hex[:12]

        if provider is not None:
            agent = self._agent_class(provider=provider)
        else:
            agent = self._agent_class(
                model=model or self._manifest.agent.model,
                api_key=api_key,
            )

        self._sessions[session_id] = agent
        return session_id

    def get(self, session_id: str) -> "BaseAgent":
        """Get the agent instance for a session.

        Raises:
            KeyError: If the session does not exist.
        """
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"Session not found: {session_id}")

    def delete(self, session_id: str) -> None:
        """Delete a session.

        Raises:
            KeyError: If the session does not exist.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        del self._sessions[session_id]

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())
