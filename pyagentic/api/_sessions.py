"""
In-memory session manager. Each session gets its own agent instance with
independent state and conversation history.
"""

import uuid
from typing import TYPE_CHECKING, Any, Optional

from pyagentic.api._build import build_agent

if TYPE_CHECKING:
    from pyagentic._base._agent._agent import BaseAgent


class SessionManager:
    """Manages agent sessions keyed by session ID.

    Each session holds its own agent instance, so state and conversation
    history are isolated per session. Agents are constructed from a
    construct-model payload (mirroring the constructor) plus developer-supplied
    dependencies, via :func:`pyagentic.api._build.build_agent`.
    """

    def __init__(
        self,
        agent_class: "type[BaseAgent]",
        default_model: Optional[str] = None,
        dependencies: Optional[list] = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            agent_class (type[BaseAgent]): The agent class to instantiate for
                each session.
            default_model (Optional[str]): Model string used when a session is
                created without one in its construct payload. ``None`` falls
                back to the agent class's own default model.
            dependencies (Optional[list]): Instances or factories used to satisfy
                the agent tree's ``Depends[T]`` fields (resolved by type).
        """
        self._agent_class = agent_class
        self._default_model = default_model
        self._dependencies = dependencies or []
        self._sessions: dict[str, "BaseAgent"] = {}

    def create(self, *, construct_data: Any = None) -> str:
        """Create a new session with a fresh agent instance.

        Args:
            construct_data (Any): An instance of the agent's
                ``__construct_model__`` (or a dict of the same shape) carrying
                the serializable construction data (state, nested linked-agent
                state, ``model``/``api_key``). ``None`` builds with defaults.

        Returns:
            str: The new session ID.
        """
        session_id = uuid.uuid4().hex[:12]

        agent = build_agent(
            self._agent_class,
            construct_data,
            self._dependencies,
            default_model=self._default_model,
        )

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
