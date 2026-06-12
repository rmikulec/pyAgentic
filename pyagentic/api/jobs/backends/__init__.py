"""
Execution backend implementations and the factory.

Only the in-process backend is shipped: it runs ``agent.step()`` in the serve
process's own event loop with zero external dependencies.
"""

from typing import TYPE_CHECKING, Optional

from pyagentic._base._agent._agent import BaseAgent
from pyagentic.api.jobs.backends._base import EmitFn, ExecutionBackend
from pyagentic.api.jobs.backends._in_process import InProcessBackend

if TYPE_CHECKING:
    from pyagentic.api._sessions import SessionManager

__all__ = ["EmitFn", "ExecutionBackend", "InProcessBackend", "build_backend"]


def build_backend(
    agent_class: type[BaseAgent],
    sessions: "SessionManager",
    *,
    max_concurrency: int = 8,
    default_model: Optional[str] = None,
    dependencies: Optional[list] = None,
) -> ExecutionBackend:
    """Build the in-process ExecutionBackend.

    Args:
        agent_class (type[BaseAgent]): The agent class being served.
        sessions (SessionManager): Live session registry.
        max_concurrency (int): Max concurrent agent runs.
        default_model (Optional[str]): Model for sessionless agents.
        dependencies (Optional[list]): Providers for ``Depends[T]`` fields used
            when building a fresh agent for a sessionless job.

    Returns:
        ExecutionBackend: A configured :class:`InProcessBackend`.
    """
    return InProcessBackend(
        agent_class,
        sessions,
        max_concurrency=max_concurrency,
        default_model=default_model,
        dependencies=dependencies,
    )
