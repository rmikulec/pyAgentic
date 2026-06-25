"""
Default execution backend: runs agent.step() inside the server process's
own event loop. Zero external dependencies — no Docker, no cloud.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Optional

from pyagentic._base._agent._agent import BaseAgent
from pyagentic.models.llm import LLMResponse
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.api.jobs._models import (
    BackendHealth,
    JobRecord,
    RunOutcome,
    _build_prompt,
)
from pyagentic.api.jobs.backends._base import EmitFn, LocalExecutionBackend
from pyagentic.api._build import build_agent

if TYPE_CHECKING:
    from pyagentic.api._sessions import SessionManager


class InProcessBackend(LocalExecutionBackend):
    """Runs jobs as coroutines in the current event loop.

    Session-bound jobs run on the session's live in-memory agent (the
    orchestrator serializes them, so the agent is never run concurrently);
    sessionless jobs get a fresh default agent that is closed after the run.
    """

    def __init__(
        self,
        agent_class: type[BaseAgent],
        sessions: "SessionManager",
        *,
        max_concurrency: int = 8,
        default_model: Optional[str] = None,
        dependencies: Optional[list] = None,
    ) -> None:
        """Create the backend.

        Args:
            agent_class (type[BaseAgent]): Agent class to instantiate for
                sessionless jobs.
            sessions (SessionManager): Live session registry for session-bound jobs.
            max_concurrency (int): Max concurrent agent runs.
            default_model (Optional[str]): Model for sessionless agents.
            dependencies (Optional[list]): Providers for the agent's ``Depends[T]``
                fields when building a fresh agent for a sessionless job.
        """
        super().__init__()
        self._agent_class = agent_class
        self._sessions = sessions
        self._default_model = default_model
        self._dependencies = dependencies or []
        self._max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _execute(
        self,
        job: JobRecord,
        emit: EmitFn,
        session_id: Optional[str],
    ) -> RunOutcome:
        """Drive agent.step() on the session's agent (or a fresh one), emitting each update."""
        StreamEventModel = self._agent_class.__stream_event_model__
        LLMEvent = StreamEventModel.__llm_event__
        ToolEvent = StreamEventModel.__tool_event__
        AgentEvent = StreamEventModel.__agent_event__

        # Session-bound jobs reuse the live agent (owned by the SessionManager,
        # so we don't close it); sessionless jobs get a throwaway agent.
        if session_id is not None:
            agent = self._sessions.get(session_id)
            own_agent = False
        else:
            agent = build_agent(
                self._agent_class,
                job.construct_payload,
                self._dependencies,
                default_model=self._default_model,
            )
            own_agent = True

        prompt = _build_prompt(job.request)
        result_json: Optional[str] = None
        try:
            async with self._sem:
                async for update in agent.step(prompt):
                    if isinstance(update, LLMResponse):
                        event = LLMEvent(data=update)
                    elif isinstance(update, ToolResponse):
                        event = ToolEvent(data=update)
                    elif isinstance(update, AgentResponse):
                        event = AgentEvent(data=update)
                    else:
                        payload = (
                            update.model_dump()
                            if hasattr(update, "model_dump")
                            else str(update)
                        )
                        await emit("update", json.dumps(payload, default=str))
                        continue
                    await emit(event.event, event.model_dump_json())
                    if isinstance(update, AgentResponse):
                        result_json = update.model_dump_json()
            return RunOutcome(result_json=result_json)
        finally:
            if own_agent:
                await agent.close()

    async def health(self) -> BackendHealth:
        """Report current in-flight run count.

        Returns:
            BackendHealth: Health snapshot with running-job diagnostics.
        """
        return BackendHealth(
            ok=True,
            detail={
                "kind": "in_process",
                "running_jobs": len(self._tasks),
                "max_concurrency": self._max_concurrency,
            },
        )
