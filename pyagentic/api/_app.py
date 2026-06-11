"""
FastAPI app and router factories for serving PyAgentic agents over HTTP.

Two entry points:
  - ``create_router(agent)`` builds an ``APIRouter`` for a single agent, to mount
    onto your own FastAPI app.
  - ``create_app(agents)`` builds a standalone ``FastAPI`` app serving one or
    several agents (each under its own prefix).

Routes are derived from the metaclass-generated models on the agent class:
  - __request_model__        →  request body on chat endpoints
  - __response_model__       →  response_model on chat endpoint
  - __stream_event_model__   →  typed SSE events on stream endpoint
  - __state_class__          →  response_model on state endpoint
"""

import asyncio
import contextlib
import json
import re
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyagentic._base._agent._agent import BaseAgent
from pyagentic.models.llm import LLMResponse
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.api._config import AgentsConfig, load_config
from pyagentic.api._sessions import SessionManager

# Agents may be passed as a single class, a list of classes, or a
# {url_prefix: class} mapping.
AgentsArg = Union[type[BaseAgent], list[type[BaseAgent]], dict[str, type[BaseAgent]]]


class CreateSessionRequest(BaseModel):
    """Request body for creating a new agent session.

    Attributes:
        model (Optional[str]): LLM model string override
            (e.g. ``'openai::gpt-4o'``).
        api_key (Optional[str]): API key for the model provider.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None


_REQUIRED_AGENT_ATTRS = (
    "__request_model__",
    "__response_model__",
    "__stream_event_model__",
    "__state_class__",
    "__tool_defs__",
    "__state_defs__",
    "__linked_agents__",
)


def _validate_agent_class(cls: type) -> None:
    """Verify that *cls* has the metaclass-generated attributes required by the server.

    Args:
        cls (type): The class to validate.

    Raises:
        TypeError: If *cls* is missing required metaclass attributes.
    """
    missing = [attr for attr in _REQUIRED_AGENT_ATTRS if not hasattr(cls, attr)]
    if missing:
        raise TypeError(
            f"{cls.__name__} is not a valid PyAgentic agent class. "
            f"Missing attributes: {', '.join(missing)}. "
            f"Ensure it inherits from BaseAgent (which uses AgentMeta)."
        )


def _default_prefix(agent_class: type[BaseAgent]) -> str:
    """Derive a URL prefix from an agent class name (e.g. ResearchAgent -> /research)."""
    name = agent_class.__name__
    if name.endswith("Agent") and len(name) > len("Agent"):
        name = name[: -len("Agent")]
    # Insert a hyphen at lower/digit -> upper boundaries (CamelCase -> kebab).
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name).lower()
    # Collapse any remaining non-alphanumeric runs into single hyphens.
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return "/" + name


def _store_path_for(base: str, prefix: str, multi: bool) -> str:
    """Derive a per-agent job-store path so multiple agents don't share one DB file."""
    if base == ":memory:" or not multi:
        return base
    slug = prefix.strip("/").replace("/", "-") or "root"
    p = Path(base)
    return str(p.with_name(f"{p.stem}-{slug}{p.suffix}"))


def _normalize_agents(agents: AgentsArg) -> dict[str, type[BaseAgent]]:
    """Normalize the ``agents`` argument into a {prefix: class} mapping.

    A single class is mounted at the root (""); a list derives a prefix from
    each class name; a dict is used as-is.
    """
    if isinstance(agents, dict):
        mapping = dict(agents)
    elif isinstance(agents, (list, tuple)):
        if len(agents) == 1:
            mapping = {"": agents[0]}
        else:
            mapping = {}
            for cls in agents:
                prefix = _default_prefix(cls)
                if prefix in mapping:
                    raise ValueError(
                        f"Multiple agents map to prefix '{prefix}'. "
                        f"Pass a {{prefix: agent}} dict to disambiguate."
                    )
                mapping[prefix] = cls
    else:
        mapping = {"": agents}

    if not mapping:
        raise ValueError("create_app requires at least one agent.")
    return mapping


def create_router(
    agent_class: type[BaseAgent],
    *,
    model: Optional[str] = None,
    name: Optional[str] = None,
    version: str = "0.1.0",
) -> APIRouter:
    """Build a FastAPI ``APIRouter`` exposing a single agent's endpoints.

    Mount it onto an existing FastAPI app::

        app.include_router(create_router(MyAgent), prefix="/agent")

    The router owns its own :class:`SessionManager`, exposed as
    ``router.sessions`` so callers can share it (e.g. with :func:`mount_mcp`).

    Routes registered (relative to the mount point):
      - ``GET /`` agent info, ``GET /health``, ``GET /schema``
      - ``POST/GET /sessions``, ``DELETE /sessions/{id}``
      - ``POST /sessions/{id}/chat``, ``POST /sessions/{id}/chat/stream``
      - ``GET /sessions/{id}/state``

    Args:
        agent_class (type[BaseAgent]): The agent class to serve.
        model (Optional[str]): Default model for sessions. ``None`` falls back to
            the agent's own default.
        name (Optional[str]): Display name for the info route. Defaults to the
            class name.
        version (str): Version string for the info route.

    Returns:
        APIRouter: A router with all agent routes registered, with its
            ``SessionManager`` attached as ``router.sessions``.

    Raises:
        TypeError: If agent_class is missing required metaclass attributes.
    """
    _validate_agent_class(agent_class)

    name = name or agent_class.__name__
    RequestModel = agent_class.__request_model__
    ResponseModel = agent_class.__response_model__
    StreamEventModel = agent_class.__stream_event_model__
    StateModel = agent_class.__state_class__

    router = APIRouter()
    sessions = SessionManager(agent_class, default_model=model)

    # ---- info routes ----

    @router.get("/")
    async def agent_info() -> dict:
        """Return basic agent metadata."""
        return {
            "name": name,
            "version": version,
            "agent_class": agent_class.__name__,
            "tools": list(agent_class.__tool_defs__.keys()),
            "state_fields": list(agent_class.__state_defs__.keys()),
            "linked_agents": list(agent_class.__linked_agents__.keys()),
        }

    @router.get("/health")
    async def health() -> dict:
        """Liveness probe endpoint."""
        return {"status": "ok"}

    @router.get("/schema")
    async def schema() -> dict:
        """Return JSON schemas for request, response, stream event, and state models."""
        return {
            "request": RequestModel.model_json_schema(),
            "response": ResponseModel.model_json_schema(),
            "stream_event": StreamEventModel.model_json_schema(),
            "state": StateModel.model_json_schema(),
        }

    # ---- session routes ----

    @router.post("/sessions", status_code=201)
    async def create_session(req: Optional[CreateSessionRequest] = None) -> dict:
        """Create a new agent session."""
        model = req.model if req else None
        api_key = req.api_key if req else None
        session_id = sessions.create(model=model, api_key=api_key)
        return {"session_id": session_id}

    @router.get("/sessions")
    async def list_sessions() -> dict:
        """List all active session IDs."""
        return {"sessions": sessions.list_sessions()}

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        """Delete an existing session."""
        try:
            sessions.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    # ---- chat routes ----

    @router.post("/sessions/{session_id}/chat", response_model=ResponseModel)
    async def chat(session_id: str, req: RequestModel):  # type: ignore[valid-type]
        """Send a message and receive a complete agent response."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

        kwargs = {field: getattr(req, field) for field in req.model_fields}
        response = await agent(**kwargs)
        return response

    @router.post(
        "/sessions/{session_id}/chat/stream",
        response_model=StreamEventModel,
        responses={
            200: {
                "description": (
                    "SSE stream. Each line is `event: <type>\\ndata: <json>`. "
                    "Possible event types: llm_response, tool_response, agent_response."
                ),
            }
        },
    )
    async def chat_stream(session_id: str, req: RequestModel):  # type: ignore[valid-type]
        """Send a message and receive agent responses as an SSE stream."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

        # Build a single prompt string from the request fields for step().
        # NOTE: step() currently only accepts a string argument. For agents
        # with multiple input fields, we serialize to JSON to preserve
        # structure better than naive stringification.
        kwargs = {field: getattr(req, field) for field in req.model_fields}
        non_none = {k: v for k, v in kwargs.items() if v is not None}
        if len(non_none) == 1:
            prompt = str(next(iter(non_none.values())))
        else:
            prompt = json.dumps(non_none, default=str)

        # Grab the typed event wrappers for constructing SSE payloads
        LLMEvent = StreamEventModel.__llm_event__
        ToolEvent = StreamEventModel.__tool_event__
        AgentEvent = StreamEventModel.__agent_event__

        async def event_generator():
            """Yield SSE-formatted events from the agent step iterator."""
            async for update in agent.step(prompt):
                if isinstance(update, LLMResponse):
                    event = LLMEvent(data=update)
                elif isinstance(update, ToolResponse):
                    event = ToolEvent(data=update)
                elif isinstance(update, AgentResponse):
                    event = AgentEvent(data=update)
                else:
                    # Fallback for unexpected types
                    payload = update.model_dump() if hasattr(update, "model_dump") else str(update)
                    yield f"event: update\ndata: {json.dumps(payload, default=str)}\n\n"
                    continue

                yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ---- state route ----

    @router.get("/sessions/{session_id}/state", response_model=StateModel)
    async def get_state(session_id: str):
        """Return the current state of the agent for a session."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.state

    # Expose the session manager so callers can share it (e.g. with mount_mcp).
    router.sessions = sessions

    return router


def _wire_jobs_lifespan(app: FastAPI, orchestrators: list) -> None:
    """Start/stop the per-agent job orchestrators inside the app's lifespan.

    Wraps the host app's existing lifespan (the same pattern ``mount_mcp`` uses)
    so each orchestrator's store is initialized, in-flight jobs are recovered,
    and the TTL purge loop runs while the app is up — then torn down on shutdown.
    """
    prev_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def _lifespan_with_jobs(host_app):
        purge_tasks = []
        for orch in orchestrators:
            await orch.ensure_started()
            await orch.recover()
            purge_tasks.append(asyncio.create_task(orch.purge_loop()))
        try:
            async with prev_lifespan(host_app):
                yield
        finally:
            for task in purge_tasks:
                task.cancel()
            for orch in orchestrators:
                await orch.shutdown()
                await orch.store.close()

    app.router.lifespan_context = _lifespan_with_jobs


def create_app(
    agents: AgentsArg,
    *,
    config: Optional[AgentsConfig] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    model: Optional[str] = None,
    mcp: bool = False,
    jobs: bool = False,
) -> FastAPI:
    """Build a standalone FastAPI app serving one or several agents.

    A single agent is mounted at the root; multiple agents are each mounted
    under a prefix (derived from the class name, or supplied via a
    ``{prefix: agent}`` dict), with a top-level ``GET /`` index and
    ``GET /health``::

        create_app(MyAgent)                          # served at /
        create_app([Research, Writer])               # /research/*, /writer/*
        create_app({"/a": AgentA, "/b": AgentB})     # explicit prefixes

    App metadata (name/version/description/default model) defaults to the
    ``[app]`` section of ``./agents.toml`` when present; explicit keyword
    arguments override it.

    Args:
        agents (AgentsArg): A single agent class, a list of classes, or a
            ``{prefix: class}`` mapping.
        config (Optional[AgentsConfig]): Parsed config. If ``None``,
            ``./agents.toml`` is loaded if present (else defaults).
        name (Optional[str]): App name (FastAPI title). Overrides config.
        version (Optional[str]): App version. Overrides config.
        description (Optional[str]): App description. Overrides config.
        model (Optional[str]): Default model for all agents' sessions. Overrides
            config.
        mcp (bool): If True, mount an MCP endpoint per agent at ``<prefix>/mcp``.
        jobs (bool): If True (or ``[jobs].enabled`` in config), mount a durable
            async job system per agent at ``<prefix>/jobs`` — runs submitted there
            execute in the background and survive client timeouts/reconnects.

    Returns:
        FastAPI: A configured FastAPI app.

    Raises:
        TypeError: If any agent class is missing required metaclass attributes.
        ValueError: If no agents are given or two agents collide on a prefix.
    """
    load_dotenv()

    if config is None:
        config = load_config()
    name = name or config.app.name
    version = version or config.app.version
    description = description if description is not None else config.app.description
    model = model or config.app.model
    jobs_enabled = jobs or config.jobs.enabled

    mapping = _normalize_agents(agents)
    mounted_at_root = "" in mapping
    multi = len(mapping) > 1

    app = FastAPI(title=name, version=version, description=description)

    from pyagentic.api._mcp_server import mount_mcp

    index = []
    orchestrators = []
    for prefix, agent_class in mapping.items():
        router = create_router(agent_class, model=model, name=name, version=version)
        app.include_router(router, prefix=prefix)
        index.append({"agent_class": agent_class.__name__, "prefix": prefix or "/"})
        if mcp:
            mount_mcp(
                app,
                agent_class,
                name=name,
                version=version,
                sessions=router.sessions,
                path=(prefix + "/mcp"),
            )
        if jobs_enabled:
            from pyagentic.api.jobs import JobOrchestrator, build_backend, build_store
            from pyagentic.api.jobs._routes import build_jobs_router

            store = build_store(_store_path_for(config.jobs.store, prefix, multi))
            backend = build_backend(
                agent_class,
                router.sessions,
                max_concurrency=config.jobs.max_concurrency,
                default_model=model,
            )
            orchestrator = JobOrchestrator(store, backend, config.jobs, router.sessions)
            app.include_router(
                build_jobs_router(orchestrator, agent_class, router.sessions),
                prefix=prefix,
            )
            orchestrators.append(orchestrator)

    if orchestrators:
        _wire_jobs_lifespan(app, orchestrators)

    # When no agent occupies the root, add a top-level index + health probe.
    if not mounted_at_root:

        @app.get("/")
        async def app_index() -> dict:
            """List the agents mounted on this app."""
            return {"name": name, "version": version, "agents": index}

        @app.get("/health")
        async def app_health() -> dict:
            """Liveness probe endpoint."""
            return {"status": "ok"}

    return app
