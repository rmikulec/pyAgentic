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
from pyagentic.api._config import AgentsConfig, JobsConfig, load_config
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


def _slugify(value: str) -> str:
    """Convert a CamelCase or arbitrary name into a kebab-case URL/route slug."""
    # Insert a hyphen at lower/digit -> upper boundaries (CamelCase -> kebab).
    slug = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value).lower()
    # Collapse any remaining non-alphanumeric runs into single hyphens.
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "agent"


def _default_prefix(agent_class: type[BaseAgent]) -> str:
    """Derive a URL prefix from an agent class name (e.g. ResearchAgent -> /research)."""
    name = agent_class.__name__
    if name.endswith("Agent") and len(name) > len("Agent"):
        name = name[: -len("Agent")]
    return "/" + _slugify(name)


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
    tags: Optional[list[str]] = None,
    sessions: bool = True,
    jobs: bool = False,
    jobs_config: Optional["JobsConfig"] = None,
) -> APIRouter:
    """Build a FastAPI ``APIRouter`` exposing a single agent's endpoints.

    Mount it onto an existing FastAPI app under whatever prefix you like::

        app.include_router(create_router(MyAgent, name="my-agent"), prefix="/agents/my-agent")

    The router owns its own :class:`SessionManager`, exposed as
    ``router.sessions`` so callers can share it (e.g. with :func:`mount_mcp`).
    When ``jobs`` is enabled, the job orchestrator is exposed as
    ``router.orchestrator`` (otherwise ``None``) so callers can wire its
    lifecycle into their app's lifespan.

    Every route is given a unique ``name`` derived from ``name`` (e.g.
    ``my-agent_chat``), so multiple agents can be mounted on one app without
    colliding and individual routes remain addressable via ``url_path_for``.

    Routes registered (relative to the mount point):
      - ``GET /`` agent info, ``GET /health``, ``GET /schema``
      - when ``sessions``: ``POST/GET /sessions``, ``DELETE /sessions/{id}``,
        ``POST /sessions/{id}/chat``, ``POST /sessions/{id}/chat/stream``,
        ``GET /sessions/{id}/state``
      - when ``jobs``: ``POST/GET /jobs``, ``GET /jobs/{id}``,
        ``GET /jobs/{id}/updates``, ``GET /jobs/{id}/stream``,
        ``POST /jobs/{id}/cancel``

    Args:
        agent_class (type[BaseAgent]): The agent class to serve.
        model (Optional[str]): Default model for sessions. ``None`` falls back to
            the agent's own default.
        name (Optional[str]): Display name for the info route and basis for
            route names/tags. Defaults to the class name.
        version (str): Version string for the info route.
        tags (Optional[list[str]]): OpenAPI tags applied to every route on the
            router (groups them in the docs). ``None`` leaves routes untagged.
        sessions (bool): If True, mount the session-based routes (create/list/
            delete sessions plus synchronous chat, streaming chat, and state).
            If False, those routes are omitted — interaction happens only via
            ``jobs``.
        jobs (bool): If True, mount a durable async job system at ``/jobs`` whose
            orchestrator is exposed as ``router.orchestrator``. The caller is
            responsible for starting/stopping that orchestrator (see
            :func:`create_app`, which does this automatically).
        jobs_config (Optional[JobsConfig]): Configuration for the job system when
            ``jobs`` is True. Defaults to :class:`JobsConfig` defaults.

    Returns:
        APIRouter: A router with the selected agent routes registered, with its
            ``SessionManager`` attached as ``router.sessions`` and its job
            orchestrator (or ``None``) as ``router.orchestrator``.

    Raises:
        TypeError: If agent_class is missing required metaclass attributes.
    """
    _validate_agent_class(agent_class)

    name = name or agent_class.__name__
    slug = _slugify(name)
    RequestModel = agent_class.__request_model__
    ResponseModel = agent_class.__response_model__
    StreamEventModel = agent_class.__stream_event_model__
    StateModel = agent_class.__state_class__

    router = APIRouter(tags=tags)
    session_manager = SessionManager(agent_class, default_model=model)

    # ---- info routes ----

    @router.get("/", name=f"{slug}_info")
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

    @router.get("/health", name=f"{slug}_health")
    async def health() -> dict:
        """Liveness probe endpoint."""
        return {"status": "ok"}

    @router.get("/schema", name=f"{slug}_schema")
    async def schema() -> dict:
        """Return JSON schemas for request, response, stream event, and state models."""
        return {
            "request": RequestModel.model_json_schema(),
            "response": ResponseModel.model_json_schema(),
            "stream_event": StreamEventModel.model_json_schema(),
            "state": StateModel.model_json_schema(),
        }

    if sessions:
        _register_session_routes(
            router,
            slug,
            session_manager,
            RequestModel,
            ResponseModel,
            StreamEventModel,
            StateModel,
        )

    # Expose the session manager so callers can share it (e.g. with mount_mcp).
    router.sessions = session_manager

    orchestrator = None
    if jobs:
        orchestrator = _mount_jobs_router(
            router,
            agent_class,
            session_manager,
            jobs_config or JobsConfig(),
            model,
            tags,
        )
    router.orchestrator = orchestrator

    return router


def _mount_jobs_router(
    router: APIRouter,
    agent_class: type[BaseAgent],
    sessions: SessionManager,
    jobs_config: "JobsConfig",
    default_model: Optional[str],
    tags: Optional[list[str]],
) -> "JobOrchestrator":
    """Build a job orchestrator and include its /jobs router; return the orchestrator."""
    from pyagentic.api.jobs import JobOrchestrator, build_backend, build_store
    from pyagentic.api.jobs._routes import build_jobs_router

    store = build_store(jobs_config.store)
    backend = build_backend(
        agent_class,
        sessions,
        max_concurrency=jobs_config.max_concurrency,
        default_model=default_model,
    )
    orchestrator = JobOrchestrator(store, backend, jobs_config, sessions)
    router.include_router(
        build_jobs_router(orchestrator, agent_class, sessions), tags=tags
    )
    return orchestrator


def _register_session_routes(
    router: APIRouter,
    slug: str,
    sessions: SessionManager,
    RequestModel: type,
    ResponseModel: type,
    StreamEventModel: type,
    StateModel: type,
) -> None:
    """Register the session-based routes (sessions CRUD, chat, stream, state)."""

    # ---- session routes ----

    @router.post("/sessions", status_code=201, name=f"{slug}_create_session")
    async def create_session(req: Optional[CreateSessionRequest] = None) -> dict:
        """Create a new agent session."""
        model = req.model if req else None
        api_key = req.api_key if req else None
        session_id = sessions.create(model=model, api_key=api_key)
        return {"session_id": session_id}

    @router.get("/sessions", name=f"{slug}_list_sessions")
    async def list_sessions() -> dict:
        """List all active session IDs."""
        return {"sessions": sessions.list_sessions()}

    @router.delete("/sessions/{session_id}", name=f"{slug}_delete_session")
    async def delete_session(session_id: str) -> dict:
        """Delete an existing session."""
        try:
            sessions.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    # ---- chat routes ----

    @router.post(
        "/sessions/{session_id}/chat",
        response_model=ResponseModel,
        name=f"{slug}_chat",
    )
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
        name=f"{slug}_chat_stream",
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

    @router.get(
        "/sessions/{session_id}/state",
        response_model=StateModel,
        name=f"{slug}_get_state",
    )
    async def get_state(session_id: str):
        """Return the current state of the agent for a session."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.state


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
    sessions: bool = True,
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
        sessions (bool): If True (default), mount the session-based routes
            (sessions CRUD plus synchronous/streaming chat and state) for each
            agent. If False, those routes are omitted — pair with ``jobs`` to
            serve agents purely through the async job API.
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
        # The root agent inherits the app's display name; agents under a prefix
        # take a name derived from that prefix, so each router's route names stay
        # unique across the app.
        agent_name = name if prefix == "" else prefix.strip("/").replace("/", "-")
        jobs_cfg = None
        if jobs_enabled:
            jobs_cfg = config.jobs.model_copy(
                update={"store": _store_path_for(config.jobs.store, prefix, multi)}
            )
        router = create_router(
            agent_class,
            model=model,
            name=agent_name,
            version=version,
            sessions=sessions,
            jobs=jobs_enabled,
            jobs_config=jobs_cfg,
        )
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
        if router.orchestrator is not None:
            orchestrators.append(router.orchestrator)

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
