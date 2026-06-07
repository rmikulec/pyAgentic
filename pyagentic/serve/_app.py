"""
FastAPI app factory and routes for serving a PyAgentic agent over HTTP.

The app leverages the metaclass-generated models on the agent class:
  - __request_model__        →  request body on chat endpoints
  - __response_model__       →  response_model on chat endpoint
  - __stream_event_model__   →  typed SSE events on stream endpoint
  - __state_class__          →  response_model on state endpoint
"""

import json
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyagentic._base._agent._agent import BaseAgent
from pyagentic.models.llm import LLMResponse
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.serve._manifest import Manifest
from pyagentic.serve._sessions import SessionManager


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


def create_app(
    agent_class: type[BaseAgent],
    manifest: Manifest,
    mcp: bool = False,
) -> FastAPI:
    """Build a FastAPI application wired to the given agent class.

    The generated API automatically reflects the agent's input/output schemas:
      - Chat request body is derived from the agent's ``__call__`` signature
      - Chat response body uses the agent's ``__response_model__``
      - Stream events use the agent's ``__stream_event_model__``
      - State endpoint uses the agent's ``__state_class__``

    Args:
        agent_class (type[BaseAgent]): The agent class to serve.
        manifest (Manifest): Parsed pyagentic.toml manifest.
        mcp (bool): If True, mount an MCP server endpoint at ``/mcp``.

    Returns:
        FastAPI: A configured FastAPI app.

    Raises:
        TypeError: If agent_class is missing required metaclass attributes.
    """
    load_dotenv()
    _validate_agent_class(agent_class)

    RequestModel = agent_class.__request_model__
    ResponseModel = agent_class.__response_model__
    StreamEventModel = agent_class.__stream_event_model__
    StateModel = agent_class.__state_class__

    app = FastAPI(
        title=manifest.project.name,
        version=manifest.project.version,
        description=manifest.project.description,
    )
    sessions = SessionManager(agent_class, manifest)

    # ---- info routes ----

    @app.get("/")
    async def agent_info() -> dict:
        """Return basic agent metadata."""
        return {
            "name": manifest.project.name,
            "version": manifest.project.version,
            "agent_class": agent_class.__name__,
            "tools": list(agent_class.__tool_defs__.keys()),
            "state_fields": list(agent_class.__state_defs__.keys()),
            "linked_agents": list(agent_class.__linked_agents__.keys()),
        }

    @app.get("/health")
    async def health() -> dict:
        """Liveness probe endpoint."""
        return {"status": "ok"}

    @app.get("/schema")
    async def schema() -> dict:
        """Return JSON schemas for request, response, stream event, and state models."""
        return {
            "request": RequestModel.model_json_schema(),
            "response": ResponseModel.model_json_schema(),
            "stream_event": StreamEventModel.model_json_schema(),
            "state": StateModel.model_json_schema(),
        }

    # ---- session routes ----

    @app.post("/sessions", status_code=201)
    async def create_session(req: Optional[CreateSessionRequest] = None) -> dict:
        """Create a new agent session."""
        model = req.model if req else None
        api_key = req.api_key if req else None
        session_id = sessions.create(model=model, api_key=api_key)
        return {"session_id": session_id}

    @app.get("/sessions")
    async def list_sessions() -> dict:
        """List all active session IDs."""
        return {"sessions": sessions.list_sessions()}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        """Delete an existing session."""
        try:
            sessions.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    # ---- chat routes ----

    @app.post("/sessions/{session_id}/chat", response_model=ResponseModel)
    async def chat(session_id: str, req: RequestModel):  # type: ignore[valid-type]
        """Send a message and receive a complete agent response."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

        kwargs = {field: getattr(req, field) for field in req.model_fields}
        response = await agent(**kwargs)
        return response

    @app.post(
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

    @app.get("/sessions/{session_id}/state", response_model=StateModel)
    async def get_state(session_id: str):
        """Return the current state of the agent for a session."""
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.state

    if mcp:
        from pyagentic.serve._mcp_server import _mount_mcp

        _mount_mcp(app, agent_class, sessions, manifest)

    return app
