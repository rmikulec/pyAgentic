"""
FastAPI app factory and routes for serving a PyAgentic agent over HTTP.

The app leverages the metaclass-generated models on the agent class:
  - __request_model__        →  request body on chat endpoints
  - __response_model__       →  response_model on chat endpoint
  - __stream_event_model__   →  typed SSE events on stream endpoint
  - __state_class__          →  response_model on state endpoint
"""

from dotenv import load_dotenv
load_dotenv()

import json
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyagentic.serve._manifest import Manifest
from pyagentic.serve._sessions import SessionManager
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.models.llm import LLMResponse


class CreateSessionRequest(BaseModel):
    """Request body for creating a new agent session.

    Attributes:
        model (Optional[str]): LLM model string override
            (e.g. ``'openai::gpt-4o'``).
        api_key (Optional[str]): API key for the model provider.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None

from pyagentic._base._agent._agent import BaseAgent


def create_app(
    agent_class: type[BaseAgent],
    manifest: Manifest,
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

    Returns:
        FastAPI: A configured FastAPI app.
    """
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
        kwargs = {field: getattr(req, field) for field in req.model_fields}
        if len(kwargs) == 1:
            prompt = str(next(iter(kwargs.values())))
        else:
            prompt = "\n".join(f"{k}: {v}" for k, v in kwargs.items() if v is not None)

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

    return app
