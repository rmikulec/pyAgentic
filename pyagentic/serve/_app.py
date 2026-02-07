"""
FastAPI app factory and routes for serving a PyAgentic agent over HTTP.
"""

import json
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyagentic.serve._manifest import Manifest
from pyagentic.serve._sessions import SessionManager
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.models.llm import LLMResponse

if TYPE_CHECKING:
    from pyagentic._base._agent._agent import BaseAgent


class ChatRequest(BaseModel):
    message: str


def create_app(
    agent_class: "type[BaseAgent]",
    manifest: Manifest,
) -> FastAPI:
    """Build a FastAPI application wired to the given agent class.

    Args:
        agent_class: The agent class to serve.
        manifest: Parsed pyagentic.toml manifest.

    Returns:
        A configured FastAPI app.
    """
    app = FastAPI(
        title=manifest.project.name,
        version=manifest.project.version,
        description=manifest.project.description,
    )
    sessions = SessionManager(agent_class, manifest)

    # ---- info routes ----

    @app.get("/")
    async def agent_info() -> dict:
        tools = list(agent_class.__tool_defs__.keys())
        state_fields = list(agent_class.__state_defs__.keys())
        linked = list(agent_class.__linked_agents__.keys())
        return {
            "name": manifest.project.name,
            "version": manifest.project.version,
            "agent_class": agent_class.__name__,
            "tools": tools,
            "state_fields": state_fields,
            "linked_agents": linked,
        }

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/schema")
    async def schema() -> dict:
        return agent_class.__response_model__.model_json_schema()

    # ---- session routes ----

    @app.post("/sessions", status_code=201)
    async def create_session() -> dict:
        session_id = sessions.create()
        return {"session_id": session_id}

    @app.get("/sessions")
    async def list_sessions() -> dict:
        return {"sessions": sessions.list_sessions()}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        try:
            sessions.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": session_id}

    # ---- chat routes ----

    @app.post("/sessions/{session_id}/chat")
    async def chat(session_id: str, req: ChatRequest) -> dict:
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

        response = await agent.run(req.message)
        return response.model_dump()

    @app.post("/sessions/{session_id}/chat/stream")
    async def chat_stream(session_id: str, req: ChatRequest):
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

        async def event_generator():
            async for update in agent.step(req.message):
                if isinstance(update, LLMResponse):
                    event_type = "llm_response"
                    data = update.model_dump()
                elif isinstance(update, AgentResponse):
                    event_type = "agent_response"
                    data = update.model_dump()
                elif isinstance(update, ToolResponse):
                    event_type = "tool_response"
                    data = update.model_dump()
                else:
                    event_type = "update"
                    data = update.model_dump() if hasattr(update, "model_dump") else str(update)

                payload = json.dumps(data, default=str)
                yield f"event: {event_type}\ndata: {payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ---- state route ----

    @app.get("/sessions/{session_id}/state")
    async def get_state(session_id: str) -> dict:
        try:
            agent = sessions.get(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.state.model_dump()

    return app
