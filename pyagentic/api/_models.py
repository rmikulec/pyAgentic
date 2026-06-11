"""
Request and response models for the HTTP API surface.

These cover the static (agent-independent) endpoints — info, health, schema,
sessions CRUD, and the multi-agent app index. The chat, stream, and state
endpoints are typed directly from each agent's metaclass-generated models
(``__request_model__`` / ``__response_model__`` / ``__stream_event_model__`` /
``__state_class__``), so they are not duplicated here.
"""

from typing import Optional

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request body for creating a new agent session.

    Attributes:
        model (Optional[str]): LLM model string override
            (e.g. ``'openai::gpt-4o'``).
        api_key (Optional[str]): API key for the model provider.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None


class CreateSessionResponse(BaseModel):
    """Response returned after creating a session.

    Attributes:
        session_id (str): Identifier of the newly created session.
    """

    session_id: str


class ListSessionsResponse(BaseModel):
    """Response listing the active session IDs.

    Attributes:
        sessions (list[str]): Active session identifiers.
    """

    sessions: list[str] = Field(default_factory=list)


class DeleteSessionResponse(BaseModel):
    """Response returned after deleting a session.

    Attributes:
        deleted (str): Identifier of the deleted session.
    """

    deleted: str


class HealthResponse(BaseModel):
    """Liveness probe response.

    Attributes:
        status (str): Always ``"ok"`` when the service is up.
    """

    status: str = "ok"


class AgentInfo(BaseModel):
    """Metadata describing a single mounted agent.

    Attributes:
        name (str): Display name for the agent.
        version (str): Version string for the deployment.
        agent_class (str): The agent class name.
        tools (list[str]): Names of the tools the agent exposes.
        state_fields (list[str]): Names of the agent's state fields.
        linked_agents (list[str]): Names of agents linked to this one.
    """

    name: str
    version: str
    agent_class: str
    tools: list[str] = Field(default_factory=list)
    state_fields: list[str] = Field(default_factory=list)
    linked_agents: list[str] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    """JSON schemas for an agent's request/response/stream/state models.

    Attributes:
        request (dict): JSON schema of the agent's request model.
        response (dict): JSON schema of the agent's response model.
        stream_event (dict): JSON schema of the agent's stream event model.
        state (dict): JSON schema of the agent's state model.
    """

    request: dict
    response: dict
    stream_event: dict
    state: dict


class AppAgentEntry(BaseModel):
    """One agent entry in the multi-agent app index.

    Attributes:
        agent_class (str): The agent class name.
        prefix (str): URL prefix the agent is mounted under.
    """

    agent_class: str
    prefix: str


class AppIndex(BaseModel):
    """Top-level index of agents mounted on a multi-agent app.

    Attributes:
        name (str): App name.
        version (str): App version.
        agents (list[AppAgentEntry]): The mounted agents.
    """

    name: str
    version: str
    agents: list[AppAgentEntry] = Field(default_factory=list)
