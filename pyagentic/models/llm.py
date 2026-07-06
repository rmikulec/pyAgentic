from typing import Annotated, Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    Base message class for representing LLM conversation messages.

    Stores message metadata including type, role, and content. Can be converted
    to a dictionary format for API communication.
    """

    kind: str = "message"
    type: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self, exclude_none: bool = True):
        """
        Convert message to dictionary format.

        Args:
            exclude_none (bool): Whether to exclude fields with None values. Defaults to True.

        Returns:
            dict: Dictionary representation of the message
        """
        d = {}

        for field, value in self.__dict__.items():
            if field == "kind":
                continue
            if value is not None:
                d[field] = value
            elif not exclude_none:
                d[field] = value
        return d


class SystemMessage(Message):
    """
    System prompt message. Only appears in compiled message views; the agent
    never stores it in history since the system prompt is re-rendered per call.
    """

    kind: Literal["system"] = "system"
    role: Optional[str] = "system"


class UserMessage(Message):
    """
    A user-turn message.
    """

    kind: Literal["user"] = "user"
    role: Optional[str] = "user"


class AssistantMessage(Message):
    """
    A plain assistant text message.
    """

    kind: Literal["assistant"] = "assistant"
    role: Optional[str] = "assistant"


class ToolCallMessage(Message):
    """
    Records the LLM requesting a tool invocation.

    Providers convert this to their wire format (e.g. OpenAI `function_call`
    items, Anthropic `tool_use` blocks) at request time.
    """

    kind: Literal["tool_call"] = "tool_call"
    role: Optional[str] = "assistant"
    id: str
    name: str
    arguments: str  # JSON string from the model


class ToolResultMessage(Message):
    """
    Records the result of a tool invocation; `content` holds the stringified
    output. Carries both the originating call id and the tool name (some
    providers, e.g. Gemini, key results by name rather than id).
    """

    kind: Literal["tool_result"] = "tool_result"
    role: Optional[str] = "tool"
    tool_call_id: str
    name: str


class AgentCallMessage(ToolCallMessage):
    """
    A tool call that targets a linked agent. Subclass of ToolCallMessage so
    generic tool policies match it; agent-specific policies can target it directly.
    """

    kind: Literal["agent_call"] = "agent_call"


class AgentResultMessage(ToolResultMessage):
    """
    The result of a linked-agent call. Subclass of ToolResultMessage so generic
    tool policies match it; agent-specific policies can target it directly.
    """

    kind: Literal["agent_result"] = "agent_result"


class CompactionSummaryMessage(AssistantMessage):
    """
    An assistant message produced by a compaction policy, replacing a span of
    older history with a summary. `compacted_count` records how many messages
    it replaced.
    """

    kind: Literal["compaction_summary"] = "compaction_summary"
    compacted_count: int = 0


AnyMessage = Annotated[
    Union[
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolCallMessage,
        ToolResultMessage,
        AgentCallMessage,
        AgentResultMessage,
        CompactionSummaryMessage,
    ],
    Field(discriminator="kind"),
]
"""Discriminated union of all semantic message types, for round-trip (de)serialization."""


class ToolCall(BaseModel):
    """
    Represents a tool/function call request from the LLM.

    Contains the tool identifier, name, and JSON-encoded arguments that the
    LLM wants to invoke.
    """

    id: str
    name: str
    arguments: str  # JSON string from the model


class UsageInfo(BaseModel):
    """
    Token usage information for an LLM API call.

    Tracks input tokens, output tokens, and total tokens consumed, along with
    optional metadata about token usage.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    output_tokens_metadata: Optional[dict] = None
    input_tokens_metadata: Optional[dict] = None


class LLMResponse(BaseModel):
    """
    Unified response format from any LLM provider.

    Encapsulates text output, tool calls, parsed structured data, reasoning traces,
    finish reason, token usage, and the raw provider response.
    """

    text: str | None = None
    tool_calls: List[ToolCall] | None = None
    parsed: BaseModel | None = None
    reasoning: list[dict] | None = None
    finish_reason: Optional[str] = None
    usage: UsageInfo = None
    raw: Optional[Any] = None


class ProviderInfo(BaseModel):
    """
    Metadata about the LLM provider and model used for a request.

    Stores the provider name, model identifier, and any additional provider-specific
    attributes or configuration.
    """

    name: str
    model: str
    attributes: dict = None
