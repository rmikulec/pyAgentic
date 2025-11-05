from typing import Any, List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    """
    Base message class for representing LLM conversation messages.

    Stores message metadata including type, role, and content. Can be converted
    to a dictionary format for API communication.
    """

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
            if value is not None:
                d[field] = value
            elif not exclude_none:
                d[field] = value
        return d


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
