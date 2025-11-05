"""
Abstract base class for LLM provider implementations.

This module defines the interface that all LLM providers must implement to be
compatible with the pyagentic framework.
"""

from typing import Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic.models.llm import Message, LLMResponse, ToolCall


class LLMProvider(ABC):
    """
    Abstract base class for all LLM provider implementations.

    This class defines the interface that concrete LLM providers must implement
    to integrate with the pyagentic agent framework. Providers handle communication
    with specific language model APIs and translate between the framework's
    internal representation and the provider's API format.

    Attributes:
        __llm_name__: Human-readable name for the provider
        __supports_tool_calls__: Whether the provider supports function/tool calling
        __supports_structured_outputs__: Whether the provider supports structured response formats
    """

    __llm_name__ = "base"
    __supports_tool_calls__ = True
    __supports_structured_outputs__ = True

    _model: str = None

    @abstractmethod
    def __init__(self, model: str, api_key: str, *, base_url: str = False, **kwargs):
        """
        Initialize the LLM provider with authentication and configuration.

        Args:
            model: The model identifier/name to use for this provider
            api_key: Authentication key for the provider's API
            base_url: Optional custom base URL for the API endpoint
            **kwargs: Additional provider-specific configuration options
        """
        ...

    @abstractmethod
    def to_tool_call_message(self, tool_call: ToolCall) -> Message:
        """
        Convert a tool call to a provider-specific message format.

        Args:
            tool_call: The tool call to convert

        Returns:
            Message formatted for this provider's API
        """
        ...

    @abstractmethod
    def to_tool_call_result_message(self, result, id_) -> Message:
        """
        Convert a tool call result to a provider-specific message format.

        Args:
            result: The result/output from the tool execution
            id_: The tool call ID to associate with this result

        Returns:
            Message formatted for this provider's API
        """
        ...

    @abstractmethod
    async def generate(
        self,
        state: _AgentState,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the language model.

        Args:
            state: The agent state containing conversation history and system messages
            tool_defs: Optional list of tool definitions the model can use
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional provider-specific generation parameters

        Returns:
            LLMResponse containing the generated text, tool calls, and metadata
        """
        ...
