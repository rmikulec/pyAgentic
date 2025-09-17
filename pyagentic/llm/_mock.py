"""
Mock provider implementation for testing and development purposes.

This module provides a mock LLM provider that can be used for testing agents
without making actual API calls to external language model services.
"""

from typing import Optional, Type
from pydantic import BaseModel

from pyagentic.llm._provider import LLMProvider, LLMResponse

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._context import _AgentContext
from pyagentic.models.llm import Message, LLMResponse, ToolCall


class _MockProvider(LLMProvider):
    """
    Mock LLM provider for testing and development.

    This provider returns fixed responses without making actual API calls,
    useful for testing agent behavior and development workflows without
    incurring API costs or requiring network connectivity.

    Note: This is intended for internal testing purposes only.
    """
    # TODO: Can implement logic here to load in test cases and return depending on
    #   the latest message, or something more sophisticated
    __supports_tool_calls__ = True
    __supports_structured_outputs__ = True

    def __init__(self, model: str, api_key: str, *, base_url: str = False, **kwargs):
        """
        Initialize the mock provider.

        Args:
            model: Model identifier (stored but not used)
            api_key: API key (stored but not used)
            base_url: Base URL (ignored)
            **kwargs: Additional arguments (ignored)
        """
        self.model = model

    def to_tool_call_message(self, tool_call: ToolCall) -> Message:
        """
        Convert a tool call to a mock message format.

        Args:
            tool_call: The tool call to convert

        Returns:
            Simple Message with fixed content
        """
        return Message(role="assistant", content="Tool call message")

    def to_tool_call_result_message(self, result, id_) -> Message:
        """
        Convert a tool result to a mock message format.

        Args:
            result: The tool execution result (ignored)
            id_: The tool call ID (ignored)

        Returns:
            Simple Message with fixed content
        """
        return Message(role="assistant", content="Tool call result message")

    async def generate(
        self,
        context: _AgentContext,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a mock response without calling any external APIs.

        Returns a fixed response suitable for testing. In the future, this could
        be enhanced to return different responses based on input context or
        load test cases from configuration.

        Args:
            context: Agent context (currently ignored)
            tool_defs: Available tools (currently ignored)
            response_format: Structured output format (currently ignored)
            **kwargs: Additional parameters (currently ignored)

        Returns:
            LLMResponse with fixed test content
        """

        return LLMResponse(
            text="test",
            tool_calls=[],
            finish_reason="stop",
        )
