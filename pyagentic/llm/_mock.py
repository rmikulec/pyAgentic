"""
Mock provider implementation for testing and development purposes.

This module provides a mock LLM provider that can be used for testing agents
without making actual API calls to external language model services.
"""

from typing import Optional, Type, Any
from pydantic import BaseModel

from pyagentic.llm._provider import LLMProvider

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic.models.llm import Message, LLMResponse, ToolCall, ProviderInfo


class _MockMessage(Message):
    tool_call: Any = None
    tool_result: Any = None
    tool_call_id: Any = None


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

        self._info = ProviderInfo(name="_mock", model=self.model, attributes=kwargs)

    def to_tool_call_message(self, tool_call: ToolCall) -> Message:
        """
        Convert a tool call to a mock message format.

        Args:
            tool_call: The tool call to convert

        Returns:
            Simple Message with fixed content
        """
        return _MockMessage(
            type="tool_call",
            content="Tool call message",
            tool_call=tool_call,
            tool_call_id=tool_call.id,
        )

    def to_tool_call_result_message(self, result, id_) -> Message:
        """
        Convert a tool result to a mock message format.

        Args:
            result: The tool execution result (ignored)
            id_: The tool call ID (ignored)

        Returns:
            Simple Message with fixed content
        """
        return _MockMessage(
            type="tool_result",
            content="Tool call result message",
            tool_result=result,
            tool_call_id=id_,
        )

    async def generate(
        self,
        state: _AgentState,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a mock response without calling any external APIs.

        Returns a fixed response suitable for testing. In the future, this could
        be enhanced to return different responses based on input state or
        load test cases from configuration.

        Args:
            state: Agent state (currently ignored)
            tool_defs: Available tools (currently ignored)
            response_format: Structured output format (currently ignored)
            **kwargs: Additional parameters (currently ignored)

        Returns:
            LLMResponse with fixed test content
        """
        latest_message = state._messages[-1].content

        return LLMResponse(
            text=f"user said {latest_message}",
            tool_calls=[],
            finish_reason="stop",
        )
