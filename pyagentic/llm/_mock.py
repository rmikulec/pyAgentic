"""
Mock provider implementation for testing and development purposes.

This module provides a mock LLM provider that can be used for testing agents
without making actual API calls to external language model services.
"""

from typing import Optional, Type
from pydantic import BaseModel

from pyagentic.llm._provider import LLMProvider

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic.models.llm import LLMResponse, ProviderInfo, UsageInfo


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
        # Optional FIFO queue of canned responses; when non-empty, generate()
        # pops and returns these instead of the default echo response.
        self.responses: list[LLMResponse] = []

        self._info = ProviderInfo(name="_mock", model=self.model, attributes=kwargs)

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

        Pops from the `responses` queue when canned responses were provided;
        otherwise echoes the latest context message. Usage is reported as a
        naive character count of the context so token-triggered policies can
        be tested deterministically.

        Args:
            state: Agent state providing the compiled message context
            tool_defs: Available tools (currently ignored)
            response_format: Structured output format (currently ignored)
            **kwargs: Additional parameters (currently ignored)

        Returns:
            LLMResponse with canned or echoed test content
        """
        if self.responses:
            return self.responses.pop(0)

        latest_message = state._context[-1].content if state._context else ""
        input_size = sum(len(message.content or "") for message in state._context)

        return LLMResponse(
            text=f"user said {latest_message}",
            tool_calls=[],
            finish_reason="stop",
            usage=UsageInfo(
                input_tokens=input_size,
                output_tokens=0,
                total_tokens=input_size,
            ),
        )
