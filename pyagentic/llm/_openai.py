"""
OpenAI provider implementation for the pyagentic framework.

This module provides integration with OpenAI's API for text generation and tool calling,
supporting both standard and structured output generation.
"""

import openai
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ParsedResponse as OpenAIParsedResponse

from typing import List, Optional, Type
from pydantic import BaseModel

from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition
from pyagentic.llm._provider import LLMProvider
from pyagentic.models.llm import (
    ProviderInfo,
    LLMResponse,
    Message,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    UsageInfo,
)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation for language model inference.

    Provides integration with OpenAI's API supporting both standard text generation
    and structured output parsing. Handles tool calling and function execution
    through OpenAI's function calling capabilities.
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        Initialize the OpenAI provider with the specified model and API key.

        Args:
            model: OpenAI model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key for authentication
            **kwargs: Additional arguments passed to the OpenAI client
        """
        self._model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, **kwargs)
        self._info = ProviderInfo(name="openai", model=model, attributes=kwargs)

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert semantic messages to OpenAI Responses API input items.

        Args:
            messages (List[Message]): Semantic message history from the agent context.

        Returns:
            List[dict]: Input items formatted for OpenAI's Responses API.
        """
        items = []
        for message in messages:
            if isinstance(message, ToolCallMessage):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": message.id,
                        "name": message.name,
                        "arguments": message.arguments,
                    }
                )
            elif isinstance(message, ToolResultMessage):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content or "",
                    }
                )
            else:
                items.append({"role": message.role, "content": message.content or ""})
        return items

    async def generate(
        self,
        state: _AgentState,
        *,
        tool_defs: Optional[List[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI's API.

        Supports both standard text generation and structured output parsing.
        When response_format is provided, uses OpenAI's structured output capabilities
        to ensure the response conforms to the specified Pydantic model.

        Args:
            state: Agent state containing conversation history and system messages
            tool_defs: List of available tools the model can call
            response_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters for the OpenAI API call

        Returns:
            LLMResponse containing generated text, parsed data, tool calls, and metadata
        """

        if tool_defs is None:
            tool_defs = []

        if response_format:
            response: OpenAIParsedResponse[Type[BaseModel]] = await self.client.responses.parse(
                model=self._model,
                instructions=state.system_message,
                input=self._convert_messages(state._context),
                tools=[tool.to_openai_spec() for tool in tool_defs],
                text_format=response_format,
                **kwargs,
            )
            parsed = response.output_parsed if response.output_parsed else None
            text = parsed.model_dump_json(indent=2) if parsed else None
            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            tool_calls = [rx for rx in response.output if rx.type == "function_call"]

            return LLMResponse(
                text=text,
                parsed=parsed,
                tool_calls=[
                    ToolCall(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)
                    for tool_call in tool_calls
                ],
                reasoning=reasoning,
                raw=response,
                usage=UsageInfo(**response.usage.model_dump()),
            )
        else:
            response: OpenAIResponse = await self.client.responses.create(
                model=self._model,
                instructions=state.system_message,
                input=self._convert_messages(state._context),
                tools=[tool.to_openai_spec() for tool in tool_defs],
                **kwargs,
            )

            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            tool_calls = [rx for rx in response.output if rx.type == "function_call"]

            return LLMResponse(
                text=response.output_text,
                tool_calls=[
                    ToolCall(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)
                    for tool_call in tool_calls
                ],
                reasoning=reasoning,
                raw=response,
                usage=UsageInfo(**response.usage.model_dump()),
            )
