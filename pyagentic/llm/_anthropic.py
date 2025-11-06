"""
Anthropic provider implementation for the pyagentic framework.

This module provides integration with Anthropic's Claude API for text generation
and tool calling capabilities.
"""

import anthropic
import json

from typing import List, Optional, Type
from pydantic import BaseModel
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition
from pyagentic.llm._provider import LLMProvider
from pyagentic.models.llm import ProviderInfo, LLMResponse, ToolCall, Message


class AnthropicMessage(Message):
    """
    Anthropic-specific message format extending the base Message class.

    Includes additional fields required for Anthropic's API format including
    tool use handling and proper message structuring for Claude models.
    """

    # Tool Usage
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[str] = None
    tool_use_id: Optional[str] = None


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation for Claude language models.

    Provides integration with Anthropic's Claude API supporting text generation
    and tool calling. Note that this provider does not support structured outputs
    natively and implements them through tool calling mechanisms.
    """

    __supports_structured_outputs__ = False

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        Initialize the Anthropic provider with the specified model and API key.

        Args:
            model: Anthropic model identifier (e.g., 'claude-3-sonnet-20240229')
            api_key: Anthropic API key for authentication
            **kwargs: Additional arguments passed to the Anthropic client
        """
        self._model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, **kwargs)
        self._info = ProviderInfo(name="anthropic", model=model, attributes=kwargs)

    def to_tool_call_message(self, tool_call: ToolCall):
        """
        Convert a tool call to Anthropic's tool use message format.

        Args:
            tool_call: The tool call to convert

        Returns:
            AnthropicMessage formatted for Anthropic's tool use API
        """
        return AnthropicMessage(
            type="tool_use",
            id=tool_call.id,
            name=tool_call.name,
            input=json.loads(tool_call.arguments),
        )

    def to_tool_call_result_message(self, result, id_):
        """
        Convert a tool execution result to Anthropic's tool result message format.

        Args:
            result: The output from the tool execution
            id_: The tool use ID to associate with this result

        Returns:
            AnthropicMessage containing the tool execution result
        """
        return AnthropicMessage(type="tool_result", tool_use_id=id_, content=result)

    async def generate(
        self,
        state: _AgentState,
        *,
        tool_defs: Optional[List[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using Anthropic's Claude API.

        Handles message formatting, system message extraction, and tool calling.
        For structured outputs, attempts to parse tool call results into the
        specified Pydantic model format.

        Args:
            state: Agent state containing conversation history and system messages
            tool_defs: List of available tools the model can call
            response_format: Optional Pydantic model for structured output (limited support)
            **kwargs: Additional parameters for the Anthropic API call

        Returns:
            LLMResponse containing generated text, parsed data, tool calls, and metadata
        """
        # Convert messages to Anthropic format
        messages = []

        for message in state._messages:
            msg_dict = message.to_dict()
            if msg_dict.get("type") == "tool_use":
                messages.append({"role": "assistant", "content": [{**msg_dict}]})
            elif msg_dict.get("type") == "tool_result":
                messages.append({"role": "user", "content": [{**msg_dict}]})
            else:
                messages.append(msg_dict)

        # Prepare request parameters
        request_params = {
            "model": self._model,
            "messages": messages,
            **kwargs,
        }

        if "max_tokens" not in request_params:
            request_params["max_tokens"] = 1024

        request_params["system"] = state.system_message

        if tool_defs:
            request_params["tools"] = [tool.to_anthropic_spec() for tool in tool_defs]

        # Make the API call
        async with self.client.messages.stream(**request_params) as stream:
            response = await stream.get_final_message()

        # Parse response
        text_content = ""
        tool_calls = []

        for content_block in response.content:
            if content_block.type == "text":
                text_content += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        arguments=json.dumps(content_block.input),
                    )
                )

        # Handle structured output
        parsed = None
        if response_format and tool_calls:
            # Assume first tool call contains structured data
            try:
                parsed = response_format(**tool_calls[0].arguments)
                text_content = parsed.model_dump_json(indent=2)
            except Exception:
                # Fallback to text content if parsing fails
                pass

        return LLMResponse(
            text=text_content if text_content else None,
            parsed=parsed,
            tool_calls=tool_calls,
            raw=response,
        )

    def extract_usage_info(self, response):
        return super().extract_usage_info(response)
