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
from pyagentic.models.llm import (
    ProviderInfo,
    LLMResponse,
    Message,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
)


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider implementation for Claude language models.

    Provides integration with Anthropic's Claude API supporting text generation,
    tool calling, and structured outputs via ``output_config.format``.
    """

    __supports_structured_outputs__ = True

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

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        """
        Convert semantic messages to Anthropic's message format.

        Anthropic requires consecutive tool_use blocks merged into one assistant
        message and consecutive tool_result blocks merged into one user message
        (needed for parallel tool calls).

        Args:
            messages (List[Message]): Semantic message history from the agent context.

        Returns:
            List[dict]: Messages formatted for Anthropic's API.
        """
        converted = []
        for message in messages:
            if isinstance(message, ToolCallMessage):
                block = {
                    "type": "tool_use",
                    "id": message.id,
                    "name": message.name,
                    "input": json.loads(message.arguments) if message.arguments else {},
                }
                if (
                    converted
                    and converted[-1]["role"] == "assistant"
                    and isinstance(converted[-1].get("content"), list)
                ):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "assistant", "content": [block]})
            elif isinstance(message, ToolResultMessage):
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content or "",
                }
                if (
                    converted
                    and converted[-1]["role"] == "user"
                    and isinstance(converted[-1].get("content"), list)
                ):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": message.role, "content": message.content or ""})
        return converted

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
        messages = self._convert_messages(state._context)

        # Prepare request parameters
        request_params = {
            "model": self._model,
            "messages": messages,
            **kwargs,
        }

        if "max_tokens" not in request_params:
            request_params["max_tokens"] = 4096

        request_params["system"] = state.system_message

        if tool_defs:
            request_params["tools"] = [tool.to_anthropic_spec() for tool in tool_defs]

        # When a structured output is requested and no tools need calling,
        # use Anthropic's output_config to constrain the response to JSON.
        if response_format and not tool_defs:
            schema = response_format.model_json_schema()
            # Anthropic requires additionalProperties: false on all objects
            self._enforce_additional_properties(schema)
            request_params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }

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
        if response_format and text_content and not tool_calls:
            try:
                parsed = response_format.model_validate_json(text_content)
                text_content = parsed.model_dump_json(indent=2)
            except Exception:
                pass

        return LLMResponse(
            text=text_content if text_content else None,
            parsed=parsed,
            tool_calls=tool_calls,
            raw=response,
        )

    @staticmethod
    def _enforce_additional_properties(schema: dict) -> None:
        """Recursively set additionalProperties: false on all object schemas."""
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for prop in schema.get("properties", {}).values():
                AnthropicProvider._enforce_additional_properties(prop)
            if "required" not in schema:
                schema["required"] = list(schema.get("properties", {}).keys())
        if "items" in schema:
            AnthropicProvider._enforce_additional_properties(schema["items"])
        for key in ("anyOf", "allOf"):
            for sub in schema.get(key, []):
                AnthropicProvider._enforce_additional_properties(sub)
        # Resolve $defs references inline
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                AnthropicProvider._enforce_additional_properties(def_schema)

    def extract_usage_info(self, response):
        return super().extract_usage_info(response)
