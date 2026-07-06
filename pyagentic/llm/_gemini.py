"""
Gemini provider implementation for the pyagentic framework.

This module provides integration with Google's Gemini API for text generation
and tool calling capabilities using the google-generativeai library.
"""

import google.generativeai as genai
import json

from typing import List, Optional, Type
from pydantic import BaseModel
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._agent._agent_state import _AgentState
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


class GeminiProvider(LLMProvider):
    """
    Gemini provider implementation for Google's Gemini language models.

    Provides integration with Google's Gemini API supporting text generation
    and tool calling through the google-generativeai library.
    """

    __llm_name__ = "gemini"
    __supports_structured_outputs__ = False

    def __init__(self, model: str, api_key: str, **kwargs):
        """
        Initialize the Gemini provider with the specified model and API key.

        Args:
            model: Gemini model identifier (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash')
            api_key: Google API key for authentication
            **kwargs: Additional arguments for the Gemini configuration
        """
        self._model = model
        genai.configure(api_key=api_key)

        # Extract generation config from kwargs if provided
        self.generation_config = kwargs.pop("generation_config", {})
        self.safety_settings = kwargs.pop("safety_settings", None)

        self.client = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
        self._info = ProviderInfo(name="gemini", model=model, attributes=kwargs)

    def _convert_messages_to_gemini_format(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[dict]]:
        """
        Convert semantic messages to Gemini's content format.

        Args:
            messages: List of Message objects from the agent context

        Returns:
            Tuple of (system_instruction, converted_messages)
        """
        gemini_messages = []
        system_instruction = None

        for message in messages:
            if isinstance(message, ToolCallMessage):
                gemini_messages.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": message.name,
                                    "args": (
                                        json.loads(message.arguments) if message.arguments else {}
                                    ),
                                }
                            }
                        ],
                    }
                )
            elif isinstance(message, ToolResultMessage):
                # Gemini keys function responses by function NAME, not call id
                gemini_messages.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": message.name,
                                    "response": {"result": message.content},
                                }
                            }
                        ],
                    }
                )
            elif message.role == "system":
                system_instruction = message.content
            else:
                role = message.role or "user"
                # Map roles: assistant -> model, user -> user
                if role == "assistant":
                    role = "model"

                if message.content:
                    gemini_messages.append({"role": role, "parts": [{"text": message.content}]})

        return system_instruction, gemini_messages

    def _tool_defs_to_gemini_format(self, tool_defs: List[_ToolDefinition]) -> List[dict]:
        """
        Convert tool definitions to Gemini's function declaration format.

        Args:
            tool_defs: List of tool definitions
            context: Agent context for resolving parameters

        Returns:
            List of Gemini function declarations
        """
        gemini_tools = []

        for tool_def in tool_defs:
            openai_spec = tool_def.to_openai_spec()

            # Convert OpenAI spec to Gemini format
            gemini_func = {
                "name": openai_spec["name"],
                "description": openai_spec["description"],
                "parameters": openai_spec["parameters"],
            }

            gemini_tools.append(gemini_func)

        return gemini_tools

    async def generate(
        self,
        state: _AgentState,
        *,
        tool_defs: Optional[List[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response using Google's Gemini API.

        Handles message formatting, system instruction extraction, and tool calling.
        Note: Structured outputs are not natively supported and will be handled
        through JSON mode if response_format is provided.

        Args:
            context: Agent context containing conversation history and system messages
            tool_defs: List of available tools the model can call
            response_format: Optional Pydantic model for structured output (limited support)
            **kwargs: Additional parameters for the Gemini API call

        Returns:
            LLMResponse containing generated text, tool calls, and metadata
        """
        if tool_defs is None:
            tool_defs = []

        # Convert messages to Gemini format
        system_instruction, gemini_messages = self._convert_messages_to_gemini_format(
            state._context
        )

        # Update model configuration if system instruction exists
        if system_instruction:
            self.client._system_instruction = system_instruction

        # Prepare tools if provided
        tools = None
        if tool_defs:
            gemini_funcs = self._tool_defs_to_gemini_format(tool_defs)
            tools = [{"function_declarations": gemini_funcs}]

        # Handle structured output request
        generation_config = self.generation_config.copy()
        if response_format:
            # Enable JSON mode for structured outputs
            generation_config["response_mime_type"] = "application/json"
            if hasattr(response_format, "model_json_schema"):
                generation_config["response_schema"] = response_format.model_json_schema()

        # Generate response using the model directly
        # If we have message history, use chat mode
        if gemini_messages:
            chat = self.client.start_chat(
                history=gemini_messages[:-1] if len(gemini_messages) > 1 else []
            )
            last_message = (
                gemini_messages[-1]["parts"][0]["text"] if gemini_messages[-1].get("parts") else ""
            )

            response = await chat.send_message_async(
                last_message,
                tools=tools,
                generation_config=generation_config if generation_config else None,
                **kwargs,
            )
        else:
            # No message history, use direct generation
            response = await self.client.generate_content_async(
                "",
                tools=tools,
                generation_config=generation_config if generation_config else None,
                **kwargs,
            )

        # Parse response
        text_content = ""
        tool_calls = []
        parsed = None

        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text_content += part.text
            elif hasattr(part, "function_call") and part.function_call:
                # Extract function call
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=fc.name,  # Gemini doesn't provide separate IDs
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)),
                    )
                )

        # Handle structured output parsing
        if response_format and text_content:
            try:
                parsed_data = json.loads(text_content)
                parsed = response_format(**parsed_data)
            except Exception:
                # Fallback if parsing fails
                pass

        # Extract usage information
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = UsageInfo(
                input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
            )

        return LLMResponse(
            text=text_content if text_content else None,
            parsed=parsed,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
        )
