import anthropic
import json

from dataclasses import dataclass
from typing import List, Optional, Type
from pydantic import BaseModel
from pyagentic._base._context import _AgentContext
from pyagentic._base._tool import _ToolDefinition
from pyagentic.llm._backend import LLMBackend
from pyagentic.models.llm import BackendInfo, Response, ToolCall, Message


@dataclass
class AnthropicMessage(Message):
    # Base
    type: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    # Tool Usage
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[str] = None
    tool_use_id: Optional[str] = None


class AnthropicBackend(LLMBackend):
    __supports_structured_outputs__ = False

    """
    Anthropic Backend
    """

    def __init__(self, model: str, api_key: str, *, base_url: str = None, **kwargs):
        self._model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
        self._info = BackendInfo(name="anthropic", model=model)
        self._default_extra = kwargs or {}

    def to_tool_call_message(self, tool_call: ToolCall):
        return AnthropicMessage(
            type="tool_use",
            id=tool_call.id,
            name=tool_call.name,
            input=json.loads(tool_call.arguments),
        )

    def to_tool_call_result_message(self, result, id_):
        return AnthropicMessage(type="tool_result", tool_use_id=id_, content=result)

    async def generate(
        self,
        context: _AgentContext,
        *,
        tool_defs: Optional[List[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Response:
        # Convert messages to Anthropic format
        messages = []
        system_message = None

        for message in context.messages:
            msg_dict = message.to_dict()
            if msg_dict.get("role") == "system":
                system_message = msg_dict.get("content")
            elif msg_dict.get("type") == "tool_use":
                messages.append({"role": "assistant", "content": [{**msg_dict}]})
            elif msg_dict.get("type") == "tool_result":
                messages.append({"role": "user", "content": [{**msg_dict}]})
            else:
                messages.append(msg_dict)

        # Prepare request parameters
        request_params = {
            "model": self._model,
            "messages": messages,
            **self._default_extra,
            **kwargs,
        }

        if system_message:
            request_params["system"] = system_message

        if tool_defs:
            request_params["tools"] = [tool.to_anthropic(context) for tool in tool_defs]

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

        return Response(
            text=text_content if text_content else None,
            parsed=parsed,
            tool_calls=tool_calls,
            raw=response,
        )
