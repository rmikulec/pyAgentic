import openai
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ParsedResponse as OpenAIParsedResponse

from dataclasses import dataclass
from typing import List, Optional, Type
from pydantic import BaseModel

from pyagentic._base._context import _AgentContext
from pyagentic._base._tool import _ToolDefinition
from pyagentic.llm._provider import LLMProvider
from pyagentic.models.llm import ProviderInfo, LLMResponse, ToolCall, Message


@dataclass
class OpenAIMessage(Message):
    # Base
    type: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None
    # Tool Usage
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    output: Optional[str] = None


class OpenAIProvider(LLMProvider):
    """
    OpenAI Backend
    """

    def __init__(self, model: str, api_key: str, **kwargs):
        self._model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, **kwargs)
        self._info = ProviderInfo(name="openai", model=model, attributes=kwargs)

    def to_tool_call_message(self, tool_call: ToolCall):
        return OpenAIMessage(
            type="function_call",
            call_id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )

    def to_tool_call_result_message(self, result, id_):
        return OpenAIMessage(type="function_call_output", call_id=id_, output=result)

    async def generate(
        self,
        context: _AgentContext,
        *,
        tool_defs: Optional[List[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:

        if tool_defs is None:
            tool_defs = []

        if response_format:
            response: OpenAIParsedResponse[Type[BaseModel]] = await self.client.responses.parse(
                model=self._model,
                input=[message.to_dict() for message in context.messages],
                tools=[tool.to_openai(context) for tool in tool_defs],
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
            )
        else:
            response: OpenAIResponse = await self.client.responses.create(
                model=self._model,
                input=[message.to_dict() for message in context.messages],
                tools=[tool.to_openai(context) for tool in tool_defs],
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
            )
