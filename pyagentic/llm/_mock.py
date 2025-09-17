from typing import Optional, Type
from pydantic import BaseModel

from pyagentic.llm._provider import LLMProvider, LLMResponse

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._context import _AgentContext
from pyagentic.models.llm import Message, LLMResponse, ToolCall


class _MockProvider(LLMProvider):
    # TODO: Can implement logic here to load in test cases and return depending on
    #   the latest message, or something more sophisticated
    __supports_tool_calls__ = True
    __supports_structured_outputs__ = True

    def __init__(self, model: str, api_key: str, *, base_url: str = False, **kwargs):
        self.model = model

    def to_tool_call_message(self, tool_call: ToolCall) -> Message:
        return Message(role="assistant", content="Tool call message")

    def to_tool_call_result_message(self, result, id_) -> Message:
        return Message(role="assistant", content="Tool call result message")

    async def generate(
        self,
        context: _AgentContext,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse:

        return LLMResponse(
            text="test",
            tool_calls=[],
            finish_reason="stop",
        )
