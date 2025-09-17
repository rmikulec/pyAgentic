from typing import List, Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._context import _AgentContext
from pyagentic.models.llm import Message, LLMResponse, ToolCall


class LLMProvider(ABC):
    __llm_name__ = "base"
    __supports_tool_calls__ = True
    __supports_structured_outputs__ = True

    @abstractmethod
    def __init__(self, model: str, api_key: str, *, base_url: str = False, **kwargs): ...

    @abstractmethod
    def to_tool_call_message(self, tool_call: ToolCall) -> Message: ...

    @abstractmethod
    def to_tool_call_result_message(self, result, id_) -> Message: ...

    @abstractmethod
    async def generate(
        self,
        context: _AgentContext,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> LLMResponse: ...
