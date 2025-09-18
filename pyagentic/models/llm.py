from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    type: Optional[str] = None
    role: Optional[str] = None
    content: Optional[str] = None

    def to_dict(self, exclude_none: bool = True):
        d = {}

        for field, value in self.__dict__.items():
            if value is not None:
                d[field] = value
            elif not exclude_none:
                d[field] = value
        return d


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON string from the mode


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    output_tokens_metadata: Optional[dict] = None
    input_tokens_metadata: Optional[dict] = None


class LLMResponse(BaseModel):
    text: str
    tool_calls: List[ToolCall]
    parsed: BaseModel = None
    reasoning: list[dict] = None
    finish_reason: Optional[str] = None
    usage: UsageInfo = None
    raw: Optional[Any] = None


class ProviderInfo(BaseModel):
    name: str
    model: str
    attributes: dict = None