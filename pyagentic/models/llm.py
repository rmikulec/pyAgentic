from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


@dataclass
class Message:
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


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string from the mode


@dataclass
class LLMResponse:
    text: str
    tool_calls: List[ToolCall]
    parsed: BaseModel = None
    reasoning: list[dict] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw: Optional[Any] = None


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    model: str
    attributes: dict = None
