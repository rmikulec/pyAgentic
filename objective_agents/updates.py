from pydantic import BaseModel
from typing import Literal
from enum import Enum


class Status(Enum):
    GENERATING = "generating"
    PROCESSING = "processing"
    SUCCEDED = "succeded"
    ERROR = "error"


class EmitUpdate(BaseModel):
    type: Literal["base"] = "base"
    status: Status


class AiUpdate(EmitUpdate):
    type: Literal["ai_response"] = "ai_response"
    message: str = None


class ToolUpdate(EmitUpdate):
    type: Literal["tool_update"] = "tool_update"
    tool_call: str = None
    tool_args: dict = None
