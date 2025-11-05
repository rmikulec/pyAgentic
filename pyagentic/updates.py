from pydantic import BaseModel
from typing import Literal
from enum import Enum


class Status(Enum):
    """
    Enumeration of possible agent execution statuses.
    """
    GENERATING = "generating"
    PROCESSING = "processing"
    SUCCEDED = "succeded"
    ERROR = "error"


class EmitUpdate(BaseModel):
    """
    Base update model for emitting agent execution status changes.
    """
    type: Literal["base"] = "base"
    status: Status


class AiUpdate(EmitUpdate):
    """
    Update model for AI response events, includes the generated message.
    """
    type: Literal["ai_response"] = "ai_response"
    message: str = None


class ToolUpdate(EmitUpdate):
    """
    Update model for tool execution events, includes tool name and arguments.
    """
    type: Literal["tool_update"] = "tool_update"
    tool_call: str = None
    tool_args: dict = None
