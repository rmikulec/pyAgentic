from typing import Any, TypeVar, Generic
from dataclasses import dataclass

from pydantic import BaseModel

from pyagentic._base._info import AgentInfo
from pyagentic._base._agent import BaseAgent

T = TypeVar("T")


class Link(Generic[T]):
    """
    Generic type for annotating agent state fields.

    Usage:
        local_repository: State[LocalRepository]
        conversation: State[ConversationState] = spec.State(persist=True)
    """

    def __class_getitem__(cls, item):
        # Return a special marker type that metaclass can detect
        return type(
            f"State[{item.__name__}]",
            (),
            {"__origin__": Link, "__args__": (item,), "__linked_agent__": item},
        )


@dataclass
class _LinkedAgentDefinition:
    agent: BaseAgent
    info: AgentInfo = None
