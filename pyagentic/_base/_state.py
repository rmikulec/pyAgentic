from typing import Any, TypeVar, Generic, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from pyagentic._base._info import StateInfo

T = TypeVar("T")


class State(Generic[T]):
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
            {"__origin__": State, "__args__": (item,), "__state_model__": item},
        )


@dataclass
class _StateDefinition:
    model: BaseModel
    info: StateInfo = None
