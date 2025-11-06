from typing import Any, TypeVar, Generic
from dataclasses import dataclass

from pydantic import BaseModel

from pyagentic._base._info import StateInfo

T = TypeVar("T")


class State(Generic[T]):
    """
    Type annotation for defining persistent state fields in agents.

    State fields hold data that persists across tool calls and LLM interactions.
    Use Pydantic models as the state type to ensure type safety and validation.

    Args:
        T: A Pydantic BaseModel class defining the structure of the state

    Example:
        ```python
        from pydantic import BaseModel

        class ConversationState(BaseModel):
            user_name: str
            message_count: int = 0

        class ChatAgent(BaseAgent):
            __system_message__ = "You are a chatbot"

            # Simple state field
            conversation: State[ConversationState]

            # State field with default value
            logs: State[list] = spec.State(default_factory=list)
        ```
    """

    def __class_getitem__(cls, item):
        """
        Creates a generic State type for a given model class.

        Args:
            item: The model class to wrap as a state field

        Returns:
            type: Special marker type that the metaclass can detect and process
        """
        # Return a special marker type that metaclass can detect
        return type(
            f"State[{item.__name__}]",
            (),
            {"__origin__": State, "__args__": (item,), "__state_model__": item},
        )


@dataclass
class _StateDefinition:
    """
    Internal definition of a state field combining model type and metadata.

    Pairs a Pydantic model class with StateInfo containing defaults and policies.
    """

    model: BaseModel
    info: StateInfo = None
