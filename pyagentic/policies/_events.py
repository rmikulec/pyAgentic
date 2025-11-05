from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from datetime import datetime


class EventKind(Enum):
    INIT = auto()
    GET = auto()
    SET = auto()
    DELETE = auto()
    TIMER = auto()


@dataclass
class Event:
    """Base event class shared by all event types."""

    kind: EventKind
    name: Optional[str] = None
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} kind={self.kind.name} value={self.value!r}>"

    # Allows easy cloning or modification without mutation
    def with_value(self, value: Any) -> "Event":
        """Return a shallow copy of this event with a new value."""
        return self.__class__(**{**self.__dict__, "value": value})


@dataclass
class GetEvent(Event):
    """Represents reading a value from state."""

    kind: EventKind = EventKind.GET


@dataclass
class SetEvent(Event):
    """Represents writing or updating a value in state."""

    kind: EventKind = EventKind.SET
    previous: Any = None  # Optional context: previous stored value
