from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from datetime import datetime


class EventKind(Enum):
    INIT = auto()
    GET = auto()
    SET = auto()
    DELETE = auto()
    TIMER = auto()
    CUSTOM = auto()


@dataclass
class Event:
    """A structured, serializable event object for policy triggers."""

    kind: EventKind
    name: str  # Name of the state field
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = datetime.now()
    context: dict[str, Any] | None = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
