from typing import Protocol, TypeVar, Any
from pyagentic.policies._events import GetEvent, SetEvent

T = TypeVar("T")


class Policy(Protocol[T]):
    """
    Base interface for all State policies.

    Each method receives:
        - event: contextual metadata (field name, timestamp, etc.)
        - value: the current value being processed

    Each may:
        - Return a transformed value of the same type `T`
        - Return None to indicate no change
        - Raise an exception to block or veto the operation
    """

    def on_get(self, event: GetEvent, value: T) -> T | None: ...
    async def background_get(self, event: GetEvent, value: T) -> T | None: ...
    def on_set(self, event: SetEvent, value: T) -> T | None: ...
    async def background_set(self, event: SetEvent, value: T) -> T | None: ...
