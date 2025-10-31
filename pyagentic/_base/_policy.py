# pyagentic/_base/_policy.py
from typing import Protocol, runtime_checkable
from pyagentic._base._event import Event


@runtime_checkable
class Policy(Protocol):
    """Base interface for all State policies."""

    async def handle_event(self, event: Event) -> None: ...
