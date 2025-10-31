# pyagentic/policies/core.py
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from pyagentic._base._event import Event, EventKind
from pyagentic._base._policy import Policy


class HistoryPolicy:
    """Tracks all changes for a given state field."""

    def __init__(self, max_length: int | None = 100):
        self.max_length = max_length
        self._history: list[dict[str, Any]] = []

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:
            self._history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "name": event.name,
                    "old": event.old_value,
                    "new": event.new_value,
                }
            )
            if self.max_length and len(self._history) > self.max_length:
                self._history.pop(0)

    def get_history(self) -> list[dict[str, Any]]:
        return self._history


class AutoPersistPolicy:
    """Persists the state field to JSON whenever it changes."""

    def __init__(self, path: str):
        self.path = Path(path)

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:
            data = {
                "timestamp": event.timestamp.isoformat(),
                "name": event.name,
                "new_value": event.new_value,
            }

            existing = []
            if self.path.exists():
                try:
                    existing = json.loads(self.path.read_text())
                except json.JSONDecodeError:
                    existing = []

            existing.append(data)
            self.path.write_text(json.dumps(existing, indent=2))


class ValidatePolicy:
    """Validates state values with a provided predicate."""

    def __init__(self, predicate: Callable[[Any], bool], message: str):
        self.predicate = predicate
        self.message = message

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET and not self.predicate(event.new_value):
            raise ValueError(f"Invalid value for {event.name}: {self.message}")


class ReactivePolicy:
    """Automatically sets another field when this field changes."""

    def __init__(self, target_field: str, transform: Callable[[Any], Any] | None = None):
        self.target_field = target_field
        self.transform = transform or (lambda x: x)

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:
            state = event.context.get("state")
            if not state:
                return
            new_value = self.transform(event.new_value)
            state.set(self.target_field, new_value)


class SummarizePolicy:
    """Calls a summarizer agent when a text field exceeds a threshold length."""

    def __init__(self, summarizer_agent, max_length: int = 500):
        self.summarizer = summarizer_agent
        self.max_length = max_length

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET and isinstance(event.new_value, str):
            if len(event.new_value) > self.max_length:
                summary = await self.summarizer(f"Summarize this text:\n{event.new_value}")
                state = event.context.get("state")
                if state:
                    state.set(event.name, summary.final_output)


class EmitUpdatePolicy:
    """Sends live updates to an emitter (WebSocket, event bus, etc.)."""

    def __init__(self, emitter: Callable[[dict[str, Any]], Any]):
        self.emitter = emitter

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:
            payload = {
                "event": "state_update",
                "field": event.name,
                "value": event.new_value,
                "timestamp": event.timestamp.isoformat(),
            }
            result = self.emitter(payload)
            if asyncio.iscoroutine(result):
                await result


class TracePolicy:
    """Integrates with the AgentTracer to record all state transitions."""

    def __init__(self, tracer):
        self.tracer = tracer

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:
            self.tracer.record_attribute(
                f"state.{event.name}",
                {"old": event.old_value, "new": event.new_value},
            )


class TTLPolicy:
    """Clears a state field after `ttl_seconds` seconds."""

    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds

    async def handle_event(self, event: Event):
        if event.kind == EventKind.SET:

            async def _expire():
                await asyncio.sleep(self.ttl)
                state = event.context.get("state")
                if state:
                    state.set(event.name, None)

            asyncio.create_task(_expire())


class DebugPolicy:
    """Sends live updates to an emitter (WebSocket, event bus, etc.)."""

    def __init__(self, log: Callable):
        self.log = log

    def handle_event(self, event: Event):
        print("Called")
        if event.kind == EventKind.SET:
            payload = {
                "event": "state_update",
                "field": event.name,
                "value": event.new_value,
                "timestamp": event.timestamp.isoformat(),
            }
            self.log(payload)
        elif event.kind == EventKind.GET:
            payload = {
                "event": "state_get",
                "field": event.name,
                "value": event.new_value,
                "timestamp": event.timestamp.isoformat(),
            }
            self.log(payload)
