from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, AsyncIterator
import time
import uuid
import contextvars


class SpanKind(str, Enum):
    
    AGENT = "agent"
    TOOL = "tool"
    INFERENCE = "inference"
    STEP = "step"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


@dataclass
class Span:
    """A light, tracer-agnostic span handle used by PyAgentic."""
    name: str
    kind: SpanKind
    context: SpanContext
    start_ns: int
    end_ns: Optional[int] = None
    status: SpanStatus = SpanStatus.OK
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "pyagentic_current_span", default=None
)


class AgentTracer(ABC):
    """
    Tracer interface for PyAgentic.

    Implementations should override start_span/end_span/add_event/set_attributes/record_exception/record_tokens.
    """

    # ---------- low-level lifecycle ----------
    @abstractmethod
    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        ...

    @abstractmethod
    def end_span(self, span: Span) -> None:
        ...

    # ---------- enrichment ----------
    @abstractmethod
    def add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def record_exception(self, span: Span, exc: BaseException) -> None:
        ...

    def record_tokens(
        self,
        span: Span,
        *,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Default token recording as an event; impls may override to native metrics."""
        payload = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        if attributes:
            payload.update(attributes)
        self.add_event(span, "llm.tokens", payload)

    # ---------- ergonomics: async contexts ----------
    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind,
        *,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> AsyncIterator[Span]:
        parent = parent or _current_span.get()
        span = self.start_span(name=name, kind=kind, parent=parent, attributes=attributes)
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as e:
            span.status = SpanStatus.ERROR
            self.record_exception(span, e)
            raise
        finally:
            _current_span.reset(token)
            self.end_span(span)

    @asynccontextmanager
    async def agent(self, name: str, parent: Optional[Span] = None, **attrs: Any) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.AGENT, parent=parent, attributes=attrs) as s:
            yield s

    @asynccontextmanager
    async def tool(self, name: str, parent: Optional[Span] = None, **attrs: Any) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.TOOL, parent=parent, attributes=attrs) as s:
            yield s

    @asynccontextmanager
    async def inference(self, name: str = "completion", **attrs: Any) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.INFERENCE, attributes=attrs) as s:
            yield s