import inspect

from functools import wraps
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncIterator, Callable
import contextvars

from pyagentic.models.tracing import Span, SpanStatus, SpanKind, SpanContext


_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "pyagentic_current_span", default=None
)


class AgentTracer(ABC):
    """
    Tracer interface for PyAgentic.

    Implementations should override start_span/end_span/add_event/set_attributes/record_exception/record_tokens.
    """

    @property
    def current_span(self) -> Optional[Span]:
        return _current_span.get()

    def set_attributes(self, **kwargs):
        self._set_attributes(
            self.current_span,
            kwargs
        )

    def _add_event(self, name, **kwargs):
        self._add_event(
            self.current_span,
            name,
            kwargs
        )
    
    def record_exception(self, exception: str):
        self._record_exception(
            self.current_span,
            exc=exception
        )

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
    def _add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def _set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def _record_exception(self, span: Span, exc: BaseException) -> None:
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
        self._add_event(span, "llm.tokens", payload)

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
            self._record_exception(span, e)
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


def traced(kind: SpanKind, name: Optional[str] = None, attrs: Optional[dict] = None):
    """
    Decorate async or sync callables to run inside a span.
    Uses Agent.tracer and current_span ContextVar for nesting.
    """
    def outer(fn: Callable):
        is_async = inspect.iscoroutinefunction(fn)

        @wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            tracer: AgentTracer = self.tracer
            span_name = name or f"{self.__class__.__name__}.{fn.__name__}"
            async with tracer.span(span_name, kind, attributes=attrs):
                return await fn(self, *args, **kwargs)


        return async_wrapper
    return outer