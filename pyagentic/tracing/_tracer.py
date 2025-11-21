import contextvars

from functools import wraps
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, AsyncIterator, Callable

from pyagentic.models.tracing import Span, SpanStatus, SpanKind


_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "pyagentic_current_span", default=None
)


class AgentTracer(ABC):
    """
    Tracer interface for PyAgentic.

    Implementations should override start_span/end_span/add_event/
        set_attributes/record_exception/record_tokens.
    """

    @property
    def current_span(self) -> Optional[Span]:
        """
        Returns the currently active span from context.

        Returns:
            Optional[Span]: The current span, or None if no span is active
        """
        return _current_span.get()

    def set_attributes(self, **kwargs):
        """
        Set attributes on the current span.

        Args:
            **kwargs: Arbitrary key-value pairs to attach as span attributes
        """
        self._set_attributes(self.current_span, kwargs)

    def add_event(self, name, **kwargs):
        """
        Add a point-in-time event to the current span.

        Args:
            name (str): Name of the event
            **kwargs: Event attributes as key-value pairs
        """
        self._add_event(self.current_span, name, kwargs)

    def record_exception(self, exception: str):
        """
        Record an exception on the current span.

        Args:
            exception (str): Exception message or description
        """
        self._record_exception(self.current_span, exc=exception)

    # ---------- low-level lifecycle ----------
    @abstractmethod
    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name (str): Name of the span
            kind (SpanKind): Type of span (agent, tool, inference, step)
            parent (Optional[Span]): Parent span for nesting. Defaults to None.
            attributes (Optional[Dict[str, Any]]): Initial span attributes. Defaults to None.

        Returns:
            Span: The newly created span
        """
        ...

    @abstractmethod
    def end_span(self, span: Span) -> None:
        """
        End a span and record its completion time.

        Args:
            span (Span): The span to end
        """
        ...

    # ---------- enrichment ----------
    @abstractmethod
    def _add_event(
        self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an event to a specific span (internal implementation).

        Args:
            span (Span): The span to add the event to
            name (str): Name of the event
            attributes (Optional[Dict[str, Any]]): Event attributes. Defaults to None.
        """
        ...

    @abstractmethod
    def _set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        """
        Set attributes on a specific span (internal implementation).

        Args:
            span (Span): The span to update
            attributes (Dict[str, Any]): Attributes to set
        """
        ...

    @abstractmethod
    def _record_exception(self, span: Span, exc: BaseException) -> None:
        """
        Record an exception on a specific span (internal implementation).

        Args:
            span (Span): The span to record the exception on
            exc (BaseException): The exception to record
        """
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
        """
        Record token usage information for an LLM call.

        Default implementation records tokens as an event; implementations may
        override to use native metrics.

        Args:
            span (Span): The span to record tokens on
            model (Optional[str]): Model identifier. Defaults to None.
            prompt_tokens (Optional[int]): Number of input tokens. Defaults to None.
            completion_tokens (Optional[int]): Number of output tokens. Defaults to None.
            total_tokens (Optional[int]): Total tokens used. Defaults to None.
            attributes (Optional[Dict[str, Any]]): Additional attributes. Defaults to None.
        """
        payload = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
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
    async def agent(
        self, name: str, parent: Optional[Span] = None, **attrs: Any
    ) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.AGENT, parent=parent, attributes=attrs) as s:
            yield s

    @asynccontextmanager
    async def tool(
        self, name: str, parent: Optional[Span] = None, **attrs: Any
    ) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.TOOL, parent=parent, attributes=attrs) as s:
            yield s

    @asynccontextmanager
    async def inference(self, name: str = "completion", **attrs: Any) -> AsyncIterator[Span]:
        async with self.span(name, SpanKind.INFERENCE, attributes=attrs) as s:
            yield s


def traced(kind: SpanKind, name: Optional[str] = None, attrs: Optional[dict] = None):
    """
    Decorator to run async methods inside a traced span.

    Uses Agent.tracer and current_span ContextVar for automatic span nesting.

    Args:
        kind (SpanKind): Type of span to create
        name (Optional[str]): Custom span name. Defaults to ClassName.method_name.
        attrs (Optional[dict]): Initial span attributes. Defaults to None.

    Returns:
        Callable: Decorated function that runs within a span context
    """

    def outer(fn: Callable):

        @wraps(fn)
        async def async_wrapper(self, *args, **kwargs):
            span_name = name or f"{self.__class__.__name__}.{fn.__name__}"
            async with self.tracer.span(span_name, kind, attributes=attrs):
                self.tracer.set_attributes(input=kwargs)
                try:
                    output = await fn(self, *args, **kwargs)
                except Exception as e:
                    self.tracer.record_exception(str(e))
                    raise e
                self.tracer.set_attributes(output=output)
                return output

        return async_wrapper

    return outer
