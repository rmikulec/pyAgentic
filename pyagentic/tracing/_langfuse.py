# pyagentic/tracing/langfuse.py
from __future__ import annotations

from typing import Optional, Dict, Any
import time
import uuid

from pyagentic.models.tracing import Span, SpanKind, SpanStatus, SpanContext
from pyagentic.tracing._tracer import AgentTracer

try:
    from langfuse import Langfuse
    from langfuse.client import StatefulTraceClient, StatefulObservationClient
except Exception:  # pragma: no cover - optional dep
    Langfuse = None
    StatefulTraceClient = None
    StatefulObservationClient = None


class LangfuseTracer(AgentTracer):
    """
    Maps PyAgentic spans to Langfuse Traces/Observations.

    - First AGENT span in a trace becomes the Langfuse "trace"
    - Other spans become "observations" under that trace
    """

    def __init__(self, *, client: Optional[Langfuse] = None, **kwargs: Any) -> None:
        if client is None and Langfuse is None:
            raise RuntimeError("langfuse is not installed: `pip install langfuse`")
        self.client = client or Langfuse(**kwargs)
        # span_id -> (trace_or_obs_client, is_trace)
        self._handles: Dict[str, tuple[StatefulTraceClient | StatefulObservationClient, bool]] = {}

    def _mk_ids(self, parent: Optional[Span]) -> tuple[str, str, Optional[str]]:
        trace_id = parent.context.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex
        parent_span_id = parent.context.span_id if parent else None
        return trace_id, span_id, parent_span_id

    def start_span(self, name: str, kind: SpanKind, parent: Optional[Span] = None, attributes: Optional[Dict[str, Any]] = None) -> Span:
        trace_id, span_id, parent_span_id = self._mk_ids(parent)
        ctx = SpanContext(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id)
        span = Span(name=name, kind=kind, context=ctx, start_ns=time.monotonic_ns(), attributes=dict(attributes or {}))

        # Create or attach to Langfuse objects
        if parent is None or kind == SpanKind.AGENT and parent is None:
            trace = self.client.trace(
                id=trace_id,
                name=name,
                input=attributes.get("input") if attributes else None,
                metadata=attributes,
            )
            self._handles[span_id] = (trace, True)
        else:
            parent_handle, parent_is_trace = self._handles[parent.context.span_id]
            trace = parent_handle if parent_is_trace else parent_handle.trace
            obs = trace.observation(
                name=name,
                type=kind.value,  # 'agent' | 'tool' | 'llm' | 'step'
                metadata=attributes,
            )
            self._handles[span_id] = (obs, False)

        return span

    def end_span(self, span: Span) -> None:
        span.end_ns = time.monotonic_ns()
        handle, is_trace = self._handles.get(span.context.span_id, (None, False))
        if handle is None:
            return
        duration_ms = (span.end_ns - span.start_ns) / 1e6
        if is_trace:
            handle.update(
                name=span.name,
                metadata=span.attributes,
                status_message=span.error,
                duration=duration_ms,
                level="ERROR" if span.status == SpanStatus.ERROR else "DEFAULT",
            )
        else:
            handle.update(
                name=span.name,
                metadata=span.attributes,
                status_message=span.error,
                duration=duration_ms,
                level="ERROR" if span.status == SpanStatus.ERROR else "DEFAULT",
            )

    def _add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        handle, _ = self._handles.get(span.context.span_id, (None, False))
        if handle is not None:
            handle.event(name=name, metadata=attributes or {})

    def _set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        span.attributes.update(attributes)
        handle, is_trace = self._handles.get(span.context.span_id, (None, False))
        if handle is not None:
            handle.update(metadata=span.attributes)

    def _record_exception(self, span: Span, exc: BaseException) -> None:
        span.error = f"{type(exc).__name__}: {exc}"
        self._add_event(span, "exception", {"type": type(exc).__name__, "message": str(exc)})
