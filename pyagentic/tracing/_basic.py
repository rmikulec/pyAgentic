# pyagentic/tracing/dictionary.py
from __future__ import annotations

import time
import uuid
import threading
from collections import defaultdict
from typing import Optional, Dict, Any, List

from pyagentic.tracing._tracer import AgentTracer, Span, SpanKind, SpanStatus, SpanContext


class BasicTracer(AgentTracer):
    """
    In-memory tracer that stores spans and events in Python dictionaries.

    - Start/End spans are recorded with timing and attributes
    - Events are appended per-span with timestamps
    - Export traces as a serializable dict (optionally clearing storage)
    """

    def __init__(self) -> None:
        # span_id -> Span (live object, updated in-place)
        self._spans: Dict[str, Span] = {}
        # span_id -> list[dict] events
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # trace_id -> list[span_id] (creation order)
        self._trace_index: Dict[str, List[str]] = defaultdict(list)
        # parent_span_id -> list[child_span_id]
        self._children: Dict[str, List[str]] = defaultdict(list)
        # simple lock for thread/async safety around shared structures
        self._lock = threading.RLock()

    # ---------- AgentTracer interface ----------

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        trace_id = parent.context.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex
        parent_span_id = parent.context.span_id if parent else None

        ctx = SpanContext(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id)
        span = Span(
            name=name,
            kind=kind,
            context=ctx,
            start_ns=time.monotonic_ns(),
            attributes=dict(attributes or {}),
        )

        with self._lock:
            self._spans[span_id] = span
            self._trace_index[trace_id].append(span_id)
            if parent_span_id:
                self._children[parent_span_id].append(span_id)

        return span

    def end_span(self, span: Span) -> None:
        with self._lock:
            span.end_ns = time.monotonic_ns()

    def _add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        evt = {
            "name": name,
            "ts_ns": time.monotonic_ns(),
            "attributes": dict(attributes or {}),
        }
        with self._lock:
            self._events[span.context.span_id].append(evt)

    def _set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        with self._lock:
            span.attributes.update(attributes)

    def _record_exception(self, span: Span, exc: BaseException) -> None:
        with self._lock:
            span.status = SpanStatus.ERROR
            span.error = f"{type(exc).__name__}: {exc}"
        # also record as an event
        self._add_event(
            span,
            "exception",
            {"type": type(exc).__name__, "message": str(exc)},
        )

    # ---------- Utilities / Export ----------

    def get_trace_ids(self) -> List[str]:
        with self._lock:
            return list(self._trace_index.keys())

    def get_span(self, span_id: str) -> Optional[Span]:
        with self._lock:
            return self._spans.get(span_id)

    def export_trace(self, trace_id: str, *, reset: bool = False) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict for a single trace:
        {
          "trace_id": "...",
          "spans": [
            {
              "span_id": "...",
              "parent_span_id": "...",
              "name": "...",
              "kind": "agent|tool|llm|step",
              "status": "ok|error",
              "start_ns": int,
              "end_ns": int|None,
              "duration_ms": float|None,
              "attributes": {...},
              "error": "..."|None,
              "events": [{"name": "...", "ts_ns": int, "attributes": {...}}, ...],
              "children": ["child_span_id", ...]
            },
            ...
          ]
        }
        """
        with self._lock:
            span_ids = list(self._trace_index.get(trace_id, []))
            # sort by start time for readability
            span_ids.sort(key=lambda sid: self._spans[sid].start_ns if sid in self._spans else 0)

            spans_out: List[Dict[str, Any]] = []
            for sid in span_ids:
                sp = self._spans.get(sid)
                if sp is None:
                    continue
                duration_ms = (
                    (sp.end_ns - sp.start_ns) / 1e6 if sp.end_ns is not None else None
                )
                spans_out.append(
                    {
                        "span_id": sp.context.span_id,
                        "parent_span_id": sp.context.parent_span_id,
                        "name": sp.name,
                        "kind": sp.kind.value,
                        "status": sp.status.value if isinstance(sp.status, SpanStatus) else str(sp.status),
                        "start_ns": sp.start_ns,
                        "end_ns": sp.end_ns,
                        "duration_ms": duration_ms,
                        "attributes": dict(sp.attributes),
                        "error": sp.error,
                        "events": list(self._events.get(sid, [])),
                        "children": list(self._children.get(sid, [])),
                    }
                )

            out = {"trace_id": trace_id, "spans": spans_out}

            if reset:
                # remove all data for this trace
                for sid in span_ids:
                    self._spans.pop(sid, None)
                    self._events.pop(sid, None)
                    self._children.pop(sid, None)
                self._trace_index.pop(trace_id, None)

            return out

    def export_all(self, *, reset: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            trace_ids = list(self._trace_index.keys())
        traces = [self.export_trace(tid, reset=reset) for tid in trace_ids]
        return traces

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._events.clear()
            self._trace_index.clear()
            self._children.clear()
