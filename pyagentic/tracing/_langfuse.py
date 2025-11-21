# pyagentic/tracing/langfuse_tracer.py
from __future__ import annotations

import time
import uuid
import threading
from typing import Optional, Dict, Any, List, Tuple

from pyagentic.tracing._tracer import AgentTracer
from pyagentic.models.tracing import Span, SpanContext, SpanKind, SpanStatus

try:
    # Langfuse Python SDK v3 (OTel-based)
    from langfuse import get_client  # type: ignore

    _LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover
    _LANGFUSE_AVAILABLE = False


class LangfuseTracer(AgentTracer):
    """
    Tracer that forwards spans/events to Langfuse via the Python SDK v3.

    - Uses manual observations so we can start/end spans explicitly.
    - Maps SpanKind.LLM -> Langfuse Generation; others -> Langfuse Span.
    - Attributes are merged via Langfuse `metadata` updates.
    - Events are sent as Langfuse "events" attached to the span.
    - Maintains a minimal in-memory index of *live* spans (for lifecycle),
      but does not retain completed data (use the Langfuse UI/API to query).

    Notes:
    - Configure Langfuse via environment variables (recommended) and the SDK's
      `get_client()` will pick them up.
    - Observation API surface used here: `start_span/start_generation`, `.update(...)`,
      `.end()`, and `.event(...)`.
    """

    def __init__(self) -> None:
        """
        Initialize the Langfuse tracer with the SDK client.

        Raises:
            RuntimeError: If Langfuse SDK is not installed or not configured
        """
        if not _LANGFUSE_AVAILABLE:
            raise RuntimeError(
                "Langfuse SDK not available. Install with `pip install langfuse` "
                "and ensure environment variables are set."
            )

        # langfuse client (lazy init to avoid import side effects during tests)
        self._client = get_client()  # uses env config; see docs.

        # span_id -> (wrapped_observation, kind)
        self._wrapped: Dict[str, Tuple[object, SpanKind]] = {}

        # book-keeping similar to BasicTracer but only for *live* spans:
        self._spans: Dict[str, Span] = {}
        self._trace_index: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

        self._lock = threading.RLock()

    # ---------- AgentTracer interface ----------

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Create a Langfuse observation and a local Span wrapper.

        - If `kind` is LLM, start a Generation (so model/usage can be attached).
        - Otherwise start a generic Span.
        - Parent/child: if a parent Span is provided, we create the child via the
          parent's wrapped observation, preserving nesting.
        """
        attrs = dict(attributes or {})

        # We create a pyagentic Span immediately for caller ergonomics.
        trace_id = parent.context.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex
        parent_span_id = parent.context.span_id if parent else None

        ctx = SpanContext(trace_id=trace_id, span_id=span_id, parent_span_id=parent_span_id)
        py_span = Span(
            name=name,
            kind=kind,
            context=ctx,
            start_ns=time.monotonic_ns(),
            attributes=attrs.copy(),
        )

        # Create Langfuse observation (manual start; we'll call .end() later)
        if parent is not None:
            parent_wrapped = self._wrapped.get(parent.context.span_id)
            if parent_wrapped is None:
                # Fallback: no known parent wrapper; start at root.
                wrapped = self._start_root_observation(name, kind)
            else:
                p_obs, _ = parent_wrapped
                # Child spans must be created from the parent object (manual API).
                if kind == SpanKind.INFERENCE:
                    # Prefer "generation" for LLM work; model can be passed
                    #   later via ._set_attributes
                    wrapped = getattr(p_obs, "start_generation", getattr(p_obs, "start_span"))(
                        name=name
                    )
                else:
                    wrapped = getattr(p_obs, "start_span")(name=name)
        else:
            wrapped = self._start_root_observation(name, kind)

        # Initial attribute merge -> Langfuse metadata
        if attrs:
            try:
                wrapped.update(metadata=attrs)  # Arbitrary JSON metadata.
            except Exception:
                pass  # don't break tracing for metadata issues

        with self._lock:
            self._spans[span_id] = py_span
            self._wrapped[span_id] = (wrapped, kind)
            self._trace_index.setdefault(trace_id, []).append(span_id)
            if parent_span_id:
                self._children.setdefault(parent_span_id, []).append(span_id)

        return py_span

    def end_span(self, span: Span) -> None:
        """
        End a span and notify Langfuse.

        Args:
            span (Span): The span to end
        """
        with self._lock:
            wrapped_tuple = self._wrapped.get(span.context.span_id)
        if wrapped_tuple is not None:
            wrapped, _ = wrapped_tuple
            try:
                wrapped.end()  # Important for manual observations.
            except Exception:
                pass
        with self._lock:
            span.end_ns = time.monotonic_ns()
            # We intentionally do NOT delete from _wrapped/_spans immediately;
            # callers may still update attributes/events right after end. The
            # Langfuse SDK will merge updates within its allowed window.

    # ---------- Internal helpers used by AgentTracer base ----------

    def _add_event(
        self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Attach a point-in-time event to the observation.
        """
        meta = dict(attributes or {})
        with self._lock:
            wrapped_tuple = self._wrapped.get(span.context.span_id)
        if wrapped_tuple is None:
            return
        wrapped, _ = wrapped_tuple
        try:
            # Langfuse events are created from the observation; we immediately end them.
            evt = wrapped.event(name=name, metadata=meta)  # create event
            # Best-effort close; SDKs commonly end events via .end(...).
            end_kwargs = {}
            if "output" in meta:
                end_kwargs["output"] = meta["output"]
            evt.end(**end_kwargs)
        except Exception:
            pass

    def _set_attributes(self, span: Span, attributes: Dict[str, Any]) -> None:
        with self._lock:
            wrapped_tuple = self._wrapped.get(span.context.span_id)
        if wrapped_tuple is None:
            return

        wrapped, wrapped_kind = wrapped_tuple

        # Copy so we can pop safely without KeyErrors
        attrs = dict(attributes)

        # Pull these if present; leave others in attrs -> metadata
        usage_details = attrs.pop("usage_details", None)
        model = attrs.pop("model", None)
        input_ = attrs.pop("input", None)
        output = attrs.pop("output", None)
        model_params = attrs.pop("model_parameters", None)  # optional, useful to pass through

        # Build kwargs for .update() selectively
        update_kwargs: Dict[str, Any] = {}
        if input_ is not None:
            update_kwargs["input"] = input_
        if output is not None:
            update_kwargs["output"] = output

        # Only generations accept model/usage/model_parameters in Langfuse v3
        if wrapped_kind == SpanKind.INFERENCE:
            if model is not None:
                update_kwargs["model"] = model
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            if model_params is not None:
                update_kwargs["model_parameters"] = model_params

        if attrs:  # remaining keys -> metadata
            update_kwargs["metadata"] = attrs

        try:
            if update_kwargs:
                wrapped.update(**update_kwargs)
        except Exception:
            # best-effort; don't break tracing
            pass

        # Keep local mirror in sync
        with self._lock:
            # Merge everything (including input/output/etc.) into span.attributes for export/debug
            span.attributes.update(dict(attributes))

    def _record_exception(self, span: Span, exc: BaseException) -> None:
        """
        Mark error, attach exception event, and pass it to the underlying span.
        """
        # Local bookkeeping
        with self._lock:
            span.status = SpanStatus.ERROR
            span.error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            wrapped_tuple = self._wrapped.get(span.context.span_id)
        if wrapped_tuple is None:
            return
        wrapped, _ = wrapped_tuple
        try:
            # Record as an event for visibility in the UI
            self._add_event(span, "exception", {"type": type(exc).__name__, "message": str(exc)})
            # Also forward to underlying OTel span if exposed
            if hasattr(wrapped, "record_exception"):
                wrapped.record_exception(exc)  # OTel compatibility.
        except Exception:
            pass

    # ---------- Export / Maintenance ----------

    def get_trace_ids(self) -> List[str]:
        """
        Get all trace IDs currently tracked locally.

        Returns:
            List[str]: List of trace IDs
        """
        with self._lock:
            return list(self._trace_index.keys())

    def get_span(self, span_id: str) -> Optional[Span]:
        """
        Retrieve a span by its ID from local cache.

        Args:
            span_id (str): The span ID to look up

        Returns:
            Optional[Span]: The span, or None if not found
        """
        with self._lock:
            return self._spans.get(span_id)

    def export_trace(self, trace_id: str, *, reset: bool = False) -> Dict[str, Any]:
        """
        Export minimal trace data from local cache.

        Note: This is not a full history—use Langfuse UI/API for authoritative data.

        Args:
            trace_id (str): The trace ID to export
            reset (bool): Whether to clear the trace from local cache. Defaults to False.

        Returns:
            Dict[str, Any]: Local trace data snapshot
        """
        with self._lock:
            span_ids = list(self._trace_index.get(trace_id, []))
            span_ids.sort(key=lambda sid: self._spans[sid].start_ns if sid in self._spans else 0)

            spans_out: List[Dict[str, Any]] = []
            for sid in span_ids:
                sp = self._spans.get(sid)
                if sp is None:
                    continue
                duration_ms = ((sp.end_ns - sp.start_ns) / 1e6) if sp.end_ns is not None else None
                spans_out.append(
                    {
                        "span_id": sp.context.span_id,
                        "parent_span_id": sp.context.parent_span_id,
                        "name": sp.name,
                        "kind": sp.kind.value,
                        "status": (
                            sp.status.value
                            if isinstance(sp.status, SpanStatus)
                            else str(sp.status)
                        ),
                        "start_ns": sp.start_ns,
                        "end_ns": sp.end_ns,
                        "duration_ms": duration_ms,
                        "attributes": dict(sp.attributes),
                        "error": sp.error,
                        # events not stored locally in this tracer
                        "events": [],
                        "children": list(self._children.get(sid, [])),
                    }
                )

            out = {"trace_id": trace_id, "spans": spans_out}

            if reset:
                for sid in span_ids:
                    self._spans.pop(sid, None)
                    self._wrapped.pop(sid, None)
                    self._children.pop(sid, None)
                self._trace_index.pop(trace_id, None)

            return out

    def export_all(self, *, reset: bool = False) -> List[Dict[str, Any]]:
        """
        Export all traces from local cache.

        Args:
            reset (bool): Whether to clear all traces from local cache. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of local trace data snapshots
        """
        with self._lock:
            trace_ids = list(self._trace_index.keys())
        return [self.export_trace(tid, reset=reset) for tid in trace_ids]

    def clear(self) -> None:
        """
        Clear all local caches (does not affect data already sent to Langfuse).
        """
        with self._lock:
            self._spans.clear()
            self._wrapped.clear()
            self._trace_index.clear()
            self._children.clear()

    # ---------- Private ----------

    def _start_root_observation(self, name: str, kind: SpanKind):
        """
        Start a root observation on the Langfuse client.

        Uses the manual API to align with explicit start/end in AgentTracer.

        Args:
            name (str): Name of the observation
            kind (SpanKind): Type of span (determines whether to create generation or span)

        Returns:
            Langfuse observation object
        """
        # LLM spans become Generations so model/usage/cost can be attached later.
        if kind == SpanKind.INFERENCE:
            # Manual start; caller may later .update(model="...")
            #   on the wrapper via _set_attributes
            return self._client.start_generation(name=name)
        else:
            return self._client.start_span(name=name)
