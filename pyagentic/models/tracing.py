from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SpanKind(str, Enum):
    """
    Enumeration of span types for tracing agent execution.

    Categorizes different types of operations in the agent lifecycle for
    observability and debugging purposes.
    """

    AGENT = "agent"
    TOOL = "tool"
    INFERENCE = "inference"
    STEP = "step"


class SpanStatus(str, Enum):
    """
    Enumeration of span status values.

    Indicates whether a traced operation completed successfully or encountered an error.
    """

    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanContext:
    """
    Immutable context information for a span.

    Links a span to its trace and parent span for building execution hierarchies.
    """

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
