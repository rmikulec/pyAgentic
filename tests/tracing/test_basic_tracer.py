import pytest
import asyncio
import time
import threading
from unittest.mock import patch

from pyagentic.tracing import BasicTracer
from pyagentic.models.tracing import SpanKind, SpanStatus, Span, SpanContext


class TestBasicTracer:
    """Test suite for BasicTracer functionality."""

    @pytest.fixture
    def tracer(self):
        return BasicTracer()

    def test_tracer_initialization(self, tracer):
        """Test that BasicTracer initializes with empty state."""
        assert len(tracer._spans) == 0
        assert len(tracer._events) == 0
        assert len(tracer._trace_index) == 0
        assert len(tracer._children) == 0
        assert tracer._lock is not None

    def test_start_span_root(self, tracer):
        """Test starting a root span (no parent)."""
        span = tracer.start_span("test-root", SpanKind.AGENT)

        assert span.name == "test-root"
        assert span.kind == SpanKind.AGENT
        assert span.context.parent_span_id is None
        assert span.context.trace_id is not None
        assert span.context.span_id is not None
        assert span.start_ns > 0
        assert span.end_ns is None
        assert span.status == SpanStatus.OK
        assert span.attributes == {}
        assert span.error is None

    def test_start_span_with_attributes(self, tracer):
        """Test starting a span with initial attributes."""
        attrs = {"model": "gpt-4", "temperature": 0.7}
        span = tracer.start_span("test-span", SpanKind.INFERENCE, attributes=attrs)

        assert span.attributes == attrs

    def test_start_child_span(self, tracer):
        """Test starting a child span with parent."""
        parent = tracer.start_span("parent", SpanKind.AGENT)
        child = tracer.start_span("child", SpanKind.TOOL, parent=parent)

        assert child.context.parent_span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id
        assert child.context.span_id != parent.context.span_id

    def test_end_span(self, tracer):
        """Test ending a span sets end timestamp."""
        span = tracer.start_span("test-span", SpanKind.AGENT)
        assert span.end_ns is None

        tracer.end_span(span)
        assert span.end_ns is not None
        assert span.end_ns > span.start_ns

    def test_set_attributes(self, tracer):
        """Test setting attributes on a span."""
        span = tracer.start_span("test-span", SpanKind.AGENT)
        tracer._set_attributes(span, {"key1": "value1", "key2": 42})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42

    def test_add_event(self, tracer):
        """Test adding events to a span."""
        span = tracer.start_span("test-span", SpanKind.AGENT)

        tracer._add_event(span, "test-event", {"data": "test"})

        events = tracer._events[span.context.span_id]
        assert len(events) == 1
        assert events[0]["name"] == "test-event"
        assert events[0]["attributes"]["data"] == "test"
        assert "ts_ns" in events[0]

    def test_record_exception(self, tracer):
        """Test recording an exception."""
        span = tracer.start_span("test-span", SpanKind.AGENT)
        exception = ValueError("Test error")

        tracer._record_exception(span, exception)

        assert span.status == SpanStatus.ERROR
        assert span.error == "ValueError: Test error"

        # Check that exception event was added
        events = tracer._events[span.context.span_id]
        assert len(events) == 1
        assert events[0]["name"] == "exception"
        assert events[0]["attributes"]["type"] == "ValueError"
        assert events[0]["attributes"]["message"] == "Test error"

    def test_current_span_context_var(self, tracer):
        """Test that current_span returns None initially."""
        assert tracer.current_span is None

    def test_record_tokens(self, tracer):
        """Test recording token usage."""
        span = tracer.start_span("test-span", SpanKind.INFERENCE)

        tracer.record_tokens(
            span,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        events = tracer._events[span.context.span_id]
        assert len(events) == 1
        event = events[0]
        assert event["name"] == "llm.tokens"
        assert event["attributes"]["model"] == "gpt-4"
        assert event["attributes"]["prompt_tokens"] == 100
        assert event["attributes"]["completion_tokens"] == 50
        assert event["attributes"]["total_tokens"] == 150

    def test_get_trace_ids(self, tracer):
        """Test getting all trace IDs."""
        span1 = tracer.start_span("span1", SpanKind.AGENT)
        span2 = tracer.start_span("span2", SpanKind.AGENT)

        trace_ids = tracer.get_trace_ids()
        assert len(trace_ids) == 2
        assert span1.context.trace_id in trace_ids
        assert span2.context.trace_id in trace_ids

    def test_get_span(self, tracer):
        """Test retrieving a span by ID."""
        span = tracer.start_span("test-span", SpanKind.AGENT)

        retrieved = tracer.get_span(span.context.span_id)
        assert retrieved is span

        # Test non-existent span
        assert tracer.get_span("nonexistent") is None

    def test_export_trace(self, tracer):
        """Test exporting a single trace."""
        parent = tracer.start_span("parent", SpanKind.AGENT, attributes={"test": "value"})
        child = tracer.start_span("child", SpanKind.TOOL, parent=parent)
        tracer._add_event(parent, "test-event", {"event_data": "test"})
        tracer.end_span(child)
        tracer.end_span(parent)

        export = tracer.export_trace(parent.context.trace_id)

        assert export["trace_id"] == parent.context.trace_id
        assert len(export["spans"]) == 2

        # Check parent span
        parent_export = next(s for s in export["spans"] if s["span_id"] == parent.context.span_id)
        assert parent_export["name"] == "parent"
        assert parent_export["kind"] == "agent"
        assert parent_export["status"] == "ok"
        assert parent_export["attributes"]["test"] == "value"
        assert len(parent_export["events"]) == 1
        assert parent_export["events"][0]["name"] == "test-event"
        assert child.context.span_id in parent_export["children"]

    def test_export_trace_with_reset(self, tracer):
        """Test exporting a trace and clearing it."""
        span = tracer.start_span("test-span", SpanKind.AGENT)
        trace_id = span.context.trace_id

        export = tracer.export_trace(trace_id, reset=True)

        # Check export contains data
        assert export["trace_id"] == trace_id
        assert len(export["spans"]) == 1

        # Check data was cleared
        assert trace_id not in tracer._trace_index
        assert span.context.span_id not in tracer._spans

    def test_export_all(self, tracer):
        """Test exporting all traces."""
        span1 = tracer.start_span("span1", SpanKind.AGENT)
        span2 = tracer.start_span("span2", SpanKind.AGENT)

        exports = tracer.export_all()

        assert len(exports) == 2
        trace_ids = [e["trace_id"] for e in exports]
        assert span1.context.trace_id in trace_ids
        assert span2.context.trace_id in trace_ids

    def test_clear(self, tracer):
        """Test clearing all trace data."""
        tracer.start_span("test-span", SpanKind.AGENT)

        assert len(tracer._spans) > 0
        tracer.clear()

        assert len(tracer._spans) == 0
        assert len(tracer._events) == 0
        assert len(tracer._trace_index) == 0
        assert len(tracer._children) == 0

    def test_thread_safety(self, tracer):
        """Test that the tracer is thread-safe."""
        spans = []

        def create_spans():
            for i in range(10):
                span = tracer.start_span(f"span-{i}", SpanKind.AGENT)
                spans.append(span)
                tracer.end_span(span)

        threads = [threading.Thread(target=create_spans) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 30 spans total
        assert len(tracer._spans) == 30

    @pytest.mark.asyncio
    async def test_span_context_manager(self, tracer):
        """Test the async span context manager."""
        async with tracer.span("test-span", SpanKind.AGENT, attributes={"test": "value"}) as span:
            assert span.name == "test-span"
            assert span.kind == SpanKind.AGENT
            assert span.attributes["test"] == "value"
            assert tracer.current_span is span

        # After context exit, span should be ended
        assert span.end_ns is not None
        assert tracer.current_span is None

    @pytest.mark.asyncio
    async def test_span_context_manager_with_exception(self, tracer):
        """Test span context manager handles exceptions properly."""
        with pytest.raises(ValueError):
            async with tracer.span("test-span", SpanKind.AGENT) as span:
                raise ValueError("Test error")

        # Span should be marked as error
        assert span.status == SpanStatus.ERROR
        assert "ValueError: Test error" in span.error
        assert span.end_ns is not None

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, tracer):
        """Test the agent-specific context manager."""
        async with tracer.agent("test-agent", test_attr="value") as span:
            assert span.name == "test-agent"
            assert span.kind == SpanKind.AGENT
            assert span.attributes["test_attr"] == "value"

    @pytest.mark.asyncio
    async def test_tool_context_manager(self, tracer):
        """Test the tool-specific context manager."""
        async with tracer.tool("test-tool", tool_param="param") as span:
            assert span.name == "test-tool"
            assert span.kind == SpanKind.TOOL
            assert span.attributes["tool_param"] == "param"

    @pytest.mark.asyncio
    async def test_inference_context_manager(self, tracer):
        """Test the inference-specific context manager."""
        async with tracer.inference("completion", model="gpt-4") as span:
            assert span.name == "completion"
            assert span.kind == SpanKind.INFERENCE
            assert span.attributes["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_nested_spans(self, tracer):
        """Test nested span contexts."""
        async with tracer.agent("parent-agent") as parent:
            assert tracer.current_span is parent

            async with tracer.tool("child-tool") as child:
                assert tracer.current_span is child
                assert child.context.parent_span_id == parent.context.span_id
                assert child.context.trace_id == parent.context.trace_id

            # After child context, parent should be current again
            assert tracer.current_span is parent

        # After all contexts, no current span
        assert tracer.current_span is None