import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from pyagentic.tracing._tracer import AgentTracer, traced, _current_span
from pyagentic.models.tracing import SpanKind, SpanStatus, Span, SpanContext


class MockTracer(AgentTracer):
    """Mock implementation of AgentTracer for testing."""

    def __init__(self):
        self.spans = {}
        self.events = []
        self.attributes = {}
        self.exceptions = []
        self.ended_spans = []

    def start_span(self, name, kind, parent=None, attributes=None):
        span_id = f"span-{len(self.spans)}"
        trace_id = parent.context.trace_id if parent else f"trace-{len(self.spans)}"
        parent_span_id = parent.context.span_id if parent else None

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )

        span = Span(
            name=name,
            kind=kind,
            context=context,
            start_ns=1000000000,  # Fixed for testing
            attributes=dict(attributes or {})
        )

        self.spans[span_id] = span
        return span

    def end_span(self, span):
        self.ended_spans.append(span.context.span_id)
        span.end_ns = 2000000000  # Fixed for testing

    def _add_event(self, span, name, attributes=None):
        self.events.append({
            'span_id': span.context.span_id,
            'name': name,
            'attributes': attributes or {}
        })

    def _set_attributes(self, span, attributes):
        if span.context.span_id not in self.attributes:
            self.attributes[span.context.span_id] = {}
        self.attributes[span.context.span_id].update(attributes)
        span.attributes.update(attributes)

    def _record_exception(self, span, exc):
        self.exceptions.append({
            'span_id': span.context.span_id,
            'exception': exc
        })
        span.status = SpanStatus.ERROR
        span.error = str(exc)


class TestAgentTracer:
    """Test suite for AgentTracer abstract base class."""

    @pytest.fixture
    def tracer(self):
        return MockTracer()

    def test_current_span_initially_none(self, tracer):
        """Test that current_span is None initially."""
        assert tracer.current_span is None

    def test_set_attributes_with_current_span(self, tracer):
        """Test setting attributes using current span."""
        span = tracer.start_span("test", SpanKind.AGENT)

        # Mock current span
        token = _current_span.set(span)
        try:
            tracer.set_attributes(key="value", number=42)

            # Check that attributes were set on the span
            assert span.attributes["key"] == "value"
            assert span.attributes["number"] == 42
            assert tracer.attributes[span.context.span_id]["key"] == "value"
            assert tracer.attributes[span.context.span_id]["number"] == 42
        finally:
            _current_span.reset(token)

    def test_add_event_with_current_span(self, tracer):
        """Test adding event using current span."""
        span = tracer.start_span("test", SpanKind.AGENT)

        token = _current_span.set(span)
        try:
            tracer.add_event("test-event", data="test")

            # Check that event was added
            assert len(tracer.events) == 1
            event = tracer.events[0]
            assert event["span_id"] == span.context.span_id
            assert event["name"] == "test-event"
            assert event["attributes"]["data"] == "test"
        finally:
            _current_span.reset(token)

    def test_record_exception_with_current_span(self, tracer):
        """Test recording exception using current span."""
        span = tracer.start_span("test", SpanKind.AGENT)

        token = _current_span.set(span)
        try:
            tracer.record_exception("Test error")

            # Check that exception was recorded
            assert len(tracer.exceptions) == 1
            exc = tracer.exceptions[0]
            assert exc["span_id"] == span.context.span_id
            assert exc["exception"] == "Test error"
            assert span.status == SpanStatus.ERROR
            assert span.error == "Test error"
        finally:
            _current_span.reset(token)

    def test_record_tokens(self, tracer):
        """Test recording token usage."""
        span = tracer.start_span("test", SpanKind.INFERENCE)

        tracer.record_tokens(
            span,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            attributes={"custom": "value"}
        )

        # Check that token event was added
        assert len(tracer.events) == 1
        event = tracer.events[0]
        assert event["span_id"] == span.context.span_id
        assert event["name"] == "llm.tokens"
        assert event["attributes"]["model"] == "gpt-4"
        assert event["attributes"]["prompt_tokens"] == 100
        assert event["attributes"]["completion_tokens"] == 50
        assert event["attributes"]["total_tokens"] == 150
        assert event["attributes"]["custom"] == "value"

    @pytest.mark.asyncio
    async def test_span_context_manager(self, tracer):
        """Test the async span context manager."""
        async with tracer.span("test-span", SpanKind.AGENT, attributes={"test": "value"}) as span:
            assert span.name == "test-span"
            assert span.kind == SpanKind.AGENT
            assert span.attributes["test"] == "value"
            assert tracer.current_span is span

        # Check span was ended
        assert span.context.span_id in tracer.ended_spans
        assert tracer.current_span is None

    @pytest.mark.asyncio
    async def test_span_context_manager_with_parent(self, tracer):
        """Test span context manager with explicit parent."""
        parent = tracer.start_span("parent", SpanKind.AGENT)

        async with tracer.span("child", SpanKind.TOOL, parent=parent) as child:
            assert child.context.parent_span_id == parent.context.span_id
            assert child.context.trace_id == parent.context.trace_id

    @pytest.mark.asyncio
    async def test_span_context_manager_with_exception(self, tracer):
        """Test span context manager handles exceptions."""
        with pytest.raises(ValueError):
            async with tracer.span("test", SpanKind.AGENT) as span:
                raise ValueError("Test error")

        # Check exception was recorded
        assert span.status == SpanStatus.ERROR
        assert len(tracer.exceptions) == 1
        assert tracer.exceptions[0]["exception"].args[0] == "Test error"
        assert span.context.span_id in tracer.ended_spans

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, tracer):
        """Test agent-specific context manager."""
        async with tracer.agent("test-agent", model="gpt-4") as span:
            assert span.name == "test-agent"
            assert span.kind == SpanKind.AGENT
            assert span.attributes["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_tool_context_manager(self, tracer):
        """Test tool-specific context manager."""
        async with tracer.tool("test-tool", tool_id="123") as span:
            assert span.name == "test-tool"
            assert span.kind == SpanKind.TOOL
            assert span.attributes["tool_id"] == "123"

    @pytest.mark.asyncio
    async def test_inference_context_manager(self, tracer):
        """Test inference-specific context manager."""
        async with tracer.inference("completion", model="gpt-4") as span:
            assert span.name == "completion"
            assert span.kind == SpanKind.INFERENCE
            assert span.attributes["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_nested_spans_with_context_vars(self, tracer):
        """Test nested spans maintain proper context variable state."""
        async with tracer.agent("parent") as parent:
            assert tracer.current_span is parent

            async with tracer.tool("child") as child:
                assert tracer.current_span is child
                assert child.context.parent_span_id == parent.context.span_id

                async with tracer.inference("grandchild") as grandchild:
                    assert tracer.current_span is grandchild
                    assert grandchild.context.parent_span_id == child.context.span_id
                    assert grandchild.context.trace_id == parent.context.trace_id

                # Back to child context
                assert tracer.current_span is child

            # Back to parent context
            assert tracer.current_span is parent

        # No current span after all contexts
        assert tracer.current_span is None


class TestTracedDecorator:
    """Test suite for the @traced decorator."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with tracer."""
        agent = Mock()
        agent.tracer = MockTracer()
        agent.__class__.__name__ = "TestAgent"
        return agent

    @pytest.mark.asyncio
    async def test_traced_decorator_basic(self, mock_agent):
        """Test basic functionality of @traced decorator."""
        @traced(SpanKind.TOOL, "custom-name")
        async def test_method(self, arg1, arg2="default"):
            return f"result: {arg1}-{arg2}"

        result = await test_method(mock_agent, "test", arg2="value")

        assert result == "result: test-value"

        # Check span was created
        tracer = mock_agent.tracer
        assert len(tracer.spans) == 1
        span = list(tracer.spans.values())[0]
        assert span.name == "custom-name"
        assert span.kind == SpanKind.TOOL

        # Check input/output attributes were set
        assert "input" in tracer.attributes[span.context.span_id]
        assert "output" in tracer.attributes[span.context.span_id]
        assert tracer.attributes[span.context.span_id]["input"]["arg2"] == "value"
        assert tracer.attributes[span.context.span_id]["output"] == "result: test-value"

    @pytest.mark.asyncio
    async def test_traced_decorator_default_name(self, mock_agent):
        """Test @traced decorator with default span name."""
        @traced(SpanKind.AGENT)
        async def my_method(self):
            return "success"

        await my_method(mock_agent)

        tracer = mock_agent.tracer
        span = list(tracer.spans.values())[0]
        assert span.name == "TestAgent.my_method"

    @pytest.mark.asyncio
    async def test_traced_decorator_with_attributes(self, mock_agent):
        """Test @traced decorator with custom attributes."""
        attrs = {"custom": "value", "number": 42}

        @traced(SpanKind.INFERENCE, attrs=attrs)
        async def inference_method(self, prompt):
            return f"response to: {prompt}"

        await inference_method(mock_agent, "test prompt")

        tracer = mock_agent.tracer
        span = list(tracer.spans.values())[0]
        assert span.attributes["custom"] == "value"
        assert span.attributes["number"] == 42

    @pytest.mark.asyncio
    async def test_traced_decorator_with_exception(self, mock_agent):
        """Test @traced decorator handles exceptions."""
        @traced(SpanKind.TOOL)
        async def failing_method(self):
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            await failing_method(mock_agent)

        tracer = mock_agent.tracer
        # The decorator records exception both as string and as exception object
        assert len(tracer.exceptions) >= 1
        exc = tracer.exceptions[0]
        assert "Something went wrong" in str(exc["exception"])

    @pytest.mark.asyncio
    async def test_traced_decorator_nested_calls(self, mock_agent):
        """Test nested calls with @traced decorator."""
        @traced(SpanKind.AGENT)
        async def outer_method(self):
            return await inner_method(self)

        @traced(SpanKind.TOOL)
        async def inner_method(self):
            return "inner result"

        result = await outer_method(mock_agent)

        assert result == "inner result"

        # Should have created 2 spans
        tracer = mock_agent.tracer
        assert len(tracer.spans) == 2

        # Check span hierarchy
        spans = list(tracer.spans.values())
        outer_span = next(s for s in spans if s.name == "TestAgent.outer_method")
        inner_span = next(s for s in spans if s.name == "TestAgent.inner_method")

        # Inner span should be child of outer span
        assert inner_span.context.parent_span_id == outer_span.context.span_id