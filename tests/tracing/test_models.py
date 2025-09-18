import pytest
from dataclasses import FrozenInstanceError

from pyagentic.models.tracing import SpanKind, SpanStatus, SpanContext, Span


class TestSpanKind:
    """Test suite for SpanKind enum."""

    def test_span_kind_values(self):
        """Test that SpanKind has correct values."""
        assert SpanKind.AGENT == "agent"
        assert SpanKind.TOOL == "tool"
        assert SpanKind.INFERENCE == "inference"
        assert SpanKind.STEP == "step"

    def test_span_kind_is_string(self):
        """Test that SpanKind values are strings."""
        for kind in SpanKind:
            assert isinstance(kind.value, str)

    def test_span_kind_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {"AGENT", "TOOL", "INFERENCE", "STEP"}
        actual_members = {member.name for member in SpanKind}
        assert actual_members == expected_members


class TestSpanStatus:
    """Test suite for SpanStatus enum."""

    def test_span_status_values(self):
        """Test that SpanStatus has correct values."""
        assert SpanStatus.OK == "ok"
        assert SpanStatus.ERROR == "error"

    def test_span_status_is_string(self):
        """Test that SpanStatus values are strings."""
        for status in SpanStatus:
            assert isinstance(status.value, str)

    def test_span_status_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {"OK", "ERROR"}
        actual_members = {member.name for member in SpanStatus}
        assert actual_members == expected_members


class TestSpanContext:
    """Test suite for SpanContext dataclass."""

    def test_span_context_creation(self):
        """Test creating a SpanContext with required fields."""
        context = SpanContext(trace_id="trace123", span_id="span456")

        assert context.trace_id == "trace123"
        assert context.span_id == "span456"
        assert context.parent_span_id is None

    def test_span_context_with_parent(self):
        """Test creating a SpanContext with parent."""
        context = SpanContext(
            trace_id="trace123",
            span_id="span456",
            parent_span_id="parent789"
        )

        assert context.trace_id == "trace123"
        assert context.span_id == "span456"
        assert context.parent_span_id == "parent789"

    def test_span_context_is_frozen(self):
        """Test that SpanContext is immutable (frozen)."""
        context = SpanContext(trace_id="trace123", span_id="span456")

        with pytest.raises(FrozenInstanceError):
            context.trace_id = "new_trace"

        with pytest.raises(FrozenInstanceError):
            context.span_id = "new_span"

        with pytest.raises(FrozenInstanceError):
            context.parent_span_id = "new_parent"

    def test_span_context_equality(self):
        """Test SpanContext equality comparison."""
        context1 = SpanContext(trace_id="trace123", span_id="span456")
        context2 = SpanContext(trace_id="trace123", span_id="span456")
        context3 = SpanContext(trace_id="trace123", span_id="span789")

        assert context1 == context2
        assert context1 != context3

    def test_span_context_hash(self):
        """Test that SpanContext is hashable."""
        context = SpanContext(trace_id="trace123", span_id="span456")

        # Should be able to use as dict key
        test_dict = {context: "value"}
        assert test_dict[context] == "value"

        # Should be able to add to set
        test_set = {context}
        assert context in test_set


class TestSpan:
    """Test suite for Span dataclass."""

    def test_span_creation_minimal(self):
        """Test creating a Span with minimal required fields."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000
        )

        assert span.name == "test-span"
        assert span.kind == SpanKind.AGENT
        assert span.context == context
        assert span.start_ns == 1000000000
        assert span.end_ns is None
        assert span.status == SpanStatus.OK
        assert span.attributes == {}
        assert span.error is None

    def test_span_creation_full(self):
        """Test creating a Span with all fields."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        attributes = {"model": "gpt-4", "temperature": 0.7}

        span = Span(
            name="test-span",
            kind=SpanKind.INFERENCE,
            context=context,
            start_ns=1000000000,
            end_ns=2000000000,
            status=SpanStatus.ERROR,
            attributes=attributes,
            error="Test error"
        )

        assert span.name == "test-span"
        assert span.kind == SpanKind.INFERENCE
        assert span.context == context
        assert span.start_ns == 1000000000
        assert span.end_ns == 2000000000
        assert span.status == SpanStatus.ERROR
        assert span.attributes == attributes
        assert span.error == "Test error"

    def test_span_attributes_default_factory(self):
        """Test that attributes default to empty dict and are mutable."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000
        )

        # Should start with empty dict
        assert span.attributes == {}

        # Should be able to modify
        span.attributes["key"] = "value"
        assert span.attributes["key"] == "value"

    def test_span_mutability(self):
        """Test that Span fields can be modified after creation."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000
        )

        # Should be able to modify mutable fields
        span.end_ns = 2000000000
        span.status = SpanStatus.ERROR
        span.error = "Test error"
        span.attributes["new_key"] = "new_value"

        assert span.end_ns == 2000000000
        assert span.status == SpanStatus.ERROR
        assert span.error == "Test error"
        assert span.attributes["new_key"] == "new_value"

    def test_span_context_immutability(self):
        """Test that Span's context remains immutable."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000
        )

        # Context should still be frozen
        with pytest.raises(FrozenInstanceError):
            span.context.trace_id = "new_trace"

    def test_span_duration_calculation(self):
        """Test calculating span duration."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000,
            end_ns=1500000000
        )

        # Duration in nanoseconds
        duration_ns = span.end_ns - span.start_ns
        assert duration_ns == 500000000

        # Duration in milliseconds
        duration_ms = duration_ns / 1e6
        assert duration_ms == 500.0

    def test_span_different_kinds(self):
        """Test creating spans with different kinds."""
        context = SpanContext(trace_id="trace123", span_id="span456")

        for kind in SpanKind:
            span = Span(
                name=f"test-{kind.value}",
                kind=kind,
                context=context,
                start_ns=1000000000
            )
            assert span.kind == kind
            assert span.name == f"test-{kind.value}"

    def test_span_different_statuses(self):
        """Test setting different span statuses."""
        context = SpanContext(trace_id="trace123", span_id="span456")
        span = Span(
            name="test-span",
            kind=SpanKind.AGENT,
            context=context,
            start_ns=1000000000
        )

        # Test OK status (default)
        assert span.status == SpanStatus.OK

        # Test ERROR status
        span.status = SpanStatus.ERROR
        assert span.status == SpanStatus.ERROR