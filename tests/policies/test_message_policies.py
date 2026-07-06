"""Unit tests for the prewritten message-context policies."""

import pytest

from pyagentic.llm._mock import _MockProvider
from pyagentic.models.llm import (
    AgentResultMessage,
    AssistantMessage,
    CompactionSummaryMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
    UsageInfo,
    LLMResponse,
)
from pyagentic.policies import (
    CompactionPolicy,
    SlidingWindowPolicy,
    ToolEvictionPolicy,
    ToolOutputClipPolicy,
)
from pyagentic.policies._events import AppendEvent, CompileEvent


def _tool_pair(i: int, content: str = "result"):
    """Build a tool call/result message pair"""
    return [
        ToolCallMessage(id=f"call_{i}", name="tool", arguments="{}"),
        ToolResultMessage(tool_call_id=f"call_{i}", name="tool", content=content),
    ]


# ---- ToolOutputClipPolicy ----


def test_clip_policy_clips_oversized_tool_results():
    """Oversized tool result content is clipped with a marker suffix"""
    policy = ToolOutputClipPolicy(max_chars=10, suffix="…[clipped]")
    message = ToolResultMessage(tool_call_id="c1", name="t", content="x" * 100)

    result = policy.on_append(AppendEvent(name="messages", value=message), message)

    assert result.content == "x" * 10 + "…[clipped]"
    assert result.tool_call_id == "c1"


def test_clip_policy_ignores_small_results_and_other_messages():
    """Small tool results and non-tool messages pass through unchanged (None)"""
    policy = ToolOutputClipPolicy(max_chars=10)

    small = ToolResultMessage(tool_call_id="c1", name="t", content="tiny")
    assert policy.on_append(AppendEvent(value=small), small) is None

    user = UserMessage(content="x" * 100)
    assert policy.on_append(AppendEvent(value=user), user) is None


# ---- ToolEvictionPolicy ----


@pytest.mark.asyncio
async def test_eviction_keeps_last_n_and_stubs_rest():
    """All but the last N tool results are stubbed; pairing is preserved"""
    policy = ToolEvictionPolicy(keep_last_n=1, stub="[evicted]")
    items = [
        UserMessage(content="go"),
        *_tool_pair(1, "old result"),
        *_tool_pair(2, "recent result"),
    ]

    result = await policy.on_compile(CompileEvent(name="messages"), items)

    # Same shape: nothing deleted, call/result ids intact
    assert len(result) == len(items)
    assert result[2].content == "[evicted]"
    assert result[2].tool_call_id == "call_1"
    assert result[4].content == "recent result"


@pytest.mark.asyncio
async def test_eviction_is_idempotent():
    """A second pass over an already-evicted context returns None (no change)"""
    policy = ToolEvictionPolicy(keep_last_n=1, stub="[evicted]")
    items = [*_tool_pair(1, "old"), *_tool_pair(2, "new")]

    first = await policy.on_compile(CompileEvent(name="messages"), items)
    second = await policy.on_compile(CompileEvent(name="messages"), first)

    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_eviction_can_exempt_agent_results():
    """include_agent_results=False leaves linked-agent results intact"""
    policy = ToolEvictionPolicy(keep_last_n=0, stub="[evicted]", include_agent_results=False)
    items = [
        AgentResultMessage(tool_call_id="a1", name="helper", content="agent says"),
        *_tool_pair(1, "tool result"),
    ]

    result = await policy.on_compile(CompileEvent(name="messages"), items)

    assert result[0].content == "agent says"
    assert result[2].content == "[evicted]"


# ---- SlidingWindowPolicy ----


@pytest.mark.asyncio
async def test_sliding_window_drops_from_front():
    """Context is bounded to max_messages, dropping oldest first"""
    policy = SlidingWindowPolicy(max_messages=2)
    items = [
        UserMessage(content="one"),
        AssistantMessage(content="two"),
        UserMessage(content="three"),
        AssistantMessage(content="four"),
    ]

    result = await policy.on_compile(CompileEvent(name="messages"), items)

    assert [m.content for m in result] == ["three", "four"]


@pytest.mark.asyncio
async def test_sliding_window_never_orphans_tool_results():
    """The cut advances past tool results whose calls were dropped"""
    policy = SlidingWindowPolicy(max_messages=4)
    # 6 items, cut lands exactly on call_1's RESULT -> must advance past it
    items = [
        UserMessage(content="go"),
        *_tool_pair(1),
        *_tool_pair(2),
        AssistantMessage(content="done"),
    ]
    result = await policy.on_compile(CompileEvent(name="messages"), items)

    # No result without its call at the front
    assert not isinstance(result[0], ToolResultMessage)
    for message in result:
        if isinstance(message, ToolResultMessage):
            assert any(
                isinstance(m, ToolCallMessage) and m.id == message.tool_call_id for m in result
            )


@pytest.mark.asyncio
async def test_sliding_window_noop_under_budget():
    """Under-budget context returns None (no change)"""
    policy = SlidingWindowPolicy(max_messages=10)
    items = [UserMessage(content="hi")]
    assert await policy.on_compile(CompileEvent(name="messages"), items) is None


# ---- CompactionPolicy ----


def _compile_event(provider, input_tokens):
    return CompileEvent(
        name="messages",
        provider=provider,
        last_usage=UsageInfo(
            input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens
        ),
    )


@pytest.mark.asyncio
async def test_compaction_triggers_over_threshold():
    """Above the token threshold, older history collapses into a summary message"""
    provider = _MockProvider(model="test-model", api_key="k")
    provider.responses.append(LLMResponse(text="the summary", tool_calls=[]))
    policy = CompactionPolicy(max_input_tokens=100, keep_recent=2)

    items = [
        UserMessage(content="old question"),
        AssistantMessage(content="old answer"),
        *_tool_pair(1),
        UserMessage(content="recent question"),
        AssistantMessage(content="recent answer"),
    ]

    result = await policy.on_compile(_compile_event(provider, input_tokens=500), items)

    assert isinstance(result[0], CompactionSummaryMessage)
    assert "the summary" in result[0].content
    assert result[0].compacted_count == 4
    assert [m.content for m in result[1:]] == ["recent question", "recent answer"]


@pytest.mark.asyncio
async def test_compaction_noop_under_threshold():
    """Below the token threshold, nothing happens"""
    provider = _MockProvider(model="test-model", api_key="k")
    policy = CompactionPolicy(max_input_tokens=1000, keep_recent=2)

    items = [
        UserMessage(content="q1"),
        AssistantMessage(content="a1"),
        UserMessage(content="q2"),
    ]

    assert await policy.on_compile(_compile_event(provider, input_tokens=50), items) is None


@pytest.mark.asyncio
async def test_compaction_respects_pair_boundary():
    """The compaction cut never splits a tool call from its result"""
    provider = _MockProvider(model="test-model", api_key="k")
    provider.responses.append(LLMResponse(text="summary", tool_calls=[]))
    policy = CompactionPolicy(max_input_tokens=100, keep_recent=2)

    items = [
        UserMessage(content="go"),
        *_tool_pair(1),  # keep_recent=2 naively cuts between call_1 and its result
        AssistantMessage(content="done"),
    ]

    result = await policy.on_compile(_compile_event(provider, input_tokens=500), items)

    for message in result:
        if isinstance(message, ToolResultMessage):
            assert any(
                isinstance(m, ToolCallMessage) and m.id == message.tool_call_id for m in result
            )
