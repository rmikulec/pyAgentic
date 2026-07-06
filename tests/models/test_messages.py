import pydantic
import pytest

from pyagentic.models.llm import (
    AnyMessage,
    AgentCallMessage,
    AgentResultMessage,
    AssistantMessage,
    CompactionSummaryMessage,
    Message,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)


def test_semantic_messages_preset_roles():
    """Each semantic type defaults to the correct role"""
    assert SystemMessage(content="s").role == "system"
    assert UserMessage(content="u").role == "user"
    assert AssistantMessage(content="a").role == "assistant"
    assert ToolCallMessage(id="1", name="t", arguments="{}").role == "assistant"
    assert ToolResultMessage(tool_call_id="1", name="t", content="ok").role == "tool"


def test_semantic_messages_are_messages():
    """All semantic types remain Message subclasses"""
    for msg in [
        SystemMessage(),
        UserMessage(),
        AssistantMessage(),
        ToolCallMessage(id="1", name="t", arguments="{}"),
        ToolResultMessage(tool_call_id="1", name="t"),
        AgentCallMessage(id="1", name="a", arguments="{}"),
        AgentResultMessage(tool_call_id="1", name="a"),
        CompactionSummaryMessage(content="summary"),
    ]:
        assert isinstance(msg, Message)


def test_agent_messages_are_tool_messages():
    """Agent call/result subclass the tool types so generic tool policies match them"""
    call = AgentCallMessage(id="1", name="helper", arguments="{}")
    result = AgentResultMessage(tool_call_id="1", name="helper", content="done")

    assert isinstance(call, ToolCallMessage)
    assert isinstance(result, ToolResultMessage)
    assert call.kind == "agent_call"
    assert result.kind == "agent_result"


def test_compaction_summary_is_assistant_message():
    """CompactionSummaryMessage subclasses AssistantMessage"""
    msg = CompactionSummaryMessage(content="summary of 12 messages", compacted_count=12)
    assert isinstance(msg, AssistantMessage)
    assert msg.role == "assistant"
    assert msg.compacted_count == 12


def test_any_message_round_trip():
    """AnyMessage discriminated union round-trips each semantic type"""
    adapter = pydantic.TypeAdapter(AnyMessage)

    originals = [
        UserMessage(content="hi"),
        AssistantMessage(content="hello"),
        ToolCallMessage(id="c1", name="search", arguments='{"q": "x"}'),
        ToolResultMessage(tool_call_id="c1", name="search", content="found"),
        AgentCallMessage(id="c2", name="helper", arguments="{}"),
        AgentResultMessage(tool_call_id="c2", name="helper", content="done"),
        CompactionSummaryMessage(content="summary", compacted_count=3),
    ]

    for original in originals:
        restored = adapter.validate_python(original.model_dump())
        assert type(restored) is type(original)
        assert restored == original


def test_to_dict_excludes_kind_and_none():
    """to_dict drops the internal discriminator and None fields"""
    msg = ToolResultMessage(tool_call_id="c1", name="search", content="found")
    d = msg.to_dict()

    assert "kind" not in d
    assert d["tool_call_id"] == "c1"
    assert d["name"] == "search"
    assert d["content"] == "found"

    plain = Message(role="user", content="hi")
    assert plain.to_dict() == {"role": "user", "content": "hi"}


def test_tool_messages_require_identity_fields():
    """ToolCallMessage/ToolResultMessage require their identity fields"""
    with pytest.raises(pydantic.ValidationError):
        ToolCallMessage(name="t", arguments="{}")
    with pytest.raises(pydantic.ValidationError):
        ToolResultMessage(name="t")
