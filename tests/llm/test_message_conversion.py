import pytest

from pyagentic.llm import OpenAIProvider, AnthropicProvider, GeminiProvider
from pyagentic.models.llm import (
    AgentCallMessage,
    AgentResultMessage,
    AssistantMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)


@pytest.fixture
def history():
    """A canonical history: user turn, two parallel tool calls + results, final answer"""
    return [
        UserMessage(content="What's the weather in NYC and LA?"),
        ToolCallMessage(id="call_1", name="get_weather", arguments='{"city": "NYC"}'),
        ToolCallMessage(id="call_2", name="get_weather", arguments='{"city": "LA"}'),
        ToolResultMessage(tool_call_id="call_1", name="get_weather", content="72F"),
        ToolResultMessage(tool_call_id="call_2", name="get_weather", content="85F"),
        AssistantMessage(content="NYC is 72F, LA is 85F"),
    ]


def test_openai_conversion(history):
    """OpenAI converter emits Responses API items"""
    provider = OpenAIProvider(model="gpt-4o", api_key="fake")
    items = provider._convert_messages(history)

    assert items[0] == {"role": "user", "content": "What's the weather in NYC and LA?"}
    assert items[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"city": "NYC"}',
    }
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "72F",
    }
    assert items[5] == {"role": "assistant", "content": "NYC is 72F, LA is 85F"}


def test_anthropic_conversion_merges_parallel_blocks(history):
    """Anthropic converter merges consecutive tool_use/tool_result into single messages"""
    provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="fake")
    messages = provider._convert_messages(history)

    # user, merged assistant tool_use, merged user tool_result, assistant text
    assert len(messages) == 4

    assert messages[0] == {"role": "user", "content": "What's the weather in NYC and LA?"}

    tool_use = messages[1]
    assert tool_use["role"] == "assistant"
    assert [b["type"] for b in tool_use["content"]] == ["tool_use", "tool_use"]
    assert tool_use["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "NYC"},
    }

    tool_result = messages[2]
    assert tool_result["role"] == "user"
    assert [b["type"] for b in tool_result["content"]] == ["tool_result", "tool_result"]
    assert tool_result["content"][1] == {
        "type": "tool_result",
        "tool_use_id": "call_2",
        "content": "85F",
    }

    assert messages[3] == {"role": "assistant", "content": "NYC is 72F, LA is 85F"}


def test_gemini_conversion_uses_tool_name(history):
    """Gemini converter keys function_response by tool NAME, not call id (regression)"""
    provider = GeminiProvider(model="gemini-1.5-pro", api_key="fake")
    system_instruction, messages = provider._convert_messages_to_gemini_format(history)

    assert system_instruction is None
    assert messages[0] == {
        "role": "user",
        "parts": [{"text": "What's the weather in NYC and LA?"}],
    }
    assert messages[1] == {
        "role": "model",
        "parts": [{"function_call": {"name": "get_weather", "args": {"city": "NYC"}}}],
    }
    assert messages[3] == {
        "role": "function",
        "parts": [{"function_response": {"name": "get_weather", "response": {"result": "72F"}}}],
    }
    # assistant maps to model
    assert messages[5] == {"role": "model", "parts": [{"text": "NYC is 72F, LA is 85F"}]}


def test_agent_messages_convert_like_tool_messages():
    """AgentCallMessage/AgentResultMessage convert identically to their tool counterparts"""
    provider = OpenAIProvider(model="gpt-4o", api_key="fake")
    items = provider._convert_messages(
        [
            AgentCallMessage(id="call_9", name="helper_agent", arguments="{}"),
            AgentResultMessage(tool_call_id="call_9", name="helper_agent", content="done"),
        ]
    )

    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call_9"
    assert items[1]["type"] == "function_call_output"
    assert items[1]["output"] == "done"
