import pytest
import asyncio

from pyagentic._base._agent import Agent
from pyagentic._base._context import ContextItem, _AgentContext
from pyagentic._base._tool import tool, _ToolDefinition
from pyagentic._base._exceptions import SystemMessageNotDeclared


def test_agent_class_declaration_raises_no_system_message():
    with pytest.raises(SystemMessageNotDeclared) as e:

        class TestAgent(Agent):
            pass

    assert "SystemMessageNotDeclared" in str(e)


def test_agent_class_declaration_no_context_or_tools():
    class TestAgent(Agent):
        __system_message__ = "This is a test"

    assert not TestAgent.__context_attrs__
    assert not TestAgent.__tool_defs__


def test_agent_class_declaration_context():
    class TestAgent(Agent):
        __system_message__ = "This is a test"

        item: str = ContextItem(default="test")

    assert "item" in TestAgent.__context_attrs__
    assert TestAgent.__context_attrs__["item"][0] == str
    assert isinstance(TestAgent.__context_attrs__["item"][1], ContextItem)


def test_agent_class_declaration_tool():
    class TestAgent(Agent):
        __system_message__ = "This is a test"

        @tool("testing in agent")
        def test(self) -> str:
            pass

    assert "test" in TestAgent.__tool_defs__
    assert isinstance(TestAgent.__tool_defs__["test"], _ToolDefinition)
    assert TestAgent.__tool_defs__["test"].description == "testing in agent"


def test_agent_creation(mock_agent):
    assert issubclass(mock_agent.__class__, Agent)


def test_agent_context_loaded(mock_agent):
    assert issubclass(mock_agent.context.__class__, _AgentContext)


def test_agent_computed_context_attributes(mock_agent):
    expected = {
        "instructions": "This is a mock agent",
        "input_template": None,
        "_messages": [],
        "int_default": 4,
        "str_default": "test",
        "int_factory": mock_agent.context.int_factory,
        "str_factory": mock_agent.context.str_factory,
        "default_override": "overriden",
        "random_computed": mock_agent.context.random_computed,
    }

    for name, value in expected.items():
        attr_value = getattr(mock_agent.context, name, None)
        if name == "random_computed":
            assert attr_value > 1000, (
                f"Value not set properly for {name}\n" f"Expected > 1000" f"Recieved: {attr_value}"
            )
        else:
            assert attr_value == value, (
                f"Value not set properly for {name}\n"
                f"Expected: {value}\n"
                f"Recieved: {attr_value}"
            )


def test_agent_static_tool_ref(mock_agent):
    openai_tools = asyncio.run(mock_agent._get_tool_defs())
    for cur_tool in openai_tools:
        if cur_tool.name == "test_ref":
            tool_def = cur_tool.to_openai_spec(mock_agent.context)
            assert tool_def["parameters"]["properties"]["letter"]["enum"] == ["a", "b", "c"]
            break


def test_agent_dynamic_tool_ref(mock_agent):
    openai_tools = asyncio.run(mock_agent._get_tool_defs())
    for cur_tool in openai_tools:
        if cur_tool.name == "test_computed_ref":
            tool_def = cur_tool.to_openai_spec(mock_agent.context)
            assert tool_def["parameters"]["properties"]["letter"]["enum"] == ["a", "b", "c"]
            break
