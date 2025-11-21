import pytest
import asyncio
from pydantic import BaseModel

from pyagentic import BaseAgent, tool, spec, State, ref
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._exceptions import SystemMessageNotDeclared


def test_agent_class_declaration_raises_no_system_message():
    """Test that BaseAgent subclass without __system_message__ raises exception"""
    with pytest.raises(SystemMessageNotDeclared) as e:

        class TestAgent(BaseAgent):
            pass

    assert "SystemMessageNotDeclared" in str(e)


def test_agent_class_declaration_no_state_or_tools():
    """Test that a minimal BaseAgent subclass can be created"""

    class TestAgent(BaseAgent):
        __system_message__ = "This is a test"

    assert not TestAgent.__state_defs__
    assert not TestAgent.__tool_defs__


def test_agent_class_declaration_state():
    """Test that State fields are properly registered in __state_defs__"""

    class TestStateModel(BaseModel):
        value: str = "test"

    class TestAgent(BaseAgent):
        __system_message__ = "This is a test"

        item: State[TestStateModel] = spec.State(
            default_factory=lambda: TestStateModel(value="test")
        )

    assert "item" in TestAgent.__state_defs__
    assert TestAgent.__state_defs__["item"].model == TestStateModel


def test_agent_class_declaration_tool():
    """Test that @tool decorated methods are properly registered in __tool_defs__"""

    class TestAgent(BaseAgent):
        __system_message__ = "This is a test"

        @tool("testing in agent")
        def test(self) -> str:
            return "test"

    assert "test" in TestAgent.__tool_defs__
    assert isinstance(TestAgent.__tool_defs__["test"], _ToolDefinition)
    assert TestAgent.__tool_defs__["test"].description == "testing in agent"


def test_agent_creation(mock_agent):
    """Test that agent instance is properly created"""
    assert issubclass(mock_agent.__class__, BaseAgent)


def test_agent_state_loaded(mock_agent):
    """Test that agent.state is properly initialized"""
    assert issubclass(mock_agent.state.__class__, _AgentState)


def test_agent_state_attributes(mock_agent):
    """Test that state attributes are accessible and have correct values"""
    assert mock_agent.state.instructions == "This is a mock agent"
    assert mock_agent.state.input_template == ""
    assert mock_agent.state._messages == []
    assert mock_agent.state.int_default.value == 4
    assert mock_agent.state.str_default.value == "test"
    assert 0 <= mock_agent.state.int_factory.value <= 100
    assert "Random: " in mock_agent.state.str_factory.value
    assert mock_agent.state.default_override.value == "overriden"


def test_agent_state_override(mock_agent):
    """Test that state can be overridden after initialization"""
    from tests.conftest import StrStateModel

    mock_agent.state.default_override = StrStateModel(value="new value")
    assert mock_agent.state.default_override.value == "new value"


def test_agent_static_tool_ref(mock_agent):
    """Test that tool parameters with ref() resolve correctly"""
    openai_tools = asyncio.run(mock_agent._get_tool_defs())
    for cur_tool in openai_tools:
        if cur_tool.name == "test_ref":
            resolved_tool = cur_tool.resolve(mock_agent.agent_reference)
            tool_def = resolved_tool.to_openai_spec()
            assert tool_def["parameters"]["properties"]["letter"]["enum"] == [
                "a",
                "b",
                "c",
            ]
            break


def test_agent_dynamic_tool_ref(mock_agent):
    """Test that tool parameters with static values work correctly"""
    openai_tools = asyncio.run(mock_agent._get_tool_defs())
    for cur_tool in openai_tools:
        if cur_tool.name == "test_computed_ref":
            resolved_tool = cur_tool.resolve(mock_agent.agent_reference)
            tool_def = resolved_tool.to_openai_spec()
            assert tool_def["parameters"]["properties"]["letter"]["enum"] == [
                "a",
                "b",
                "c",
            ]
            break


def test_agent_has_state_class(mock_agent):
    """Test that agent has a generated state class"""
    assert mock_agent.__state_class__ is not None
    assert issubclass(mock_agent.__state_class__, _AgentState)


def test_agent_provider_setup(mock_agent):
    """Test that provider is properly set up"""
    assert mock_agent.provider is not None
    assert mock_agent.provider.__class__.__name__ in ["OpenAIProvider", "AnthropicProvider", "_MockProvider"]
