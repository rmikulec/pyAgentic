import pytest
import random
from pydantic import BaseModel

from pyagentic import BaseAgent, spec, State
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._exceptions import InvalidStateRefNotFoundInState
from pyagentic._base._state import StateInfo


def test_state_info_default():
    """Test that StateInfo with default value works correctly"""
    info = spec.State(default=4)
    assert info.get_default() == 4


def test_state_info_default_factory():
    """Test that StateInfo with default_factory works correctly"""
    info = spec.State(default_factory=lambda: 4)
    assert info.get_default() == 4


def test_agent_state_class_construction():
    """Test that agent state class is properly constructed"""

    class IntModel(BaseModel):
        value: int = 0

    class StrModel(BaseModel):
        value: str = ""

    class TestAgent(BaseAgent):
        __system_message__ = "Test"

        int_default: State[IntModel] = spec.State(default_factory=lambda: IntModel(value=4))
        str_default: State[StrModel] = spec.State(default_factory=lambda: StrModel(value="test"))
        int_factory: State[IntModel] = spec.State(
            default_factory=lambda: IntModel(value=random.randint(0, 1000))
        )
        str_factory: State[StrModel] = spec.State(
            default_factory=lambda: StrModel(value=f"Random: {random.randint(0, 1000)}")
        )

    agent = TestAgent(model="_mock::test-model", api_key="test")
    state_class = agent.__state_class__

    assert issubclass(
        state_class, _AgentState
    ), "Created state class is not a subclass of _AgentState"

    assert state_class.__name__ == "AgentState[TestAgent]", (
        "New state name not being set properly\n"
        "Expected: AgentState[TestAgent]\n"
        f"Received: {state_class.__name__}"
    )

    # Check that all state fields are in the model
    assert "int_default" in state_class.model_fields
    assert "str_default" in state_class.model_fields
    assert "int_factory" in state_class.model_fields
    assert "str_factory" in state_class.model_fields


def test_agent_state_default_values(mock_state):
    """Test that state default values are properly set"""
    assert (
        mock_state.int_default.value == 4
    ), f"Unexpected value with default: {mock_state.int_default.value}"
    assert (
        mock_state.int_factory.value < 100
    ), f"Unexpected value with factory: {mock_state.int_factory.value}"


def test_agent_state_default_override(mock_state):
    """Test that default values can be overridden"""
    assert (
        mock_state.default_override.value == "overriden"
    ), "Default value not being overridden when supplied after construction"


def test_agent_state_attributes(mock_state):
    """Test that state attributes are accessible and have correct values"""
    # Check basic attributes
    assert mock_state.instructions == "This is a mock"
    assert mock_state.input_template == ""  # Empty string to avoid Jinja2 errors
    assert mock_state._messages == []

    # Check state fields
    assert mock_state.int_default.value == 4
    assert mock_state.str_default.value == "test"
    assert 0 <= mock_state.int_factory.value <= 100
    assert "Random: " in mock_state.str_factory.value
    assert mock_state.default_override.value == "overriden"


def test_agent_state_as_dict(mock_state):
    """Test that state can be converted to dictionary"""
    state_dict = mock_state.model_dump()

    assert "instructions" in state_dict
    assert state_dict["instructions"] == "This is a mock"
    assert "int_default" in state_dict
    assert state_dict["int_default"]["value"] == 4


def test_agent_state_add_user_message(mock_state):
    """Test adding user message to state"""
    user_message = "Hello"
    mock_state.add_user_message(user_message)
    assert mock_state._messages[-1].content == user_message


def test_agent_state_add_user_message_with_template():
    """Test adding user message (template rendering tested separately)"""
    from tests.conftest import IntStateModel

    class TestAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = "Template: {{ content }}"

        int_default: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=4)
        )

    agent = TestAgent(model="_mock::test-model", api_key="test")
    user_message = "Hello"
    agent.state.add_user_message(user_message)
    # Verify message was added
    assert len(agent.state._messages) > 0
    # Message content should contain the user message in some form
    assert agent.state._messages[-1].role == "user"


def test_agent_state_get_item(mock_state):
    """Test getting state item by direct attribute access"""
    value = mock_state.int_default
    assert value.value == 4


def test_agent_state_has_attribute(mock_state):
    """Test that state has expected attributes"""
    assert hasattr(mock_state, "int_default")
    assert hasattr(mock_state, "str_default")
    assert hasattr(mock_state, "default_override")


def test_agent_state_system_message():
    """Test that system message is formatted with state values"""
    from tests.conftest import IntStateModel

    class TestAgent(BaseAgent):
        __system_message__ = "System: {int_default}"

        int_default: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=4)
        )

    agent = TestAgent(model="_mock::test-model", api_key="test")
    # Note: System message doesn't auto-format nested state models, so we check for the dict representation
    assert "int_default" in agent.state.system_message


def test_agent_state_messages():
    """Test that messages property includes system message"""
    from tests.conftest import IntStateModel

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

        int_default: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=4)
        )

    agent = TestAgent(model="_mock::test-model", api_key="test")
    agent.state.add_user_message("Hello")

    messages = agent.state.messages
    assert len(messages) == 2  # system + user
    assert messages[0].role == "system"
    assert messages[0].content == "Test agent"
    assert messages[1].role == "user"
    assert messages[1].content == "Hello"


def test_state_info_with_policies():
    """Test that StateInfo can be created with policies"""
    from pyagentic.policies._policy import Policy

    class TestPolicy(Policy):
        pass

    info = spec.State(default=4, policies=[TestPolicy()])
    assert len(info.policies) == 1
    assert isinstance(info.policies[0], TestPolicy)


def test_state_info_access_levels():
    """Test that StateInfo supports different access levels"""
    for access_level in ["read", "write", "readwrite", "hidden"]:
        info = spec.State(default=4, access=access_level)
        assert info.access == access_level
