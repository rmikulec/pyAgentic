import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, tool, spec, State, ref
from pyagentic._base._validation import _AgentConstructionValidator, AgentValidationError


def test_validator_basic():
    """Test that validator can be instantiated with a simple agent"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"
        __input_template__ = ""

    # Validator should be creatable
    validator = _AgentConstructionValidator(TestAgent)
    assert validator is not None
    assert validator.AgentClass == TestAgent


def test_validator_with_state():
    """Test validator can be created with agent that has state fields"""

    class TestState(BaseModel):
        value: int

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent with state"
        __input_template__ = ""

        test_state: State[TestState] = spec.State(default_factory=lambda: TestState(value=42))

    validator = _AgentConstructionValidator(TestAgent)
    assert validator is not None


def test_validator_with_tools():
    """Test validator can be created with agent that has tools"""

    class TestState(BaseModel):
        value: int

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"
        __input_template__ = ""

        test_state: State[TestState] = spec.State(default_factory=lambda: TestState(value=42))

        @tool("Test tool with ref")
        def test_tool(self, value: int = spec.Param(default=ref.self.test_state.value)) -> str:
            return f"Value: {value}"

    validator = _AgentConstructionValidator(TestAgent)
    assert validator is not None


def test_validator_creates_sample_agent():
    """Test that validator creates a sample agent for validation"""

    class TestState(BaseModel):
        value: int

    class TestAgent(BaseAgent):
        __system_message__ = "Test"

        test_state: State[TestState] = spec.State(default_factory=lambda: TestState(value=1))

    validator = _AgentConstructionValidator(TestAgent)
    assert validator.sample_agent is not None
    assert isinstance(validator.sample_agent, TestAgent)


def test_validator_accumulates_problems():
    """Test that validator has problems list"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    validator = _AgentConstructionValidator(TestAgent)
    assert hasattr(validator, "problems")
    assert isinstance(validator.problems, list)
