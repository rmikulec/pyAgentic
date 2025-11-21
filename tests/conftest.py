import pytest
import random
from pydantic import BaseModel

from pyagentic import BaseAgent, tool, spec, State, ref


# Pydantic models for state
class IntStateModel(BaseModel):
    value: int = 4


class StrStateModel(BaseModel):
    value: str = "test"


class EnumValuesModel(BaseModel):
    values: list[str] = []


@pytest.fixture
def mock_state():
    """Creates a mock state instance for testing state-related functionality"""

    class MockAgent(BaseAgent):
        __system_message__ = "This is a mock"
        __input_template__ = ""  # Prevent None from being passed to Jinja2

        int_default: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=4)
        )
        str_default: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value="test")
        )
        int_factory: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=random.randint(0, 100))
        )
        str_factory: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value=f"Random: {random.randint(100, 200)}")
        )
        default_override: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value="testing")
        )

    # Create instance with overridden default
    agent = MockAgent(
        model="_mock::test-model",
        api_key="MyKey",
    )
    agent.state.default_override = StrStateModel(value="overriden")

    yield agent.state


@pytest.fixture
def mock_agent():
    """Creates a mock agent instance for testing agent functionality"""

    class MockAgent(BaseAgent):
        __system_message__ = "This is a mock agent"
        __input_template__ = ""  # Prevent None from being passed to Jinja2

        int_default: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=4)
        )
        str_default: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value="test")
        )
        int_factory: State[IntStateModel] = spec.State(
            default_factory=lambda: IntStateModel(value=random.randint(0, 100))
        )
        str_factory: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value=f"Random: {random.randint(100, 200)}")
        )
        default_override: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value="testing")
        )
        test_enum_values: State[EnumValuesModel] = spec.State(
            default_factory=lambda: EnumValuesModel(values=[])
        )

        @tool("Testing from mock agent")
        def test(self) -> str:
            return "test"

        @tool("Testing params in tool with ref")
        def test_ref(
            self, letter: str = spec.Param(values=ref.self.test_enum_values.values)
        ) -> str:
            return f"letter: {letter}"

        @tool("testing ref with computed")
        def test_computed_ref(self, letter: str = spec.Param(values=["a", "b", "c"])) -> str:
            return f"letter: {letter}"

    agent = MockAgent(
        model="_mock::test-model",
        api_key="MyKey",
    )
    agent.state.default_override = StrStateModel(value="overriden")
    agent.state.test_enum_values = EnumValuesModel(values=["a", "b", "c"])

    yield agent
