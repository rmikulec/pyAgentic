import pytest
import random

from pyagentic._base._context import (
    _AgentContext,
    ContextItem,
    computed_context,
    ContextRef,
)
from pyagentic._base._tool import tool
from pyagentic._base._params import ParamInfo
from pyagentic._base._agent import Agent


@pytest.fixture
def mock_context():
    @computed_context
    def test(self):
        return random.randint(1000, 10000)

    MockContext = _AgentContext.make_ctx_class(
        "Mock",
        ctx_map={
            "int_default": (int, ContextItem(default=4)),
            "str_default": (str, ContextItem(default="test")),
            "int_factory": (int, ContextItem(default_factory=lambda: random.randint(0, 100))),
            "str_factory": (
                str,
                ContextItem(default_factory=lambda: f"Random: {random.randint(100, 200)}"),
            ),
            "random_computed": (computed_context, test),
            "default_override": (str, ContextItem(default="testing")),
        },
    )
    yield MockContext(instructions="This is a mock", default_override="overriden")


@pytest.fixture
def mock_agent():
    class MockAgent(Agent):
        __system_message__ = "This is a mock agent"

        int_default: int = ContextItem(default=4)
        str_default: str = ContextItem(default="test")
        int_factory: int = ContextItem(default_factory=lambda: random.randint(0, 100))
        str_factory: str = ContextItem(
            default_factory=lambda: f"Random: {random.randint(100, 200)}"
        )
        default_override: str = ContextItem(default="testing")
        test_enum_values: list[str] = ContextItem(default_factory=list)

        @computed_context
        def random_computed(self):
            return self.int_default * random.randint(1000, 10000)

        @computed_context
        def computed_values(self):
            return ["a", "b", "c"]

        @tool("Testing from mock agent")
        def test(self) -> str:
            pass

        @tool("Testing params in tool with ref")
        def test_ref(self, letter: str = ParamInfo(values=ContextRef("test_enum_values"))) -> str:
            pass

        @tool("testing ref with computed")
        def test_computed_ref(
            self, letter: str = ParamInfo(values=ContextRef("computed_values"))
        ) -> str:
            pass

    yield MockAgent(
        model="mock",
        api_key="MyKey",
        default_override="overriden",
        test_enum_values=["a", "b", "c"],
    )
