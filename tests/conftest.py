import pytest
import random

from objective_agents._base._context import _AgentContext, ContextItem, computed_context


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
