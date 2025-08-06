import pytest
import random

from objective_agents._base._context import _AgentContext, ContextItem


@pytest.fixture
def mock_context():
    MockContext = _AgentContext.make_ctx_class(
        "Mock",
        ctx_map={
            "int_default": (int, ContextItem(default=4)),
            "str_default": (str, ContextItem(default="test")),
            "int_factory": (int, ContextItem(default_factory=lambda: random.randint(0, 1000))),
            "str_factory": (
                str,
                ContextItem(default_factory=lambda: f"Random: {random.randint(0, 1000)}"),
            ),
        },
    )
    yield MockContext(instructions="This is a mock")
