import pytest
import random
from dataclasses import is_dataclass
from deepdiff import DeepDiff

from pyagentic._base._context import (
    ContextItem,
    computed_context,
    _AgentContext,
    ContextRef,
)
from pyagentic._base._exceptions import InvalidContextRefNotFoundInContext


def test_context_item_default():
    item = ContextItem(default=4)
    assert item.get_default_value() == 4


def test_context_item_default_factory():
    item = ContextItem(default_factory=lambda: 4)
    assert item.get_default_value() == 4


def test_computed_context_flag():
    @computed_context
    def test():
        pass

    assert getattr(
        test, "_is_context", False
    ), "computed_context not properly flagging function as context"


def test_agent_context_class_construction():
    context_attrs = {
        "int_default": (int, ContextItem(default=4)),
        "str_default": (str, ContextItem(default="test")),
        "int_factory": (int, ContextItem(default_factory=lambda: random.randint(0, 1000))),
        "str_factory": (
            str,
            ContextItem(default_factory=lambda: f"Random: {random.randint(0, 1000)}"),
        ),
    }
    TestClass = _AgentContext.make_ctx_class("Test", context_attrs)

    assert is_dataclass(TestClass), "Created Context class is not a dataclass"

    assert TestClass.__name__ == "TestContext", (
        "New context name not being set properly\n"
        "Expected: TestContext\n"
        f"Recieved: {TestClass.__name__}"
    )
    annots = {"int_default": int, "str_default": str, "int_factory": int, "str_factory": str}
    assert TestClass.__annotations__ == annots, (
        "annotations not being set properly\n"
        f"Expected: {annots}\n"
        f"Recieved: {TestClass.__annotations__}"
    )
    assert "int_factory" in TestClass.__dataclass_fields__


def test_agent_context_class_construction_with_computed_context():
    @computed_context
    def test():
        return 5

    context_attrs = {
        "computed": (computed_context, test),
    }
    TestClass = _AgentContext.make_ctx_class("Test", context_attrs)

    assert hasattr(
        TestClass, "computed"
    ), "Computed context method failed to be bounded to context"


# These tests will use the context created in conftest to make tests simplier
def test_agent_context_default_values(mock_context):
    assert (
        mock_context.int_default == 4
    ), f"Unexpected value with default: {mock_context.int_default}"
    assert (
        mock_context.int_factory < 100
    ), f"Unexpected_value with factory: {mock_context.int_factory}"


def test_agent_context_default_override(mock_context):
    assert (
        mock_context.default_override == "overriden"
    ), "Default value not being overriden when supplied in constructor"


def test_agent_context_computed_field(mock_context):
    assert mock_context.random_computed > 1000, "Computed context not set properly"


def test_agent_context_attributes(mock_context):
    expected = {
        "instructions": "This is a mock",
        "input_template": None,
        "_messages": [],
        "int_default": 4,
        "str_default": "test",
        "int_factory": mock_context.int_factory,
        "str_factory": mock_context.str_factory,
        "default_override": "overriden",
        "random_computed": mock_context.random_computed,
    }

    for name, value in expected.items():
        attr_value = getattr(mock_context, name, None)
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


def test_agent_context_as_dict(mock_context):
    expected = {
        "instructions": "This is a mock",
        "input_template": None,
        "_messages": [],
        "int_default": 4,
        "str_default": "test",
        "int_factory": mock_context.int_factory,
        "str_factory": f"Random: {mock_context.str_factory}",
        "default_override": "overriden",
        "random_computed": mock_context.random_computed,
    }

    diff = DeepDiff(expected, expected, ignore_order=True)

    assert not diff, f"Dict export does not match expected: \n {diff.pretty()}"


def test_agent_context_add_user_message(mock_context):
    user_message = "Hello"
    mock_context.add_user_message(user_message)
    assert mock_context.messages[-1]["content"] == user_message


def test_agent_context_add_user_message_with_template(mock_context):
    mock_context.input_template = "Template: {user_message}"
    user_message = "Hello"
    mock_context.add_user_message(user_message)
    assert mock_context.messages[-1]["content"] == f"Template: {user_message}"


def test_agent_context_get_item(mock_context):
    assert mock_context.get("int_default") == 4


def test_agent_context_get_item_raises_not_found(mock_context):
    with pytest.raises(InvalidContextRefNotFoundInContext) as e:
        mock_context.get("not real")
    assert "InvalidContextRefNotFoundInContext" in str(e)


def test_agent_context_system_message(mock_context):
    mock_context.instructions = "System: {int_default}"
    assert mock_context.system_message == "System: 4"


def test_agent_context_system_message_dynamic(mock_context):
    mock_context.instructions = "System: {random_computed}"
    assert mock_context.system_message != mock_context.system_message


def test_agent_context_messages_dynamic(mock_context):
    mock_context.instructions = "System: {random_computed}"
    first = mock_context.messages[0]["content"]
    second = mock_context.messages[0]["content"]
    assert first != second


def test_context_ref_resolve(mock_context):
    ref = ContextRef("int_default")
    assert ref.resolve(mock_context) == 4


def test_context_ref_dynamic(mock_context):
    ref = ContextRef("random_computed")
    first = ref.resolve(mock_context)
    second = ref.resolve(mock_context)
    assert first != second
