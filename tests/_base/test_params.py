from deepdiff import DeepDiff

from pyagentic._base._params import Param, ParamInfo
from pyagentic._base._context import ContextRef


def test_param_creation():

    class Test(Param):
        field: int

    type_, info = Test.__attributes__["field"]
    assert type_ == int, f"Typing is not being set properly, got: {type_}, expected: {int}"
    assert isinstance(
        info, ParamInfo
    ), "Default ParamInfo not being set when not supplied in definition"


def test_param_creation_with_info():

    class Test(Param):
        field: int = ParamInfo(description="This is a test")

    _, info = Test.__attributes__["field"]

    assert info.description == "This is a test", "Supplied ParamInfo is not being set when given"


def test_param_openai_export(mock_context):

    class ExportTest(Param):
        field: int
        field_with_info: str = ParamInfo(description="This is a test")
        required_field: str = ParamInfo(required=True)
        field_with_default: str = ParamInfo(default="default")

    openai_param = ExportTest.to_openai(mock_context)
    expected = {
        "type": "object",
        "properties": {
            "field": {"type": "integer"},
            "field_with_info": {"type": "string", "description": "This is a test"},
            "required_field": {"type": "string"},
            "field_with_default": {"type": "string"},
        },
        "required": ["required_field"],
    }

    diff = DeepDiff(openai_param, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"


def test_param_info_context_resolve(mock_context):
    info = ParamInfo(description=ContextRef("str_default"))

    resolved_info = info.resolve(mock_context)

    assert resolved_info.description == "test", (
        "The resolved description did not match that in the mock context\n"
        f"Expected: {mock_context.str_default}\n"
        f"Recieved: {resolved_info.description}\n"
    )


def test_param_openai_export_with_context_resolve(mock_context):
    class ResolveTest(Param):
        field: str = ParamInfo(description=ContextRef("str_default"))

    openai_param = ResolveTest.to_openai(mock_context)
    export_description = openai_param["properties"]["field"]["description"]

    assert export_description == "test", (
        "The resolved description did not match that in the mock context\n"
        f"Expected: {mock_context.str_default}\n"
        f"Recieved: {export_description}\n"
    )
