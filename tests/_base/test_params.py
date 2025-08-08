import pytest
from deepdiff import DeepDiff

from pyagentic._base._params import Param, ParamInfo
from pyagentic._base._context import ContextRef


def test_param_declaration():

    class Test(Param):
        field: int

    type_, info = Test.__attributes__["field"]
    assert type_ == int, f"Typing is not being set properly, got: {type_}, expected: {int}"
    assert isinstance(
        info, ParamInfo
    ), "Default ParamInfo not being set when not supplied in definition"


def test_param_declaration_with_info():

    class Test(Param):
        field: int = ParamInfo(description="This is a test")

    _, info = Test.__attributes__["field"]

    assert info.description == "This is a test", "Supplied ParamInfo is not being set when given"

def test_param_declaration_nested_param():

    class Nested(Param):
        field: int

    class Test(Param):
        param_field: Nested

    type_, info = Test.__attributes__["param_field"]
    assert type_ == Nested
    assert isinstance(
        info, ParamInfo
    ), "Default ParamInfo not being set when not supplied in definition"

def test_param_creation():

    class Test(Param):
        field: int

    test = Test(field=2)

    assert test.field == 2

def test_param_creation_invalid_type():

    class Test(Param):
        field: int

    with pytest.raises(TypeError) as e:
        test = Test(field="string")
    
    assert "Field 'field' expected" in str(e)


def test_param_creation_unexpected_field():

    class Test(Param):
        field: int

    with pytest.raises(TypeError) as e:
        test = Test(field=1, another_field=2)

    assert "Unexpected fields for" in str(e)

def test_param_creation_with_info():
    
    class Test(Param):
        field: int = ParamInfo(default=2)

    test = Test()

    assert test.field == 2

def test_param_creation_nested_param():

    class Nested(Param):
        field: int

    class Test(Param):
        param_field: Nested

    nested = Nested(field=2)
    test = Test(param_field=nested)

    assert test.param_field == nested


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
        field: str = ParamInfo(description=ContextRef("str_default"), values=["a"])

    openai_param = ResolveTest.to_openai(mock_context)
    export_description = openai_param["properties"]["field"]["description"]
    export_values = openai_param["properties"]["field"]["enum"]

    assert export_description == "test", (
        "The resolved description did not match that in the mock context\n"
        f"Expected: {mock_context.str_default}\n"
        f"Recieved: {export_description}\n"
    )

    assert export_values == ["a"], (
        "The resolved description did not match that in the mock context\n"
        f"Expected: {mock_context.str_default}\n"
        f"Recieved: {export_description}\n"
    )
