import pytest
from deepdiff import DeepDiff

from objective_agents._base._tool import tool, _ToolDefinition
from objective_agents._base._params import Param, ParamInfo
from objective_agents._base._exceptions import ToolDeclarationFailed


def test_tool_declaration():

    @tool("This is for a test")
    def test() -> str:
        pass

    assert isinstance(
        test.__tool_def__, _ToolDefinition
    ), "Tool Definition not properly appened to function"
    tool_def: _ToolDefinition = test.__tool_def__
    assert tool_def.name == "test", "Name not properly set in tool definition"
    assert (
        tool_def.description == "This is for a test"
    ), "description not properly set in tool definition"


def test_tool_declaration_enforces_string_return():
    with pytest.raises(ToolDeclarationFailed) as e:

        @tool("no string return test")
        def test():
            pass

    assert "Method must have a return type of `str`" in str(e)


def test_tool_declarion_with_bare_string():

    @tool("Primitive Test")
    def test(primitive: str) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters

    assert (
        "primitive" in params
    ), f"Function attributes not being added to tool def params, got: {params.keys()}, expected: ['a', 'b']"  # noqa E501

    type_, info = params["primitive"]
    assert type_ == str and isinstance(
        info, ParamInfo
    ), f"Default value not properly being casted to ParamInfo for tool def, got: {type(info)}, expected: ParamInfo"  # noqa E501


def test_tool_declaration_with_annotated_non_string_primitive():
    @tool("Annotated Primitive Test")
    def test(a: int = ParamInfo(description="this is a test")) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters
    type_, info = params["a"]
    assert type_ == int, "Non-string typing not properly being set in tool def"
    assert (
        info.description == "this is a test"
    ), "Given ParamInfo not being set in tool def, got: {info}, expected: ParamInfo(description='this is a test)"  # noqa E501


def test_tool_declaration_with_param():
    class TestParam(Param):
        param_primitive: int

    @tool("Param Test")
    def test(param: TestParam) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters
    type_, info = params["param"]

    assert (
        type_ == TestParam
    ), "Attribute not being set as given Param class in tool def, got: {type_}, expected: TestParam"  # noqa E501


def test_tool_declaration_with_annotated_param():
    class TestParam(Param):
        param_primitive: int

    @tool("Annotated Param Test")
    def test(annotated_param: TestParam = ParamInfo(description="this is a test")) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters
    type_, info = params["annotated_param"]

    assert (
        type_ == TestParam
    ), "Attribute not being set as given Param class in tool def, got: {type_}, expected: TestParam"  # noqa E501
    assert (
        info.description == "this is a test"
    ), "Given ParamInfo not being set in tool def, got: {info}, expected: ParamInfo(description='this is a test)"  # noqa E501


def test_tool_openai_export():
    class TestParam(Param):
        param_primitive: int

    @tool("OpenAI Export Test")
    def test(
        primitive: str,
        param: TestParam,
        annotated_primitive: int = ParamInfo(description="this is an annotated primitive"),
        annotated_param: TestParam = ParamInfo(description="this is an annotated param"),
        required_primitive: int = ParamInfo(required=True),
        required_param: TestParam = ParamInfo(required=True),
    ) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai()
    expected = {
        "type": "function",
        "name": "test",
        "description": "OpenAI Export Test",
        "parameters": {
            "type": "object",
            "properties": {
                "primitive": {"type": "string"},
                "param": {
                    "type": "object",
                    "properties": {"param_primitive": {"type": "integer"}},
                    "required": [],
                },
                "annotated_primitive": {
                    "type": "integer",
                    "description": "this is an annotated primitive",
                },
                "annotated_param": {
                    "type": "object",
                    "properties": {"param_primitive": {"type": "integer"}},
                    "required": [],
                    "description": "this is an annotated param",
                },
                "required_primitive": {"type": "integer"},
                "required_param": {
                    "type": "object",
                    "properties": {"param_primitive": {"type": "integer"}},
                    "required": [],
                },
            },
        },
        "required": ["required_primitive", "required_param"],
    }

    diff = DeepDiff(openai_tool, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"


def test_tool_openai_export_multiple_params():
    class TestParamA(Param):
        param_primitive_a: int

    class TestParamB(Param):
        param_primitive_b: str

    @tool("OpenAI Export Test")
    def test(
        param_a: TestParamA,
        param_b: TestParamB,
    ) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai()
    expected = {
        "type": "function",
        "name": "test",
        "description": "OpenAI Export Test",
        "parameters": {
            "type": "object",
            "properties": {
                "param_a": {
                    "type": "object",
                    "properties": {"param_primitive_a": {"type": "integer"}},
                    "required": [],
                },
                "param_b": {
                    "type": "object",
                    "properties": {"param_primitive_b": {"type": "string"}},
                    "required": [],
                },
            },
        },
        "required": [],
    }

    diff = DeepDiff(openai_tool, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"
