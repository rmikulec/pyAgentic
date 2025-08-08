import pytest
from deepdiff import DeepDiff

from pyagentic._base._tool import tool, _ToolDefinition
from pyagentic._base._params import Param, ParamInfo
from pyagentic._base._context import ContextRef
from pyagentic._base._exceptions import ToolDeclarationFailed


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


def test_tool_declarion_with_listed_string():

    @tool("Primitive Test")
    def test(primitive: list[str]) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters

    assert (
        "primitive" in params
    ), f"Function attributes not being added to tool def params, got: {params.keys()}, expected: ['a', 'b']"  # noqa E501

    type_, info = params["primitive"]
    assert type_ == list[str] and isinstance(
        info, ParamInfo
    ), f"Default value not properly being casted to ParamInfo for tool def, got: {type(info)}, expected: ParamInfo"  # noqa E501


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


def test_tool_declaration_with_listed_param():
    class TestParam(Param):
        param_primitive: int

    @tool("Param Test")
    def test(param: list[TestParam]) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters
    type_, info = params["param"]

    assert (
        type_ == list[TestParam]
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


def test_tool_declarion_with_listed_string():

    @tool("Primitive Test")
    def test(values: list[str]) -> str:
        pass

    params: list[Param] = test.__tool_def__.parameters

    assert (
        "values" in params
    ), f"Function attributes not being added to tool def params, got: {params.keys()}, expected: ['a', 'b']"  # noqa E501

    type_, info = params["values"]
    assert type_ == list[str] and isinstance(
        info, ParamInfo
    ), f"Default value not properly being casted to ParamInfo for tool def, got: {type(info)}, expected: ParamInfo"  # noqa E501


def test_tool_compile_args():
    class TestParam(Param):
        field: int

    @tool("Testing compile args")
    def test(
        primitive: str,
        listed_primitive: list[str],
        param: TestParam,
        listed_param: list[TestParam],
    ) -> str:
        pass

    kwargs = {
        "primitive": "test",
        "listed_primitive": ["a", "b"],
        "param": {"field": 1},
        "listed_param": [{"field": 2}, {"field": 3}],
    }

    compiled_args = test.__tool_def__.compile_args(**kwargs)

    assert type(compiled_args["primitive"]) == str
    assert compiled_args["primitive"] == "test"

    assert type(compiled_args["listed_primitive"]) == list
    assert compiled_args["listed_primitive"] == ["a", "b"]

    assert isinstance(compiled_args["param"], TestParam)
    assert compiled_args["param"].field == 1

    for i, param in enumerate(compiled_args["listed_param"]):
        assert isinstance(param, TestParam)
        assert param.field == 2 + i


def test_tool_compile_args_nested_param():
    class NestedParam(Param):
        field: int

    class TestParam(Param):
        param_field: NestedParam

    @tool("Testing compile args with nested params")
    def test(param: TestParam, listed_param: list[TestParam]) -> str:
        pass

    kwargs = {
        "param": {"param_field": {"field": 1}},
        "listed_param": [
            {"param_field": {"field": 2}},
            {"param_field": {"field": 3}},
        ],
    }
    compiled_args = test.__tool_def__.compile_args(**kwargs)


def test_tool_openai_export(mock_context):
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

    openai_tool = test.__tool_def__.to_openai(mock_context)
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


def test_tool_openai_export_multiple_params(mock_context):
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

    openai_tool = test.__tool_def__.to_openai(mock_context)
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


def test_tool_openai_export_listed_primitive(mock_context):
    @tool("OpenAI Export Test")
    def test(list_: list[str]) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai(mock_context)
    expected = {
        "type": "function",
        "name": "test",
        "description": "OpenAI Export Test",
        "parameters": {
            "type": "object",
            "properties": {
                "list_": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
        },
        "required": [],
    }

    diff = DeepDiff(openai_tool, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"


def test_tool_openai_export_listed_param(mock_context):
    class TestParam(Param):
        value: int

    @tool("OpenAI Export Test")
    def test(list_: list[TestParam]) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai(mock_context)
    expected = {
        "type": "function",
        "name": "test",
        "description": "OpenAI Export Test",
        "parameters": {
            "type": "object",
            "properties": {
                "list_": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": [],
                    },
                },
            },
        },
        "required": [],
    }

    diff = DeepDiff(openai_tool, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"


def test_tool_openai_export_context_resolve(mock_context):
    class ResolveParam(Param):
        field: str = ParamInfo(description=ContextRef("str_default"))

    @tool("Testing context resolve on tool level")
    def test(param: ResolveParam) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai(mock_context)
    export_description = openai_tool["parameters"]["properties"]["param"]["properties"]["field"][
        "description"
    ]
    assert export_description == "test", (
        "The resolved description did not match that in the mock context\n"
        f"Expected: {mock_context.str_default}\n"
        f"Recieved: {export_description}\n"
    )
