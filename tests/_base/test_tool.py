import pytest
import asyncio
from deepdiff import DeepDiff
from pydantic import BaseModel

from pyagentic import tool, spec, BaseAgent, ref, State
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._info import ParamInfo
from pyagentic._base._exceptions import InvalidToolDefinition


def test_tool_declaration():
    """Test basic tool declaration"""

    @tool("This is for a test")
    def test() -> str:
        pass

    assert isinstance(
        test.__tool_def__, _ToolDefinition
    ), "Tool Definition not properly appended to function"
    tool_def: _ToolDefinition = test.__tool_def__
    assert tool_def.name == "test", "Name not properly set in tool definition"
    assert (
        tool_def.description == "This is for a test"
    ), "description not properly set in tool definition"


def test_tool_declaration_enforces_string_return():
    """Test that tool decorator works with proper return type"""
    # Tools with string return type should work fine
    @tool("string return test")
    def test() -> str:
        return "test"

    assert test.__tool_def__ is not None
    assert test.__tool_def__.name == "test"


def test_tool_declaration_with_bare_string():
    """Test tool with simple string parameter"""

    @tool("Primitive Test")
    def test(primitive: str) -> str:
        pass

    params: dict = test.__tool_def__.parameters

    assert (
        "primitive" in params
    ), f"Function attributes not being added to tool def params, got: {params.keys()}, expected: ['primitive']"

    type_, info = params["primitive"]
    assert type_ == str and isinstance(
        info, ParamInfo
    ), f"Default value not properly being casted to ParamInfo for tool def, got: {type(info)}, expected: ParamInfo"


def test_tool_declaration_with_annotated_non_string_primitive():
    """Test tool with annotated non-string primitive parameter"""

    @tool("Annotated Primitive Test")
    def test(a: int = spec.Param(description="this is a test")) -> str:
        pass

    params: dict = test.__tool_def__.parameters
    type_, info = params["a"]
    assert type_ == int, "Non-string typing not properly being set in tool def"
    assert (
        info.description == "this is a test"
    ), f"Given ParamInfo not being set in tool def, got: {info}, expected: ParamInfo(description='this is a test')"


def test_tool_declaration_with_pydantic_model():
    """Test tool with Pydantic model parameter"""

    class TestParam(BaseModel):
        param_primitive: int

    @tool("Param Test")
    def test(param: TestParam) -> str:
        pass

    params: dict = test.__tool_def__.parameters
    type_, info = params["param"]

    assert (
        type_ == TestParam
    ), f"Attribute not being set as given Pydantic model class in tool def, got: {type_}, expected: TestParam"


def test_tool_declaration_with_listed_pydantic_model():
    """Test tool with list of Pydantic models parameter"""

    class TestParam(BaseModel):
        param_primitive: int

    @tool("Param Test")
    def test(param: list[TestParam]) -> str:
        pass

    params: dict = test.__tool_def__.parameters
    type_, info = params["param"]

    assert (
        type_ == list[TestParam]
    ), f"Attribute not being set as list of Pydantic model class in tool def, got: {type_}, expected: list[TestParam]"


def test_tool_declaration_with_annotated_pydantic_model():
    """Test tool with annotated Pydantic model parameter"""

    class TestParam(BaseModel):
        param_primitive: int

    @tool("Annotated Param Test")
    def test(annotated_param: TestParam = spec.Param(description="this is a test")) -> str:
        pass

    params: dict = test.__tool_def__.parameters
    type_, info = params["annotated_param"]

    assert (
        type_ == TestParam
    ), f"Attribute not being set as given Pydantic model class in tool def, got: {type_}, expected: TestParam"
    assert (
        info.description == "this is a test"
    ), f"Given ParamInfo not being set in tool def, got: {info}, expected: ParamInfo(description='this is a test')"


def test_tool_declaration_with_listed_string():
    """Test tool with list of strings parameter"""

    @tool("Primitive Test")
    def test(values: list[str]) -> str:
        pass

    params: dict = test.__tool_def__.parameters

    assert (
        "values" in params
    ), f"Function attributes not being added to tool def params, got: {params.keys()}, expected: ['values']"

    type_, info = params["values"]
    assert type_ == list[str] and isinstance(
        info, ParamInfo
    ), f"Default value not properly being casted to ParamInfo for tool def, got: {type(info)}, expected: ParamInfo"


def test_tool_compile_args():
    """Test compiling tool arguments from raw kwargs"""

    class TestParam(BaseModel):
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
    """Test compiling tool arguments with nested Pydantic models"""

    class NestedParam(BaseModel):
        field: int

    class TestParam(BaseModel):
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

    assert isinstance(compiled_args["param"], TestParam)
    assert isinstance(compiled_args["param"].param_field, NestedParam)
    assert compiled_args["param"].param_field.field == 1


def test_tool_openai_export(mock_state):
    """Test OpenAI spec export for tools"""

    class TestParam(BaseModel):
        param_primitive: int

    @tool("OpenAI Export Test")
    def test(
        annotated_primitive: int = spec.Param(description="this is an annotated primitive"),
        annotated_param: TestParam = spec.Param(description="this is an annotated param"),
        required_primitive: int = spec.Param(required=True),
        required_param: TestParam = spec.Param(required=True),
    ) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai_spec()

    # Check basic structure
    assert openai_tool["type"] == "function"
    assert openai_tool["name"] == "test"
    assert openai_tool["description"] == "OpenAI Export Test"

    # Check parameters exist
    params = openai_tool["parameters"]["properties"]
    assert "annotated_primitive" in params
    assert "annotated_param" in params
    assert "required_primitive" in params
    assert "required_param" in params

    # Check descriptions
    assert params["annotated_primitive"]["description"] == "this is an annotated primitive"
    assert params["annotated_param"]["description"] == "this is an annotated param"

    # Check required fields
    assert "required_primitive" in openai_tool["parameters"]["required"]
    assert "required_param" in openai_tool["parameters"]["required"]


def test_tool_openai_export_multiple_params():
    """Test OpenAI spec export with multiple Pydantic model parameters"""

    class TestParamA(BaseModel):
        param_primitive_a: int

    class TestParamB(BaseModel):
        param_primitive_b: str

    @tool("OpenAI Export Test")
    def test(
        param_a: TestParamA,
        param_b: TestParamB,
    ) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai_spec()

    # Check basic structure
    assert openai_tool["type"] == "function"
    assert openai_tool["name"] == "test"

    # Check both params exist
    params = openai_tool["parameters"]["properties"]
    assert "param_a" in params
    assert "param_b" in params

    # Check param types
    assert params["param_a"]["type"] == "object"
    assert params["param_b"]["type"] == "object"


def test_tool_openai_export_listed_primitive():
    """Test OpenAI spec export with list of primitives"""

    @tool("OpenAI Export Test")
    def test(list_: list[str]) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai_spec()

    # Check list parameter exists and has correct structure
    params = openai_tool["parameters"]["properties"]
    assert "list_" in params
    assert params["list_"]["type"] == "array"
    assert params["list_"]["items"]["type"] == "string"


def test_tool_openai_export_listed_param():
    """Test OpenAI spec export with list of Pydantic models"""

    class TestParam(BaseModel):
        value: int

    @tool("OpenAI Export Test")
    def test(list_: list[TestParam]) -> str:
        pass

    openai_tool = test.__tool_def__.to_openai_spec()

    # Check list of objects
    params = openai_tool["parameters"]["properties"]
    assert "list_" in params
    assert params["list_"]["type"] == "array"
    assert params["list_"]["items"]["type"] == "object"
    assert "value" in params["list_"]["items"]["properties"]


def test_tool_openai_export_ref_resolve(mock_state):
    """Test that ref() references are properly resolved in tool specs"""
    from tests.conftest import StrStateModel

    class TestAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

        str_default: State[StrStateModel] = spec.State(
            default_factory=lambda: StrStateModel(value="test")
        )

        @tool("Testing ref resolve on tool level")
        def test(self, value: str = spec.Param(description=ref.self.str_default.value)) -> str:
            return value

    agent = TestAgent(model="_mock::test-model", api_key="test")

    tools = asyncio.run(agent._get_tool_defs())
    resolved_tool = tools[0].resolve(agent.agent_reference)
    openai_tool = resolved_tool.to_openai_spec()

    # Check that the ref was resolved
    export_description = openai_tool["parameters"]["properties"]["value"]["description"]
    assert export_description == "test", (
        "The resolved description did not match expected\n"
        f"Expected: test\n"
        f"Received: {export_description}\n"
    )
