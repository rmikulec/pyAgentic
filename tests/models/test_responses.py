from typing import get_args

from pyagentic._base._tool import tool
from pyagentic._base._params import Param, ParamInfo

from pyagentic.models.response import param_to_pydantic, ToolResponse, AgentResponse


def test_tool_response_creation():

    @tool("tool response creation")
    def test(a: int, b: str) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields
    assert "b" in ResponseClass.model_fields

    assert ResponseClass.model_fields["a"].annotation == int
    assert ResponseClass.model_fields["b"].annotation == str


def test_tool_response_creation_with_info():

    @tool("tool response creation")
    def test(
        a: int = ParamInfo(default=1, description="test"),
    ) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert ResponseClass.model_fields["a"].default == 1
    assert ResponseClass.model_fields["a"].description == "test"


def test_tool_response_creation_with_lists():

    @tool("tool response creation")
    def test(a: list[int], b: list[str]) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields
    assert "b" in ResponseClass.model_fields

    assert ResponseClass.model_fields["a"].annotation == list[int]
    assert ResponseClass.model_fields["b"].annotation == list[str]


def test_tool_response_creation_with_param():

    class TestParam(Param):
        a: int

    @tool("tool response creation")
    def test(
        a: TestParam,
    ) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields

    res_type_annotation = ResponseClass.model_fields["a"].annotation.model_json_schema()
    expected_annotation = param_to_pydantic(TestParam).model_json_schema()

    assert res_type_annotation == expected_annotation

    assert "a" in ResponseClass.model_fields
    assert ResponseClass.model_fields["a"].annotation.model_fields["a"].annotation == int


def test_tool_response_creation_with_listed_param():

    class TestParam(Param):
        a: int

    @tool("tool response creation")
    def test(a: list[TestParam]) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields

    res_type_annotation = ResponseClass.model_fields["a"].annotation
    res_type_annotation = get_args(res_type_annotation)[0].model_json_schema()
    expected_annotation = param_to_pydantic(TestParam).model_json_schema()

    assert res_type_annotation == expected_annotation
