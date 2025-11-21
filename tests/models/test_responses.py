from typing import get_args
from pydantic import BaseModel

from pyagentic._base._tool import tool
from pyagentic._base._info import ParamInfo
from pyagentic import spec

from pyagentic.models.response import ToolResponse


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
        a: int = spec.Param(default=1, description="test"),
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

    class TestParam(BaseModel):
        a: int

    @tool("tool response creation")
    def test(
        a: TestParam,
    ) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields
    # The response class should have the param field with the right structure
    assert ResponseClass.model_fields["a"].annotation.model_fields["a"].annotation == int


def test_tool_response_creation_with_listed_param():

    class TestParam(BaseModel):
        a: int

    @tool("tool response creation")
    def test(a: list[TestParam]) -> str:
        pass

    ResponseClass = ToolResponse.from_tool_def(test.__tool_def__)

    assert "a" in ResponseClass.model_fields
    # Check that it's a list type
    res_type_annotation = ResponseClass.model_fields["a"].annotation
    assert get_args(res_type_annotation)  # Should have type args for list
