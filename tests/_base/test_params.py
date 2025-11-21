import pytest
from pydantic import BaseModel

from pyagentic import spec, ref
from pyagentic._base._info import ParamInfo


def test_param_info_creation():
    """Test that ParamInfo can be created with various parameters"""
    info = spec.Param(description="This is a test")
    assert info.description == "This is a test"
    assert info.required == False
    assert info.default is None


def test_param_info_with_default():
    """Test ParamInfo with default value"""
    info = spec.Param(default=4)
    assert info.get_default() == 4


def test_param_info_with_default_factory():
    """Test ParamInfo with default_factory"""
    info = spec.Param(default_factory=lambda: 4)
    assert info.get_default() == 4


def test_param_info_required():
    """Test ParamInfo with required flag"""
    info = spec.Param(required=True)
    assert info.required == True


def test_param_info_with_values():
    """Test ParamInfo with enum values"""
    info = spec.Param(values=["a", "b", "c"])
    assert info.values == ["a", "b", "c"]


def test_param_info_with_ref():
    """Test ParamInfo with ref reference"""
    from pyagentic._base._ref import RefNode

    info = spec.Param(description=ref.self.test_field)
    assert isinstance(info.description, RefNode)


def test_param_info_resolve():
    """Test that ParamInfo.resolve resolves ref references"""
    info = spec.Param(description=ref.self.str_default.value, values=ref.self.enum_values)

    agent_reference = {"self": {"str_default": {"value": "test"}, "enum_values": ["a", "b", "c"]}}

    resolved_info = info.resolve(agent_reference)
    assert resolved_info.description == "test"
    assert resolved_info.values == ["a", "b", "c"]


def test_param_info_resolve_without_refs():
    """Test that ParamInfo.resolve works with non-ref values"""
    info = spec.Param(description="static description", values=["x", "y", "z"])

    agent_reference = {"self": {}}

    resolved_info = info.resolve(agent_reference)
    assert resolved_info.description == "static description"
    assert resolved_info.values == ["x", "y", "z"]


def test_pydantic_model_as_param():
    """Test using Pydantic BaseModel as tool parameter"""

    class TestParam(BaseModel):
        field: int
        field_with_default: str = "default"

    # Create an instance
    param = TestParam(field=2)
    assert param.field == 2
    assert param.field_with_default == "default"


def test_pydantic_model_nested():
    """Test nested Pydantic models as tool parameters"""

    class Nested(BaseModel):
        field: int

    class TestParam(BaseModel):
        param_field: Nested

    nested = Nested(field=2)
    param = TestParam(param_field=nested)
    assert param.param_field.field == 2


def test_pydantic_model_with_spec_param():
    """Test Pydantic model fields with spec.Param"""

    class TestParam(BaseModel):
        field: int = spec.Param(description="This is a field")
        required_field: str = spec.Param(required=True, description="Required field")

    # The spec.Param becomes the default value in the Pydantic model
    assert TestParam.__annotations__["field"] == int
    assert TestParam.__annotations__["required_field"] == str
