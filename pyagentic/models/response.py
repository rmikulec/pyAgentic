from pydantic import BaseModel, Field, create_model
from typing import Any, Type, get_origin, get_args, Self, Union

from openai.types.responses import Response

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._params import Param


PRIMITIVES = (bool, str, int, float, type(None))


def is_primitive(type_: Any) -> bool:
    """
    Helper function to check if a type is a python primitive
    """
    return type_ in PRIMITIVES


def param_to_pydantic(ParamClass: Type[Param]) -> Type[BaseModel]:
    """
    Converts a pyagentic Param Class to a pydantic BaseModel

    Args:
        - ParamClass(Type[Param]): A pyagentic ParamClass, this is any class that extends Param

    Returns:
        - Type[BaseModel]: A pydantic BaseModel with the same fields as the pyagentic ParamClass
    """
    fields = {}

    for attr_name, (attr_type, attr_info) in ParamClass.__attributes__.items():
        if get_origin(attr_type) == list:
            origin_type = get_args(attr_type)[0]
            if is_primitive(origin_type):
                fields[attr_name] = (
                    list[attr_type],
                    Field(default=attr_info.default, description=attr_info.description),
                )
            elif issubclass(origin_type, Param):
                SubParamModel = param_to_pydantic(origin_type)
                fields[attr_name] = (
                    list[SubParamModel],
                    Field(default=attr_info.default, description=attr_info.description),
                )
            else:
                raise Exception(f"Unsupported type: {attr_type}")
        elif is_primitive(attr_type):
            fields[attr_name] = (
                attr_type,
                Field(default=attr_info.default, description=attr_info.description),
            )
        elif issubclass(attr_type, Param):
            SubParamModel = param_to_pydantic(attr_type)
            fields[attr_name] = (
                SubParamModel,
                Field(default=attr_info.default, description=attr_info.description),
            )
        else:
            raise Exception(f"Unsupported type: {attr_type}")

    return create_model(f"{ParamClass.__name__}", **fields)


class ToolResponse(BaseModel):
    """
    Tool response class to capture both the call from openai and the result from pyagentic.

    Use `from_tool_def` to create a subclass that has the tool params as fields of the pydantic
        model.
    """

    raw_kwargs: str
    call_depth: int
    output: str

    @classmethod
    def from_tool_def(cls, tool_def: _ToolDefinition) -> Type[Self]:
        """
        Creates a subclass of `ToolResponse`, using the Tool Definition to make the kwargs
            accessible through pydantic
        """
        fields = {}
        for param_name, (param_type, param_info) in tool_def.parameters.items():
            if get_origin(param_type) == list:
                pass
            elif is_primitive(param_type):
                fields[param_name] = (
                    param_type,
                    Field(default=param_info.default, description=param_info.description),
                )
            elif issubclass(param_type, Param):
                fields[param_name] = (
                    param_to_pydantic(param_type),
                    Field(default=param_info.default, description=param_info.description),
                )
            else:
                raise Exception(f"Unsupported type: {param_type}")

        return create_model(f"ToolResponse[{tool_def.name}]", __base__=cls, **fields)


class AgentResponse(BaseModel):
    """
    Agent response class that captures the final output from an pyagentic agent and the raw
        response from openai.

    Each Agent will have a unique, predetermined Response Model that can easily be integrated in
        a fastapi app. This is done by calling `from_tool_defs`.
    """

    response: Response
    final_output: str

    @classmethod
    def from_tool_defs(
        cls, agent_name: str, tool_response_models: list[Type[ToolResponse]]
    ) -> Type[Self]:
        """
        Creates a subclass of `AgentResponse`, using Tool Definitions to create a predetermined
            schema of what the response will look like.
        """
        if tool_response_models:
            ToolResult = Union[tuple(tool_response_models)]
            return create_model(
                f"{agent_name}Response", __base__=cls, tool_responses=(list[ToolResult], ...)
            )
        else:
            return create_model(
                f"{agent_name}Response",
                __base__=cls,
            )
