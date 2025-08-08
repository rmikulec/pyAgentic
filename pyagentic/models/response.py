from pydantic import BaseModel, Field, create_model
from typing import Type, Self, Union

from openai.types.responses import Response

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._params import Param

from pyagentic._utils._typing import TypeCategory, analyze_type


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
        type_info = analyze_type(attr_type, Param)

        match type_info.category:

            case TypeCategory.PRIMITIVE:
                fields[attr_name] = (
                    attr_type,
                    Field(default=attr_info.default, description=attr_info.description),
                )
            case TypeCategory.LIST_PRIMITIVE:
                fields[attr_name] = (
                    list[attr_type],
                    Field(default=attr_info.default, description=attr_info.description),
                )
            case TypeCategory.SUBCLASS:
                SubParamModel = param_to_pydantic(attr_type)
                fields[attr_name] = (
                    SubParamModel,
                    Field(default=attr_info.default, description=attr_info.description),
                )
            case TypeCategory.LIST_SUBCLASS:
                SubParamModel = param_to_pydantic(type_info.inner_type)
                fields[attr_name] = (
                    list[SubParamModel],
                    Field(default=attr_info.default, description=attr_info.description),
                )
            case _:
                raise Exception(f"Unsupported type: {attr_type}")

    return create_model(f"{ParamClass.__name__}Model", **fields)


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
            type_info = analyze_type(param_type, Param)
            match type_info.category:

                case TypeCategory.PRIMITIVE:
                    fields[param_name] = (
                        param_type,
                        Field(default=param_info.default, description=param_info.description),
                    )
                case TypeCategory.LIST_PRIMITIVE:
                    fields[param_name] = (
                        param_type,
                        Field(default=param_info.default, description=param_info.description),
                    )
                case TypeCategory.SUBCLASS:
                    ParamSubModel = param_to_pydantic(param_type)
                    fields[param_name] = (
                        ParamSubModel,
                        Field(default=param_info.default, description=param_info.description),
                    )
                case TypeCategory.LIST_SUBCLASS:
                    ParamSubModel = param_to_pydantic(type_info.inner_type)
                    fields[param_name] = (
                        list[ParamSubModel],
                        Field(default=param_info.default, description=param_info.description),
                    )
                case _:
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
