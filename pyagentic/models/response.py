from pydantic import BaseModel, Field, create_model
from typing import Type, Self, Union, Any

from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._agent._agent_state import _AgentState

from pyagentic._utils._typing import TypeCategory, analyze_type
from pyagentic.models.llm import ProviderInfo


class ToolResponse(BaseModel):
    """
    Tool response class to capture both the call from openai and the result from pyagentic.

    Use `from_tool_def` to create a subclass that has the tool params as fields of the pydantic
        model.
    """

    raw_kwargs: str
    call_depth: int
    output: Any

    @classmethod
    def from_tool_def(cls, tool_def: _ToolDefinition) -> Type[Self]:
        """
        Creates a subclass of `ToolResponse`, using the Tool Definition to make the kwargs
            accessible through pydantic.

        Args:
            tool_def (_ToolDefinition): Tool definition containing parameter specifications

        Returns:
            Type[Self]: New ToolResponse subclass with tool parameters as fields
        """
        fields = {}
        for param_name, (param_type, param_info) in tool_def.parameters.items():
            type_info = analyze_type(param_type, BaseModel)
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
                    fields[param_name] = (
                        param_type,
                        Field(default=param_info.default, description=param_info.description),
                    )
                case TypeCategory.LIST_SUBCLASS:
                    fields[param_name] = (
                        param_type,
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

    final_output: Union[str, Type[BaseModel]]
    provider_info: ProviderInfo

    @classmethod
    def from_agent_class(
        cls,
        agent_name: str,
        tool_response_models: list[Type[ToolResponse]],
        linked_agents_response_models: list[Type[Self]],
        ResponseFormat: Union[str, Type[BaseModel]],
        StateClass: Type[_AgentState],
    ) -> Type[Self]:
        """
        Creates a subclass of `AgentResponse`, using Tool Definitions to create a predetermined
            schema of what the response will look like.

        Args:
            agent_name (str): Name of the agent class
            tool_response_models (list[Type[ToolResponse]]): List of tool response model types
            linked_agents_response_models (list[Type[Self]]): List of linked agent response types
            ResponseFormat (Union[str, Type[BaseModel]]): Expected response format specification
            StateClass (Type[_AgentState]): Agent state class type

        Returns:
            Type[Self]: New AgentResponse subclass with predetermined schema
        """
        fields = {}
        if tool_response_models:
            ToolResult = Union[tuple(tool_response_models)]
            fields["tool_responses"] = (list[ToolResult], ...)
        if linked_agents_response_models:
            AgentResult = Union[tuple(linked_agents_response_models)]
            fields["agent_responses"] = (list[AgentResult], ...)
        if ResponseFormat:
            fields["final_output"] = (ResponseFormat, ...)
        if StateClass:
            fields["state"] = (StateClass, ...)
        else:
            fields["final_output"] = (str, ...)
        return create_model(f"{agent_name}Response", __base__=cls, **fields)
