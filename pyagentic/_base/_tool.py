import inspect
from typing import Callable, Any, TypeVar, get_type_hints, Self
from collections import defaultdict
from copy import deepcopy
from pydantic import BaseModel

from pyagentic._base._params import Param, ParamInfo, _TYPE_MAP
from pyagentic._base._agent_state import _AgentState
from pyagentic._base._exceptions import InvalidToolDefinition

from pyagentic._utils._typing import TypeCategory, analyze_type


class _ToolDefinition:
    """
    Private class to handle tool definitions

    Attributes:
        name(str): Name of the tool, automatically filled out as the function name
        description(str): Description of the tool for LLM to read
        parameters(str): Dictionary containing parameters captured by the tool descriptor
        condition(str): The condition supplied determining when this tool should be included
            in the LLM inference call

    Methods:
        to_openai()->dict: Converts the definition to an "openai-ready" dictionary
        compile_args()->dict[str, Any]: Converts any raw kwargs, usually from LLM tool call, to
            match that of the tool definition. This process does the following:
                1. Fills in any default values for args not supplied
                2. Casts a raw dictionary to any arg that is a Param class
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, tuple[TypeVar, ParamInfo]],
        condition: Callable[[Any], bool] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.parameters: dict[str, tuple[TypeVar, ParamInfo]] = parameters
        self.condition = condition

    def resolve(self, agent_reference: dict) -> Self:
        new_parameters = {}

        for name, (type_, default) in self.parameters.items():
            if isinstance(default, ParamInfo):
                new_default = default.resolve(agent_reference)
            else:
                new_default = default

            new_parameters[name] = (type_, new_default)

        return self.__class__(
            name=self.name,
            description=self.description,
            parameters=new_parameters,
            condition=self.condition,
        )

    def to_openai_spec(self) -> dict:
        """
        Converts the definition to an "openai-ready" dictionary.

        Returns:
            dict: An OpenAI-compliant tool spec.
        """
        params = defaultdict(dict)
        required = []
        top_level_defs = {}

        for name, (type_, default) in self.parameters.items():
            type_info = analyze_type(type_, BaseModel)

            match type_info.category:
                case TypeCategory.PRIMITIVE:
                    params[name] = {"type": _TYPE_MAP.get(type_, "string")}

                case TypeCategory.LIST_PRIMITIVE:
                    params[name] = {
                        "type": "array",
                        "items": {"type": _TYPE_MAP.get(type_info.inner_type, "string")},
                    }

                case TypeCategory.SUBCLASS:
                    schema = deepcopy(type_.model_json_schema())

                    # Move $defs to top level
                    if "$defs" in schema:
                        top_level_defs.update(schema.pop("$defs"))

                    params[name] = schema

                case TypeCategory.LIST_SUBCLASS:
                    schema = deepcopy(type_info.inner_type.model_json_schema())

                    # Move $defs to top level
                    if "$defs" in schema:
                        top_level_defs.update(schema.pop("$defs"))

                    params[name] = {
                        "type": "array",
                        "items": schema,
                    }

            # Handle defaults and metadata
            if isinstance(default, ParamInfo):
                if default.description:
                    params[name]["description"] = default.description
                if default.required:
                    required.append(name)
                if default.values:
                    params[name]["enum"] = default.values

        # Final structure
        parameters = {
            "type": "object",
            "properties": dict(params),
            "required": required,
        }

        if top_level_defs:
            parameters["$defs"] = top_level_defs

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def to_anthropic_spec(self) -> dict:
        """
        Convert using the already-built OpenAI spec, then adapt shape to Anthropic:
          - name, description copied over
          - input_schema derived from OpenAI `parameters`
          - required moved from top-level to inside input_schema
        """
        openai_spec = self.to_openai_spec()

        # Copy to avoid mutating original
        input_schema = dict(openai_spec.get("parameters", {"type": "object", "properties": {}}))
        required = openai_spec.get("required") or []
        if required:
            # Anthropic expects `required` inside the JSON Schema object
            input_schema = {**input_schema, "required": list(required)}

        return {
            "name": openai_spec.get("name", self.name),
            "description": openai_spec.get("description", self.description),
            "input_schema": input_schema,
        }

    def to_openai_v1(self):
        openai_spec = self.to_openai_spec()
        openai_spec.pop("type")
        v1_spec = {"type": "function", "function": {**openai_spec}}
        v1_spec["function"]["strict"] = True
        return v1_spec

    def compile_args(self, **kwargs) -> dict[str, Any]:
        """
        Converts the definition to an "openai-ready" dictionary
        compile_args()->dict[str, Any]: Converts any raw kwargs, usually from LLM tool call, to
            match that of the tool definition. This process does the following:
                1. Fills in any default values for args not supplied
                2. Casts a raw dictionary to any arg that is a Param class

        Args:
            **kwargs: Recieves any arguements that will be verified and compiled

        Returns:
            dict[str, Any]: Dictionary of args that are ready to be run through the tool
        """
        compiled_args = {}

        for name, (type_, info) in self.parameters.items():
            if name in kwargs:
                type_info = analyze_type(type_, BaseModel)

                match type_info.category:
                    case TypeCategory.PRIMITIVE:
                        compiled_args[name] = kwargs[name]
                    case TypeCategory.LIST_PRIMITIVE:
                        compiled_args[name] = kwargs[name]
                    case TypeCategory.SUBCLASS:
                        param_args = kwargs[name]
                        compiled_args[name] = type_.model_validate(param_args)
                    case TypeCategory.LIST_SUBCLASS:
                        compiled_args[name] = [
                            type_info.inner_type.model_validate(param_args)
                            for param_args in kwargs[name]
                        ]
            else:
                compiled_args[name] = info.default

        return compiled_args


def tool(
    description: str,
    condition: Callable[[Any], bool] = None,
):
    """
    Decorator to mark a method as a callable tool.
    All methods marked with this descriptor **must** return a string

    Args:
        description(str): Description of the tool that will be read by the LLM
        condition(Callable): A callable that returns a boolean, determining when the tool
            will be included in the LLM inference call
    """

    def decorator(fn: Callable):
        # Check return type
        types = get_type_hints(fn)
        return_type = types.pop("return", None)
        if return_type != str and fn.__name__ != "__call__":
            raise InvalidToolDefinition(
                tool_name=fn.__name__, message="Method must have a return type of `str`"
            )

        # 2) grab default values
        sig = inspect.signature(fn)
        defaults = {
            param_name: param.default
            for param_name, param in sig.parameters.items()
            if param.default is not inspect._empty
        }

        params = {}

        for name, type_ in types.items():
            default = defaults.get(name, None)
            if isinstance(default, ParamInfo):
                params[name] = (type_, default)
            elif default is not None:
                params[name] = (type_, ParamInfo(default=default))
            else:
                params[name] = (type_, ParamInfo(required=True))

        fn.__tool_def__ = _ToolDefinition(
            name=fn.__name__,
            description=description or fn.__doc__ or "",
            parameters=params,
            condition=condition,
        )
        return fn

    return decorator
