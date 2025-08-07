import inspect
from typing import Callable, Any, TypeVar, get_type_hints
from collections import defaultdict

from pyagentic._base._params import Param, ParamInfo, _TYPE_MAP
from pyagentic._base._context import _AgentContext
from pyagentic._base._exceptions import ToolDeclarationFailed


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

    def to_openai(self, context: _AgentContext) -> dict:
        """
        Converts the definition to an "openai-ready" dictionary

        Returns:
            dict: A openai dictionary for tool calling
        """
        params = defaultdict(dict)
        required = []

        for name, attr in self.parameters.items():
            type_, default = attr

            if issubclass(type_, Param):
                params[name] = type_.to_openai(context)
            else:
                params[name] = {"type": _TYPE_MAP.get(type_, "string")}
            if isinstance(default, ParamInfo):
                resolved_default = default.resolve(context)
                if resolved_default.description:
                    params[name]["description"] = resolved_default.description
                if resolved_default.required:
                    required.append(name)
                if resolved_default.values:
                    params[name]["enum"] = resolved_default.values

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": dict(params)},
            "required": required,
        }

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
                if issubclass(type_, Param):
                    param_args = kwargs[name]
                    compiled_args[name] = type_(**param_args)
                else:
                    compiled_args[name] = kwargs[name]
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
        if return_type != str:
            raise ToolDeclarationFailed(
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
                params[name] = (type_, ParamInfo())

        fn.__tool_def__ = _ToolDefinition(
            name=fn.__name__,
            description=description or fn.__doc__ or "",
            parameters=params,
            condition=condition,
        )
        return fn

    return decorator
