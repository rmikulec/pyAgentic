import inspect
from typing import Callable, Any, TypeVar, get_type_hints
from collections import defaultdict

from objective_agents._base._params import Param, ParamInfo, _TYPE_MAP


class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, tuple[TypeVar, ParamInfo]],
        pre: Callable[[dict], dict] = None,
        post: Callable[[Any], Any] = None,
        condition: Callable[[Any], bool] = None,
    ):
        self.name: str = name
        self.description: str = description
        self.parameters: dict[str, tuple[TypeVar, ParamInfo]] = parameters
        self.pre = pre
        self.post = post
        self.condition = condition

    def to_openai(self):
        params = defaultdict(dict)
        required = []

        for name, attr in self.parameters.items():
            type_, default = attr

            if issubclass(type_, Param):
                params[name] = type_.to_openai()
            else:
                params[name] = {"type": _TYPE_MAP.get(type_, "string")}
            if isinstance(default, ParamInfo):
                if default.description:
                    params[name]["description"] = default.description
                if default.required:
                    required.append(name)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": dict(params)},
            "required": required,
        }

    def compile_args(self, **kwargs):
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
    preprocess: Callable[[dict], dict] = None,
    postprocess: Callable[[Any], Any] = None,
    condition: Callable[[Any], bool] = None,
):
    """
    Decorator to mark a method as a callable tool.
    """

    def decorator(fn: Callable):
        # Check return type
        types = get_type_hints(fn)
        return_type = types.pop("return", None)
        if return_type != str:
            raise Exception(
                f"Tool defined function - {fn.__name__} - must have a return type of `str`"
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

        fn.__tool_def__ = ToolDefinition(
            name=fn.__name__,
            description=description or fn.__doc__ or "",
            parameters=params,
            pre=preprocess,
            post=postprocess,
            condition=condition,
        )
        return fn

    return decorator
