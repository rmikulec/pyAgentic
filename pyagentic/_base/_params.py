from dataclasses import dataclass
from collections import defaultdict
from typing import get_type_hints, Any, List, Dict, Type

from typeguard import check_type, TypeCheckError

from pyagentic._base._resolver import ContextualMixin, MaybeContext
from pyagentic._base._context import _AgentContext
from pyagentic._utils._typing import analyze_type, TypeCategory

# simple mapping from Python types to JSON Schema/OpenAI types
_TYPE_MAP: Dict[Type[Any], str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


@dataclass
class ParamInfo(ContextualMixin):
    """
    Declare metadata for parameters in tool declarations and/or Parameter declarations.

    Attributes:
        description (str | None): A human-readable description of the parameter.
        required (bool): Whether this parameter must be provided by the user.
        default (Any): The default value to use if none is provided.
        values (list[str]): values to limit the input of this parameter. If used, the
            agent is forced to use on the the values in the list.

    Context-Ready Attributes:
        These attributes can be given a `ContextRef` to link them to any context items in
        the agent.

         - description
         - default
         - values
    """

    description: MaybeContext[str] = None
    required: bool = False
    default: MaybeContext[Any] = None
    values: MaybeContext[list[str]] = None


class Param:
    """
    Base class for defining structured parameters that can be converted
    into OpenAI-compatible JSON schema entries.

    Subclasses should declare class attributes with type annotations,
    optionally assigning a ParamInfo instance or a raw default value.

    On subclass creation, __attributes__ is populated mapping field names
    to (type, ParamInfo) pairs. Instances perform simple type-checked
    assignment and reject unknown fields.
    """

    __attributes__: dict[str, tuple[type, ParamInfo]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Inspect annotated attributes on the subclass and build a mapping
        of parameter definitions for OpenAI schema generation.
        """
        super().__init_subclass__(**kwargs)
        cls.__attributes__ = {}
        for name, type_ in get_type_hints(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            default = cls.__dict__.get(name, None)
            if isinstance(default, ParamInfo):
                cls.__attributes__[name] = (type_, default)
            elif default is not None:
                cls.__attributes__[name] = (type_, ParamInfo(default=default))
            else:
                cls.__attributes__[name] = (type_, ParamInfo())

    def __init__(self, **kwargs):
        """
        Instantiate a Param subclass by validating and assigning each
        annotated field, falling back to class-level defaults if absent.

        Raises:
            TypeError: if a provided value does not match the annotated type,
                       or if unexpected fields are passed.
        """
        for field_name, (field_type, field_info) in self.__attributes__.items():
            type_info = analyze_type(field_type, self.__class__.__bases__[0])
            value = kwargs.get(field_name, field_info.default)

            try:
                if not type_info.is_subclass:
                    check_type(value, field_type)
            except TypeCheckError:
                raise TypeError(f"Field '{field_name}' expected {field_type}, got {type(value)}")

            match type_info.category:

                case TypeCategory.PRIMITIVE:
                    setattr(self, field_name, value)
                case TypeCategory.LIST_PRIMITIVE:
                    setattr(self, field_name, value)
                case TypeCategory.SUBCLASS:
                    value = field_type(**value) if type(value) == dict else value
                    setattr(self, field_name, value)
                case TypeCategory.LIST_SUBCLASS:
                    listed_value = (
                        [type_info.inner_type(**param_kwargs) for param_kwargs in value]
                        if isinstance(value, list) and all(isinstance(v, dict) for v in value)
                        else value
                    )
                    setattr(self, field_name, listed_value)

        unexpected_args = [kwarg for kwarg in kwargs if kwarg not in self.__attributes__]
        if unexpected_args:
            unexpected = ", ".join(unexpected_args)
            raise TypeError(f"Unexpected fields for {self.__class__.__name__}: {unexpected}")

    def __repr__(self):
        vals = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
        return f"{type(self).__name__}({vals})"

    @classmethod
    def to_openai(cls, context: _AgentContext) -> List[Dict[str, Any]]:
        """
        Generate a JSON-schema-style dictionary suitable for OpenAI function
        parameter definitions.

        Returns:
            Dict[str, Any]: A schema object with keys:
              - "type": always "object"
              - "properties": mapping from field names to their OpenAI types
              - "required": list of names marked as required
        """
        properties: Dict[str, dict] = defaultdict(dict)
        required = []

        for name, (type_, info) in cls.__attributes__.items():
            resolved_info = info.resolve(context)
            properties[name]["type"] = _TYPE_MAP.get(type_, "string")
            if resolved_info.description:
                properties[name]["description"] = resolved_info.description
            if resolved_info.values:
                properties[name]["enum"] = resolved_info.values

            if resolved_info.required:
                required.append(name)

        return {"type": "object", "properties": dict(properties), "required": required}
