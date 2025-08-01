from dataclasses import dataclass
from collections import defaultdict
from typing import get_type_hints, Any, List, Dict, Type

# simple mapping from Python types to JSON Schema/OpenAI types
_TYPE_MAP: Dict[Type[Any], str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


@dataclass
class ParamInfo:
    description: str = None
    required: bool = False
    default: Any = None


class Param:
    __attributes__: dict[str, tuple[type, ParamInfo]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__attributes__ = {}
        for name, type_ in get_type_hints(cls).items():
            default = cls.__dict__.get(name, None)
            if isinstance(default, ParamInfo):
                cls.__attributes__[name] = (type_, default)
            elif default is not None:
                cls.__attributes__[name] = (type_, ParamInfo(default=default))
            else:
                cls.__attributes__[name] = (type_, ParamInfo())

    def __init__(self, **kwargs):
        cls = type(self)
        hints = get_type_hints(cls)

        # Assign annotated fields
        for name, typ in hints.items():
            if name in kwargs:
                value = kwargs.pop(name)
                # simple type check
                if not isinstance(value, typ) and value is not None:
                    raise TypeError(f"Field '{name}' expected {typ}, got {type(value)}")
                setattr(self, name, value)
            else:
                # use class‐level default if given, else None
                attr = getattr(cls, name, None)
                if isinstance(attr, ParamInfo):
                    default = attr.default
                else:
                    default = attr
                setattr(self, name, default)

        if kwargs:
            unexpected = ", ".join(kwargs)
            raise TypeError(f"Unexpected fields for {cls.__name__}: {unexpected}")

    def __repr__(self):
        vals = ", ".join(f"{k}={v!r}" for k, v in self.dict().items())
        return f"{type(self).__name__}({vals})"

    @classmethod
    def to_openai(cls) -> List[Dict[str, Any]]:
        """
        Produce a list of OpenAI‐compatible parameter schemas for this class.

        Each entry is a dict with:
          - name: parameter name
          - schema: {"type": ...}
          - required: whether no default was provided
          - default: the default value (if any)
        """
        properties: Dict[str, dict] = defaultdict(dict)
        required = []

        for name, (type_, info) in cls.__attributes__.items():
            properties[name]["type"] = _TYPE_MAP.get(type_, "string")
            if info.description:
                properties[name]["description"] = info.description

            if info.required:
                required.append(name)

        return {"type": "object", "properties": dict(properties), "required": required}
