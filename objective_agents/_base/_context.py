from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class ContextItem:
    default: Any = None
    default_factory: Callable = None

    def get_default_value(self):
        if self.default_factory:
            return self.default_factory()
        else:
            return self.default


def computed_context(func: callable):
    func._is_context = True
    return func


class _AgentContext:
    def __init__(self, instructions):
        self.messages = [{"role": "system", "content": instructions}]
        self._contexts: dict[str, Any] = {}

    def __getattr__(self, name):
        if name in self._contexts:
            return self._contexts[name]
        raise AttributeError(f"{self.__class__.__name__!r} context has no item {name!r}")

    def __setattr__(self, name, value):
        # internal attrs go in __dict__; everything else into _data
        if name in ("contexts", "messages", "_contexts"):
            object.__setattr__(self, name, value)
        else:
            # get the real _data dict without recursion
            object.__getattribute__(self, "_contexts")[name] = value
            self.__class__.__annotations__[name] = type(value)

    @property
    def system_message(self):
        if self.messages:
            return self.messages[0]

    @system_message.setter
    def system_message(self, value):
        if self.messages:
            self.messages[0] = value

    def add(self, name: str, value: Any):
        self._contexts[name] = value
        self.__class__.__annotations__[name] = type(value)

    @classmethod
    def make_ctx_class(cls, name: str, ctx_map: dict[str, tuple[type, Any]]):
        # Build a new namespace with module, annotations, and defaults
        namespace: dict[str, Any] = {"__module__": cls.__module__, "__annotations__": {}}

        for key, (typ, default_info) in ctx_map.items():
            # 1) Record the type hint
            namespace["__annotations__"][key] = typ
            # 2) Set a real class attribute default
            default_val = (
                default_info.get_default_value()
                if hasattr(default_info, "get_default_value")
                else default_info
            )
            namespace[key] = default_val

        # Create a new subclass named e.g. "MyAgentContext"
        return type(f"{name}Context", (cls,), namespace)


class ContextRef:
    """
    A placeholder pointing at some attribute or method
    on the agent’s context, to be resolved at schema-build time.
    """

    def __init__(self, path: str):
        self.path = path  # dot-notation into agent.context

    def resolve(self, context: _AgentContext) -> Any:
        val = context
        for part in self.path.split("."):
            val = getattr(val, part)
            # if it’s wrapped in our Context helper, drill into .value
            if hasattr(val, "value"):
                val = val.value
        return val() if callable(val) else val
