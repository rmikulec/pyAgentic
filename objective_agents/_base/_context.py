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


class AgentContext:
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


class ContextRef:
    """
    A placeholder pointing at some attribute or method
    on the agent’s context, to be resolved at schema-build time.
    """

    def __init__(self, path: str):
        self.path = path  # dot-notation into agent.context

    def resolve(self, context: AgentContext) -> Any:
        val = context
        for part in self.path.split("."):
            val = getattr(val, part)
            # if it’s wrapped in our Context helper, drill into .value
            if hasattr(val, "value"):
                val = val.value
        return val() if callable(val) else val
