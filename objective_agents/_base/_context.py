from typing import Any, Callable, Type, Self
from dataclasses import dataclass, make_dataclass, field


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


@dataclass(repr=True)
class _AgentContext:
    """
    Base context class for agents; uses dataclass for auto-generated init/signature.
    """

    instructions: str
    messages: list = field(init=False)

    def __post_init__(self):
        # Initialize messages and context store
        object.__setattr__(self, "messages", [{"role": "system", "content": self.instructions}])
        object.__setattr__(self, "_contexts", {})

    @property
    def system_message(self) -> dict:
        return self.messages[0]

    @system_message.setter
    def system_message(self, value: dict):
        self.messages[0] = value

    @classmethod
    def make_ctx_class(cls, name: str, ctx_map: dict[str, tuple[Type[Any], Any]]) -> Type[Self]:
        """
        Dynamically create a dataclass subclass with typed context fields.

        Args:
            name: base name for the new class (e.g. 'MyAgent').
            ctx_map: mapping of field name to (type, default or default_info).

        Returns:
            A new dataclass type 'NameContext'.
        """
        fields = []
        for key, (typ, default_info) in ctx_map.items():
            # Determine default value or factory
            if default_info.default_factory:
                field_ = field(default_factory=default_info.default_factory)
            elif default_info.default:
                field_ = field(default=default_info.default)
            else:
                field_ = field()

            # create a standard field
            fields.append((key, typ, field_))

        # Create and return the new dataclass
        return make_dataclass(
            cls_name=f"{name}Context",
            fields=fields,
            bases=(cls,),
            namespace={"__module__": cls.__module__},
        )


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
