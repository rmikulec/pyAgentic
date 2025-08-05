from typing import Any, Callable, Type, Self
from dataclasses import dataclass, make_dataclass, field, asdict

from objective_agents._base._exceptions import InvalidContextRefNotFoundInContext


@dataclass
class ContextItem:
    default: Any = None
    default_factory: Callable = None

    def get_default_value(self):
        if self.default_factory:
            return self.default_factory()
        else:
            return self.default


class computed_context(property):
    def __init__(self, fget):
        # initialize the property
        super().__init__(fget)
        # mark it as a context‐provider
        self._is_context = True


@dataclass(repr=True)
class _AgentContext:
    """
    Base context class for agents; uses dataclass for auto-generated init/signature.
    """

    instructions: str
    _messages: list = field(default_factory=list)

    @property
    def system_message(self):
        return self.instructions.format(**asdict(self))

    @property
    def messages(self) -> list[dict[str, str]]:
        messages = self._messages.copy()
        messages.insert(0, {"role": "system", "content": self.system_message})
        return messages

    def add_message(self, role, content):
        self._messages.append({"role": role, "content": content})

    def get(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise InvalidContextRefNotFoundInContext(name)

    @classmethod
    def make_ctx_class(cls, name: str, ctx_map: dict[str, tuple[Type[Any], Any]]) -> Type[Self]:
        """
        Dynamically create a dataclass subclass with typed context fields.

        Args:
            name: base name for the new class (e.g. 'MyAgent').
            ctx_map: mapping of field name to (type, ContextItem).

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
