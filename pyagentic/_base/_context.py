import functools
from typing import Any, Callable, Type, Self
from dataclasses import dataclass, make_dataclass, field, asdict

from pyagentic._base._exceptions import InvalidContextRefNotFoundInContext


@dataclass
class ContextItem:
    """
    A `ContextItem` is used to signal that a class attribute can be used in the context
    of an agent. Any of these values can be referenced in:
        - the agent's `instructions`
        - the agent's `input_template`
        - any `ContextRef` used in the Agent (e.g., in a `ParamInfo`)
        - the constructor of the Agent itself

    Args:
        default (Any, optional):
            The default value for this context item if no explicit value is provided.
            Defaults to `None`.
        default_factory (Callable[[], Any], optional):
            A zero-argument factory function that produces a default value.
            If provided, its return value takes precedence over `default`.
            Defaults to `None`.
    """

    default: Any = None
    default_factory: Callable = None

    def __post_init__(self):
        if not (self.default or self.default_factory):
            raise AttributeError("default or default_factory must be given")

    def get_default_value(self):
        if self.default_factory:
            return self.default_factory()
        else:
            return self.default


class computed_context:
    """
    Descriptor used to mark a method in an Agent as a computed context.

    Computed contexts work very similarly to Python's `@property` descriptor: they are
    re-computed each time they're accessed. When a computed context appears in:

      - the agent's `instructions`, its value will be refreshed on every call to the agent,
        updating the system message with the latest value.
      - the agent's `input_template`, its value will be refreshed each time a new user message is
        added, updating the prompt accordingly.

    Args:
        func (Callable[[Agent], Any]):
            The method on the Agent class that computes and returns the context value.
            It will be called with the agent instance each time the context is accessed.
    """

    def __init__(self, fget):
        functools.update_wrapper(self, fget)
        self.fget = fget
        self._is_context = True

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        # when accessed on the class, return the descriptor itself
        if instance is None:
            return self
        # when accessed on the instance, run the function against the *context* object
        # (we’ll inject this descriptor onto the Context class)
        return self.fget(instance)


@dataclass(repr=True)
class _AgentContext:
    """
    Base context class for agents; uses dataclass for auto-generated init/signature.
    """

    instructions: str
    input_template: str = None
    _messages: list = field(default_factory=list)

    def as_dict(self) -> dict:
        """
        Exports the context as a dictionary. This dictionary is not serialized, so
        any `ContextItem` or `computed_context` remains their original type.

        Returns:
            - dict: A dictionary containing all `ContextItem` and `computed_context`
                for later processing.
        """

        data = asdict(self)

        # tinject every computed_context value
        for name, attr in type(self).__dict__.items():
            if getattr(attr, "_is_context", False):
                data[name] = getattr(self, name)
        return data

    @property
    def system_message(self) -> str:
        """
        The current formatted system_message
        """
        # start with all the normal dataclass fields

        # now format your instruction template
        return self.instructions.format(**self.as_dict())

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        List of openai-ready messages with the most up-to-date system message
        """
        messages = self._messages.copy()
        messages.insert(0, {"role": "system", "content": self.system_message})
        return messages

    def add_user_message(self, message: str):
        """
        Add a user message to the message list. If a `input_template` is given then
            the message will be formatted in it as well as any context used in the template.

        To use the user message in the template, place the key `user_message`.

        Args:
            message(str): The user message to be added.
        """

        if self.input_template:
            data = self.as_dict()
            data["user_message"] = message
            content = self.input_template.format(**data)
        else:
            content = message
        self._messages.append({"role": "user", "content": content})

    def get(self, name: str) -> Any:
        """
        Retrieves an item from the context.

        Args:
            name(str): The name of the item

        Returns:
            Any: The item. If it is a computed context item, then it is computed upon retrieval.
        """
        try:
            return self.as_dict()[name]
        except KeyError:
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
        dc_fields = []  # for actual dataclass fields (ContextItem)
        namespace: dict[str, Any] = {"__module__": cls.__module__}

        for field_name, (type_, info) in ctx_map.items():
            if isinstance(info, ContextItem):
                # ---- your existing logic for setting defaults ----
                if info.default_factory is not None:
                    dc_def = field(default_factory=info.default_factory)
                else:
                    dc_def = field(default=info.default)
                dc_fields.append((field_name, type_, dc_def))

            elif isinstance(info, computed_context):
                # stick the descriptor straight into the namespace
                namespace[field_name] = info
                # also record its type for annotation
                namespace.setdefault("__annotations__", {})[field_name] = type_

            else:
                raise RuntimeError(f"Unexpected ctx_map entry for {field_name!r}: {info!r}")

        # now build the dataclass
        return make_dataclass(
            cls_name=f"{name}Context",
            fields=dc_fields,
            bases=(cls,),
            namespace=namespace,
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
