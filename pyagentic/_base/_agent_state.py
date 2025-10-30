from typing import Any, Type, Self, Optional
from pydantic import BaseModel, create_model, Field, computed_field, PrivateAttr
from jinja2 import Template

from pyagentic._base._exceptions import InvalidContextRefNotFoundInContext
from pyagentic._base._state import _StateDefinition
from pyagentic.models.llm import Message


class _AgentState(BaseModel):
    """
    Base context class for agents; uses dataclass for auto-generated init/signature.
    """

    instructions: str
    input_template: Optional[str] = None
    _messages: list[Message] = PrivateAttr(default_factory=list)
    _instructions_template: Template = PrivateAttr(default_factory=lambda: Template(source=""))

    def model_post_init(self, context):
        self._instructions_template = Template(source=self.instructions)
        return super().model_post_init(context)

    @property
    def system_message(self) -> str:
        """
        The current formatted system_message
        """
        # start with all the normal dataclass fields

        # now format your instruction template
        return self._instructions_template.render(**self.model_dump())

    @property
    def messages(self) -> list[Message]:
        """
        List of openai-ready messages with the most up-to-date system message
        """
        messages = self._messages.copy()
        messages.insert(0, Message(role="system", content=self.system_message))
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
        self._messages.append(Message(role="user", content=content))

    def get(self, name: str) -> Any:
        """
        Retrieves an item from the context.

        Args:
            name(str): The name of the item

        Returns:
            Any: The item. If it is a computed context item, then it is computed upon retrieval.
        """
        try:
            return getattr(self, name)
        except KeyError:
            raise InvalidContextRefNotFoundInContext(name)

    @classmethod
    def make_state_model(
        cls, name: str, state_definitions: dict[str, _StateDefinition]
    ) -> Type[Self]:
        """
        Dynamically create a dataclass subclass with typed context fields.

        Args:
            name: base name for the new class (e.g. 'MyAgent').
            ctx_map: mapping of field name to (type, ContextItem).

        Returns:
            A new dataclass type 'NameContext'.
        """
        pydantic_fields = {}  # for actual dataclass fields (ContextItem)

        for _name, definition in state_definitions.items():
            if isinstance(definition, _StateDefinition):
                # ---- your existing logic for setting defaults ----
                if definition.info.default_factory is not None:
                    pydantic_fields[_name] = (
                        definition.model,
                        Field(default_factory=definition.info.default_factory),
                    )
                elif definition.info.default is not None:
                    pydantic_fields[_name] = (
                        definition.model,
                        Field(default=definition.info.default),
                    )
                else:
                    pydantic_fields[_name] = (definition.model, ...)
            else:
                raise RuntimeError(f"Unexpected ctx_map entry for {_name!r}: {definition!r}")

        # now build the dataclass
        return create_model(f"AgentState[{name}]", __base__=cls, **pydantic_fields)
