import inspect
import asyncio
from typing import Any, Type, Self, Optional, ClassVar
from pydantic import BaseModel, create_model, Field, PrivateAttr
from jinja2 import Template
from typing import Optional

from pyagentic._base._exceptions import InvalidStateRefNotFoundInState
from pyagentic._base._state import _StateDefinition
from pyagentic._base._policy import Policy
from pyagentic._base._event import Event, EventKind

from pyagentic.models.llm import Message


class _AgentState(BaseModel):
    """
    Base state class for agents; uses dataclass for auto-generated init/signature.
    """

    __policies__: ClassVar[dict[str, list[Policy]]]

    instructions: str
    input_template: Optional[str] = None
    _messages: list[Message] = PrivateAttr(default_factory=list)
    _instructions_template: Template = PrivateAttr(default_factory=lambda: Template(source=""))

    def model_post_init(self, state):
        self._instructions_template = Template(source=self.instructions)
        return super().model_post_init(state)

    def get_policies(self, state_name: str) -> list[Policy]:
        """Get all policies attached to a given state field."""
        return self.__policies__.get(state_name, [])

    async def _dispatch_policies(self, event: Event, state_name: str):
        """Run all policies registered for a given state field."""
        for policy in self.get_policies(state_name):
            handler = policy.handle_event
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)

    @property
    def recent_message(self) -> Message:
        return self._messages[-1]

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
            the message will be formatted in it as well as any state used in the template.

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
        Retrieves an item from the state.

        Args:
            name(str): The name of the item

        Returns:
            Any: The item. If it is a computed state item, then it is computed upon retrieval.
        """
        try:
            value = getattr(self, name)
        except KeyError:
            raise InvalidStateRefNotFoundInState(name)

        event = Event(kind=EventKind.GET, name=name, new_value=value)
        # Fire policies synchronously (usually safe for GET)
        asyncio.create_task(self._dispatch_policies(event, name))
        return value

    def set(self, name: str, value: Any):
        """
        Sets an item from the state.

        Args:
            name(str): The name of the item

        Returns:
            Any: The item. If it is a computed state item, then it is computed upon retrieval.
        """
        try:
            old_value = getattr(self, name, value)
            setattr(self, name, value)
        except KeyError:
            raise InvalidStateRefNotFoundInState(name)
        event = Event(kind=EventKind.SET, name=name, old_value=old_value, new_value=value)
        asyncio.create_task(self._dispatch_policies(event, name))

    @classmethod
    def make_state_model(
        cls, name: str, state_definitions: dict[str, _StateDefinition]
    ) -> Type[Self]:
        """
        Dynamically create a dataclass subclass with typed state fields.

        Args:
            name: base name for the new class (e.g. 'MyAgent').
            ctx_map: mapping of field name to (type, StateItem).

        Returns:
            A new dataclass type 'NameState'.
        """
        pydantic_fields = {}  # for actual dataclass fields (StateItem)

        for _name, definition in state_definitions.items():
            if isinstance(definition, _StateDefinition):
                # ---- your existing logic for setting defaults ----
                if definition.info.default_factory is not None:
                    pydantic_fields[_name] = (
                        Optional[definition.model],
                        Field(default_factory=definition.info.default_factory),
                    )
                elif definition.info.default is not None:
                    pydantic_fields[_name] = (
                        Optional[definition.model],
                        Field(default=definition.info.default),
                    )
                else:
                    pydantic_fields[_name] = (definition.model, ...)
            else:
                raise RuntimeError(f"Unexpected ctx_map entry for {_name!r}: {definition!r}")

        # now build the dataclass
        return create_model(f"AgentState[{name}]", __base__=cls, **pydantic_fields)
