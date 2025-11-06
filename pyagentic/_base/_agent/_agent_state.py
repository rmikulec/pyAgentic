import asyncio
import threading
from typing import Any, Type, Self, Optional, ClassVar
from pydantic import BaseModel, create_model, Field, PrivateAttr
from jinja2 import Template
from typing import Optional, Literal

from pyagentic._base._exceptions import InvalidStateRefNotFoundInState
from pyagentic._base._state import _StateDefinition
from pyagentic.policies._policy import Policy
from pyagentic.policies._events import Event, EventKind, GetEvent, SetEvent

from pyagentic.models.llm import Message


class _AgentState(BaseModel):
    """
    Base state class for agents, uses Pydantic for auto-generated init and validation.
    Manages state fields, policies, and message history for agent execution.
    """

    _policy_handlers = {
        ("on", EventKind.GET): "on_get",
        ("background", EventKind.GET): "background_get",
        ("on", EventKind.SET): "on_set",
        ("background", EventKind.SET): "background_set",
    }
    _state_lock: ClassVar[threading.Lock] = PrivateAttr(default_factory=threading.Lock)
    __policies__: ClassVar[dict[str, list[Policy]]]

    instructions: str
    input_template: Optional[str] = None
    _messages: list[Message] = PrivateAttr(default_factory=list)
    _instructions_template: Template = PrivateAttr(default_factory=lambda: Template(source=""))
    _input_template: Template = PrivateAttr(default_factory=lambda: Template(source=""))

    def model_post_init(self, state):
        self._instructions_template = Template(source=self.instructions)
        self._input_template = Template(source=self.input_template)
        return super().model_post_init(state)

    def get_policies(self, state_name: str) -> list[Policy]:
        """
        Get all policies attached to a given state field.

        Args:
            state_name (str): The name of the state field

        Returns:
            list[Policy]: List of policies for the field, or empty list if none
        """
        return self.__policies__.get(state_name, [])

    def _run_policies(self, event: Event, policy_type: Literal["on", "background"]) -> Any:
        """
        Run synchronous policies as a transformation pipeline.
        Each policy receives (event, value) and returns the next transformed value.

        Args:
            event (Event): The event being processed
            policy_type (Literal["on", "background"]): The type of policy handler to run

        Returns:
            Any: The transformed value after all policies have been applied

        Raises:
            ValueError: If no handler exists for the policy type and event kind combination
            RuntimeError: If called with non-synchronous policy type
        """
        value = event.value

        policies = self.get_policies(event.name)

        if not policies:
            return value

        for policy in self.get_policies(event.name):
            handler_name = self._policy_handlers.get((policy_type, event.kind))
            if not handler_name:
                raise ValueError(f"No handler for ({policy_type}, {event.kind})")

            handler = getattr(policy, handler_name)

            if policy_type != "on":
                raise RuntimeError("_run_policies is only for synchronous 'on' policies")

            try:
                new_value = handler(event, value)
                if new_value is not None:
                    value = new_value
            except Exception as e:
                print(f"[PolicyError] {policy.__class__.__name__}.{handler_name} failed: {e}")

        return value

    async def _dispatch_policies(self, event: Event, policy_type: Literal["on", "background"]):
        """
        Run async/background policies as a transformation pipeline.

        Each async policy receives (event, value) and may return a new value.
        The final result is written back to the state safely under a lock.

        Args:
            event (Event): The event being processed
            policy_type (Literal["on", "background"]): The type of policy handler to run

        Raises:
            ValueError: If no handler exists for the policy type and event kind combination
        """
        value = event.value

        policies = self.get_policies(event.name)

        if not policies:
            return

        for policy in self.get_policies(event.name):
            handler_name = self._policy_handlers.get((policy_type, event.kind))
            if not handler_name:
                raise ValueError(f"No handler for ({policy_type}, {event.kind})")

            handler = getattr(policy, handler_name)

            try:
                maybe_new_value = await handler(event, value)
                if maybe_new_value is not None:
                    value = maybe_new_value
            except Exception as e:
                print(f"[PolicyError] {policy.__class__.__name__}.{handler_name} failed: {e}")

        # If any policy returned an updated value, apply it to state
        if value is not None:
            with self._state_lock:
                setattr(self, event.name, value)

    def get(self, name: str) -> Any:
        """
        Retrieves and transforms a value via GET policies.

        Args:
            name (str): The name of the state field to retrieve

        Returns:
            Any: The retrieved and transformed value

        Raises:
            InvalidStateRefNotFoundInState: If the state field does not exist
        """
        try:
            stored_value = getattr(self, name)
        except AttributeError:
            raise InvalidStateRefNotFoundInState(name)

        event = GetEvent(name=name, value=stored_value)
        transformed_value = self._run_policies(event, "on")
        asyncio.create_task(self._dispatch_policies(event, "background"))
        return transformed_value

    def set(self, name: str, value: Any):
        """
        Sets a value via SET policies (transform, validate, store).

        Args:
            name (str): The name of the state field to set
            value (Any): The new value to set

        Raises:
            InvalidStateRefNotFoundInState: If the state field does not exist
        """
        try:
            previous = getattr(self, name)
        except AttributeError:
            raise InvalidStateRefNotFoundInState(name)

        event = SetEvent(name=name, previous=previous, value=value)
        final_value = self._run_policies(event, "on")

        setattr(self, name, final_value)
        asyncio.create_task(self._dispatch_policies(event, "background"))

    @classmethod
    def make_state_model(
        cls, name: str, state_definitions: dict[str, _StateDefinition]
    ) -> Type[Self]:
        """
        Dynamically creates a Pydantic model subclass with typed state fields.

        Args:
            name (str): Base name for the new class (e.g. 'MyAgent')
            state_definitions (dict[str, _StateDefinition]): Mapping of field names to state definitions

        Returns:
            Type[Self]: A new Pydantic model type named 'AgentState[name]'

        Raises:
            RuntimeError: If an unexpected entry type is found in state_definitions
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

    @property
    def recent_message(self) -> Message:
        """
        Returns the most recent message in the message history.

        Returns:
            Message: The last message added to the state
        """
        return self._messages[-1]

    @property
    def system_message(self) -> str:
        """
        Returns the current formatted system message with state fields interpolated.

        Returns:
            str: The rendered system message with state values
        """
        # start with all the normal dataclass fields

        # now format your instruction template
        return self._instructions_template.render(**self.model_dump())

    @property
    def messages(self) -> list[Message]:
        """
        Returns a list of OpenAI-ready messages with the most up-to-date system message.

        Returns:
            list[Message]: Complete message list with system message prepended
        """
        messages = self._messages.copy()
        messages.insert(0, Message(role="system", content=self.system_message))
        return messages

    def add_user_message(self, message: str):
        """
        Adds a user message to the message list. If an `input_template` is given then
            the message will be formatted in it as well as any state used in the template.

        To use the user message in the template, place the key `user_message`.

        Args:
            message (str): The user message to be added
        """

        if self.input_template:
            data = self.model_dump()
            data["user_message"] = message
            content = self._input_template.render(**data)
        else:
            content = message
        self._messages.append(Message(role="user", content=content))
