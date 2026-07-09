import asyncio
import threading
from typing import Any, Type, Self, Optional, ClassVar
from pydantic import BaseModel, ConfigDict, create_model, Field, PrivateAttr
from jinja2 import Template
from typing import Optional, Literal, Callable
from transitions import Machine

from pyagentic._base._exceptions import InvalidStateRefNotFoundInState
from pyagentic._base._state import _StateDefinition
from pyagentic._base._prompts import PromptRef, PromptSource, _inline_source
from pyagentic.policies._policy import Policy
from pyagentic.policies._events import Event, EventKind, GetEvent, SetEvent, CompileEvent
from pyagentic.policies._list import PolicyList

from pyagentic.models.llm import Message, SystemMessage, UserMessage, UsageInfo


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
        ("on", EventKind.APPEND): "on_append",
        ("background", EventKind.APPEND): "background_append",
    }
    __policies__: ClassVar[dict[str, list[Policy]]] = {}
    __agent_name__: ClassVar[str] = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: str | PromptRef
    input_template: Optional[str] = "{{ user_message }}"

    _machine: Machine = PrivateAttr(default=None)
    _messages: list[Message] = PrivateAttr(default_factory=list)
    _context: list[Message] = PrivateAttr(default_factory=list)
    _last_usage: Optional[UsageInfo] = PrivateAttr(default=None)
    _prompt_source: Optional[PromptSource] = PrivateAttr(default=None)
    _instructions_template: Template = PrivateAttr(default_factory=lambda: Template(source=""))
    _input_template: Template = PrivateAttr(
        default_factory=lambda: Template(source="{{ user_message }}")
    )

    def _build_phase_machine(self, phases: list[tuple[str, str, Callable]]) -> Machine:
        if not phases:
            return None

        states = []
        for source, dest, _ in phases:
            if source not in states:
                states.append(source)
            if dest not in states:
                states.append(dest)

        machine = Machine(states=states, initial=states[0])

        for source, dest, _ in phases:
            machine.add_transition(
                trigger=f"{source}_to_{dest}",
                source=source,
                dest=dest,
            )

        self._machine = machine

    def _update_state_machine(self, phases):
        for source, dest, condition in phases:
            with self._state_lock:
                if condition(self) and self.phase == source:
                    trigger = f"{source}_to_{dest}"
                    getattr(self._machine, trigger)()

    def model_post_init(self, state):
        # Instructions declared as a PromptRef resolve here, at instantiation, so
        # every new agent instance (and every fork) picks up the latest prompt.
        # Plain strings get an "inline" PromptSource so every run still carries
        # versioned prompt metadata.
        if isinstance(self.instructions, PromptRef):
            self._prompt_source = self.instructions.resolve()
            self.instructions = self._prompt_source.text
        else:
            self._prompt_source = _inline_source(
                self.instructions, source=self.__agent_name__ or type(self).__name__
            )

        self._instructions_template = Template(source=self.instructions)
        if self.input_template:
            self._input_template = Template(source=self.input_template)

        self._state_lock = threading.Lock()

        # Bind the message context to the "messages" policy key so appends and
        # compiles route through any attached message policies
        self._context = PolicyList(self._context, state=self, name="messages")

        # Wrap list-valued state fields that have policies attached, so in-place
        # mutations (append, etc.) trigger the policy pipeline
        for field_name, policies in self.__policies__.items():
            if field_name == "messages" or not policies:
                continue
            value = getattr(self, field_name, None)
            if isinstance(value, list) and not isinstance(value, PolicyList):
                setattr(self, field_name, PolicyList(value, state=self, name=field_name))

        return super().model_post_init(state)

    def get_policies(self, state_name: str) -> list[Policy]:
        """
        Get all policies attached to a given state field.

        Args:
            state_name (str): The name of the state field

        Returns:
            list[Policy]: List of policies for the field, or empty list if none
        """
        return self.__policies__.get(state_name) or []

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

            handler = getattr(policy, handler_name, None)
            if handler is None:
                continue

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

            handler = getattr(policy, handler_name, None)
            if handler is None:
                continue

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
        self._schedule_background(event)
        return transformed_value

    def _schedule_background(self, event: Event):
        """Schedule background policies on the running loop; skip in sync contexts."""
        if not self.get_policies(event.name):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._dispatch_policies(event, "background"))

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

        # Keep policied list fields observable so in-place mutations keep
        # routing through the policy pipeline
        if (
            isinstance(final_value, list)
            and not isinstance(final_value, PolicyList)
            and self.get_policies(name)
        ):
            final_value = PolicyList(final_value, state=self, name=name)

        with self._state_lock:
            setattr(self, name, final_value)
        self._schedule_background(event)

    async def _run_compile(self, event: CompileEvent, items: list) -> list:
        """
        Run the async on_compile pipeline for one list. Each policy may return a
        transformed list (or None for no change); exceptions are logged and skipped.
        """
        value = items
        for policy in self.get_policies(event.name):
            handler = getattr(policy, "on_compile", None)
            if handler is None:
                continue
            try:
                new_value = await handler(event, value)
                if new_value is not None:
                    value = new_value
            except Exception as e:
                print(f"[PolicyError] {policy.__class__.__name__}.on_compile failed: {e}")
        return value

    async def compile_context(self, provider) -> list[Message]:
        """
        Compiles the message context (and any policied list-valued state fields)
        by running attached policies' on_compile handlers. Called right before
        each LLM inference; the transformed lists are written back, so policies
        like eviction and compaction persist their effect.

        Args:
            provider: The LLM provider about to be called, exposed to policies
                via CompileEvent (e.g. for summarization calls).

        Returns:
            list[Message]: The compiled message context providers should send.
        """
        # Compile policied list-valued state fields first so the system message
        # (rendered from state) reflects their final contents
        for field_name, policies in self.__policies__.items():
            if field_name == "messages" or not policies:
                continue
            value = getattr(self, field_name, None)
            if not isinstance(value, list):
                continue
            event = CompileEvent(
                name=field_name,
                value=list(value),
                provider=provider,
                last_usage=self._last_usage,
                state=self,
            )
            result = await self._run_compile(event, list(value))
            with self._state_lock:
                if isinstance(value, PolicyList):
                    value._set_contents(result)
                else:
                    setattr(self, field_name, result)

        if self.get_policies("messages"):
            event = CompileEvent(
                name="messages",
                value=list(self._context),
                provider=provider,
                last_usage=self._last_usage,
                system_message=self.system_message,
                state=self,
            )
            result = await self._run_compile(event, list(self._context))
            with self._state_lock:
                self._context._set_contents(result)

        return self._context

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
        model = create_model(f"AgentState[{name}]", __base__=cls, **pydantic_fields)
        # Used as the PromptSource `source` for inline (plain string) instructions
        model.__agent_name__ = name
        return model

    @property
    def phase(self) -> str:
        return self._machine.state if self._machine else None

    @property
    def recent_message(self) -> Message:
        """
        Returns the most recent message in the message history.

        Returns:
            Message: The last message added to the state
        """
        return self._messages[-1]

    @property
    def prompt_source(self) -> PromptSource:
        """
        Returns metadata about the prompt behind the instructions: the engine it
        was loaded from (source_type "local", ...), or an "inline" source named
        after the agent for instructions declared as a plain string.

        Returns:
            PromptSource: The prompt text plus source, version, and load-time metadata.
        """
        return self._prompt_source

    @property
    def system_message(self) -> str:
        """
        Returns the current formatted system message with state fields interpolated.

        Returns:
            str: The rendered system message with state values
        """
        # start with all the normal dataclass fields

        # now format your instruction template
        if self.phase:
            return self._instructions_template.render(phase=self.phase, **self.model_dump())
        else:
            return self._instructions_template.render(**self.model_dump())

    @property
    def messages(self) -> list[Message]:
        """
        Returns the compiled message context with the most up-to-date system message.

        This is the policy-shaped view providers consume; see `raw_messages` for the
        untouched history.

        Returns:
            list[Message]: Compiled message context with system message prepended
        """
        messages = self._context.copy()
        messages.insert(0, SystemMessage(content=self.system_message))
        return messages

    @property
    def raw_messages(self) -> list[Message]:
        """
        Returns the raw, append-only message history with the most up-to-date
        system message. Policies never modify this list; use it for debugging
        and auditing what actually happened.

        Returns:
            list[Message]: Raw message history with system message prepended
        """
        messages = self._messages.copy()
        messages.insert(0, SystemMessage(content=self.system_message))
        return messages

    def add_message(self, message: Message) -> None:
        """
        Adds a message to both histories: the raw append-only log (`_messages`)
        and the working context (`_context`) that providers consume.

        Args:
            message (Message): The message to record.
        """
        self._messages.append(message)
        self._context.append(message)

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
            if self.phase:
                content = self._input_template.render(phase=self.phase, **data)
            else:
                content = self._input_template.render(**data)
        else:
            content = message
        self.add_message(UserMessage(content=content))
