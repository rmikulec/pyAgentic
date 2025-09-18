import inspect
import json
from functools import wraps
from dataclasses import field
from typing import Callable, Any, TypeVar, ClassVar, Type, Self, dataclass_transform, Optional

from pydantic import BaseModel

from pyagentic.logging import get_logger
from pyagentic._base._params import ParamInfo
from pyagentic._base._tool import _ToolDefinition, tool
from pyagentic._base._context import ContextItem
from pyagentic._base._metaclasses import AgentMeta
from pyagentic._base._exceptions import InvalidLLMSetup

from pyagentic.models.response import ToolResponse, AgentResponse
from pyagentic.models.llm import Message, ToolCall, LLMResponse
from pyagentic.models.tracing import SpanKind

from pyagentic.updates import AiUpdate, Status, EmitUpdate, ToolUpdate
from pyagentic.llm._provider import LLMProvider
from pyagentic.llm import LLMProviders
from pyagentic.tracing._tracer import AgentTracer, traced
from pyagentic.tracing import BasicTracer


logger = get_logger(__name__)


async def _safe_run(fn, *args, **kwargs):
    """
    Helper function to always run a function, async or not
    """
    if inspect.iscoroutinefunction(fn):
        result = await fn(*args, **kwargs)
    else:
        result = fn(*args, **kwargs)
    return result


@dataclass_transform(field_specifiers=(ContextItem,))
class AgentExtension:
    """Inherit this in any mixin that contributes fields to the Agent __init__."""

    __annotations__: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Merge annotations from all AgentExtension bases (oldest first),
        # then let the subclass' own annotations win on key conflicts.
        merged: dict[str, Any] = {}
        for base in reversed(cls.__mro__[1:]):  # skip cls, walk up towards object
            if issubclass(base, AgentExtension):
                ann = getattr(base, "__annotations__", None)
                if ann:
                    merged.update(ann)

        merged.update(getattr(cls, "__annotations__", {}))
        # Assign a fresh dict so we don't mutate a base class' annotations
        cls.__annotations__ = dict(merged)


class Agent(metaclass=AgentMeta):
    __abstract_base__ = ClassVar[True]
    """
    Base agent class to be extended in order to define a new Agent

    Agent defintion requires the use of special function decorators in order to define the
        behavior of the agent.

        - @tool: Declares a method as a tool, allowing the agent to use it

    Agents also have default arguements that can be declared on initiation

    Args:
        - model (str): Model used for inference. Please provide in the following format:
            `<provider>::<model>`. For example, to use GPT-5, set the model to `openai::gpt-5`
            When using model, the `api_key` must also be supplied. If model or api key not given,
            then the user must specify a `provider`
        - api_key (str): The matching api key to the model string provided.
        - provider (LLMProvider): A provider instance allowing the user to configure provider's
            client deeper. Providers must have a `model` and `api_key`, then any subsequent args
            will be sent to the provider's base client. This will override `model` and `api_key`.
        - emitter (Callable): A function that will be called to recieve intermittant information
            about the agent's process. A common use case is that of a websocket, to be able
            to recieve information about the process as it is happening
        - max_call_depth (int): How many loops of tool calling the agent does on one run.
            Defaults to 1.

    Examples:

        With a model string

        ```
        agent = MyAgent(
            model="openai::gpt-4o,
            api_key=MY_API_KEY
        )
        ```

        With a provider

        ```
        from pyagentic.llm import OpenAIProvider

        agent = MyAgent(
            provider=OpenAIProvier(
                model="gpt-4o",
                api_key=MY_API_KEY,
                base_url="http://localhost:8000",
                max_retries=5
            )
        )
        ```
    """
    # Immutable Class Attributes
    __tool_defs__: ClassVar[dict[str, _ToolDefinition]]
    __context_attrs__: ClassVar[dict[str, tuple[TypeVar, ContextItem]]]
    __linked_agents__: ClassVar[dict[str, Type[Self]]]

    # User-set Class Attributes
    __system_message__: ClassVar[str]
    __description__: ClassVar[str]
    __input_template__: ClassVar[str] = None
    __response_format__: ClassVar[Type[BaseModel]] = None

    # Accesible Class Attributes
    __response_model__: ClassVar[Type[AgentResponse]] = None
    __tool_response_models__: ClassVar[dict[str, Type[ToolResponse]]]
    __call_params__: ClassVar[dict[str, tuple[TypeVar, ParamInfo]]]

    # Base Attributes
    model: str = None
    api_key: str = None
    provider: LLMProvider = None
    emitter: Callable[[Any], str] = None
    tracer: AgentTracer = None
    max_call_depth: int = 1


    def _check_llm_provider(self):
        if (not self.model and not self.api_key) and (not self.provider):
            raise InvalidLLMSetup(reason="no-provider")

        if self.provider:
            return

        try:
            values = self.model.split("::")
            assert len(values) == 2
        except AssertionError:
            raise InvalidLLMSetup(model=self.model, reason="invalid-format")

        provider, model_name = values

        try:
            assert provider.upper() in LLMProviders.__members__

            self.provider = LLMProviders[provider.upper()].value(
                model=model_name,
                api_key=self.api_key,
            )
        except AssertionError:
            valid_providers = [
                key.lower() for key in LLMProviders.__members__.keys() if key != "_MOCK"
            ]
            raise InvalidLLMSetup(
                model=self.model, reason="provider-not-found", valid_providers=valid_providers
            )

        if self.__response_format__ and not self.provider.__supports_structured_outputs__:
            raise Exception("Response format is not supported with this provider")

        if self.__tool_defs__ and not self.provider.__supports_tool_calls__:
            raise Exception("Tools are not supported with this provider")

    def __post_init__(self):
        self._check_llm_provider()

        if not self.tracer:
            self.tracer = BasicTracer()

    @traced(SpanKind.INFERENCE)
    async def _process_llm_inference(
        self,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        self.tracer.set_attributes(
            system_message=self.context.system_message,
            user_message=self.context.recent_message
        )
        """
        Processes LLM inferences by adding appropriate messages to the context, generating a
            response using the provider and handling errors.
        """
        try:
            response = await self.provider.generate(
                context=self.context,
                tool_defs=tool_defs,
                response_format=self.__response_format__,
                **kwargs,
            )
            return response
        except Exception as e:
            # Error handling mirrors your original
            logger.exception(e)
            if self.emitter:
                await _safe_run(self.emitter, EmitUpdate(status=Status.ERROR))
            self.context._messages.append(
                Message(role="assistant", content="Failed to generate a response")
            )
            return LLMResponse(text=f"The LLM failed to generate a response: {e}", tool_calls=[])

    @traced(SpanKind.AGENT)
    async def _process_agent_call(self, tool_call: ToolCall) -> AgentResponse:
        """
        Processes linked agents by adding appropriate messages to the context, calling the agent,
            handling errors, and creating an agent response.
        """
        self.tracer.set_attributes(
            agent=tool_call.name,
        )
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")
        self.context._messages.append(self.provider.to_tool_call_message(tool_call))
        agent = getattr(self, tool_call.name)
        agent.tracer = self.tracer
        try:
            kwargs = json.loads(tool_call.arguments)
            self.tracer.set_attributes(**kwargs)
            response = await agent(**kwargs)
            result = f"Agent {tool_call.name}: {response.final_output}"
            self.tracer.set_attributes(result=response.model_dump())
        except Exception as e:
            self.tracer.record_exception(str(e))
            result = f"Agent `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide context, and ask if they want to try again."  # noqa E501
            response = AgentResponse(final_output=result, provider_info=agent.provider._info)
        self.context._messages.append(
            self.provider.to_tool_call_result_message(result=result, id_=tool_call.id)
        )
        return response

    @traced(SpanKind.TOOL)
    async def _process_tool_call(self, tool_call: ToolCall, call_depth: int) -> ToolResponse:
        """
        Processes a tool call by adding appropriate messages to the context, calling the tool,
            handling errors, and creating the tool response
        """
        self.tracer.set_attributes(
            **tool_call.__dict__
        )
        self.context._messages.append(self.provider.to_tool_call_message(tool_call))
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")
        # Lookup the bound method
        try:
            tool_def = self.__tool_defs__[tool_call.name]
            handler = getattr(self, tool_call.name)
        except KeyError:
            return f"Tool {tool_call.name} not found"
        kwargs = json.loads(tool_call.arguments)

        # Run the tool, emitting updates
        try:
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(
                        status=Status.PROCESSING, tool_call=tool_call.name, tool_args=kwargs
                    ),
                )

            compiled_args = tool_def.compile_args(**kwargs)
            result = await _safe_run(handler, **compiled_args)
            self.tracer.set_attributes(result=result) 
        except Exception as e:
            self.tracer.record_exception(str(e))
            logger.exception(e)
            result = f"Tool `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide context, and ask if they want to try again."  # noqa E501
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(status=Status.ERROR, tool_call=tool_call.name, tool_args=kwargs),
                )

        # Record output for LLM
        self.context._messages.append(
            self.provider.to_tool_call_result_message(result=result, id_=tool_call.id)
        )
        ToolResponseModel = self.__tool_response_models__[tool_call.name]
        return ToolResponseModel(
            raw_kwargs=tool_call.arguments, call_depth=call_depth, output=result, **compiled_args
        )

    async def _get_tool_defs(self) -> list[_ToolDefinition]:
        """
        Creates a list of tool definitions from any methods decorated with "@tool" and any agents
            linked to the parent agent
        """
        tool_defs = []
        # iterate through registered tools
        for tool_def in self.__tool_defs__.values():
            # Check if any of the tool params use a ContextRef
            # convert to openai schema
            tool_defs.append(tool_def)
        for name, agent in self.__linked_agents__.items():
            tool_def = agent.get_tool_definition(name)
            tool_defs.append(tool_def)
        return tool_defs

    async def run(self, input_: str) -> str:
        async with self.tracer.agent(
            name=f"{self.__class__.__name__}.run",
            model=getattr(self.provider, "model", None),
            input_len=len(input_) if input_ else 0,
            max_call_depth=self.max_call_depth,
        ):
            """
            Run the agent with any given input

            Parameters:
                input_(str): The user input for the agent to process

            Returns:
                str: The output of the agent
            """
            # Prime context with the user message
            self.context.add_user_message(input_)

            # Build tools once (if yours can change each turn, move inside the loop)
            tool_defs = await self._get_tool_defs()

            # Tracking
            tool_responses: list = []
            agent_responses: list = []
            processed_call_ids: set[str] = set()

            # Optional status emit
            if self.emitter:
                await _safe_run(self.emitter, EmitUpdate(status=Status.GENERATING))

            depth = 0
            final_ai_output: str | None = None

            while depth < self.max_call_depth:
                # Ask the LLM what to do next (may return tool calls or final text)
                response = await self._process_llm_inference(tool_defs=tool_defs)

                # If the model produced a final text (no calls), we can stop
                if not response.tool_calls:
                    final_ai_output = response.parsed if response.parsed else response.text
                    self.context._messages.append(Message(role="assistant", content=response.text))
                    break

                # Execute tool/agent calls and append their results to context
                for tool_call in response.tool_calls:
                    # Avoid double-processing if model re-sends the same id
                    if tool_call.id and tool_call.id in processed_call_ids:
                        continue

                    processed_call_ids.add(tool_call.id)

                    # Route to tools vs linked agents
                    if tool_call.name in self.__tool_defs__:
                        result = await self._process_tool_call(tool_call, call_depth=depth)
                        tool_responses.append(result)

                    elif tool_call.name in self.__linked_agents__:
                        result = await self._process_agent_call(tool_call)
                        agent_responses.append(result)

                # After executing tools, advance depth and loop to let the LLM react
                depth += 1

            # If we hit depth limit and still don’t have a final text, ask once naturally
            if final_ai_output is None:
                response = await self._process_llm_inference()
                final_ai_output = response.parsed if response.parsed else response.text

            if self.emitter:
                await _safe_run(
                    self.emitter, AiUpdate(status=Status.SUCCEDED, message=final_ai_output)
                )

            response_fields = {"final_output": final_ai_output, "provider_info": self.provider._info}
            if self.__tool_defs__:
                response_fields["tool_responses"] = tool_responses
            if self.__linked_agents__:
                response_fields["agent_responses"] = agent_responses

            return self.__response_model__(**response_fields)

    async def __call__(self, user_input: str):
        return await self.run(input_=user_input)

    @classmethod
    def get_tool_definition(cls, name: str) -> _ToolDefinition:
        """
        Creates and returns a tool definition for the agent.

        This is used for linked agents, allowing each agent to be linked to another by using it
            as a tools

        Args:
            name (str): The name of the linked agent

        Returns:
            _ToolDefinition: A pyagentic tool defintion to be injected into a generate call
        """
        desc = getattr(cls, "__description__", "") or ""

        # fresh async wrapper so each class gets its own function object
        @wraps(cls.__call__)
        async def _invoke(self, *args, **kwargs):
            return await cls.__call__(self, *args, **kwargs)

        td = tool(desc)(_invoke).__tool_def__  # decorator attaches metadata to the wrapper
        td.name = name
        return td
