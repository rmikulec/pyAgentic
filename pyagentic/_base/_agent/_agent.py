import inspect
import json
from functools import wraps
from typing import Callable, Any, TypeVar, ClassVar, Type, Self, dataclass_transform, Optional

from pydantic import BaseModel, ValidationError

from pyagentic.logging import get_logger
from pyagentic._base._tool import _ToolDefinition, tool
from pyagentic._base._state import _StateDefinition
from pyagentic._base._metaclasses import AgentMeta
from pyagentic._base._exceptions import InvalidLLMSetup, InvalidToolDefinition
from pyagentic._base._info import _SpecInfo
from pyagentic._base._agent._agent_state import _AgentState

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
    Helper function to run a function regardless of whether it's async or sync.

    Args:
        fn (Callable): The function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Any: The result of the function execution
    """
    if inspect.iscoroutinefunction(fn):
        result = await fn(*args, **kwargs)
    else:
        result = fn(*args, **kwargs)
    return result


@dataclass_transform(field_specifiers=(_SpecInfo,))
class AgentExtension:
    """
    Base class for creating reusable agent mixins that add tools, state, and behaviors.

    AgentExtension allows you to package common functionality (tools, state fields, linked agents)
    into reusable components that can be mixed into multiple agent classes. This promotes code
    reuse and modular agent design.

    Example:
        ```python
        class LoggingMixin(AgentExtension):
            logs: State[list[str]] = spec.State(default_factory=list)

            @tool("Record a log message")
            def log(self, message: str) -> str:
                self.state.logs.append(message)
                return f"Logged: {message}"

        class MyAgent(BaseAgent, LoggingMixin):
            __system_message__ = "You are a helpful agent with logging"
        ```
    """

    __annotations__: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Merges annotations from all AgentExtension bases in MRO order.
        Subclass annotations override parent annotations on conflicts.

        Args:
            **kwargs: Additional keyword arguments passed to parent __init_subclass__
        """
        super().__init_subclass__(**kwargs)

        # Merge annotations from all AgentExtension bases (oldest first),
        # then let the subclass' own annotations win on key conflicts.
        merged: dict[str, Any] = {}
        for base in reversed(cls.__mro__[1:]):  # Skip cls itself, walk up towards object
            if issubclass(base, AgentExtension):
                ann = getattr(base, "__annotations__", None)
                if ann:
                    merged.update(ann)

        merged.update(getattr(cls, "__annotations__", {}))
        # Assign a fresh dict so we don't mutate a base class' annotations
        cls.__annotations__ = dict(merged)


class BaseAgent(metaclass=AgentMeta):
    """
    Base agent class to be extended to define new LLM-powered agents.

    Agent definition requires the use of special decorators and class attributes:
      - @tool: Declares a method as a tool callable by the LLM
      - __system_message__: Required class attribute defining the agent's system prompt
      - __description__: Optional description used when agent is linked to another agent
      - __input_template__: Optional template for formatting user input
      - __response_format__: Optional Pydantic model for structured output

    Args:
        model (str, optional): Model used for inference in format `<provider>::<model>`.
            For example: `openai::gpt-4o`. Requires `api_key` to also be provided.
        api_key (str, optional): API key matching the model provider
        provider (LLMProvider, optional): Pre-configured provider instance. Overrides
            `model` and `api_key` if provided.
        emitter (Callable, optional): Callback function to receive real-time updates
            about the agent's execution (useful for WebSocket streaming)
        tracer (AgentTracer, optional): Tracer instance for observability. Defaults
            to BasicTracer if not provided.
        max_call_depth (int): Maximum number of tool calling loops per run. Defaults to 1.

    Examples:
        With a model string:
        ```python
        agent = MyAgent(
            model="openai::gpt-4o",
            api_key=MY_API_KEY
        )
        ```

        With a provider:
        ```python
        from pyagentic.llm import OpenAIProvider

        agent = MyAgent(
            provider=OpenAIProvider(
                model="gpt-4o",
                api_key=MY_API_KEY,
                base_url="http://localhost:8000",
                max_retries=5
            )
        )
        ```
    """

    __abstract_base__ = ClassVar[True]
    # Immutable Class Attributes (set by metaclass)
    __tool_defs__: ClassVar[dict[str, _ToolDefinition]]  # Registered @tool methods
    __state_defs__: ClassVar[dict[str, _StateDefinition]]  # State field definitions
    __linked_agents__: ClassVar[dict[str, Type[Self]]]  # Linked agent definitions

    # User-set Class Attributes (defined in subclass)
    __system_message__: ClassVar[str]  # Required: system prompt for the agent
    __description__: ClassVar[str]  # Optional: description for linked agents
    __input_template__: ClassVar[str] = None  # Optional: template for user input
    __response_format__: ClassVar[Type[BaseModel]] = None  # Optional: structured output format

    # Generated Class Attributes (built by metaclass)
    __response_model__: ClassVar[Type[AgentResponse]] = None  # Pydantic response model
    __state_class__: ClassVar[Type[_AgentState]] = None  # Generated state class
    __tool_response_models__: ClassVar[dict[str, Type[ToolResponse]]]  # Tool response models

    # Instance Attributes
    model: str = None
    api_key: str = None
    provider: LLMProvider = None
    emitter: Callable[[Any], str] = None
    tracer: AgentTracer = None
    max_call_depth: int = 1

    def _check_llm_provider(self):
        """
        Validates and initializes the LLM provider configuration.

        Checks if either (model + api_key) or provider is supplied, then creates
        the provider instance from the model string if needed.

        Raises:
            InvalidLLMSetup: If provider configuration is invalid or incomplete
            Exception: If response_format or tools are not supported by the provider
        """
        # Ensure either (model + api_key) or provider is provided
        if (not self.model and not self.api_key) and (not self.provider):
            raise InvalidLLMSetup(reason="no-provider")

        # If provider is already set, skip initialization
        if self.provider:
            return

        # Parse model string in format "provider::model_name"
        try:
            values = self.model.split("::")
            assert len(values) == 2
        except AssertionError:
            raise InvalidLLMSetup(model=self.model, reason="invalid-format")

        provider, model_name = values

        # Look up and instantiate the provider
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

        # Verify provider capabilities match agent requirements
        if self.__response_format__ and not self.provider.__supports_structured_outputs__:
            raise Exception("Response format is not supported with this provider")

        if self.__tool_defs__ and not self.provider.__supports_tool_calls__:
            raise Exception("Tools are not supported with this provider")

    def __post_init__(self):
        """
        Post-initialization hook called after agent instance is created.

        Override this method to add custom initialization logic like setting up
        connections, loading data, or initializing resources. The base implementation
        validates the LLM provider and sets up a default tracer.

        Example:
            ```python
            class DatabaseAgent(BaseAgent):
                __system_message__ = "You query databases"
                connection: Optional[Connection] = None

                def __post_init__(self):
                    super().__post_init__()  # Always call parent
                    # Custom initialization
                    self.connection = create_db_connection()
            ```
        """
        self._check_llm_provider()

        # Use BasicTracer as default if no tracer provided
        if not self.tracer:
            self.tracer = BasicTracer()

    @property
    def agent_reference(self) -> dict:
        """
        Builds a nested dictionary of state from this agent and all linked agents.

        This is used for resolving StateRef references that may point to state
        in the current agent ("self") or in any linked agent.

        Returns:
            dict: Dictionary with "self" key for current agent state, plus keys
                for each linked agent containing their agent_reference recursively
        """
        linked_agent_references = {}

        # Recursively build references for all linked agents
        for name in self.__linked_agents__.keys():
            linked: BaseAgent = getattr(self, name)
            linked_agent_references[name] = linked.agent_reference

        return {"self": self.state.model_dump(), **linked_agent_references}

    @traced(SpanKind.INFERENCE)
    async def _process_llm_inference(
        self,
        *,
        tool_defs: Optional[list[_ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Processes LLM inference by sending messages to the provider and handling the response.

        Args:
            tool_defs (list[_ToolDefinition], optional): List of tool definitions to send to LLM
            **kwargs: Additional arguments passed to provider.generate()

        Returns:
            LLMResponse: The response from the LLM containing text and/or tool calls
        """
        self.tracer.set_attributes(
            system_message=self.state.system_message, user_message=self.state.recent_message
        )

        try:
            response = await self.provider.generate(
                state=self.state,
                tool_defs=tool_defs,
                response_format=self.__response_format__,
                **kwargs,
            )
            self.tracer.set_attributes(
                usage_details=response.usage.model_dump(), model=self.provider._model
            )
            return response
        except Exception as e:
            # Handle inference errors gracefully
            logger.exception(e)
            if self.emitter:
                await _safe_run(self.emitter, EmitUpdate(status=Status.ERROR))
            # Add error message to conversation history
            self.state._messages.append(
                Message(role="assistant", content="Failed to generate a response")
            )
            return LLMResponse(text=f"The LLM failed to generate a response: {e}", tool_calls=[])

    @traced(SpanKind.AGENT)
    async def _process_agent_call(self, tool_call: ToolCall) -> AgentResponse:
        """
        Processes a linked agent call by invoking the agent and handling its response.

        Args:
            tool_call (ToolCall): The tool call representing the agent invocation

        Returns:
            AgentResponse: The response from the linked agent
        """
        self.tracer.set_attributes(
            agent=tool_call.name,
        )
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")

        # Add tool call message to conversation history
        self.state._messages.append(self.provider.to_tool_call_message(tool_call))

        # Get the linked agent instance and share tracer
        agent = getattr(self, tool_call.name)
        agent.tracer = self.tracer

        try:
            # Parse arguments and call the linked agent
            kwargs = json.loads(tool_call.arguments)
            self.tracer.set_attributes(**kwargs)
            response = await agent(**kwargs)
            result = f"Agent {tool_call.name}: {response.final_output}"
            self.tracer.set_attributes(result=response.model_dump())
        except Exception as e:
            # Handle agent execution errors
            self.tracer.record_exception(str(e))
            result = f"Agent `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide state, and ask if they want to try again."  # noqa E501
            response = AgentResponse(final_output=result, provider_info=agent.provider._info)

        # Add agent result to conversation history
        self.state._messages.append(
            self.provider.to_tool_call_result_message(result=result, id_=tool_call.id)
        )
        return response

    @traced(SpanKind.TOOL)
    async def _process_tool_call(self, tool_call: ToolCall, call_depth: int) -> ToolResponse:
        """
        Processes a tool call by executing the tool method and handling the result.

        Args:
            tool_call (ToolCall): The tool call to execute
            call_depth (int): Current depth in the tool calling loop

        Returns:
            ToolResponse: The response from the tool execution
        """
        # Add tool call message to conversation history
        self.state._messages.append(self.provider.to_tool_call_message(tool_call))
        self.tracer.set_attributes(**tool_call.__dict__)
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")

        # Look up the tool definition and bound method
        try:
            tool_def = self.__tool_defs__[tool_call.name]
            handler = getattr(self, tool_call.name)
        except KeyError:
            return f"Tool {tool_call.name} not found"

        # Parse and validate tool arguments
        kwargs = json.loads(tool_call.arguments)
        try:
            # Compile args converts raw JSON to typed parameters (e.g., dict -> Pydantic models)
            compiled_args = tool_def.compile_args(**kwargs)
        except ValidationError as e:
            # Handle validation errors for tool arguments
            result = f"Function Args were invalid: {str(e)}"
            compiled_args = {}
            if self.emitter:
                self.tracer.record_exception(str(e))
                logger.exception(e)
                if self.emitter:
                    await _safe_run(
                        self.emitter,
                        ToolUpdate(
                            status=Status.ERROR, tool_call=tool_call.name, tool_args=kwargs
                        ),
                    )

        # Execute the tool, emitting status updates
        try:
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(
                        status=Status.PROCESSING, tool_call=tool_call.name, tool_args=kwargs
                    ),
                )
            if compiled_args:
                result = await _safe_run(handler, **compiled_args)
                result = str(result)
                self.tracer.set_attributes(result=result)
        except TypeError as e:
            self.tracer.record_exception(str(e))
            logger.exception(e)
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(status=Status.ERROR, tool_call=tool_call.name, tool_args=kwargs),
                )
            raise InvalidToolDefinition(
                tool_name=tool_call.name,
                message=f"Tool must have a serializable return type; {tool_def.return_type} failed to be casted to a string.",
            )
        except Exception as e:
            # Handle any other tool execution errors
            self.tracer.record_exception(str(e))
            logger.exception(e)
            result = f"Tool `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide state, and ask if they want to try again."  # noqa E501
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(status=Status.ERROR, tool_call=tool_call.name, tool_args=kwargs),
                )

        # Add tool result to conversation history for LLM
        self.state._messages.append(
            self.provider.to_tool_call_result_message(result=result, id_=tool_call.id)
        )

        # Build and return the structured tool response
        ToolResponseModel = self.__tool_response_models__[tool_call.name]
        return ToolResponseModel(
            raw_kwargs=tool_call.arguments, call_depth=call_depth, output=result, **compiled_args
        )

    async def _get_tool_defs(self) -> list[_ToolDefinition]:
        """
        Builds a list of tool definitions from @tool methods and linked agents.

        Resolves any StateRef references in tool parameters using the current agent_reference,
        allowing tools to dynamically reference state values.

        Returns:
            list[_ToolDefinition]: List of resolved tool definitions ready for LLM
        """
        tool_defs = []

        # Add all @tool decorated methods
        for tool_def in self.__tool_defs__.values():
            # Resolve StateRefs in parameters (e.g., ref.self.user_name -> actual value)
            tool_defs.append(tool_def.resolve(self.agent_reference))

        # Add linked agents as tools
        for name, agent in self.__linked_agents__.items():
            tool_def = agent.get_tool_definition(name)
            tool_defs.append(tool_def.resolve(self.agent_reference))

        return tool_defs

    async def run(self, input_: str) -> str:
        """
        Main execution loop for the agent. Processes user input through multiple rounds
        of LLM inference and tool/agent calls until completion or max_call_depth reached.

        The agent follows an agentic loop pattern:
        1. Send user input and conversation history to the LLM
        2. LLM decides to either call tools or respond with final output
        3. If tools are called, execute them and feed results back to LLM
        4. Repeat until max_call_depth reached or LLM provides final output

        Args:
            input_ (str): The user input/query for the agent to process

        Returns:
            AgentResponse: Structured response containing:
                - final_output: The final text or structured output from the LLM
                - state: Current agent state after execution
                - tool_responses: List of all tool calls and their outputs
                - provider_info: Information about the LLM provider used

        Example:
            ```python
            agent = MyAgent(model="openai::gpt-4o", api_key=API_KEY)
            response = await agent.run("What's the weather in San Francisco?")
            print(response.final_output)  # LLM's final answer
            print(response.tool_responses)  # Tools that were called
            ```
        """
        async with self.tracer.agent(
            name=f"{self.__class__.__name__}.run",
            model=getattr(self.provider, "model", None),
            input_len=len(input_) if input_ else 0,
            max_call_depth=self.max_call_depth,
        ):
            self.tracer.set_attributes(input=input_)

            # Add user message to conversation state
            self.state.add_user_message(input_)

            # Build tool definitions (including linked agents as tools)
            tool_defs = await self._get_tool_defs()

            # Track responses and prevent duplicate processing
            tool_responses: list = []
            agent_responses: list = []
            processed_call_ids: set[str] = set()

            # Emit initial status
            if self.emitter:
                await _safe_run(self.emitter, EmitUpdate(status=Status.GENERATING))

            # Main agentic loop: LLM -> Tools -> LLM -> ...
            depth = 0
            final_ai_output: str | None = None

            while depth < self.max_call_depth:
                # Ask the LLM what to do next (may return tool calls or final text)
                response = await self._process_llm_inference(tool_defs=tool_defs)

                # If the model produced final text without tool calls, we're done
                if not response.tool_calls:
                    final_ai_output = response.parsed if response.parsed else response.text
                    self.state._messages.append(Message(role="assistant", content=response.text))
                    break

                # Execute all tool/agent calls from this response
                for tool_call in response.tool_calls:
                    # Skip if we've already processed this call (prevents duplicates)
                    if tool_call.id and tool_call.id in processed_call_ids:
                        continue

                    processed_call_ids.add(tool_call.id)

                    # Route to either @tool methods or linked agents
                    if tool_call.name in self.__tool_defs__:
                        result = await self._process_tool_call(tool_call, call_depth=depth)
                        tool_responses.append(result)

                    elif tool_call.name in self.__linked_agents__:
                        result = await self._process_agent_call(tool_call)
                        agent_responses.append(result)

                # Increment depth and continue loop (LLM will see tool results next iteration)
                depth += 1

            # If we exhausted max_call_depth without final text, get one more response
            if final_ai_output is None:
                response = await self._process_llm_inference()
                final_ai_output = response.parsed if response.parsed else response.text

            # Emit final success status
            if self.emitter:
                await _safe_run(
                    self.emitter, AiUpdate(status=Status.SUCCEDED, message=final_ai_output)
                )

            # Build the structured response
            response_fields = {
                "final_output": final_ai_output,
                "state": self.state,
                "provider_info": self.provider._info,
            }
            # Include tool/agent responses if any were called
            if self.__tool_defs__:
                response_fields["tool_responses"] = tool_responses
            if self.__linked_agents__:
                response_fields["agent_responses"] = agent_responses

            response = self.__response_model__(**response_fields)
            self.tracer.set_attributes(output=response)
            return response

    async def __call__(self, user_input: str):
        """
        Allows the agent to be called directly as a function.

        Args:
            user_input (str): The user input to process

        Returns:
            AgentResponse: The agent's response
        """
        return await self.run(input_=user_input)

    def __getattribute__(self, name):
        """
        Custom attribute getter that redirects state field access to the state object.

        This allows accessing state fields directly on the agent (e.g., agent.user_name)
        instead of requiring agent.state.user_name. State field access goes through
        the state.get() method, which applies GET policies.

        Args:
            name (str): The attribute name to retrieve

        Returns:
            Any: The attribute value
        """
        # First check if it's a BaseAgent attribute (methods, class vars, etc.)
        if hasattr(BaseAgent, name):
            return super().__getattribute__(name)

        # Check if it's a state field - if so, redirect to state.get()
        __state_defs__ = super().__getattribute__("__state_defs__")
        if name in __state_defs__:
            state = super().__getattribute__("state")
            return state.get(name)

        # Otherwise, use normal attribute access
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        """
        Custom attribute setter that redirects state field writes to the state object.

        This allows setting state fields directly on the agent (e.g., agent.user_name = "Bob")
        instead of requiring agent.state.set("user_name", "Bob"). State field writes go
        through the state.set() method, which applies SET policies.

        Args:
            name (str): The attribute name to set
            value (Any): The value to set
        """
        # If it's a BaseAgent attribute, set it normally
        if hasattr(BaseAgent, name):
            super().__setattr__(name, value)
            return

        # Check if it's a state field - if so, redirect to state.set()
        __state_defs__ = super().__getattribute__("__state_defs__")
        if name in __state_defs__:
            state = super().__getattribute__("state")
            state.set(name, value)
        else:
            # Otherwise, set as normal instance attribute
            super().__setattr__(name, value)

    @classmethod
    def get_tool_definition(cls, name: str) -> _ToolDefinition:
        """
        Creates a tool definition for this agent class to be used as a linked agent.

        When an agent is linked to another agent, it appears as a tool that can be
        called by the LLM. This method generates the tool definition with the agent's
        __description__ as the tool description and the agent's __call__ signature
        as the tool parameters.

        Args:
            name (str): The name to use for this agent when it appears as a tool

        Returns:
            _ToolDefinition: A tool definition that can be sent to the LLM
        """
        desc = getattr(cls, "__description__", "") or ""

        # Create a fresh async wrapper function for this agent class
        # Each class needs its own function object for the @tool decorator
        @wraps(cls.__call__)
        async def _invoke(self, *args, **kwargs):
            return await cls.__call__(self, *args, **kwargs)

        # Apply @tool decorator to extract parameter info and create definition
        td = tool(desc)(_invoke).__tool_def__
        td.name = name
        return td
