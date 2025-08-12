import inspect
import json
import openai
from functools import wraps
from typing import Callable, Any, TypeVar, ClassVar, Type, Self, dataclass_transform

from pyagentic.logging import get_logger
from pyagentic._base._params import ParamInfo
from pyagentic._base._tool import _ToolDefinition, tool
from pyagentic._base._context import ContextItem
from pyagentic._base._metaclasses import AgentMeta

from pyagentic.models.response import ToolResponse, AgentResponse
from pyagentic.updates import AiUpdate, Status, EmitUpdate, ToolUpdate

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
        - model (str): The OpenAI model that will be used for inference. Defaults to value
            found in `geo_assistant.config`
        - emitter (Callable): A function that will be called to recieve intermittant information
            about the agent's process. A common use case is that of a websocket, to be able
            to recieve information about the process as it is happening
        - max_call_depth(int): How many loops of tool calling the agent does on one run.
            Defaults to 1.
    """
    # Class Attributes
    __tool_defs__: ClassVar[dict[str, _ToolDefinition]]
    __context_attrs__: ClassVar[dict[str, tuple[TypeVar, ContextItem]]]
    __system_message__: ClassVar[str]
    __description__: ClassVar[str]
    __input_template__: ClassVar[str] = None
    __response_model__: ClassVar[Type[AgentResponse]] = None
    __tool_response_models__: ClassVar[dict[str, Type[ToolResponse]]]
    __linked_agents__: ClassVar[dict[str, Type[Self]]]
    __call_params__: ClassVar[dict[str, tuple[TypeVar, ParamInfo]]]

    # Base Attributes
    model: str
    api_key: str
    emitter: Callable[[Any], str] = None
    max_call_depth: int = 1

    def __post_init__(self):
        self.client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=self.api_key)

    async def _process_agent_call(self, tool_call) -> AgentResponse:
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")
        self.context._messages.append(tool_call)
        try:
            agent = getattr(self, tool_call.name)
            kwargs = json.loads(tool_call.arguments)
            response = await agent(**kwargs)
            result = f"Agent {tool_call.name}: {response.final_output}"
        except Exception as e:
            result = f"Agent `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide context, and ask if they want to try again."  # noqa E501
        self.context._messages.append(
            {"type": "function_call_output", "call_id": tool_call.call_id, "output": result}
        )
        return response

    async def _process_tool_call(self, tool_call, call_depth) -> ToolResponse:
        self.context._messages.append(tool_call)
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
        except Exception as e:
            logger.exception(e)
            result = f"Tool `{tool_call.name}` failed: {e}. Please kindly state to the user that is failed, provide context, and ask if they want to try again."  # noqa E501
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    ToolUpdate(status=Status.ERROR, tool_call=tool_call.name, tool_args=kwargs),
                )

        # Record output for LLM
        self.context._messages.append(
            {"type": "function_call_output", "call_id": tool_call.call_id, "output": result}
        )
        ToolResponseModel = self.__tool_response_models__[tool_call.name]
        return ToolResponseModel(
            raw_kwargs=tool_call.arguments, call_depth=call_depth, output=result, **compiled_args
        )

    async def _build_tool_defs(self) -> list[dict]:
        tool_defs = []
        # iterate through registered tools
        for tool_def in self.__tool_defs__.values():
            # Check if any of the tool params use a ContextRef
            # convert to openai schema
            tool_defs.append(tool_def.to_openai(self.context))
        for name, agent in self.__linked_agents__.items():
            tool_def = agent.get_tool_definition(name)
            tool_defs.append(tool_def.to_openai(self.context))
        return tool_defs

    async def run(self, input_: str) -> str:
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
        tool_defs = await self._build_tool_defs()

        # Tracking
        tool_responses: list = []
        agent_responses: list = []
        processed_call_ids: set[str] = set()

        # Optional status emit
        if self.emitter:
            await _safe_run(self.emitter, EmitUpdate(status=Status.GENERATING))

        depth = 0
        final_ai_message: str | None = None

        while depth < self.max_call_depth:
            # Ask the LLM what to do next (may return tool calls or final text)
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=self.context.messages,
                    tools=tool_defs,
                    max_tool_calls=5,
                    parallel_tool_calls=True,
                    tool_choice="auto",
                )
            except Exception as e:
                # Error handling mirrors your original
                logger.exception(e)
                if self.emitter:
                    await _safe_run(self.emitter, EmitUpdate(status=Status.ERROR))
                self.context._messages.append(
                    {"role": "assistant", "content": "Failed to generate a response"}
                )
                return f"OpenAI failed to generate a response: {e}"

            # Persist any reasoning traces (optional)
            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            if reasoning:
                self.context._messages.extend(reasoning)

            # Collect function/tool calls from this turn
            tool_calls = [rx for rx in response.output if rx.type == "function_call"]

            # If the model produced a final text (no calls), we can stop
            if not tool_calls:
                final_ai_message = response.output_text
                break

            # Execute tool/agent calls and append their results to context
            for tool_call in tool_calls:
                # Avoid double-processing if model re-sends the same id
                call_id = getattr(tool_call, "id", None)
                if call_id and call_id in processed_call_ids:
                    continue
                if call_id:
                    processed_call_ids.add(call_id)

                if tool_call.type != "function_call":
                    continue

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
        if final_ai_message is None:
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=self.context.messages,
                )
                final_ai_message = response.output_text
            except Exception as e:
                logger.exception(e)
                if self.emitter:
                    await _safe_run(
                        self.emitter, EmitUpdate(status=Status.ERROR, message="Generation failed")
                    )
                self.context._messages.append(
                    {"role": "assistant", "content": "Failed to generate a response"}
                )
                return f"OpenAI failed to generate a response: {e}"

        # Finalize
        self.context._messages.append({"role": "assistant", "content": final_ai_message})

        if self.emitter:
            await _safe_run(
                self.emitter, AiUpdate(status=Status.SUCCEDED, message=final_ai_message)
            )

        response_fields = {"final_output": final_ai_message}
        if self.__tool_defs__:
            print(tool_responses)
            response_fields["tool_responses"] = tool_responses
        if self.__linked_agents__:
            response_fields["agent_responses"] = agent_responses

        return self.__response_model__(**response_fields)

    async def __call__(self, user_input: str):
        return await self.run(input_=user_input)

    @classmethod
    def get_tool_definition(cls, name: str) -> _ToolDefinition:
        desc = getattr(cls, "__description__", "") or ""

        # fresh async wrapper so each class gets its own function object
        @wraps(cls.__call__)
        async def _invoke(self, *args, **kwargs):
            return await cls.__call__(self, *args, **kwargs)

        td = tool(desc)(_invoke).__tool_def__  # decorator attaches metadata to the wrapper
        td.name = name
        return td
