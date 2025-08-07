import inspect
import json
import openai
from typing import Callable, Any, TypeVar, ClassVar

from pyagentic.logging import get_logger
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._context import ContextItem
from pyagentic._base._metaclasses import AgentMeta
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


class Agent(metaclass=AgentMeta):
    __abstract_base__ = True
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
    """
    # Class Attributes
    __tool_defs__: ClassVar[dict[str, _ToolDefinition]]
    __context_attrs__: ClassVar[dict[str, tuple[TypeVar, ContextItem]]]
    __system_message__: ClassVar[str]
    __input_template__: ClassVar[str] = None

    # Base Attributes
    model: str
    api_key: str
    emitter: Callable[[Any], str] = None

    def __post_init__(self):
        self.client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=self.api_key)

    async def _process_tool_call(self, tool_call) -> bool:
        if tool_call.type != "function_call":
            return False
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
        return True

    async def _build_tool_defs(self) -> list[dict]:
        tool_defs = []
        # iterate through registered tools
        for tool_def in self.__tool_defs__.values():
            # Check if any of the tool params use a ContextRef
            # convert to openai schema
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

        # Generate and insert the new system message
        self.context.add_user_message(input_)

        # Create the tool list
        tool_defs = await self._build_tool_defs()

        # Begin the first pass on generating a response from openai
        if self.emitter:
            await _safe_run(
                self.emitter,
                EmitUpdate(
                    status=Status.GENERATING,
                ),
            )
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=self.context.messages,
                tools=tool_defs,
            )
            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            tool_calls = [rx for rx in response.output if rx.type == "function_call"]
        except Exception as e:
            logger.exception(e)
            # On failure, emit an udpate, update the messages, and return a standard message
            if self.emitter:
                await _safe_run(
                    self.emitter,
                    EmitUpdate(
                        status=Status.ERROR,
                    ),
                )
            self.context._messages.append(
                {"role": "assistant", "content": "Failed to generate a response"}
            )
            return f"OpenAI failed to generate a response: {e}"

        if reasoning:
            self.context._messages.extend(reasoning)

        # Dispatch any tool calls
        made_calls = False
        for tool_call in tool_calls:
            made_calls = made_calls or (await self._process_tool_call(tool_call))

        # If tools ran, re-invoke LLM for natural reply
        if made_calls:
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=self.context.messages,
                )
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

        # Parse and finalize the Ai Response
        ai_message = response.output_text

        self.context._messages.append({"role": "assistant", "content": ai_message})

        if self.emitter:
            await _safe_run(self.emitter, AiUpdate(status=Status.SUCCEDED, message=ai_message))

        return ai_message
