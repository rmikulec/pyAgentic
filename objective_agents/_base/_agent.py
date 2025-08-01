import inspect
import json
import openai
from typing import Callable

from objective_agents.logging import get_logger
from objective_agents._base._tool import _ToolDefinition
from objective_agents.updates import AiUpdate, Status, EmitUpdate, ToolUpdate

logger = get_logger(__name__)


class SystemMessageNotDeclared(Exception):
    def __init__(self):
        super().__init__(
            "System message not declared on agent. One function must use `@system_message`"
        )


async def _safe_run(fn, *args, **kwargs):
    """
    Helper function to always run a function, async or not
    """
    if inspect.iscoroutinefunction(fn):
        result = await fn(*args, **kwargs)
    else:
        result = fn(*args, **kwargs)
    return result


class Agent:
    """
    Base agent class to be extended in order to define a new Agent

    Agent defintion requires the use of special function decorators in order to define the
        behavior of the agent.

        - @system_message: A mandatory function defining the system message. This is dynamic,
            and will be called on each iteration of `chat`.
        - @prechat: A function that will preprocess the user's message before sending it through
            the pipeline
        - @postchat: A function that will process the ai_message before returning it
        - @tool: Declares a method as a tool, allowing the agent to use it

    Agents also have default arguements that can be declared on initiation

    Args:
        - model (str): The OpenAI model that will be used for inference. Defaults to value
            found in `geo_assistant.config`
        - emitter (Callable): A function that will be called to recieve intermittant information
            about the agent's process. A common use case is that of a websocket, to be able
            to recieve information about the process as it is happening
    """

    _tools: dict[str, _ToolDefinition] = {}

    def __init__(self, model: str, api_key: str, emitter: Callable[[AiUpdate], None] = None):
        self.client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=api_key)
        self.model: str = model
        self.emitter = emitter
        self.messages: list[dict] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._tools = {}

        for name, attr in cls.__dict__.items():
            if hasattr(attr, "__tool_def__"):
                cls._tools[name] = attr.__tool_def__

    async def _process_tool_call(self, tool_call) -> bool:
        if tool_call.type != "function_call":
            return False
        self.messages.append(tool_call)
        logger.info(f"Calling {tool_call.name} with kwargs: {tool_call.arguments}")
        # Lookup the bound method
        try:
            tool_def = self._tools[tool_call.name]
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
        self.messages.append(
            {"type": "function_call_output", "call_id": tool_call.call_id, "output": result}
        )
        return True

    async def _build_tool_defs(self, user_message: str) -> list[dict]:
        return [tool_def.to_openai() for tool_def in self._tools.values()]

    @property
    def _prechat_func(self) -> Callable | None:
        for attr in dir(self.__class__):
            fn = getattr(self.__class__, attr)
            if hasattr(fn, "_is_prechat"):
                return fn
        return None

    @property
    def _postchat_func(self) -> Callable | None:
        for attr in dir(self.__class__):
            fn = getattr(self.__class__, attr)
            if hasattr(fn, "_is_postchat"):
                return fn
        return None

    async def _build_system_message(self, user_message: str) -> str:
        for attr in dir(self.__class__):
            fn = getattr(self.__class__, attr)
            if hasattr(fn, "_is_system_message"):
                return await _safe_run(fn, self, user_message)
        raise SystemMessageNotDeclared()

    async def chat(self, user_message: str) -> str:
        # First, run a prechat processing function if one was given
        if self._prechat_func:
            user_message = await _safe_run(self._prechat_func, self, user_message)

        # Generate and insert the new system message
        system_message = {
            "role": "developer",
            "content": await self._build_system_message(user_message),
        }
        if self.messages:
            self.messages[0] = system_message
        else:
            self.messages.append(system_message)
        self.messages.append({"role": "user", "content": user_message})

        # Create the tool list
        tool_defs = await self._build_tool_defs(user_message)

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
                input=self.messages,
                tools=tool_defs,
            )
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
            self.messages.append({"role": "assistant", "content": "Failed to generate a response"})
            return f"OpenAI failed to generate a response: {e}"

        # Dispatch any tool calls
        made_calls = False
        for tool_call in response.output:
            made_calls = made_calls or (await self._process_tool_call(tool_call))

        # If tools ran, re-invoke LLM for natural reply
        if made_calls:
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=self.messages,
                )
            except Exception as e:
                logger.exception(e)
                if self.emitter:
                    await _safe_run(
                        self.emitter, EmitUpdate(status=Status.ERROR, message="Generation failed")
                    )
                self.messages.append(
                    {"role": "assistant", "content": "Failed to generate a response"}
                )
                return f"OpenAI failed to generate a response: {e}"

        # Parse and finalize the Ai Response
        ai_message = response.output_text
        if self._postchat_func:
            ai_message = await _safe_run(self._postchat_func, self, ai_message)
        self.messages.append({"role": "assistant", "content": ai_message})

        if self.emitter:
            await _safe_run(self.emitter, AiUpdate(status=Status.SUCCEDED, message=ai_message))

        return ai_message
