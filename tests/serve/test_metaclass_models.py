import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, tool
from pyagentic.models.response import AgentResponse, ToolResponse
from pyagentic.models.llm import LLMResponse


def test_request_model_default_call():
    """Test that __request_model__ is generated from the default __call__(user_input: str)."""

    class SimpleAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    RequestModel = SimpleAgent.__request_model__
    assert RequestModel is not None
    assert "user_input" in RequestModel.model_fields
    # Should be able to instantiate
    req = RequestModel(user_input="hello")
    assert req.user_input == "hello"


def test_request_model_custom_call():
    """Test that __request_model__ mirrors a custom __call__ signature."""

    class CustomAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

        async def __call__(self, query: str, max_results: int = 10) -> "AgentResponse":
            return await self.run(query)

    RequestModel = CustomAgent.__request_model__
    assert "query" in RequestModel.model_fields
    assert "max_results" in RequestModel.model_fields

    # Required field
    req = RequestModel(query="test")
    assert req.query == "test"
    assert req.max_results == 10

    # Override default
    req2 = RequestModel(query="test", max_results=5)
    assert req2.max_results == 5


def test_request_model_name():
    """Test that the request model is named after the agent class."""

    class MySpecialAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    assert MySpecialAgent.__request_model__.__name__ == "MySpecialAgentRequest"


def test_stream_event_model_exists():
    """Test that __stream_event_model__ is generated."""

    class StreamAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    assert StreamAgent.__stream_event_model__ is not None


def test_stream_event_model_has_event_types():
    """Test that the stream event model has LLM, Tool, and Agent event types."""

    class EventAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

        @tool("A test tool")
        def my_tool(self) -> str:
            return "result"

    StreamModel = EventAgent.__stream_event_model__

    # Should have the individual event models attached
    assert hasattr(StreamModel, "__llm_event__")
    assert hasattr(StreamModel, "__tool_event__")
    assert hasattr(StreamModel, "__agent_event__")


def test_stream_event_llm_event():
    """Test that LLMEvent can be constructed with correct data."""

    class LLMAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    LLMEvent = LLMAgent.__stream_event_model__.__llm_event__

    # Check it has the right fields
    assert "event" in LLMEvent.model_fields
    assert "data" in LLMEvent.model_fields

    # The event field should have a default of "llm_response"
    event = LLMEvent(data=LLMResponse(content="hello", tool_calls=[]))
    assert event.event == "llm_response"


def test_stream_event_agent_event():
    """Test that AgentEvent wraps the agent's response model."""

    class ResponseAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    AgentEvent = ResponseAgent.__stream_event_model__.__agent_event__
    assert "event" in AgentEvent.model_fields
    assert "data" in AgentEvent.model_fields


def test_response_model_exists():
    """Test that __response_model__ is generated at class declaration time."""

    class RespAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    assert RespAgent.__response_model__ is not None
    assert issubclass(RespAgent.__response_model__, AgentResponse)


def test_state_class_exists():
    """Test that __state_class__ is generated at class declaration time."""

    class StateAgent(BaseAgent):
        __system_message__ = "Test"
        __input_template__ = ""

    assert StateAgent.__state_class__ is not None
