import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, tool
from pyagentic._base._exceptions import InvalidLLMSetup
from pyagentic.llm import OpenAIProvider


def test_agent_with_model_string():
    """Test creating agent with model string format"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

    agent = TestAgent(model="_mock::test-model", api_key="test-key")
    assert agent.provider is not None


def test_agent_with_provider_instance():
    """Test creating agent with provider instance"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

    provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
    agent = TestAgent(provider=provider)
    assert agent.provider == provider


def test_agent_invalid_model_format():
    """Test that invalid model format raises error"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

    with pytest.raises(InvalidLLMSetup):
        agent = TestAgent(model="invalid-format", api_key="test-key")


def test_agent_no_provider():
    """Test that creating agent without provider raises error"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

    with pytest.raises(InvalidLLMSetup):
        agent = TestAgent()


def test_agent_provider_not_found():
    """Test that invalid provider name raises error"""

    class TestAgent(BaseAgent):
        __system_message__ = "Test agent"

    with pytest.raises(InvalidLLMSetup):
        agent = TestAgent(model="invalid_provider::model", api_key="test-key")
