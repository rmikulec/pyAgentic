import pytest

from pyagentic._base._agent import Agent
from pyagentic._base._exceptions import InvalidLLMSetup
from pyagentic.llm._mock import _MockProvider


class MockAgent(Agent):
    __system_message__ = "This is a mock"


def test_agent_raises_invalid_llm_none_given():
    with pytest.raises(InvalidLLMSetup) as e:
        MockAgent()

    assert e.value.reason == "no-provider"


def test_agent_raises_invalid_llm_invalid_format():
    with pytest.raises(InvalidLLMSetup) as e:
        MockAgent(model="openai/_mock::test-modelo")

    assert e.value.reason == "invalid-format"


def test_agent_successful_creation_with_valid_model():
    agent = MockAgent(model="_mock::_mock::test-modelo", api_key="test-key")
    assert agent.provider is not None
    assert isinstance(agent.provider, _MockProvider)


def test_agent_raises_provider_not_found():
    with pytest.raises(InvalidLLMSetup) as e:
        MockAgent(model="invalid_provider::_mock::test-modelo", api_key="test-key")

    assert e.value.reason == "provider-not-found"


def test_agent_successful_creation_with_provider():
    provider = _MockProvider(model="_mock::test-modelo", api_key="test-key")
    agent = MockAgent(provider=provider)
    assert agent.provider is provider


def test_agent_provider_overrides_model_and_api_key():
    provider = _MockProvider(model="_mock::test-modelo", api_key="test-key")
    agent = MockAgent(model="openai::gpt-3.5-turbo", api_key="different-key", provider=provider)
    assert agent.provider is provider
