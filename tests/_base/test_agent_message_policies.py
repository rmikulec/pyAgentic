"""End-to-end tests for message policies attached via __message_policies__."""

import pytest

from pyagentic import BaseAgent
from pyagentic.models.llm import AssistantMessage, Message, UserMessage
from pyagentic.policies import Policy

_MODEL = "_mock::test-model"


class _SuffixPolicy(Policy):
    """Appends a marker to every user message entering the context"""

    def __init__(self, marker: str):
        self.marker = marker

    def on_append(self, event, item):
        if isinstance(item, UserMessage):
            return item.model_copy(update={"content": f"{item.content}{self.marker}"})
        return None


class _VetoAssistantPolicy(Policy):
    """Vetoes assistant messages from entering the context"""

    def on_append(self, event, item):
        if isinstance(item, AssistantMessage):
            raise ValueError("no assistant messages in context")
        return None


class _KeepLastPolicy(Policy):
    """Compile-time policy that keeps only the last message in context"""

    async def on_compile(self, event, items):
        if len(items) > 1:
            return items[-1:]
        return None


def _agent(cls, **kwargs):
    return cls(model=_MODEL, api_key="k", **kwargs)


@pytest.mark.asyncio
async def test_on_append_transforms_context_but_not_raw_history():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""
        __message_policies__ = [_SuffixPolicy(" [seen]")]

    agent = _agent(_Agent)
    response = await agent.run("hello")

    # Context (what the provider consumed) is transformed...
    assert agent.state._context[0].content == "hello [seen]"
    assert "hello [seen]" in response.final_output
    # ...while the raw log keeps the original
    assert agent.state._messages[0].content == "hello"


@pytest.mark.asyncio
async def test_policies_run_in_declaration_order():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""
        __message_policies__ = [_SuffixPolicy("-a"), _SuffixPolicy("-b")]

    agent = _agent(_Agent)
    await agent.run("x")

    assert agent.state._context[0].content == "x-a-b"


@pytest.mark.asyncio
async def test_veto_skips_context_but_keeps_raw_history():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""
        __message_policies__ = [_VetoAssistantPolicy()]

    agent = _agent(_Agent)
    response = await agent.run("hi")

    # Run completes despite the veto
    assert response.final_output
    # The assistant reply was vetoed from context but recorded raw
    assert not any(isinstance(m, AssistantMessage) for m in agent.state._context)
    assert any(isinstance(m, AssistantMessage) for m in agent.state._messages)


@pytest.mark.asyncio
async def test_on_compile_rewrites_context_persistently():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""
        __message_policies__ = [_KeepLastPolicy()]

    agent = _agent(_Agent)
    await agent.run("first")
    await agent.run("second")

    # Compile ran before the second inference: context was cut to the latest
    # user message, then the second assistant reply appended after
    assert len(agent.state._context) == 2
    assert agent.state._context[0].content == "second"
    assert isinstance(agent.state._context[1], AssistantMessage)
    # Raw history has everything: 2 user turns + 2 assistant replies
    assert len(agent.state._messages) == 4


@pytest.mark.asyncio
async def test_message_policies_inherit_to_subclasses():
    class _Parent(BaseAgent):
        __system_message__ = "parent"
        __input_template__ = ""
        __message_policies__ = [_SuffixPolicy(" [inherited]")]

    class _Child(_Parent):
        __system_message__ = "child"
        __input_template__ = ""

    agent = _agent(_Child)
    await agent.run("hello")

    assert agent.state._context[0].content == "hello [inherited]"


@pytest.mark.asyncio
async def test_fork_keeps_policies_with_fresh_history():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __description__ = "mock"
        __input_template__ = ""
        __message_policies__ = [_SuffixPolicy(" [p]")]

    agent = _agent(_Agent)
    await agent.run("original")

    fork = agent.fork()
    assert fork.state._messages == []
    assert fork.state._context == []

    await fork.run("forked")
    assert fork.state._context[0].content == "forked [p]"
    # Template untouched by the fork's run
    assert agent.state._context[0].content == "original [p]"


def test_state_field_named_messages_is_rejected():
    from pyagentic import State, spec

    with pytest.raises(ValueError, match="reserved"):

        class _Bad(BaseAgent):
            __system_message__ = "mock"
            messages: State[list] = spec.State(default_factory=list)


@pytest.mark.asyncio
async def test_no_policies_is_a_noop():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""

    agent = _agent(_Agent)
    response = await agent.run("plain")

    assert response.final_output == "user said plain"
    assert [m.content for m in agent.state._context] == [
        m.content for m in agent.state._messages
    ]


@pytest.mark.asyncio
async def test_raw_messages_property_includes_system():
    class _Agent(BaseAgent):
        __system_message__ = "the system prompt"
        __input_template__ = ""

    agent = _agent(_Agent)
    await agent.run("hi")

    assert agent.state.raw_messages[0].role == "system"
    assert agent.state.raw_messages[0].content == "the system prompt"
    assert agent.state.messages[0].role == "system"
