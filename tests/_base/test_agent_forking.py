"""Tests for linked-agent forking: isolation, construct-time reset, shared opt-in."""

import asyncio

import pytest

from pyagentic import BaseAgent, Link, State, spec
from pyagentic.models.llm import ToolCall

_MODEL = "_mock::test-model"


class _Database:
    def __init__(self) -> None:
        self.hits = 0


class _Helper(BaseAgent):
    __system_message__ = "I am a helper"
    __description__ = "Provides help"
    __input_template__ = ""

    notes: State[list] = spec.State(default_factory=list)


def _helper(**kwargs) -> _Helper:
    return _Helper(model=_MODEL, api_key="k", **kwargs)


class _Forky(BaseAgent):
    __system_message__ = "default (forked) link"
    __input_template__ = ""
    helper: Link[_Helper]


class _Sharey(BaseAgent):
    __system_message__ = "shared link"
    __input_template__ = ""
    helper: Link[_Helper] = spec.AgentLink(shared=True)


def _parent(cls, **kwargs):
    return cls(model=_MODEL, api_key="k", helper=_helper(**kwargs))


def _call(name: str, text: str) -> ToolCall:
    return ToolCall(id=text, name=name, arguments=f'{{"user_input": "{text}"}}')


# ---- fork() unit behavior ----


def test_fork_is_a_distinct_instance_of_same_class():
    template = _helper()
    fork = template.fork()
    assert isinstance(fork, _Helper)
    assert fork is not template
    assert template.fork() is not fork


def test_fork_shares_provider():
    template = _helper()
    assert template.fork().provider is template.provider


def test_fork_state_is_isolated():
    template = _helper(notes=["seed"])
    a, b = template.fork(), template.fork()
    a.state.notes.append("a")
    assert b.state.notes == ["seed"]
    assert template.state.notes == ["seed"]


def test_fork_resets_state_to_construction_time():
    template = _helper(notes=["seed"])
    template.state.notes.append("mutated")
    # The fork ignores the template's mutated state and resets to construct time.
    assert template.fork().state.notes == ["seed"]


def test_fork_shares_dependencies():
    from pyagentic import Depends

    db = _Database()

    class _NeedsDb(BaseAgent):
        __system_message__ = "x"
        __input_template__ = ""
        db: Depends[_Database]

    template = _NeedsDb(model=_MODEL, api_key="k", db=db)
    assert template.fork().db is db


# ---- _process_agent_call: fork vs shared ----


@pytest.mark.asyncio
async def test_default_link_calls_run_on_forks_template_untouched():
    parent = _parent(_Forky, notes=["seed"])
    template = parent.helper

    await parent._process_agent_call(_call("helper", "one"))
    await parent._process_agent_call(_call("helper", "two"))

    # Each call ran on a discarded fork, so the template never accumulated
    # conversation history or state.
    assert template.state._messages == []
    assert template.state.notes == ["seed"]


@pytest.mark.asyncio
async def test_shared_link_persists_state_across_calls():
    parent = _parent(_Sharey)
    template = parent.helper

    await parent._process_agent_call(_call("helper", "one"))
    after_first = len(template.state._messages)
    await parent._process_agent_call(_call("helper", "two"))

    # The shared instance is reused, so its history grows across calls.
    assert after_first > 0
    assert len(template.state._messages) > after_first


@pytest.mark.asyncio
async def test_concurrent_default_link_calls_do_not_race():
    parent = _parent(_Forky, notes=["seed"])
    template = parent.helper

    results = await asyncio.gather(
        *[parent._process_agent_call(_call("helper", f"c{i}")) for i in range(5)]
    )

    assert len(results) == 5
    assert all(r is not None for r in results)
    # No fork leaked state back into the template.
    assert template.state._messages == []
    assert template.state.notes == ["seed"]
