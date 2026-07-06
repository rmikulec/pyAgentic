"""Tests for PolicyList: policies firing on list-valued state field mutations."""

import pytest

from pyagentic import BaseAgent, State, spec
from pyagentic.policies import Policy, PolicyList

_MODEL = "_mock::test-model"


class _UppercaseAppendPolicy(Policy):
    """Uppercases every string appended to the list"""

    def on_append(self, event, item):
        if isinstance(item, str):
            return item.upper()
        return None


class _NoShortItemsPolicy(Policy):
    """Vetoes strings shorter than 3 characters"""

    def on_append(self, event, item):
        if isinstance(item, str) and len(item) < 3:
            raise ValueError("too short")
        return None


class _RecordSetPolicy(Policy):
    """Records on_set invocations (config-only: writes to a shared external log)"""

    def __init__(self, log: list):
        self.log = log

    def on_set(self, event, value):
        self.log.append(list(value) if isinstance(value, list) else value)
        return None


def _make_agent(policies):
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""

        items: State[list] = spec.State(default_factory=list, policies=policies)

    return _Agent(model=_MODEL, api_key="k")


def test_list_state_field_is_wrapped():
    agent = _make_agent([_UppercaseAppendPolicy()])
    assert isinstance(agent.state.items, PolicyList)
    # Still a list for isinstance checks and equality
    assert isinstance(agent.state.items, list)
    assert agent.state.items == []


def test_append_fires_on_append():
    agent = _make_agent([_UppercaseAppendPolicy()])

    agent.state.items.append("hello")
    agent.state.items.extend(["world", "again"])

    assert agent.state.items == ["HELLO", "WORLD", "AGAIN"]


def test_append_veto_skips_item():
    agent = _make_agent([_NoShortItemsPolicy()])

    agent.state.items.append("ok-item")
    agent.state.items.append("no")  # vetoed

    assert agent.state.items == ["ok-item"]


def test_insert_and_iadd_fire_on_append():
    agent = _make_agent([_UppercaseAppendPolicy()])

    agent.state.items.insert(0, "first")
    agent.state.items += ["second"]

    assert agent.state.items == ["FIRST", "SECOND"]


def test_in_place_mutations_fire_on_set():
    log = []
    agent = _make_agent([_RecordSetPolicy(log)])

    agent.state.items.append("a")  # append: no on_set
    agent.state.items[0] = "b"  # setitem
    agent.state.items.remove("b")  # remove
    agent.state.items.append("c")
    agent.state.items.pop()  # pop
    agent.state.items.clear()  # clear

    assert log == [["b"], [], [], []]


def test_assignment_rewraps_list():
    agent = _make_agent([_UppercaseAppendPolicy()])

    agent.state.set("items", ["preset"])
    assert isinstance(agent.state.items, PolicyList)

    agent.state.items.append("more")
    assert agent.state.items == ["preset", "MORE"]


def test_unpolicied_list_field_is_not_wrapped():
    class _Agent(BaseAgent):
        __system_message__ = "mock"
        __input_template__ = ""

        plain: State[list] = spec.State(default_factory=list)

    agent = _Agent(model=_MODEL, api_key="k")
    assert not isinstance(agent.state.plain, PolicyList)


def test_policy_with_only_some_handlers_is_skipped_gracefully():
    """A policy implementing only on_get never breaks list mutations"""

    class _GetOnly(Policy):
        def on_get(self, event, value):
            return None

    agent = _make_agent([_GetOnly()])
    agent.state.items.append("fine")
    agent.state.items[0] = "still fine"
    assert agent.state.items == ["still fine"]
