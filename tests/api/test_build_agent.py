"""Tests for build_agent and validate_dependencies (construct + DI)."""

import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, State, Link, Depends, spec
from pyagentic.api._build import build_agent, validate_dependencies


class _Topic(BaseModel):
    name: str


class _Database:
    def __init__(self, dsn: str = "memory") -> None:
        self.dsn = dsn


class _Cache:
    def __init__(self, size: int = 1) -> None:
        self.size = size


class _ResearchAgent(BaseAgent):
    __system_message__ = "research"
    __description__ = "researches"
    __input_template__ = ""

    topic: State[_Topic]
    db: Depends[_Database]


class _Orchestrator(BaseAgent):
    __system_message__ = "orchestrate"
    __input_template__ = ""

    researcher: Link[_ResearchAgent]


def _payload(**overrides):
    base = {"researcher": {"topic": {"name": "attention"}}}
    base.update(overrides)
    return _Orchestrator.__construct_model__(**base)


def _build(payload, **kwargs):
    kwargs.setdefault("default_model", "_mock::m")
    return build_agent(_Orchestrator, payload, **kwargs)


def test_build_nested_with_instance_dependency():
    db = _Database("postgres://x")
    orch = _build(_payload(), dependencies=[db])
    assert isinstance(orch, _Orchestrator)
    assert isinstance(orch.researcher, _ResearchAgent)
    assert orch.researcher.db is db


def test_default_model_propagates_to_nested_agents():
    orch = build_agent(
        _Orchestrator,
        _Orchestrator.__construct_model__(researcher={"topic": {"name": "x"}}),
        dependencies=[_Database()],
        default_model="_mock::default",
    )
    assert orch.model == "_mock::default"
    assert orch.researcher.model == "_mock::default"


def test_build_with_factory_dependency_called_fresh():
    def make_db() -> _Database:
        return _Database("factory")

    a = _build(_payload(), dependencies=[make_db])
    b = _build(_payload(), dependencies=[make_db])
    assert a.researcher.db.dsn == "factory"
    # Factories are called fresh per build.
    assert a.researcher.db is not b.researcher.db


def test_instance_wins_over_factory_on_ambiguity():
    instance = _Database("instance")

    def make_db() -> _Database:
        return _Database("factory")

    orch = _build(_payload(), dependencies=[make_db, instance])
    assert orch.researcher.db is instance


def test_unsatisfied_dependency_raises_lookup_error_at_build():
    with pytest.raises(LookupError, match="No dependency provided for 'db'"):
        _build(_payload(), dependencies=[])


def test_validate_dependencies_walks_tree_and_fails_fast():
    with pytest.raises(ValueError, match="unsatisfied dependencies"):
        validate_dependencies(_Orchestrator, dependencies=[])
    # Satisfied by an instance or a factory.
    validate_dependencies(_Orchestrator, dependencies=[_Database()])

    def make_db() -> _Database:
        return _Database()

    validate_dependencies(_Orchestrator, dependencies=[make_db])


def test_validate_collects_dependencies_from_multiple_levels():
    class _Deep(BaseAgent):
        __system_message__ = "deep"
        __input_template__ = ""
        cache: Depends[_Cache]
        orch: Link[_Orchestrator]

    # Needs both _Cache (own) and _Database (nested in orchestrator's researcher).
    with pytest.raises(ValueError, match="_Database"):
        validate_dependencies(_Deep, dependencies=[_Cache()])
    validate_dependencies(_Deep, dependencies=[_Cache(), _Database()])
