"""Tests for the metaclass-generated ``__construct_model__`` and ``Depends[T]``."""

from pydantic import BaseModel

from pyagentic import BaseAgent, State, Link, Depends, spec


class _Topic(BaseModel):
    name: str
    depth: int = 1


class _Database:
    def __init__(self, dsn: str = "memory") -> None:
        self.dsn = dsn


class _ResearchAgent(BaseAgent):
    __system_message__ = "research"
    __description__ = "researches topics"
    __input_template__ = ""

    topic: State[_Topic]
    notes: State[list] = spec.State(default_factory=list)
    db: Depends[_Database]


class _Orchestrator(BaseAgent):
    __system_message__ = "orchestrate"
    __input_template__ = ""

    researcher: Link[_ResearchAgent]


def test_dependencies_extracted_and_typed():
    """Depends[T] fields are recorded in __dependencies__ with their type."""
    assert dict(_ResearchAgent.__dependencies__) == {"db": _Database}
    # Linking agent has no Depends of its own.
    assert dict(_Orchestrator.__dependencies__) == {}


def test_depends_excluded_from_construct_model():
    """Depends fields never appear in the construct model (not client-provided)."""
    fields = _ResearchAgent.__construct_model__.model_fields
    assert "db" not in fields
    assert set(fields) == {"topic", "notes", "model", "api_key"}


def test_construct_model_required_optional_mirror_constructor():
    """Required state has no default; defaulted state and model/api_key are optional."""
    fields = _ResearchAgent.__construct_model__.model_fields
    assert fields["topic"].is_required()          # no default
    assert not fields["notes"].is_required()       # default_factory=list
    assert not fields["model"].is_required()
    assert not fields["api_key"].is_required()


def test_construct_model_nests_linked_agents():
    """A required Link nests the child's construct model and is required."""
    fields = _Orchestrator.__construct_model__.model_fields
    assert "researcher" in fields
    assert fields["researcher"].is_required()
    assert fields["researcher"].annotation is _ResearchAgent.__construct_model__


def test_construct_model_named_after_agent():
    assert _ResearchAgent.__construct_model__.__name__ == "_ResearchAgentConstruct"


def test_optional_link_is_optional_in_construct_model():
    """A Link with a default_factory becomes optional in the construct model."""

    class _Defaulted(BaseAgent):
        __system_message__ = "x"
        __input_template__ = ""

        researcher: Link[_ResearchAgent] = spec.AgentLink(
            default_factory=lambda: _ResearchAgent(
                model="_mock::m", topic=_Topic(name="x")
            )
        )

    fields = _Defaulted.__construct_model__.model_fields
    assert not fields["researcher"].is_required()


def test_depends_field_is_optional_constructor_arg():
    """A Depends field is an optional (default None) constructor kwarg."""
    import inspect

    params = inspect.signature(_ResearchAgent.__init__).parameters
    assert "db" in params
    assert params["db"].default is None
