"""
Construct agents from a (recursive) construct-model payload plus developer-supplied
dependencies.

The metaclass generates ``agent_class.__construct_model__`` — a recursive Pydantic
model mirroring the constructor (state fields, nested linked-agent construct models,
and ``model``/``api_key``). ``build_agent`` turns an instance of that model into a
live agent, recursing into linked agents and injecting ``Depends[T]`` fields by type
from a flat ``dependencies`` list.

Dependencies are passed as bare values:
  - an **instance** of the dependency type, or
  - a zero-arg **factory** whose return annotation is (a subclass of) the type.

Resolution is by type. An instance match wins over a factory; factories are called
fresh for each agent built (so each session/job gets its own).
"""

import inspect
from typing import Any, Optional

from pyagentic._base._agent._agent import BaseAgent


def _as_field_dict(construct_data: Any) -> dict:
    """Normalize a construct payload (Pydantic model or dict) to a field dict.

    Nested values (linked-agent construct models) are preserved as-is so the
    recursion can descend into them.
    """
    if construct_data is None:
        return {}
    if isinstance(construct_data, dict):
        return dict(construct_data)
    # Pydantic model: iterate declared fields, keeping nested model instances.
    model_fields = getattr(type(construct_data), "model_fields", None)
    if model_fields is not None:
        return {name: getattr(construct_data, name) for name in model_fields}
    raise TypeError(f"Unsupported construct payload type: {type(construct_data)!r}")


def _matches_factory(value: Any, dep_type: type) -> bool:
    """Return whether *value* is a zero-arg factory producing *dep_type*."""
    if not callable(value) or isinstance(value, type):
        # Plain classes are treated as instances/values, not factories — a
        # factory is expected to be an annotated callable.
        return False
    try:
        ret = inspect.signature(value).return_annotation
    except (TypeError, ValueError):
        return False
    return isinstance(ret, type) and issubclass(ret, dep_type)


def _resolve_dependency(name: str, dep_type: type, dependencies: list) -> Any:
    """Resolve a single ``Depends[T]`` slot from the dependency list.

    Args:
        name (str): The field name (used only for error messages).
        dep_type (type): The declared dependency type ``T``.
        dependencies (list): Provided instances/factories.

    Returns:
        Any: The resolved dependency instance.

    Raises:
        LookupError: If no provided value satisfies the slot.
    """
    # Instance match wins.
    for value in dependencies:
        if isinstance(value, dep_type):
            return value
    # Otherwise a factory whose return type matches; called fresh.
    for value in dependencies:
        if _matches_factory(value, dep_type):
            return value()
    raise LookupError(
        f"No dependency provided for '{name}': {dep_type.__name__}. "
        f"Pass an instance of {dep_type.__name__} or a factory returning it "
        f"in create_router(..., dependencies=[...])."
    )


def build_agent(
    agent_class: type[BaseAgent],
    construct_data: Any,
    dependencies: Optional[list] = None,
    *,
    default_model: Optional[str] = None,
    default_api_key: Optional[str] = None,
) -> BaseAgent:
    """Instantiate an agent from a construct-model payload and injected dependencies.

    Linked agents are built first (depth-first) from their nested construct
    submodels, then ``Depends[T]`` fields are resolved by type, and finally the
    agent is constructed with state, ``model``/``api_key``, the built linked
    agents, and the resolved dependencies. The ``default_model``/``default_api_key``
    propagate down the whole tree, filling in any agent whose payload omits them.

    Args:
        agent_class (type[BaseAgent]): The agent class to instantiate.
        construct_data (Any): An instance of ``agent_class.__construct_model__``
            (or a dict with the same shape).
        dependencies (Optional[list]): Instances or factories used to satisfy
            ``Depends[T]`` fields across the agent tree. Defaults to ``[]``.
        default_model (Optional[str]): Model applied to any agent in the tree that
            does not specify its own ``model``.
        default_api_key (Optional[str]): API key applied alongside ``default_model``
            for any agent that does not specify its own.

    Returns:
        BaseAgent: The fully constructed agent instance.

    Raises:
        LookupError: If a required dependency cannot be resolved.
    """
    dependencies = dependencies or []
    # A raw dict (e.g. a persisted job payload) is parsed through the construct
    # model so nested state and linked-agent submodels are coerced to typed
    # instances before construction.
    if isinstance(construct_data, dict):
        construct_data = agent_class.__construct_model__(**construct_data)
    fields = _as_field_dict(construct_data)
    kwargs: dict[str, Any] = {}

    # Build nested linked agents first (defaults propagate down the tree).
    for name, linked_def in agent_class.__linked_agents__.items():
        child_data = fields.get(name)
        if child_data is None:
            # Omitted optional link: let __init__ apply the AgentLink default.
            continue
        kwargs[name] = build_agent(
            linked_def.agent,
            child_data,
            dependencies,
            default_model=default_model,
            default_api_key=default_api_key,
        )

    # State fields supplied by the client.
    for name in agent_class.__state_defs__:
        if name in fields and fields[name] is not None:
            kwargs[name] = fields[name]

    # Provider selection scalars, falling back to the propagated defaults.
    model = fields.get("model") or default_model
    api_key = fields.get("api_key") or default_api_key
    if model is not None:
        kwargs["model"] = model
    if api_key is not None:
        kwargs["api_key"] = api_key

    # Inject dependencies by type.
    for name, dep_type in agent_class.__dependencies__.items():
        kwargs[name] = _resolve_dependency(name, dep_type, dependencies)

    return agent_class(**kwargs)


def _collect_dependency_slots(
    agent_class: type[BaseAgent],
    _seen: Optional[set] = None,
) -> list[tuple[str, type]]:
    """Walk the linked-agent tree collecting every ``Depends[T]`` slot."""
    _seen = _seen if _seen is not None else set()
    if agent_class in _seen:
        return []
    _seen.add(agent_class)

    slots: list[tuple[str, type]] = list(agent_class.__dependencies__.items())
    for linked_def in agent_class.__linked_agents__.values():
        slots.extend(_collect_dependency_slots(linked_def.agent, _seen))
    return slots


def validate_dependencies(
    agent_class: type[BaseAgent],
    dependencies: Optional[list] = None,
) -> None:
    """Verify every ``Depends[T]`` slot in the agent tree is satisfiable.

    Intended to run once at ``create_router``/``create_app`` time so misconfigured
    dependencies fail fast rather than per request. Factories are not invoked —
    only their presence/return type is checked.

    Args:
        agent_class (type[BaseAgent]): The root agent class.
        dependencies (Optional[list]): Provided instances/factories.

    Raises:
        ValueError: If any dependency slot has no matching provider.
    """
    dependencies = dependencies or []
    missing: list[str] = []
    for name, dep_type in _collect_dependency_slots(agent_class):
        satisfiable = any(isinstance(v, dep_type) for v in dependencies) or any(
            _matches_factory(v, dep_type) for v in dependencies
        )
        if not satisfiable:
            missing.append(f"'{name}': {dep_type.__name__}")
    if missing:
        raise ValueError(
            f"{agent_class.__name__} has unsatisfied dependencies: {', '.join(missing)}. "
            f"Provide them via dependencies=[...] (an instance or a factory returning the type)."
        )
