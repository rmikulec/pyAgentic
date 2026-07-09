from typing import TypeVar, Generic

T = TypeVar("T")


class Depends(Generic[T]):
    """
    Type annotation for declaring a dependency-injected agent field.

    ``Depends`` marks a field as a server-supplied dependency rather than
    serializable per-session state. Use it in an agent's class body wherever you
    would otherwise use :class:`State`, for resources that cannot (or should not)
    travel over the wire — database handles, HTTP clients, pre-configured
    providers, and similar.

    Unlike ``State`` and ``Link``, a ``Depends`` field is **excluded** from the
    agent's generated construct model: clients never provide it. Instead it is
    supplied once by the developer at :func:`pyagentic.api.create_router` /
    :func:`pyagentic.api.create_app` via the ``dependencies`` argument, and
    resolved by type across the (possibly nested) agent tree.

    Args:
        T: The dependency type. Used both as the field's type hint and as the key
            for by-type resolution at serve time.

    Example:
        ```python
        from pyagentic import BaseAgent, State, Depends

        class ResearchAgent(BaseAgent):
            __instructions__ = "You research topics"

            topic: State[TopicState]      # client-provided per session
            db:    Depends[Database]      # injected by the developer, by type

            @tool("Look up a record")
            def lookup(self, key: str) -> str:
                return self.db.get(key)
        ```
    """

    def __class_getitem__(cls, item):
        """
        Creates a generic Depends type marker for a given dependency class.

        Args:
            item: The dependency class this field requires.

        Returns:
            type: Special marker type that the metaclass detects and records in
                ``__dependencies__`` (and excludes from the construct model).
        """
        # Return a special marker type that the metaclass can detect, mirroring
        # the State[T] / Link[T] markers.
        return type(
            f"Depends[{item.__name__}]",
            (),
            {"__origin__": Depends, "__args__": (item,), "__dependency_type__": item},
        )
