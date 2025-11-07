"""
State reference system for dynamic parameter resolution in agent tools.

This module provides the `ref` object for creating lazy references to agent state
that are resolved at runtime when tools are called.
"""

from pydantic import BaseModel


class RefNode(BaseModel):
    """
    Represents a lazy dotted reference like ref.parent.conversation.goals.
    Allows building up nested references that are resolved later against an agent context.
    """

    def __init__(self, path):
        self._path = path

    def __getattr__(self, key):
        """
        Builds a new RefNode with the added key to the path.

        Args:
            key (str): The attribute key to add to the reference path

        Returns:
            RefNode: A new RefNode with the extended path
        """
        return RefNode(self._path + [key])

    def __call__(self, agent):
        """
        Resolves the reference by calling resolve with the agent context.

        Args:
            agent (dict): The agent reference dictionary to resolve against

        Returns:
            Any: The resolved value from the agent context
        """
        return self.resolve(agent)

    def __repr__(self):
        """
        Returns a string representation of the reference path.

        Returns:
            str: A string representation of the RefNode path
        """
        return f"Ref({'.'.join(self._path)})"

    def resolve(self, agent_reference: dict):
        """
        Resolves the full dotted path by traversing the agent_reference dictionary.

        Args:
            agent_reference (dict): The dictionary to traverse for resolution

        Returns:
            Any: The value at the end of the reference path
        """
        target = agent_reference
        for part in self._path:
            target = target[part]
        return target


class _RefRoot:
    """
    Root object for creating state references. Accessed via the global 'ref' instance.

    The `ref` object allows you to create dynamic references to agent state fields
    that are resolved when tools are called by the LLM. This is useful for passing
    current state values as default arguments to tools.

    Example:
        ```python
        from pyagentic import ref

        class Agent(BaseAgent):
            __system_message__ = "You are helpful"
            conversation: State[Conversation]

            @tool("Continue the conversation")
            def chat(
                self,
                message: str,
                context: str = ref.self.conversation.history  # Dynamic reference
            ) -> str:
                # context will be automatically populated with current conversation history
                return f"You said: {message}. Context: {context}"
        ```
    """

    def __getattr__(self, key):
        """
        Creates a new RefNode starting with the given key.
        Defaults to ref.self.* for convenient access to state items.

        Args:
            key (str): The starting key for the reference path

        Returns:
            RefNode: A new RefNode initialized with the key
        """
        return RefNode([key])


ref = _RefRoot()
