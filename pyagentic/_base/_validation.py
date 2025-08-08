from typing import Type, get_args
from typeguard import check_type, TypeCheckError

from pyagentic._base._context import ContextItem, ContextRef


# Temp class for typing
class Agent:
    pass


class AgentValidationError(Exception):
    def __init__(self, problems):
        message = "Agent failed to be validated: \n"
        message += "\n".join(problems)
        super().__init__(message)


class _AgentConstructionValidator:
    """
    Class to hold validation logic that needs to be checked at runtime

    Class works by using default values to construct a sample agent, then runs additional checks
        that could not be run on creation of the class
    """

    def __init__(self, AgentClass: Type["Agent"]):
        self.problems = []
        self.AgentClass = AgentClass
        self.sample_agent = self.AgentClass(
            model="testing",
            api_key="validation",
        )

    def validate(self):
        """
        Validate an Agent class

        Raises:
            AgentValidationError: A custom exception that includes all problems found in the
                validation pipelines
        """
        self._verify_default_values(self.AgentClass)
        self._verify_context_items_can_be_strings(self.AgentClass)
        self._verify_tool_context_refs(self.AgentClass)

        if self.problems:
            raise AgentValidationError(self.problems)

    def _verify_tool_context_refs(self, AgentClass: Type["Agent"]):
        """
        Verifies that all context refs used:
            - links to an item in the context
            - The linked context item has the same type as the field it is being used in
        """
        for tool_name, tool_def in AgentClass.__tool_defs__.items():
            for param_name, (param_type, param_info) in tool_def.parameters.items():
                for info_field in param_info._get_maybe_context():
                    attr = getattr(param_info, info_field.name)
                    expected_type = get_args(info_field.type)[0]
                    if isinstance(attr, ContextRef):
                        if attr.path not in AgentClass.__context_attrs__:
                            self.problems.append(
                                f"tool.{tool_name}.param.{param_name}.{info_field.name}: Ref not found in context: {attr.path}"  # noqa E501
                            )
                        sample_value = self.sample_agent.context.get(attr.path)
                        try:
                            check_type(sample_value, expected_type)
                        except TypeCheckError:
                            self.problems.append(
                                (
                                    f"tool.{tool_name}.param.{param_name}.{info_field.name}: Ref typing does not match param info field:\n"  # noqa E501
                                    f"  Expected: {expected_type}\n"
                                    f"  Recieved: {type(sample_value).__name__}\n"
                                )
                            )

    def _verify_context_items_can_be_strings(self, AgentClass: Type["Agent"]):
        """
        Verifies that all items in the context can be injected / used in the system message or
            input template
        """
        for context_name in AgentClass.__context_attrs__.keys():
            sample_value = self.sample_agent.context.get(context_name)
            try:
                str(sample_value)
            except Exception:
                self.problems.append(
                    (
                        f"context.{context_name}: Value cannot be stringified"
                        f"  Value type: {type(sample_value)}"
                    )
                )

    def _verify_default_values(self, AgentClass: Type["Agent"]):
        """
        Verifies that default values and default factories have the same type as to that
            specified by the user
        """
        defaults = {}
        for context_name, (context_type, context_item) in AgentClass.__context_attrs__.items():
            if isinstance(context_item, ContextItem):
                default = context_item.get_default_value()
                try:
                    check_type(default, context_type)
                except TypeCheckError:
                    self.problems.append(
                        (
                            f"context.{context_name}: Default value does not match context item typing:\n"  # noqa E501
                            f"  Expected: {context_type}\n"
                            f"  Recieved: {type(default)}\n"
                        )
                    )
                defaults[context_name] = default
        return defaults
