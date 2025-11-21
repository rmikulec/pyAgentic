from typing import Type, get_args
from typeguard import check_type, TypeCheckError


# Placeholder class for Agent type annotation.
# Can't import actual agent as it would cause a circular import error.
class Agent:
    pass


class AgentValidationError(Exception):
    """
    Exception raised when an Agent class fails validation checks.
    Aggregates all validation problems into a single error message.
    """

    def __init__(self, problems):
        message = "Agent failed to be validated: \n"
        message += "\n".join(problems)
        super().__init__(message)


class _AgentConstructionValidator:
    """
    Class to hold validation logic that needs to be checked at runtime.

    Works by using default values to construct a sample agent, then runs additional checks
        that could not be run on creation of the class.
    """

    def __init__(self, AgentClass: Type["Agent"]):
        self.problems = []
        self.AgentClass = AgentClass
        self.sample_agent = self.AgentClass(model="openai::validation", api_key="validation")

    def validate(self):
        """
        Validate an Agent class.

        Raises:
            AgentValidationError: A custom exception that includes all problems found in the
                validation pipeline.
        """
        self._verify_default_values(self.AgentClass)
        self._verify_state_items_can_be_strings(self.AgentClass)
        self._verify_tool_state_refs(self.AgentClass)

        if self.problems:
            raise AgentValidationError(self.problems)

    def _verify_tool_state_refs(self, AgentClass: Type["Agent"]):
        """
        Verifies that all state refs used:
          - Link to an item in the state
          - The linked state item has the same type as the field it is being used in
        """
        for tool_name, tool_def in AgentClass.__tool_defs__.items():
            for param_name, (param_type, param_info) in tool_def.parameters.items():
                for info_field in param_info._get_maybe_state():
                    attr = getattr(param_info, info_field.name)
                    expected_type = get_args(info_field.type)[0]
                    if isinstance(attr, any):
                        if attr.path not in AgentClass.__state_attrs__:
                            self.problems.append(
                                f"tool.{tool_name}.param.{param_name}.{info_field.name}: Ref not found in state: {attr.path}"  # noqa E501
                            )
                        sample_value = self.sample_agent.state.get(attr.path)
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

    def _verify_state_items_can_be_strings(self, AgentClass: Type["Agent"]):
        """
        Verifies that all items in the state can be injected and used in the system message or
            input template.
        """
        for state_name in AgentClass.__state_attrs__.keys():
            sample_value = self.sample_agent.state.get(state_name)
            try:
                str(sample_value)
            except Exception:
                self.problems.append(
                    (
                        f"state.{state_name}: Value cannot be stringified"
                        f"  Value type: {type(sample_value)}"
                    )
                )
