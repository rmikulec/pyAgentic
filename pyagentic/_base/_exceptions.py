from typing import Literal


class InvalidToolDefinition(Exception):
    """
    Exception raised when a tool definition is invalid or malformed.

    Args:
        tool_name (str): The name of the invalid tool
        message (str): Description of the validation error
    """

    def __init__(self, tool_name, message):
        message = f"Invalid Tool Definition: {tool_name}: " f"{message}"
        super().__init__(message)


class SystemMessageNotDeclared(Exception):
    """
    Exception raised when an Agent subclass is created without declaring __system_message__.
    """

    def __init__(self):
        super().__init__(
            "System message not declared on agent. Agent must be declared with `__system_message__`"  # noqa E501
        )


class UnexpectedStateItemType(Exception):
    """
    Exception raised when a state item is initialized with a value of an unexpected type.

    Args:
        name (str): The name of the state item
        expected (type): The expected type
        recieved (type): The actual type received
    """

    def __init__(self, name, expected, recieved):
        message = (
            f"Unexpected value provided for `{name}`. "
            f"Expected: {expected} - Recieved: {recieved}"
        )
        super().__init__(message)


class InvalidStateRefNotFoundInState(Exception):
    """
    Exception raised when a state reference points to a non-existent state item.

    Args:
        name (str): The name of the missing state item
    """

    def __init__(self, name):
        message = (
            f"'{name}' not found in state. "
            "Make sure it is either declared as a `StateItem` or using `computed_state`"
        )
        super().__init__(message)


class InvalidStateRefMismatchTyping(Exception):
    """
    Exception raised when a state reference has a type mismatch with the expected field type.

    Args:
        ref_path (str): The path of the state reference
        field_name (str): The name of the field with the mismatch
        recieved_type (type): The actual type of the referenced value
        expected_type (type): The expected type for the field
    """

    def __init__(self, ref_path, field_name, recieved_type, expected_type):
        message = (
            f"StateRef('{ref_path}') for {self.__class__.__name__}.{field_name}  "
            f"is of type {recieved_type}, expected {expected_type}"
        )
        super().__init__(message)


class InvalidLLMSetup(Exception):
    """
    Exception raised when LLM provider configuration is invalid or incomplete.

    Args:
        reason (Literal): The specific reason for the invalid setup
        model (str, optional): The model string that failed validation
        valid_providers (list[str], optional): List of valid provider names
    """

    def __init__(
        self,
        reason: Literal["provider-not-found", "invalid-format", "no-provider"],
        model: str = None,
        valid_providers: list[str] = None,
    ):
        self.invalid_model = model
        self.reason = reason

        match reason:
            case "provider-not-found":
                message = (
                    f"{model.split('::')[0]} is an unsupported provider."
                    f"   Please use one of {valid_providers}"
                    "   or provide a custom provider with `provider`"
                )
            case "invalid-format":
                message = f"{model} is in invalid format. Please provide model with <provider>::<model_name>"  # noqa E501
            case "no-provider":
                message = "Please provider either `model` + `api_key` or a valid `provider`"
            case _:
                message = reason

        super().__init__(message)
