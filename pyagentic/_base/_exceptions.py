from typing import Literal


class ToolDeclarationFailed(Exception):

    def __init__(self, tool_name, message):
        message = f"Tool declaration failed for {tool_name}" f"{message}"
        super().__init__(message)


class SystemMessageNotDeclared(Exception):
    def __init__(self):
        super().__init__(
            "System message not declared on agent. Agent must be declared with `__system_message__`"  # noqa E501
        )


class UnexpectedContextItemType(Exception):
    def __init__(self, name, expected, recieved):
        message = (
            f"Unexpected value provided for `{name}`. "
            f"Expected: {expected} - Recieved: {recieved}"
        )
        super().__init__(message)


class InvalidContextRefNotFoundInContext(Exception):
    def __init__(self, name):
        message = (
            f"'{name}' not found in context. "
            "Make sure it is either declared as a `ContextItem` or using `computed_context`"
        )
        super().__init__(message)


class InvalidContextRefMismatchTyping(Exception):
    def __init__(self, ref_path, field_name, recieved_type, expected_type):
        message = (
            f"ContextRef('{ref_path}') for {self.__class__.__name__}.{field_name}  "
            f"is of type {recieved_type}, expected {expected_type}"
        )
        super().__init__(message)


class InvalidLLMSetup(Exception):
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
