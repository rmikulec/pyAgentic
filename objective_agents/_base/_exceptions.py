class ToolDeclarationFailed(Exception):

    def __init__(self, tool_name, message):
        message = f"Tool declaration failed for {tool_name}" f"{message}"
        super().__init__(message)
