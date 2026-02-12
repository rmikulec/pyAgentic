"""Exception classes for PyAgentic serve infrastructure."""


class ContainerStartError(RuntimeError):
    """Docker container failed to start or pass health checks.

    Args:
        image (str): The Docker image that failed to start.
        reason (str): Description of the failure.
    """

    def __init__(self, image: str, reason: str) -> None:
        self.image = image
        self.reason = reason
        super().__init__(f"Container for '{image}' failed to start: {reason}")


class ContainerNotRunningError(RuntimeError):
    """Operation requires a running container but none is active.

    Args:
        image (str): The Docker image expected to be running.
    """

    def __init__(self, image: str) -> None:
        self.image = image
        super().__init__(f"No running container for '{image}'")


class AgentAPIError(RuntimeError):
    """The agent HTTP API returned a non-success status code.

    Args:
        status_code (int): HTTP status code from the agent API.
        detail (str): Error detail from the response body.
        endpoint (str): The API endpoint that was called.
    """

    def __init__(self, status_code: int, detail: str, endpoint: str) -> None:
        self.status_code = status_code
        self.detail = detail
        self.endpoint = endpoint
        super().__init__(
            f"Agent API error on {endpoint}: {status_code} — {detail}"
        )


class ImageNotFoundError(RuntimeError):
    """Docker image not found locally and could not be pulled.

    Args:
        image (str): The Docker image that was not found.
        reason (str): Description of the failure.
    """

    def __init__(self, image: str, reason: str) -> None:
        self.image = image
        self.reason = reason
        super().__init__(f"Image '{image}' not found: {reason}")
