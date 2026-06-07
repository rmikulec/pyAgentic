"""Shared SSE (Server-Sent Events) parsing utilities."""

import json
from typing import AsyncGenerator, Generator, Optional

import httpx


def parse_sse_sync(resp: httpx.Response) -> Generator[dict, None, None]:
    """Parse SSE lines from an httpx sync streaming response.

    Args:
        resp (httpx.Response): A streaming response to parse SSE from.

    Yields:
        dict: An SSE event with ``"event"`` (str) and ``"data"`` (dict or str)
            keys.
    """
    event_type: Optional[str] = None
    for line in resp.iter_lines():
        line = line.strip()
        if not line:
            event_type = None
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            raw = line[len("data:"):].strip()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                data = raw
            yield {"event": event_type or "message", "data": data}


async def parse_sse_async(resp: httpx.Response) -> AsyncGenerator[dict, None]:
    """Parse SSE lines from an httpx async streaming response.

    Args:
        resp (httpx.Response): An async streaming response to parse SSE from.

    Yields:
        dict: An SSE event with ``"event"`` (str) and ``"data"`` (dict or str)
            keys.
    """
    event_type: Optional[str] = None
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line:
            event_type = None
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            raw = line[len("data:"):].strip()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                data = raw
            yield {"event": event_type or "message", "data": data}
