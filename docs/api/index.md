# API

PyAgentic turns an agent class into a FastAPI application (or a router you mount
into your own app), with an HTTP API generated automatically from the agent's
tools, state, and input signature. Optionally expose it over MCP, run long calls
as durable background jobs, and generate a Dockerfile to ship it.

## Installation

The serving tools ship as an optional extra:

```bash
pip install pyagentic-core[api]
```

This installs FastAPI and uvicorn alongside the core package.

## Quickstart

Write a `main.py` that builds the app from your agent class:

```python
# main.py
from pyagentic import BaseAgent, tool
from pyagentic.api import create_app


class AssistantAgent(BaseAgent):
    __instructions__ = "You are a helpful assistant."

    @tool("Add two numbers")
    def add(self, a: int, b: int) -> str:
        return str(a + b)


app = create_app(AssistantAgent, model="openai::gpt-4o")
```

Run it with uvicorn:

```bash
uvicorn main:app --reload
```

That's the whole loop — no CLI, no scaffolding. Your agent is now served at
`http://localhost:8000` with session management, chat, streaming, and a schema
endpoint, all derived from the class.

## What you get

The generated API always reflects your agent — add a tool parameter or state
field and the schemas update with no manual OpenAPI maintenance:

| Route | Purpose |
|---|---|
| `GET /` | Agent metadata (name, tools, state fields, linked agents, dependencies) |
| `GET /health` | Liveness probe |
| `GET /schema` | JSON schemas for construct/request/response/stream-event/state |
| `POST /sessions` | Create an isolated session (body mirrors the agent's constructor) |
| `POST /sessions/{id}/chat` | Send a message, get a complete response |
| `POST /sessions/{id}/chat/stream` | Stream typed SSE events as the agent works |
| `GET /sessions/{id}/state` | Current agent state for a session |

Creating a session constructs the agent: the body carries its state and any
linked sub-agents (`POST /sessions`), while non-serializable resources are
injected server-side with [`Depends[...]`](creating-an-app.md#dependencies).

## Guides

<div class="grid cards" markdown>

- :material-application-cog: **[Creating an App](creating-an-app.md)**

    ---

    Build an app or router from one or many agents, configure it with
    `agents.toml`, and mount it into an existing FastAPI app.

    [:octicons-arrow-right-24: Build it](creating-an-app.md)

- :material-play-circle: **[Running](running.md)**

    ---

    Run with uvicorn and explore the auto-generated HTTP API: sessions, chat,
    streaming, and state.

    [:octicons-arrow-right-24: Run locally](running.md)

- :material-clock-fast: **[Async Jobs](jobs.md)**

    ---

    Run agents that take longer than an HTTP request: submit a durable job and
    stream its updates, surviving timeouts and reconnects.

    [:octicons-arrow-right-24: Go async](jobs.md)

- :material-docker: **[Deploying](deploying.md)**

    ---

    Generate a Dockerfile from `agents.toml` and ship your agent as a container.

    [:octicons-arrow-right-24: Ship it](deploying.md)

</div>
