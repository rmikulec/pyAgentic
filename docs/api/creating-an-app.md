# Creating an App

`create_app` builds a standalone FastAPI application from one or more agents.
`create_router` builds an `APIRouter` you can mount into an app you already
have. Both derive their routes from the agent's metaclass-generated models.

## A single agent

Pass the class; it's served at the root:

```python
from pyagentic.api import create_app

app = create_app(MyAgent, model="openai::gpt-4o")
```

`model` is the default LLM used for sessions that don't override it. `name`,
`version`, and `description` set the FastAPI metadata (and default from
`agents.toml`, see below).

## Multiple agents

Pass a list and each agent is mounted under a prefix derived from its class name
(`ResearchAgent` → `/research`, `WriterAgent` → `/writer`). A top-level
`GET /` index lists what's mounted, and `GET /health` is added:

```python
app = create_app([ResearchAgent, WriterAgent])
# /research/sessions, /research/chat, ...
# /writer/sessions,   /writer/chat,   ...
```

For explicit prefixes, pass a `{prefix: agent}` mapping:

```python
app = create_app({"/research": ResearchAgent, "/writer": WriterAgent})
```

Each agent gets its own isolated `SessionManager` — sessions, state, and (if
enabled) jobs never cross between agents.

!!! note "Linked agents stay subordinate"
    Agents connected with `Link[...]` are called through their parent agent's
    tools, not exposed as separate endpoints. "Multiple agents" here means
    several independent top-level agents on one app.

## Mounting into an existing app

Use `create_router` to add an agent to a FastAPI app you already run:

```python
from fastapi import FastAPI
from pyagentic.api import create_router

app = FastAPI()
app.include_router(create_router(MyAgent, model="openai::gpt-4o"), prefix="/bot")
```

The router owns its `SessionManager`, exposed as `router.sessions` so you can
share it (for example with `mount_mcp`).

## Dependencies

Agents often need resources that can't travel over HTTP — a database handle, an
HTTP client, a pre-configured provider. Declare these with `Depends[...]` in the
agent class, exactly where you'd otherwise put `State[...]`:

```python
from pyagentic import BaseAgent, State, Depends

class ResearchAgent(BaseAgent):
    __instructions__ = "You research topics."

    topic: State[Topic]        # client-provided per session
    db:    Depends[Database]    # injected server-side, never sent by clients
```

A `Depends[T]` field is **excluded** from the session/job request body. Instead
you supply it once when building the app, via `dependencies=` — a list of
instances or zero-arg factories, resolved **by type**:

```python
app = create_app(ResearchAgent, dependencies=[Database(dsn), make_client])
```

- An **instance** (`Database(dsn)`) is shared across every session.
- A **factory** — a zero-arg callable annotated with its return type
  (`def make_client() -> Client: ...`) — is called fresh for each agent built,
  so every session and job gets its own.

Each `Depends[T]` slot — including those declared on linked sub-agents — is
matched to a provider of type `T`. Missing or mismatched dependencies fail fast
when the app is built, not on the first request.

For a multi-agent app, pass a flat list (applied to every agent) or a dict keyed
by mount prefix to scope providers per agent:

```python
app = create_app(
    {"/research": ResearchAgent, "/writer": WriterAgent},
    dependencies={"/research": [Database(dsn)], "/writer": [make_client]},
)
```

Linked agents are built from the request body — a session's body nests each
`Link[...]` sub-agent's construction under its field name — while their
`Depends[...]` fields are injected from the same `dependencies` list.

## Configuration: `agents.toml`

Put an `agents.toml` next to your `pyproject.toml`. `create_app` reads its
`[app]` section for defaults; explicit keyword arguments override it.

```toml
[app]
name = "my-agents"
version = "0.1.0"
description = "My agent service"
model = "openai::gpt-4o"   # default model for sessions
```

```python
app = create_app(MyAgent)          # name/version/model come from agents.toml
app = create_app(MyAgent, model="anthropic::claude-sonnet-4-6")  # override
```

`agents.toml` also has `[deploy]` (see [Deploying](deploying.md)) and `[jobs]`
(see [Async Jobs](jobs.md)) sections.

## Exposing the agent over MCP

Pass `mcp=True` to also mount a [Model Context Protocol](https://modelcontextprotocol.io)
endpoint per agent at `<prefix>/mcp`, sharing the same sessions as the HTTP routes:

```python
app = create_app(MyAgent, mcp=True)   # MCP server at /mcp
```

To mount MCP onto your own app, use `mount_mcp`:

```python
from pyagentic.api import create_router, mount_mcp

router = create_router(MyAgent)
app.include_router(router, prefix="/bot")
mount_mcp(app, MyAgent, sessions=router.sessions, path="/bot/mcp")
```

`mount_mcp` requires the `fastmcp` extra (`pip install pyagentic-core[mcp]`).

## Next steps

- [Run your app](running.md) and explore the HTTP API
- Enable [durable async jobs](jobs.md) for long-running calls
- [Deploy](deploying.md) it as a container
