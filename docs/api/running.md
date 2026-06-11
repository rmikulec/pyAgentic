# Running

Your app is an ordinary ASGI application, so you run it with uvicorn (or any
ASGI server). The API is generated from your agent's class — tools, state, and
input signature all become typed endpoints with zero configuration.

## Run with uvicorn

```bash
uvicorn main:app --reload
```

`main:app` points at the `app = create_app(...)` in your `main.py`. Use
`--reload` for auto-restart during development, and `--host`/`--port` to change
the bind address:

```bash
uvicorn main:app --host 0.0.0.0 --port 3000
```

The interactive OpenAPI docs are available at `http://localhost:8000/docs`.

## API endpoints

All request and response schemas are derived from your agent's
metaclass-generated models — adding a tool parameter or state field
automatically updates the API. (For a multi-agent app, these live under each
agent's prefix, e.g. `/research/sessions`.)

### Info routes

```bash
# Agent metadata: name, version, tools, state fields, linked agents
curl http://localhost:8000/

# Liveness probe
curl http://localhost:8000/health

# JSON schemas for request, response, stream event, and state models
curl http://localhost:8000/schema
```

### Session management

Sessions provide isolated agent instances with independent state and
conversation history. Each session holds its own agent, so state changes in one
session don't affect others.

```bash
# Create a new session (returns session_id)
curl -X POST http://localhost:8000/sessions

# Create with a model or API key override
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai::gpt-4o", "api_key": "sk-..."}'

# List all active session IDs
curl http://localhost:8000/sessions

# Delete a session
curl -X DELETE http://localhost:8000/sessions/{session_id}
```

Sessions are stored in memory — restarting the server clears them.

### Chat

The chat request body is automatically derived from your agent's `__call__`
signature. With the default signature, the body is `{"user_input": "..."}`; if
you override `__call__` with custom parameters, the schema updates to match.

```bash
curl -X POST http://localhost:8000/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello!"}'
```

### Streaming

The stream endpoint returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
with typed events as the agent works:

```bash
curl -N -X POST http://localhost:8000/sessions/{session_id}/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Research AI safety"}'
```

```
event: llm_response
data: {"event": "llm_response", "data": {...}}

event: tool_response
data: {"event": "tool_response", "data": {...}}

event: agent_response
data: {"event": "agent_response", "data": {...}}
```

- `llm_response` — Fired after each LLM inference. The model's text and any tool calls.
- `tool_response` — Fired after each tool execution. The tool name, arguments, and result.
- `agent_response` — Fired once at the end with the complete `AgentResponse`.

This maps directly to the three response types from
[`agent.step()`](../execution-modes.md): `LLMResponse`, `ToolResponse`, and
`AgentResponse`.

!!! tip "Long-running agents"
    `/chat` and `/chat/stream` run within the HTTP request — fine for quick
    calls, but a multi-minute agent run can outlive client or proxy timeouts.
    For those, submit a [durable job](jobs.md) instead and stream its updates.

### State

```bash
# Get the current agent state for a session
curl http://localhost:8000/sessions/{session_id}/state
```

Returns the full state model, serialized using the agent's `__state_class__`.

## How it works

`create_app` extracts the agent's metaclass-generated models and wires them to
the routes:

- `__request_model__` → chat request body
- `__response_model__` → chat response body
- `__stream_event_model__` → typed SSE events
- `__state_class__` → state endpoint

A `SessionManager` (an in-memory store holding one agent instance per session)
backs the session routes. The streaming endpoint drives `agent.step()`; the
synchronous endpoint calls `agent(**kwargs)`.

## Next steps

- Add [durable async jobs](jobs.md) for long-running calls
- [Deploy](deploying.md) your app as a Docker image
- Review the [architecture reference](../reference/architecture.md) for runtime internals
