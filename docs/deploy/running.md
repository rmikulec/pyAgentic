# Run Your Project

The `pyagentic run` command starts your agent either as a FastAPI HTTP server or as an interactive REPL for local testing. The server's API is generated automatically from your agent's class definition — tools, state, and input signature all become typed endpoints with zero configuration.

## HTTP Server

```bash
pyagentic run
```

This reads `pyagentic.toml`, loads your agent class, and starts a FastAPI server with uvicorn:

```
Serving my-agent v0.1.0
Agent: MyAgent
Listening on http://0.0.0.0:8000
```

### Options

```bash
pyagentic run --host 127.0.0.1 --port 3000 --reload
```

- `--host`, `-h` — Bind address. Falls back to `[server].host` in the manifest.
- `--port`, `-p` — Bind port. Falls back to `[server].port` in the manifest.
- `--reload` — Enable uvicorn auto-reload on file changes (development mode).
- `--repl` — Start an interactive REPL instead of the HTTP server.

## Interactive REPL

For quick local testing without HTTP:

```bash
pyagentic run --repl
```

```
PyAgentic REPL — my-agent v0.1.0
Agent: MyAgent
Type 'exit' or 'quit' to stop.

>>> Hello, what can you do?

I'm a helpful assistant! I can...
```

The REPL creates a single agent instance and sends each line of input through `agent.run()`. Type `exit`, `quit`, or press `Ctrl+C` to stop.

## API Endpoints

When running as an HTTP server, the following endpoints are available. All request and response schemas are derived from your agent's metaclass-generated models — adding a tool parameter or state field automatically updates the API.

### Info Routes

```bash
# Agent metadata: name, version, tools, state fields, linked agents
curl http://localhost:8000/

# Liveness probe
curl http://localhost:8000/health

# JSON schemas for request, response, stream event, and state models
curl http://localhost:8000/schema
```

### Session Management

Sessions provide isolated agent instances with independent state and conversation history. Each session holds its own agent, so state changes in one session don't affect others.

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

### Chat

The chat request body is automatically derived from your agent's `__call__` signature. If you use the default signature, the body is `{"user_input": "..."}`. If you override `__call__` with custom parameters, the schema updates to match.

```bash
# Synchronous — send a message, get a complete response
curl -X POST http://localhost:8000/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello!"}'
```

### Streaming

The stream endpoint returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) with typed events as the agent works:

```bash
curl -N http://localhost:8000/sessions/{session_id}/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Research AI safety"}'
```

Each event is typed so you can handle them differently in your client:

```
event: llm_response
data: {"event": "llm_response", "data": {...}}

event: tool_response
data: {"event": "tool_response", "data": {...}}

event: agent_response
data: {"event": "agent_response", "data": {...}}
```

- `llm_response` — Fired after each LLM inference. Contains the model's text and any tool calls it wants to make.
- `tool_response` — Fired after each tool execution. Contains the tool name, arguments, and result.
- `agent_response` — Fired once at the end with the complete `AgentResponse`.

This maps directly to the three response types from [`agent.step()`](../execution-modes.md#step-agentstep): `LLMResponse`, `ToolResponse`, and `AgentResponse`.

### State

```bash
# Get the current agent state for a session
curl http://localhost:8000/sessions/{session_id}/state
```

Returns the full state model including computed fields, serialized using the agent's `__state_class__`.

## How It Works

Under the hood, `pyagentic run` does four things:

1. **Loads the manifest** — `pyagentic.toml` is parsed into a `Manifest` model.
2. **Discovers the agent class** — The `[agent].entry` field (e.g. `my_agent:MyAgent`) is resolved via `importlib`.
3. **Builds a FastAPI app** — Routes are wired to the agent's metaclass-generated models:
    - `__request_model__` → chat request body
    - `__response_model__` → chat response body
    - `__stream_event_model__` → typed SSE events
    - `__state_class__` → state endpoint
4. **Starts uvicorn** on the configured host and port.

This means the API schemas always reflect your agent's current tools, state, and input types with no manual OpenAPI maintenance.

## Next Steps

- [Build & deploy](building.md) your agent as a Docker image
- Learn about [execution modes](../execution-modes.md) for direct Python usage
- Explore [tools](../tools.md) and [state management](../states.md) to add capabilities
