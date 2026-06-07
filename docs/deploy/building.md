# Build & Deploy

Once your agent is working locally, you can package it as a Docker image and push it to any container registry. The CLI handles Dockerfile generation, build context management, and publishing — your project directory is never modified.

## Building

```bash
pyagentic build
```

This reads the manifest, generates a Dockerfile, copies your project into a temporary build context, and runs `docker build`:

```
Building image for my-agent v0.1.0...
Image built: my-agent:0.1.0
```

### Options

```bash
# Custom tag
pyagentic build --tag my-agent:latest

# Clean build without cache
pyagentic build --no-cache
```

- `--tag`, `-t` — Docker image tag. Defaults to `<name>:<version>` from the manifest.
- `--no-cache` — Pass `--no-cache` to `docker build`.

### Generated Dockerfile

The build command generates a Dockerfile based on your manifest's `[build]` section:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system pyagentic-core[deploy]

COPY . .
EXPOSE 8000
CMD ["pyagentic", "run", "--host", "0.0.0.0"]
```

If your manifest includes extra `[build].dependencies`, they're installed as an additional step:

```toml
[build]
dependencies = ["pandas", "numpy"]
```

```dockerfile
# Added automatically:
RUN uv pip install --system "pandas" "numpy"
```

### Build Context

The build uses a temporary directory — your project tree is never touched. Files are copied in, excluding hidden files (`.env`, `.git/`) and `__pycache__/`. If `requirements.txt` is missing, a default one containing `pyagentic-core[deploy]` is generated.

## Publishing

```bash
pyagentic publish
```

This will:

1. Check if the image exists locally — if not, build it first
2. Tag the image for the target registry
3. Push via `docker push`

```
Pushing my-agent:0.1.0...
Published: my-agent:0.1.0
```

### Options

```bash
# Push to GitHub Container Registry
pyagentic publish --registry ghcr.io/myorg

# Push a specific tag to a registry
pyagentic publish --tag my-agent:latest --registry ghcr.io/myorg
```

- `--registry`, `-r` — Target registry prefix (e.g. `ghcr.io/myorg`). Defaults to Docker Hub.
- `--tag`, `-t` — Image tag override. Defaults to `<name>:<version>` from the manifest.

Make sure you're logged in to your target registry before publishing:

```bash
# Docker Hub
docker login

# GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## Running the Image

Once built, run the image to verify everything works:

```bash
docker run -p 8000:8000 --env-file .env my-agent:0.1.0
```

The container starts the FastAPI server exactly as `pyagentic run` would. Pass environment variables via `--env-file` or `-e`:

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  my-agent:0.1.0
```

## Request Lifecycle

Understanding what happens between an HTTP request and the agent's response helps when debugging or extending the server.

### Startup

When `pyagentic run` starts (or the Docker container boots), `create_app()` builds the FastAPI application:

1. The manifest is parsed into a `Manifest` model
2. The agent class is resolved from `[agent].entry` via `importlib`
3. The metaclass-generated models (`__request_model__`, `__response_model__`, `__stream_event_model__`, `__state_class__`) are extracted
4. A `SessionManager` is created — an in-memory store holding one agent instance per session
5. All routes are wired to the session manager

### Session Lifecycle

```
create session ──▶ chat / stream (repeat) ──▶ delete session
     │                     │                        │
  new agent          same instance              removed from
  instance           + accumulating state       memory
```

Each session gets its own agent instance rather than sharing one. This means conversation history is isolated, state changes don't leak between sessions, and sessions can use different models or API keys.

Sessions are stored in memory — restarting the server clears all sessions.

### Chat Request Flow

When a chat request hits `/sessions/{id}/chat`:

1. The session ID is resolved in the `SessionManager` (404 if not found)
2. Request body fields are unpacked into keyword arguments
3. `agent(**kwargs)` is called, triggering the full [runtime execution flow](../reference/architecture.md#runtime-phase):
    - User message added to conversation history
    - LLM called with system message, history, and tool schemas
    - If tool calls are requested, they're executed and the loop continues up to `max_call_depth`
4. The `AgentResponse` is serialized and returned

The streaming endpoint (`/sessions/{id}/chat/stream`) follows the same path but uses `agent.step()` instead, yielding `LLMResponse`, `ToolResponse`, and `AgentResponse` events incrementally as SSE.

## Next Steps

- Review the [architecture reference](../reference/architecture.md) for the full runtime internals
- Learn about [execution modes](../execution-modes.md) for calling agents directly from Python
- Explore [observability](../observability.md) to trace and monitor agent behavior in production
