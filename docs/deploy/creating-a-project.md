# Create a Project

The `pyagentic init` command scaffolds a complete agent project — an agent module, a manifest, dependencies, and a virtual environment — so you can go from zero to a running agent in seconds.

## Quick Start

```bash
pyagentic init my-agent
cd my-agent
```

This creates a directory with everything you need:

```
my-agent/
├── my_agent.py          # Your agent class
├── pyagentic.toml       # Project manifest
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── .venv/               # Virtual environment (auto-created)
```

## Templates

Two project templates are available. Both produce a fully runnable project; they differ in how much starter code you get.

### Minimal (default)

A bare-bones agent with a system message and a commented-out tool example:

```bash
pyagentic init my-agent
```

```python
from pyagentic import BaseAgent


class MyAgent(BaseAgent):
    __system_message__ = "You are a helpful assistant."

    # Add tools with the @tool decorator:
    #
    # from pyagentic import tool
    #
    # @tool("Describe what this tool does")
    # def my_tool(self, query: str) -> str:
    #     return f"Result for {query}"
```

### Full

A more complete starting point with state fields, two tools, and a richer manifest description:

```bash
pyagentic init my-agent --template full
```

```python
from pyagentic import BaseAgent, tool, State, spec


class MyAgent(BaseAgent):
    """A research agent that can search and summarize information."""

    __system_message__ = """You are a helpful research assistant.
Use your tools to find and summarize information for the user.
Always cite your sources when providing information."""

    notes: State[list] = spec.State(
        default_factory=list,
        access="readwrite",
        get_description="Retrieve all saved research notes",
        set_description="Update the research notes",
    )

    @tool("Search for information on a topic and return a summary")
    def search(self, query: str) -> str:
        """Replace this with a real search implementation."""
        return f"[Placeholder] Search results for: {query}"

    @tool("Save a research note for later reference")
    def save_note(self, note: str) -> str:
        self.notes.append(note)
        return f"Note saved. Total notes: {len(self.notes)}"
```

## Options

```bash
pyagentic init my-agent --template full --no-venv
```

- `--template`, `-t` — Template to use: `minimal` (default) or `full`.
- `--no-venv` — Skip virtual environment creation and dependency installation. Useful if you manage your own environment with `uv`, `conda`, etc.

## What Gets Generated

### `pyagentic.toml`

The project manifest. The CLI reads this file to discover your agent, configure the server, and drive the Docker build:

```toml
[project]
name = "my-agent"
version = "0.1.0"
description = "A helpful assistant"

[agent]
entry = "my_agent:MyAgent"   # module:ClassName
model = "openai::gpt-4o"     # default LLM

[server]
host = "0.0.0.0"
port = 8000

[build]
python_version = "3.13"
dependencies = []             # extra pip packages for the image

[env]
required = ["OPENAI_API_KEY"] # env vars that must be set at runtime
```

- `[project]` — Name, version, and description metadata.
- `[agent]` — Entry point in `module:ClassName` format and default LLM model. PyAgentic resolves the entry point at startup via `importlib`, so your agent module just needs to be importable from the project root.
- `[server]` — Host and port for the HTTP server.
- `[build]` — Python version and extra pip dependencies for the Docker image.
- `[env]` — Environment variable names that must be set at runtime.

### `requirements.txt`

Pre-populated with `pyagentic-core[deploy]`. Add any additional dependencies your agent needs here — they'll be picked up by both the virtual environment setup and the Docker build.

### `.env.example`

A template listing the environment variables your agent requires. Copy it to `.env` and fill in real values:

```bash
cp .env.example .env
```

The generated `.gitignore` already excludes `.env`, so secrets won't be committed accidentally.

### Virtual Environment

By default, `pyagentic init` creates a `.venv/` and installs `requirements.txt` automatically. It uses `uv` if available on your `PATH`, falling back to `python -m venv` + `pip`.

## Project Name Conventions

The project name you provide is automatically converted into a Python module name and class name:

```bash
pyagentic init my-agent
# Module: my_agent.py
# Class:  MyAgent
# Entry:  my_agent:MyAgent
```

Hyphens and spaces become underscores for the module; each word is capitalized for the class. The entry point in `pyagentic.toml` is set automatically.

## Next Steps

Once your project is scaffolded:

1. Edit the agent class in your module file — add tools, state, and a system message
2. Set your environment variables in `.env`
3. [Run your agent](running.md) with `pyagentic run`
