# Deploy

PyAgentic includes a CLI and server framework that takes your agent from a Python class to a running HTTP service. Scaffold a project, serve it locally, and ship it as a Docker image — all without leaving the terminal.

## Installation

The deploy tools ship as an optional extra:

```bash
pip install pyagentic-core[deploy]
```

This installs the `pyagentic` CLI along with FastAPI, uvicorn, and Docker utilities.

## Guides

<div class="grid cards" markdown>

- :material-folder-plus: **[Create a Project](creating-a-project.md)**

    ---

    Scaffold a new agent project with templates, a manifest, and a virtual environment.

    [:octicons-arrow-right-24: Get started](creating-a-project.md)

- :material-play-circle: **[Run Your Project](running.md)**

    ---

    Start your agent as a FastAPI server or interactive REPL, explore the auto-generated API.

    [:octicons-arrow-right-24: Run locally](running.md)

- :material-docker: **[Build & Deploy](building.md)**

    ---

    Package your agent as a Docker image, push it to a registry, and understand the request lifecycle.

    [:octicons-arrow-right-24: Ship it](building.md)

</div>
