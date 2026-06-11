# Deploying

Your app is a standard ASGI application, so it deploys like any FastAPI service:
run `uvicorn main:app` in a container. PyAgentic can generate a Dockerfile for
you from `agents.toml`.

## The `[deploy]` section

Add a `[deploy]` section to `agents.toml` (beside your `pyproject.toml`):

```toml
[deploy]
target = "main:app"        # the ASGI app uvicorn serves
python_version = "3.13"
dependencies = []          # extra pip packages to install in the image
port = 8000
env = ["OPENAI_API_KEY"]   # environment variables required at runtime
```

## Generating a Dockerfile

```python
from pyagentic.api import write_dockerfile

write_dockerfile()   # reads ./agents.toml, writes ./Dockerfile
```

Or get the contents as a string with `generate_dockerfile()`. The generated
Dockerfile installs your project and runs it with uvicorn:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN pip install uv

RUN uv pip install --system "pyagentic-core[api]"

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Extra `[deploy].dependencies` are installed as an additional step:

```toml
[deploy]
dependencies = ["pandas", "numpy"]
```

```dockerfile
# Added automatically:
RUN uv pip install --system "pandas" "numpy"
```

## Building and running

Build and run with your normal Docker tooling:

```bash
docker build -t my-agents:0.1.0 .
docker run -p 8000:8000 --env-file .env my-agents:0.1.0
```

Pass secrets via `--env-file` or `-e` (the variables you listed in
`[deploy].env`):

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... my-agents:0.1.0
```

!!! tip "Durable jobs in a container"
    If you use [async jobs](jobs.md), set `[jobs].store` to a path on a mounted
    volume so job records survive container restarts. The default
    `.pyagentic/jobs.db` lives in the container's filesystem and is lost when the
    container is removed.

## Next steps

- Review [async jobs](jobs.md) for long-running deployments
- Explore [observability](../observability.md) to trace agents in production
