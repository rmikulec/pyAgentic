# Async Jobs

A `/chat` request runs the agent inside the HTTP request and returns when it's
done. That's fine for quick calls, but an agent that researches, calls tools,
and chains for several minutes will outlast client read timeouts, load-balancer
idle limits, and flaky connections.

The job system solves this. You **submit** an agent run, it executes in the
background as a **durable record**, and you **stream or poll** its updates
later — surviving timeouts, disconnects, and reconnects.

## Enabling jobs

Jobs are opt-in. Turn them on in code:

```python
from pyagentic.api import create_app

app = create_app(MyAgent, model="openai::gpt-4o", jobs=True)
```

or in `agents.toml`:

```toml
[jobs]
enabled = true
store = ".pyagentic/jobs.db"   # SQLite path; ":memory:" for ephemeral
admission_cap = 16             # max jobs in-flight at once
max_concurrency = 8            # max concurrent agent runs
ttl = "24h"                    # how long terminal job records are kept
cleanup_interval_seconds = 300
```

This mounts a `/jobs` API per agent (under each agent's prefix in a multi-agent
app) and persists job records and their update logs to SQLite, so they survive a
server restart.

## The flow

```python
import httpx

# 1. Submit — returns immediately with a job id (202)
r = httpx.post("http://localhost:8000/jobs",
               json={"input": {"user_input": "Research AI safety"}})
job_id = r.json()["job_id"]

# 2. Stream updates as the agent works (resumable — see below)
with httpx.stream("GET", f"http://localhost:8000/jobs/{job_id}/stream") as s:
    for line in s.iter_lines():
        print(line)

# 3. Or just fetch the final result when you're ready
result = httpx.get(f"http://localhost:8000/jobs/{job_id}").json()
```

## Endpoints

| Route | Purpose |
|---|---|
| `POST /jobs` | Submit a run. Body `{"input": {<chat fields>}, "session_id": <optional>}`. Returns `202 {job_id, status}`. |
| `GET /jobs` | List jobs, newest first. Filter with `?status=` and `?session_id=`. |
| `GET /jobs/{id}` | Job status and, when terminal, its result. |
| `GET /jobs/{id}/updates?since=<seq>` | The update log past a sequence cursor. |
| `GET /jobs/{id}/stream` | SSE stream of updates: replay from cursor, then live tail. |
| `POST /jobs/{id}/cancel` | Cancel a queued or running job. |

The `input` object is your agent's normal chat request body. Pass a
`session_id` (from `POST /sessions`) to run the job against an existing session;
omit it for a one-off run.

## Surviving timeouts: replay-from-cursor

Every update carries a monotonic, gapless sequence number (`seq`), emitted on the
SSE stream as the frame `id`. If the connection drops, reconnect with the
`Last-Event-ID` header (or `?since=<seq>`) and the stream replays **only** what
you missed, then resumes the live tail:

```
event: llm_response
id: 0
data: {...}

event: tool_response
id: 1
data: {...}
```

```bash
# Reconnect after seq 1 — replays seq 2 onward, then tails live
curl -N http://localhost:8000/jobs/{id}/stream \
  -H "Last-Event-ID: 1"
```

The stream closes after a terminal event (`agent_response`, `job_failed`, or
`job_cancelled`). Because the store is the single source of truth, a client can
disconnect for any reason and pick up exactly where it left off — the agent run
keeps going server-side regardless.

## How it works

- **Durable store** — a `JobStore` (SQLite by default) holds each job record and
  its append-only update log. Restarting the server keeps finished jobs and
  their logs; an in-flight `running` job is marked failed on restart (execution
  is not resumed), and `queued` jobs are re-dispatched.
- **In-process backend** — runs `agent.step()` in the server's event loop and
  emits each update. Session-bound jobs run on the session's live agent;
  the orchestrator serializes a session's jobs FIFO (one in-flight per session).
- **Admission cap** — a single process-wide cap bounds concurrent runs; jobs
  beyond it stay `queued` until a slot frees.

## Next steps

- [Deploy](deploying.md) your app — set a persistent `[jobs].store` path in the container
- Revisit the [synchronous chat endpoints](running.md#chat) for quick calls
