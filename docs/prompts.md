# Prompt Management

Every agent declares its instructions — the system prompt the LLM receives — with the `__instructions__` class attribute. Instructions can be a plain string, but as agents mature, prompts tend to outgrow the code: they get edited more often than the logic around them, they need versioning, and non-developers may need to change them. PyAgentic's prompt engines let agents pull their instructions from a managed source instead of a hardcoded string.

## Naming

Two terms, one for each side of the rendering step:

- **Instructions** — the raw template you declare: a string or a prompt-engine reference. Available at runtime as `agent.state.instructions`.
- **System message** — the rendered result actually sent to the LLM, with state interpolated into the template. Available as `agent.state.system_message`.

```python
class ResearchAgent(BaseAgent):
    __instructions__ = "You research {{ topic }}"

    topic: State[str] = spec.State(default="transformers")

agent.state.instructions    # "You research {{ topic }}"
agent.state.system_message  # "You research transformers"
```

!!! note "Deprecated: `__system_message__`"
    Older versions used `__system_message__` for what is now `__instructions__`. The old name still works but emits a `DeprecationWarning` — new code should declare `__instructions__`.

## Prompt Engines

A `PromptEngine` is a source of managed prompts. Engines load prompt text by key and return a `PromptSource` carrying the text plus metadata (where it came from, which version, when it was loaded). Instead of assigning a string, assign a reference:

```python
from pyagentic import BaseAgent, LocalPromptEngine

prompts = LocalPromptEngine(".prompts")

class ResearchAgent(BaseAgent):
    __instructions__ = prompts.ref("researcher")
```

`ref(key)` returns a `PromptRef` — a deferred pointer that is resolved **when the agent is instantiated**, not when the class is defined:

- Every `ResearchAgent(...)` re-reads the prompt, so edits are picked up by new instances without restarting anything.
- The prompt is stable for the lifetime of an instance — it never changes mid-conversation.
- `agent.fork()` re-resolves too, so forked linked-agent calls also get the latest version.

To pin a specific version instead of tracking the latest:

```python
__instructions__ = prompts.ref("researcher", version="v2")
```

## LocalPromptEngine

`LocalPromptEngine` serves prompts from files on the local file system. A **pattern** defines how your prompt storage is laid out under the root directory (default: `.prompts/` in the working directory). The pattern is a template with a required `{key}` placeholder and an optional `{version}` placeholder.

### Flat layout (default)

The default pattern is `"{key}.md"` — one file per prompt, versioned by content hash:

```
.prompts/
├── researcher.md
└── summarizer.md
```

```python
prompts = LocalPromptEngine(".prompts")
prompts.load("researcher")  # reads .prompts/researcher.md
```

Any flat layout works by changing the pattern, e.g. `pattern="{key}.txt"` or `pattern="agents/{key}/prompt.md"`.

### Versioned layout

When the pattern contains `{version}`, the file structure holds the versions and the engine derives versioning from it:

```
.prompts/
└── researcher/
    ├── v1.md
    ├── v2.md
    └── v10.md
```

```python
prompts = LocalPromptEngine(".prompts", pattern="{key}/{version}.md")

prompts.load("researcher")                # latest → v10
prompts.load("researcher", version="v1")  # pinned
```

An unpinned load picks the latest version using natural sorting, so `v10` correctly orders after `v2`. Version names are up to you — `v1`, `2026-07-08`, `1.0.3` all work. Flat versioned layouts like `pattern="{key}_{version}.md"` work the same way.

A missing key (or pinned version) raises `PromptNotFound`. Requesting a `version` on a pattern without `{version}` raises `ValueError`.

## Prompt provenance on responses

Every agent response records which prompt produced it in the `prompt` field, as a `PromptSource`:

```python
response = await agent.run("hello")

response.prompt.text         # the raw template that was loaded
response.prompt.source       # ".prompts/researcher/v2.md"
response.prompt.source_type  # "local"
response.prompt.version      # "v2"
response.prompt.loaded_at    # when it was resolved
```

This works for plain-string instructions too: they produce an `inline` source named after the agent, with a content-hash version:

```python
class PlainAgent(BaseAgent):
    __instructions__ = "You are plain"

response.prompt.source_type  # "inline"
response.prompt.source       # "PlainAgent"
response.prompt.version      # "3f2a9c1b04de" (content hash)
```

Because the version is always populated — from the file structure or a content hash — you can group or diff runs by prompt version regardless of where the prompt lives. The same metadata is available on the agent as `agent.state.prompt_source`.

## Writing a custom engine

Engines for other backends (a database, object storage, a prompt-management service) subclass `PromptEngine` and implement one method:

```python
from pyagentic import PromptEngine, PromptSource

class MyServiceEngine(PromptEngine):
    def load(self, key: str, version: str | None = None) -> PromptSource:
        record = my_service.get_prompt(key, version=version or "latest")
        return PromptSource(
            text=record.text,
            source=f"my-service/{key}",
            source_type="my-service",
            version=record.version,
            loaded_at=record.fetched_at,
        )
```

`ref()`, instantiation-time resolution, and response provenance all come from the base class — a custom engine only decides how to fetch text and what to call a version.
