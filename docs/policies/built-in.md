# Built-in Policies

PyAgentic ships four ready-to-use policies, all focused on **context management** —
keeping the message history sent to the LLM small so long-running agents don't
blow up their context window (or your token bill).

All of them live in `pyagentic.policies`:

```python
from pyagentic.policies import (
    ToolOutputClipPolicy,
    ToolEvictionPolicy,
    SlidingWindowPolicy,
    CompactionPolicy,
)
```

Attach them to an agent with the `__message_policies__` class attribute:

```python
class ResearchAgent(BaseAgent):
    __system_message__ = "You research topics using tools."
    __message_policies__ = [
        ToolOutputClipPolicy(max_chars=8000),
        ToolEvictionPolicy(keep_last_n=5),
        CompactionPolicy(max_input_tokens=80_000),
    ]
```

They compose — the list above clips oversized tool outputs the moment they enter
context, stubs out stale tool results as the conversation moves on, and
summarizes old history if the context still crosses the token threshold.

!!! note
    Policies only ever shape the **working context** (`state._context`) that
    providers consume. The raw history (`state._messages` / `state.raw_messages`)
    always keeps every message untouched, for debugging and auditing. See the
    [Policies overview](index.md#dual-history-raw-log-vs-working-context) for
    how the dual history works.

---

## ToolOutputClipPolicy

Clips oversized tool results **at append time**, so a huge output never occupies
context in the first place. This is the cheapest, highest-impact guard — a single
tool call returning a large JSON payload is the most common cause of context
explosion.

```python
ToolOutputClipPolicy(
    max_chars=8000,                     # max content length for a tool result
    suffix="\n…[output clipped]",       # marker so the model knows it saw a slice
)
```

| Parameter | Default | Description |
|---|---|---|
| `max_chars` | `8000` | Maximum content length for a `ToolResultMessage`. |
| `suffix` | `"\n…[output clipped]"` | Appended to clipped content. |

**Behavior**

- Runs on `on_append`; only `ToolResultMessage` (and its subclass
  `AgentResultMessage`) are affected — user/assistant messages pass through.
- The raw, unclipped result is still recorded in the raw history.

**When to use:** always, essentially. Any agent whose tools can return
unbounded output (API calls, file reads, search results).

---

## ToolEvictionPolicy

Evicts old tool results from the context, keeping only the most recent N intact.
Old results are usually only useful for the turn they served — after that they're
dead weight.

```python
ToolEvictionPolicy(
    keep_last_n=5,                                   # recent results kept intact
    stub="[tool result evicted to save context]",    # replacement content
    include_agent_results=True,                      # also evict linked-agent results
)
```

| Parameter | Default | Description |
|---|---|---|
| `keep_last_n` | `5` | Number of most-recent tool results kept verbatim. |
| `stub` | `"[tool result evicted to save context]"` | Replacement content for evicted results. |
| `include_agent_results` | `True` | Whether `AgentResultMessage`s are also subject to eviction. |

**Behavior**

- Runs on `on_compile`, right before each inference.
- **Stubs, never deletes**: the message and its `tool_call_id` survive, because
  providers reject histories with orphaned call/result pairs.
- Idempotent — already-stubbed results are left alone, so it does no repeated
  work on later turns.
- Set `include_agent_results=False` to exempt linked-agent responses (useful
  when sub-agent findings need to stay in context longer than raw tool output).

**When to use:** tool-heavy agents with many calls per conversation. Pair it
with `ToolOutputClipPolicy` — clipping bounds each result, eviction bounds how
many results linger.

---

## SlidingWindowPolicy

Bounds the context to the most recent `max_messages` messages, dropping from the
front.

```python
SlidingWindowPolicy(max_messages=50)
```

| Parameter | Default | Description |
|---|---|---|
| `max_messages` | `50` | Maximum number of messages kept in context. |

**Behavior**

- Runs on `on_compile`.
- The cut is **pair-boundary-safe**: it advances past any tool results whose
  calls were dropped, so no result ever survives without the call that produced
  it.
- Blunt but predictable — older turns disappear entirely rather than being
  summarized.

**When to use:** simple bounded-memory agents where old turns genuinely stop
mattering (chat companions, per-session assistants). Prefer `CompactionPolicy`
when older context contains facts the agent must not forget.

---

## CompactionPolicy

Summarizes older history into a single `CompactionSummaryMessage` when the
context grows past a token threshold — the "keep the gist, drop the transcript"
strategy.

```python
CompactionPolicy(
    max_input_tokens=100_000,    # threshold that triggers compaction
    keep_recent=10,              # most-recent messages kept verbatim
    summary_prompt=DEFAULT_COMPACTION_PROMPT,
)
```

| Parameter | Default | Description |
|---|---|---|
| `max_input_tokens` | `100_000` | Input-token threshold that triggers compaction. |
| `keep_recent` | `10` | Number of most-recent messages kept verbatim. |
| `summary_prompt` | `DEFAULT_COMPACTION_PROMPT` | System prompt for the summarization call. |

**Behavior**

- Runs on `on_compile`. The trigger is the **previous inference's reported
  input tokens** (`event.last_usage`) — no tokenizer dependency. When usage
  isn't available, it falls back to a chars/4 estimate.
- Splits history at `len - keep_recent`, nudged forward so no tool call/result
  pair straddles the boundary, renders the older half into a transcript, and
  summarizes it with **one LLM call** via the same provider the agent uses.
- The result replaces the old messages with a single
  `CompactionSummaryMessage(compacted_count=N)`. Because the compiled context
  is written back, compaction fires **once per threshold crossing**, not every
  turn, and prior summaries are folded into the next one.
- Override `summary_prompt` to control what survives compaction (e.g. "always
  preserve file paths and error messages" for a coding agent). The default
  prompt (`pyagentic.policies.DEFAULT_COMPACTION_PROMPT`) preserves key facts,
  user goals, decisions, tool findings, and open tasks.

**When to use:** long-running agents where old context carries facts that must
survive (research, planning, multi-session work). Costs one extra LLM call per
compaction.

!!! tip "State survives compaction for free"
    Anything stored in typed `State` fields is rendered into the system prompt
    from state, not from message history — so it is never lost to compaction.
    Put durable facts in state; let the transcript be compactable.

---

## Choosing and combining

| Goal | Policy |
|---|---|
| One tool call returns megabytes | `ToolOutputClipPolicy` |
| Many tool calls accumulate | `ToolEvictionPolicy` |
| Hard cap on history length, forgetting is fine | `SlidingWindowPolicy` |
| Long sessions, must remember the gist | `CompactionPolicy` |
| Linked-agent responses flooding the parent | `ToolOutputClipPolicy` / `ToolEvictionPolicy` (they match `AgentResultMessage` too) |

A sensible default stack for a tool-using agent:

```python
__message_policies__ = [
    ToolOutputClipPolicy(max_chars=8000),       # bound each result (append time)
    ToolEvictionPolicy(keep_last_n=5),          # bound how many linger (compile time)
    CompactionPolicy(max_input_tokens=80_000),  # last resort when context still grows
]
```

Order matters at compile time: policies run in declaration order, each seeing
the previous one's output. Put cheap, targeted policies before expensive,
sweeping ones so compaction sees an already-slimmed context.

All built-in policies are **stateless** (config-only), as required — policy
instances are shared across agent instances and forks. To write your own, see
[Writing Custom Policies](custom.md).
