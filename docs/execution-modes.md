# Execution Modes

PyAgentic provides three different ways to execute your agents, each suited for different use cases. Understanding these modes helps you choose the right execution pattern for your application.

## Overview

| Mode | Method | Returns | Use Case |
|------|--------|---------|----------|
| **Call** | `agent(...)` | `AgentResponse` | Customizable interface, used when linking agents |
| **Run** | `agent.run("message")` | `AgentResponse` | Direct execution with message string |
| **Step** | `agent.step("message")` | `AsyncGenerator` | Streaming responses, real-time updates |

## Call: `agent(...)`

The `__call__` method provides a customizable interface for your agent. By default, it accepts a single `user_input` string and forwards it to `run()`, but you can override it to accept any typed parameters you need.

### Default Behavior

```python
agent = ResearchAgent(
    model="openai::gpt-4o",
    api_key=API_KEY
)

# Call the agent directly with a message
response = await agent("Find papers on AI and climate change")
print(response.final_output)
```

### Customizing `__call__` for Structured Input

The real power of `__call__` is that you can override it to accept typed parameters that match your agent's purpose. These parameters automatically become tool parameters when the agent is used as a linked agent:

```python
from typing import Optional

class CoursePlannerAgent(BaseAgent):
    __system_message__ = "You design course curricula"
    __description__ = "Creates structured course plans based on learning goals"
    __response_format__ = CoursePlan

    async def __call__(
        self,
        goal: str,
        experience: str,
        context: Optional[str] = None
    ) -> CoursePlan:
        """
        Generate a course plan based on structured inputs.

        Args:
            goal: The student's learning objective
            experience: Description of their current skill level
            context: Optional additional context or preferences
        """
        # Build a structured prompt from the parameters
        prompt_parts = [
            f"Goal: {goal}",
            f"Experience: {experience}",
        ]
        if context:
            prompt_parts.append(f"Additional Context: {context}")

        user_input = "\n".join(prompt_parts)
        return await self.run(input_=user_input).final_output

# Now you can call it with structured parameters
planner = CoursePlannerAgent(model="openai::gpt-4o", api_key=API_KEY)
course = await planner(
    goal="Learn machine learning",
    experience="Beginner programmer with Python knowledge",
    context="Prefer hands-on projects"
)
```

### Why This Matters for Agent Linking

When you link an agent to another agent, PyAgentic extracts the parameters from the `__call__` signature and uses them as tool parameters. This means the LLM will see your structured parameters instead of just a generic "user_input" string:

```python
class AssistantAgent(BaseAgent):
    __system_message__ = "You help students with learning plans"

    # Link the course planner
    planner: CoursePlannerAgent

# When the LLM wants to use the planner, it sees:
# Tool: planner(goal: str, experience: str, context: Optional[str])
# Instead of: planner(user_input: str)

assistant = AssistantAgent(
    model="openai::gpt-4o",
    api_key=API_KEY,
    planner=planner
)

# The assistant can intelligently call the planner with structured data
response = await assistant("Help me learn ML - I'm a beginner with Python")
# The LLM will call: planner(goal="machine learning", experience="beginner with Python", context=None)
```

### When to Use Custom `__call__`

- When your agent has a specific interface contract (like `goal` and `experience`)
- When you're building agents that will be linked to other agents
- When you want to enforce a structured input schema
- When you need to transform input parameters before processing

### When to Use Default `__call__`

- For simple conversational agents
- When you don't need structured parameters
- For quick prototypes
- When the agent won't be used as a linked agent

## Run: `agent.run("message")`

The `run()` method is the direct execution method that always takes a single message string. It's what `__call__` uses by default, and it's the low-level execution interface.

```python
# Direct execution with a message
response = await agent.run("What's the weather?")
```

### When to Use

- When you need to explicitly pass a formatted message string
- In internal methods where you've already formatted the input
- When bypassing a custom `__call__` implementation
- For consistency in code that always uses explicit method calls

## Step: `agent.step("message")`

The most powerful execution mode, `step()` returns an async generator that yields responses as they happen. This enables real-time streaming and fine-grained control over the agent's execution.

```python
async for response in agent.step("Research AI and climate change"):
    if isinstance(response, LLMResponse):
        print(f"LLM thinking: {response.text}")
    elif isinstance(response, ToolResponse):
        print(f"Tool '{response.tool_name}' called with {response.raw_kwargs}")
        print(f"Result: {response.output}")
    elif isinstance(response, AgentResponse):
        print(f"Final answer: {response.final_output}")
```

### When to Use

- Building interactive UIs that show real-time progress
- Streaming responses to users as the agent works
- Debugging complex multi-step agent workflows
- Implementing custom retry logic or intervention
- Monitoring tool execution in real-time

### What You Get

The generator yields three types of responses in sequence:

#### 1. `LLMResponse` - Each LLM Inference

Yielded each time the LLM is called (can happen multiple times per run):

```python
LLMResponse(
    text="I'll search for papers on that topic",
    tool_calls=[...],  # Tool calls the LLM wants to make
    parsed=None,       # Structured output (if using response_format)
    usage=UsageInfo(...)  # Token usage stats
)
```

#### 2. `ToolResponse` - Each Tool Call

Yielded for every tool that gets executed:

```python
ToolResponse(
    output="Found 5 papers...",  # The tool's return value
    call_depth=0,                # How deep in the tool loop
    raw_kwargs='{"query": "AI climate"}',  # Original JSON args
    # Plus all the tool's typed parameters...
)
```

#### 3. `AgentResponse` - Final Result

Yielded once at the very end with the complete execution summary:

```python
AgentResponse(
    final_output="Here's what I found...",
    state=<agent state>,
    tool_responses=[...],  # All tools that were called
    provider_info=<provider info>
)
```

### Response Flow Example

Here's what the response stream looks like for a typical agent run:

```python
async for response in agent.step("Find and analyze papers on AI"):
    # First: LLM decides to call search tool
    # → LLMResponse(tool_calls=[ToolCall(name="search", ...)])

    # Second: Search tool executes
    # → ToolResponse(output="Found 5 papers...")

    # Third: LLM decides to call read_paper tool
    # → LLMResponse(tool_calls=[ToolCall(name="read_paper", ...)])

    # Fourth: Read tool executes
    # → ToolResponse(output="Paper content...")

    # Fifth: LLM provides final analysis
    # → LLMResponse(text="Based on these papers...")

    # Finally: Complete response
    # → AgentResponse(final_output="Based on these papers...", ...)
```