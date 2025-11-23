# Phases

Phases are a finite state machine (FSM) system that allows agents to progress through different operational states. This powerful feature enables you to create agents with structured workflows, conditional tool availability, and context-aware behavior based on the current execution phase.

## Overview

Phases allow you to:

- **Structure agent workflows** into distinct operational stages
- **Control tool availability** based on the current phase
- **Customize prompts** dynamically using phase-aware templates
- **Define state transitions** with custom condition functions
- **Build complex multi-stage agents** with clear progression logic

## Basic Usage

Define phases on your agent class using the `phases` class variable:

```python
from pyagentic import BaseAgent, tool, spec

class ResearchAgent(BaseAgent):
    __system_message__ = """
    You are a research agent.
    Current phase: {{ phase }}
    {% if phase == "planning" %}
    Focus on creating a research plan.
    {% elif phase == "research" %}
    Execute the research plan and gather data.
    {% elif phase == "analysis" %}
    Analyze the gathered data and form conclusions.
    {% endif %}
    """

    research_plan: spec.State[str] = spec.State(default="")
    data: spec.State[list[str]] = spec.State(default_factory=list)

    # Define phase transitions: (from_state, to_state, condition_function)
    phases = [
        ("planning", "research", lambda self: bool(self.state.research_plan)),
        ("research", "analysis", lambda self: len(self.state.data) >= 3),
    ]

    @tool("Create a research plan", phases=["planning"])
    def create_plan(self, topic: str, approach: str) -> str:
        self.research_plan = f"Research {topic} using {approach}"
        return f"Plan created: {self.research_plan}"

    @tool("Gather research data", phases=["research"])
    def gather_data(self, query: str) -> str:
        # Simulate data gathering
        result = f"Data for: {query}"
        self.data.append(result)
        return result

    @tool("Analyze data", phases=["analysis"])
    def analyze(self) -> str:
        return f"Analysis of {len(self.data)} data points"
```

## Phase Definition

Phases are defined as a list of tuples with three elements:

```python
phases = [
    (source_state, destination_state, condition_function),
    # ...
]
```

- **source_state** (`str`): The current phase name
- **destination_state** (`str`): The target phase name
- **condition_function** (`Callable`): A function that receives the agent instance and returns `True` when the transition should occur

### Condition Functions

Condition functions determine when a phase transition should happen. They receive the agent instance (`self`) and have access to all state and properties:

```python
# Transition when a specific state field is populated
("planning", "execution", lambda self: bool(self.state.plan))

# Transition based on multiple conditions
("gathering", "analysis", lambda self: len(self.state.items) > 5 and self.state.quality_check)

# Transition based on complex logic
("draft", "review", lambda self: self._is_complete())
```

## Phase-Aware Tools

Tools can be restricted to specific phases using the `phases` parameter in the `@tool` decorator:

```python
@tool("Tool only available in planning phase", phases=["planning"])
def planning_tool(self) -> str:
    return "This only works during planning"

@tool("Tool available in multiple phases", phases=["research", "analysis"])
def multi_phase_tool(self) -> str:
    return "Available in research and analysis"

@tool("Tool available in all phases")  # No phases parameter
def universal_tool(self) -> str:
    return "Always available"
```

When an agent is in a specific phase, only tools that:
1. Specify that phase in their `phases` list, OR
2. Don't specify any phases (available in all phases)

will be exposed to the LLM.

## Accessing Current Phase

You can access the current phase in several ways:

### In System Messages

Use the `{{ phase }}` variable in your system message template:

```python
__system_message__ = """
You are in the {{ phase }} phase.
{% if phase == "planning" %}
Create a detailed plan.
{% elif phase == "execution" %}
Execute the plan step by step.
{% endif %}
"""
```

### In Input Templates

The phase is also available in input templates:

```python
__input_template__ = """
User request: {{ user_message }}
Current phase: {{ phase }}
"""
```

### In Code

Access the phase directly from the agent state:

```python
current_phase = self.state.phase
# or
current_phase = agent.state.phase
```

## Phase Lifecycle

### Initialization

When an agent with phases is initialized:

1. The phase machine is built from the `phases` list
2. States are extracted automatically from the tuples
3. The initial state is set to the first source state in the list
4. Transition triggers are created for each phase pair

### State Transitions

Phase transitions are evaluated automatically:

- **After LLM inference**: Following each call to the LLM
- **After tool execution**: After each tool completes
- **After agent calls**: When linked agents finish execution

The transition evaluation checks each condition function in order and triggers the first transition whose condition returns `True`.

### Manual Transitions

While transitions typically happen automatically, you can also trigger them manually if needed:

```python
# The machine creates transition methods automatically
self.state._machine.planning_to_research()
```

## Complete Example

Here's a comprehensive example showing a multi-phase document writing agent:

```python
from pyagentic import BaseAgent, tool, spec

class DocumentWriterAgent(BaseAgent):
    __system_message__ = """
    You are a document writer in {{ phase }} phase.

    {% if phase == "outline" %}
    Create a detailed outline for the document.
    {% elif phase == "drafting" %}
    Write the content based on the outline.
    {% elif phase == "review" %}
    Review and polish the draft.
    {% elif phase == "final" %}
    The document is complete.
    {% endif %}
    """

    outline: spec.State[str] = spec.State(default="")
    draft: spec.State[str] = spec.State(default="")
    final_doc: spec.State[str] = spec.State(default="")

    phases = [
        ("outline", "drafting", lambda self: len(self.state.outline) > 100),
        ("drafting", "review", lambda self: len(self.state.draft) > 500),
        ("review", "final", lambda self: bool(self.state.final_doc)),
    ]

    @tool("Create document outline", phases=["outline"])
    def create_outline(self, topic: str, sections: list[str]) -> str:
        outline = f"# {topic}\n\n"
        for section in sections:
            outline += f"## {section}\n"
        self.outline = outline
        return f"Outline created with {len(sections)} sections"

    @tool("Write draft content", phases=["drafting"])
    def write_draft(self, section: str, content: str) -> str:
        self.draft += f"\n\n## {section}\n\n{content}"
        return f"Added content to section: {section}"

    @tool("Review and finalize", phases=["review"])
    def finalize_document(self, revisions: str) -> str:
        self.final_doc = self.draft + f"\n\n---\nRevisions: {revisions}"
        return "Document finalized"

    @tool("Export document", phases=["final"])
    def export(self, format: str) -> str:
        return f"Exported document in {format} format"
```
