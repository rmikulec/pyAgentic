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

## Phases with Linked Agents

Linked agents can be restricted to specific phases using the `phases` parameter in `spec.AgentLink()`, similar to how tools can be phase-restricted. This allows you to compose multi-agent systems where different agents are available at different stages of the workflow:

```python
from pyagentic import BaseAgent, tool, spec, Link

class PlannerAgent(BaseAgent):
    __system_message__ = "I create detailed plans"
    __description__ = "Planning specialist for strategy development"

class ExecutorAgent(BaseAgent):
    __system_message__ = "I execute plans"
    __description__ = "Execution specialist for implementing plans"

class ReviewerAgent(BaseAgent):
    __system_message__ = "I review completed work"
    __description__ = "Quality assurance and review specialist"

class ProjectAgent(BaseAgent):
    __system_message__ = """
    You manage projects through phases: {{ phase }}
    """

    plan: spec.State[str] = spec.State(default="")
    execution_complete: spec.State[bool] = spec.State(default=False)

    # Define phase transitions
    phases = [
        ("planning", "execution", lambda self: bool(self.plan)),
        ("execution", "review", lambda self: self.execution_complete),
    ]

    # Each linked agent is only available during specific phases
    planner: Link[PlannerAgent] = spec.AgentLink(phases=["planning"])
    executor: Link[ExecutorAgent] = spec.AgentLink(phases=["execution"])
    reviewer: Link[ReviewerAgent] = spec.AgentLink(phases=["review"])

    @tool("Save the project plan", phases=["planning"])
    def save_plan(self, plan_text: str) -> str:
        self.plan = plan_text
        return "Plan saved"

    @tool("Mark execution complete", phases=["execution"])
    def complete_execution(self) -> str:
        self.execution_complete = True
        return "Execution marked complete"
```

When using phase-restricted linked agents:
- Linked agents only appear as available tools during their specified phases
- Omitting the `phases` parameter makes the agent available in all phases
- Multiple phases can be specified: `phases=["planning", "review"]`
- Phase restrictions work alongside the `condition` parameter for fine-grained control

For more details on linked agents, see the [Agent Linking documentation](agent-linking.md).

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