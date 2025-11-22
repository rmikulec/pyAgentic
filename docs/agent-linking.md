# Agent Linking

Agent linking allows you to build multi-agent systems where agents can call other agents as specialized tools. This enables complex workflows where different agents handle their areas of expertise while a coordinator orchestrates the overall process.

## How Agent Linking Works

When you declare an agent as an attribute of another agent class, PyAgentic automatically makes it available as a tool. The linked agent appears in the parent's toolset with its `__description__` as the tool description. When the LLM decides to use that "tool," PyAgentic seamlessly calls the linked agent and integrates its response.

This happens at the class definition level - each agent knows about its linked agents before any instances are created, ensuring type safety and predictable behavior.

## Basic Linking

The simplest way to link agents is by declaring them as typed attributes in your agent class. You can use either the `Link[T]` descriptor or direct type annotation:

```python
from pyagentic import BaseAgent, Link, tool

class DatabaseAgent(BaseAgent):
    __system_message__ = "I query databases"
    __description__ = "Retrieves and analyzes data from databases"

    @tool("Execute SQL query")
    def query(self, sql: str) -> str: ...

class ReportAgent(BaseAgent):
    __system_message__ = "I generate business reports"

    # Using Link[T] (recommended for advanced features)
    database: Link[DatabaseAgent]

    # OR direct annotation (backward compatible)
    # database: DatabaseAgent

# The report agent can now automatically call the database agent
response = await report_agent("Create a sales report for Q4")
```

When the report agent runs, the LLM sees the database agent as an available tool in its toolset. If the LLM determines it needs database information to complete the task, it will automatically call the database agent. PyAgentic handles all the communication, context passing, and response integration behind the scenes.

Both `Link[DatabaseAgent]` and direct `DatabaseAgent` annotations work identically for basic linking. The `Link[T]` syntax becomes powerful when combined with `spec.AgentLink()` for advanced configuration (covered below).

## Multiple Linked Agents

Real-world applications often require coordination between multiple specialized agents. You can link as many agents as needed to create sophisticated workflows where each agent contributes its expertise:

```python
class EmailAgent(BaseAgent):
    __system_message__ = "I send emails"
    __description__ = "Sends and manages email communications"

class CalendarAgent(BaseAgent):
    __system_message__ = "I manage calendars"
    __description__ = "Schedules meetings and manages calendar events"

class AssistantAgent(BaseAgent):
    __system_message__ = "I help with daily tasks"

    email: EmailAgent
    calendar: CalendarAgent

response = await assistant.run("Schedule a meeting with John and send him the details")
```

With multiple linked agents, the coordinator can intelligently decide which agents to call and in what order. In this example, the assistant might first call the calendar agent to schedule the meeting, then use that information to call the email agent to send the details. The LLM automatically determines the optimal workflow based on the task requirements.

## Advanced Link Configuration with `spec.AgentLink()`

Just like `State` fields can be configured with `spec.State()`, linked agents can be configured with `spec.AgentLink()` for advanced features like defaults, factories, and conditional linking. This follows the same descriptor pattern used throughout PyAgentic.

### Default Agent Instances

Provide a default agent instance that will be used if none is provided during initialization:

```python
from pyagentic import BaseAgent, Link, spec

class AnalysisAgent(BaseAgent):
    __system_message__ = "I analyze data"
    __description__ = "Performs data analysis"

# Create a pre-configured analyzer
default_analyzer = AnalysisAgent(model="gpt-4", api_key="sk-...")

class ReportAgent(BaseAgent):
    __system_message__ = "I generate reports"

    # Will use default_analyzer if no analyzer is provided
    analyzer: Link[AnalysisAgent] = spec.AgentLink(default=default_analyzer)

# Can use the default
report_agent = ReportAgent(model="gpt-4", api_key="sk-...")
# Or provide a custom analyzer
custom_analyzer = AnalysisAgent(model="claude-3", api_key="sk-...")
report_agent = ReportAgent(
    model="gpt-4",
    api_key="sk-...",
    analyzer=custom_analyzer
)
```

### Default Factory for Dynamic Creation

Use `default_factory` to create agent instances on-demand, similar to how `spec.State()` works:

```python
class SearchAgent(BaseAgent):
    __system_message__ = "I search databases"
    __description__ = "Database search specialist"

def create_searcher():
    """Factory function to create search agents"""
    return SearchAgent(
        model="gpt-4-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_iterations=5
    )

class DataAgent(BaseAgent):
    __system_message__ = "I manage data operations"

    # Automatically creates a searcher if not provided
    searcher: Link[SearchAgent] = spec.AgentLink(default_factory=create_searcher)

# Searcher is automatically created
data_agent = DataAgent(model="gpt-4", api_key="sk-...")
# data_agent.searcher is now a SearchAgent instance
```

This is particularly useful when:
- Agent configuration depends on environment variables or runtime context
- You want to avoid creating expensive agent instances until they're needed
- Different instances of the parent agent should have independent child agents

### Conditional Linking

Link agents conditionally based on runtime state, enabling dynamic agent composition:

```python
from pyagentic import State

class ExpertAgent(BaseAgent):
    __system_message__ = "I provide expert analysis"
    __description__ = "Expert consultant for complex problems"

class SmartAgent(BaseAgent):
    __system_message__ = "I handle tasks with optional expert help"

    # State field to control expert availability
    needs_expert: State[bool] = spec.State(default=False)
    complexity_level: State[int] = spec.State(default=1)

    # Expert is only available when needed
    expert: Link[ExpertAgent] = spec.AgentLink(
        condition=lambda self: self.needs_expert
    )

    # Or use more complex conditions
    advanced_expert: Link[ExpertAgent] = spec.AgentLink(
        condition=lambda self: self.complexity_level > 7
    )

    @tool("Mark task as complex")
    def mark_complex(self) -> str:
        self.needs_expert = True
        return "Expert help is now available"
```

The `condition` parameter accepts a callable that receives `self` and returns a boolean. The linked agent will only appear as an available tool when the condition evaluates to `True`.

### Combining Configuration Options

You can combine `default_factory` with `condition` for sophisticated agent composition:

```python
class SpecialistAgent(BaseAgent):
    __system_message__ = "I'm a specialist"
    __description__ = "Specialist for complex tasks"

def create_specialist():
    return SpecialistAgent(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )

class OrchestratorAgent(BaseAgent):
    __system_message__ = "I orchestrate workflows"

    task_complexity: State[int] = spec.State(default=1)

    # Specialist is auto-created when complexity is high
    specialist: Link[SpecialistAgent] = spec.AgentLink(
        default_factory=create_specialist,
        condition=lambda self: self.task_complexity > 5
    )
```

### Link Descriptor Pattern Summary

The `Link` descriptor pattern mirrors the `State` descriptor pattern:

| Feature | State Example | Link Example |
|---------|---------------|--------------|
| **Basic usage** | `user_name: State[str]` | `helper: Link[HelperAgent]` |
| **With descriptor** | `logs: State[list] = spec.State(...)` | `analyzer: Link[AnalysisAgent] = spec.AgentLink(...)` |
| **Default value** | `spec.State(default="Guest")` | `spec.AgentLink(default=agent_instance)` |
| **Factory** | `spec.State(default_factory=list)` | `spec.AgentLink(default_factory=create_agent)` |
| **Advanced config** | Access control, policies | Conditional linking |

Both patterns:
- Use marker types (`State[T]` and `Link[T]`)
- Support `default` and `default_factory`
- Are processed by the metaclass during class creation
- Provide type safety and IDE autocomplete

## Custom Agent Parameters

By default, when a linked agent is called, it receives the full user input as a single string. However, you can customize this behavior to create more sophisticated interactions by implementing custom `__call__` methods and using Pydantic models to define structured input parameters.

### Basic Custom __call__

Overriding the `__call__` method gives you complete control over how your linked agent processes input. This allows you to define specific parameters that the parent agent can pass:

```python
class AnalysisAgent(BaseAgent):
    __system_message__ = "I analyze data"
    __description__ = "Performs statistical analysis on datasets"

    async def __call__(self, data: str, analysis_type: str = "basic") -> str: ...

class ReportAgent(BaseAgent):
    __system_message__ = "I generate reports"
    analyzer: AnalysisAgent

# The LLM can now call the analyzer with specific parameters
response = await report_agent("Create a detailed analysis report")
```

With this setup, the parent agent's LLM can call the analysis agent with specific parameters like `analysis_type="advanced"`, giving it precise control over the linked agent's behavior.

### Using Pydantic Models for Structured Input

For more complex scenarios with multiple parameters, validation, and documentation, use Pydantic `BaseModel` classes. These provide type safety, validation, and automatic schema generation:

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., description="The search query to execute")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    include_metadata: bool = Field(default=True, description="Whether to include metadata in results")

class SearchAgent(BaseAgent):
    __system_message__ = "I search databases"
    __description__ = "Searches databases with advanced filtering"

    async def __call__(self, params: SearchParams) -> str:
        # Access validated parameters
        results = self.search_db(params.query, params.max_results)
        return f"Found {len(results)} results for '{params.query}'"

class DataAgent(BaseAgent):
    __system_message__ = "I manage data operations"
    searcher: SearchAgent

# The LLM can now call the searcher with structured parameters
result = await data_agent("Find customer records for 'John Smith'")
```

When using Pydantic `BaseModel` classes, PyAgentic automatically generates the proper OpenAI tool schema, complete with parameter types, defaults, descriptions, and validation rules. This makes the linked agent's interface clear to the calling LLM and ensures type safety throughout the system.

## Best Practices

### Use `Link[T]` for Advanced Features

While direct type annotations work for simple cases, prefer `Link[T]` when you need:
- Default instances or factories
- Conditional linking
- Better documentation and IDE support

```python
# ✅ Good: Using Link[T] with advanced features
class OrchestratorAgent(BaseAgent):
    researcher: Link[ResearchAgent] = spec.AgentLink(
        default_factory=create_researcher,
        condition=lambda self: self.needs_research
    )

# ✅ Also good: Simple direct annotation for basic cases
class SimpleAgent(BaseAgent):
    helper: HelperAgent
```

### Clear Descriptions

Since linked agents appear as tools to the parent agent, their `__description__` attribute becomes crucial. This description is what the LLM uses to decide when and how to use the linked agent:

```python
class DatabaseAgent(BaseAgent):
    __system_message__ = "I query databases"
    __description__ = "Retrieves and analyzes data from SQL databases"
```

A well-written description should be specific enough to guide the LLM's decision-making while being concise. Think of it as the "tool tooltip" that helps the parent agent understand the linked agent's capabilities and appropriate use cases.

### Use Default Factories for Independent Instances

When each parent agent should have its own child agent instance, use `default_factory`:

```python
# ✅ Good: Each parent gets its own searcher
class DataAgent(BaseAgent):
    searcher: Link[SearchAgent] = spec.AgentLink(
        default_factory=lambda: SearchAgent(model="gpt-4", api_key="...")
    )

# ❌ Avoid: Sharing one instance across all parents (usually wrong)
shared_searcher = SearchAgent(model="gpt-4", api_key="...")

class DataAgent(BaseAgent):
    searcher: Link[SearchAgent] = spec.AgentLink(default=shared_searcher)
```

### Leverage Conditional Linking

Use conditional linking to make agent composition dynamic based on state:

```python
class AdaptiveAgent(BaseAgent):
    difficulty: State[str] = spec.State(default="easy")

    # Basic helper always available
    helper: Link[HelperAgent]

    # Expert only available for hard tasks
    expert: Link[ExpertAgent] = spec.AgentLink(
        condition=lambda self: self.difficulty == "hard"
    )
```

This makes your agent's toolset adapt to the situation, improving efficiency and reducing hallucination.

### Keep Agents Focused

The single responsibility principle applies strongly to linked agents. Each agent should have a clear, focused purpose that makes it easy for other agents to understand when to use it:

```python
class EmailAgent(BaseAgent):
    __description__ = "Sends and manages emails"

class CalendarAgent(BaseAgent):
    __description__ = "Manages calendar events and scheduling"

class AssistantAgent(BaseAgent):
    email: Link[EmailAgent]
    calendar: Link[CalendarAgent]
```

Focused agents are easier to compose, test, and maintain. They also make the overall system behavior more predictable since each agent's role is clearly defined.
