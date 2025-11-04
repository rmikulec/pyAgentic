# Agent Linking

Agent linking allows you to build multi-agent systems where agents can call other agents as specialized tools. This enables complex workflows where different agents handle their areas of expertise while a coordinator orchestrates the overall process.

## How Agent Linking Works

When you declare an agent as an attribute of another agent class, PyAgentic automatically makes it available as a tool. The linked agent appears in the parent's toolset with its `__description__` as the tool description. When the LLM decides to use that "tool," PyAgentic seamlessly calls the linked agent and integrates its response.

This happens at the class definition level - each agent knows about its linked agents before any instances are created, ensuring type safety and predictable behavior.

## Basic Linking

The simplest way to link agents is by declaring them as typed attributes in your agent class. This creates a direct relationship where the parent agent can call the child agent as needed:

```python
class DatabaseAgent(BaseAgent):
    __system_message__ = "I query databases"
    __description__ = "Retrieves and analyzes data from databases"

    @tool("Execute SQL query")
    def query(self, sql: str) -> str: ...

class ReportAgent(BaseAgent):
    __system_message__ = "I generate business reports"
    database: DatabaseAgent  # Linked agent

# The report agent can now automatically call the database agent
response = await report_agent("Create a sales report for Q4")
```

When the report agent runs, the LLM sees the database agent as an available tool in its toolset. If the LLM determines it needs database information to complete the task, it will automatically call the database agent. PyAgentic handles all the communication, context passing, and response integration behind the scenes.

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

## Custom Agent Parameters

By default, when a linked agent is called, it receives the full user input as a single string. However, you can customize this behavior to create more sophisticated interactions by implementing custom `__call__` methods and using `Param` classes to define structured input parameters.

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

### Using Param for Structured Input

For more complex scenarios with multiple parameters, validation, and documentation, use `Param` classes. These provide type safety and automatic schema generation:

```python
from pyagentic import Param

class SearchParams(Param):
    query: str
    max_results: int = 10
    include_metadata: bool = True

class SearchAgent(BaseAgent):
    __system_message__ = "I search databases"
    __description__ = "Searches databases with advanced filtering"

    async def __call__(self, params: SearchParams) -> str: ...

class DataAgent(BaseAgent):
    __system_message__ = "I manage data operations"
    searcher: SearchAgent

# The LLM can now call the searcher with structured parameters
result = await data_agent("Find customer records for 'John Smith'")
```

When using `Param` classes, PyAgentic automatically generates the proper OpenAI tool schema, complete with parameter types, defaults, and descriptions. This makes the linked agent's interface clear to the calling LLM and ensures type safety throughout the system.

## Best Practices

### Clear Descriptions

Since linked agents appear as tools to the parent agent, their `__description__` attribute becomes crucial. This description is what the LLM uses to decide when and how to use the linked agent:

```python
class DatabaseAgent(BaseAgent):
    __system_message__ = "I query databases"
    __description__ = "Retrieves and analyzes data from SQL databases"
```

A well-written description should be specific enough to guide the LLM's decision-making while being concise. Think of it as the "tool tooltip" that helps the parent agent understand the linked agent's capabilities and appropriate use cases.

### Keep Agents Focused

The single responsibility principle applies strongly to linked agents. Each agent should have a clear, focused purpose that makes it easy for other agents to understand when to use it:

```python
class EmailAgent(BaseAgent):
    __description__ = "Sends and manages emails"

class CalendarAgent(BaseAgent):
    __description__ = "Manages calendar events and scheduling"

class AssistantAgent(BaseAgent):
    email: EmailAgent
    calendar: CalendarAgent
```

Focused agents are easier to compose, test, and maintain. They also make the overall system behavior more predictable since each agent's role is clearly defined.
