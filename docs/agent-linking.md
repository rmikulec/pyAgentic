# Agent Linking in PyAgentic

PyAgentic provides a powerful agent linking system that allows agents to call other agents as tools. This enables you to build complex multi-agent systems where agents can collaborate, delegate tasks, and leverage each other's specialized capabilities.

## Table of Contents

- [What is Agent Linking?](#what-is-agent-linking)
- [Basic Agent Linking](#basic-agent-linking)
- [How Agent Linking Works](#how-agent-linking-works)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)
- [What Not to Do](#what-not-to-do)
- [Advanced Patterns](#advanced-patterns)

## What is Agent Linking?

Agent linking allows one agent to invoke another agent as if it were a tool. When you declare a linked agent, PyAgentic automatically:

1. Makes the linked agent available as a tool in the parent agent's toolset
2. Handles communication between agents seamlessly
3. Manages context and response propagation
4. Provides type safety and inheritance support

## Basic Agent Linking

To link agents, simply declare them as type-annotated attributes in your agent class:

```python
from pyagentic import Agent

class SpecialistAgent(Agent):
    __system_message__ = "I am a domain specialist."
    __description__ = "Provides specialized knowledge and analysis"
    
    @tool("Analyze specialized data")
    def analyze(self, data: str) -> str:
        return f"Specialized analysis of: {data}"

class CoordinatorAgent(Agent):
    __system_message__ = "I coordinate with specialists to solve complex problems."
    
    # Link the specialist agent
    specialist: SpecialistAgent

# Usage
specialist = SpecialistAgent(model="gpt-4", api_key=API_KEY)
coordinator = CoordinatorAgent(
    model="gpt-4", 
    api_key=API_KEY, 
    specialist=specialist
)

# The coordinator can now call the specialist automatically
response = await coordinator.run("Please analyze this complex dataset: [data]")
```

## How Agent Linking Works

When an agent is linked:

1. **Tool Registration**: The linked agent appears as a tool with its `__description__` as the tool description
2. **Automatic Invocation**: When the LLM calls the linked agent tool, PyAgentic automatically invokes the linked agent's `run()` method
3. **Response Integration**: The linked agent's response is integrated back into the parent agent's context
4. **Error Handling**: Failures in linked agents are gracefully handled and reported

```python
class DataAgent(Agent):
    __system_message__ = "I process data files."
    __description__ = "Reads, processes, and validates data files"

class AnalysisAgent(Agent):
    __system_message__ = "I perform statistical analysis."
    __description__ = "Performs statistical analysis and generates insights"

class ReportAgent(Agent):
    __system_message__ = "I coordinate data processing and analysis to generate reports."
    
    data_processor: DataAgent
    analyzer: AnalysisAgent
    
    @tool("Generate final report")
    def generate_report(self, title: str) -> str:
        return f"Generated report: {title}"

# Create the pipeline
data_agent = DataAgent(model="gpt-4", api_key=API_KEY)
analysis_agent = AnalysisAgent(model="gpt-4", api_key=API_KEY)
report_agent = ReportAgent(
    model="gpt-4", 
    api_key=API_KEY,
    data_processor=data_agent,
    analyzer=analysis_agent
)

# The report agent can coordinate both agents automatically
result = await report_agent.run("Create a comprehensive report from the sales data")
```

## Use Cases

### 1. Specialized Task Delegation

Break complex problems into specialized sub-tasks:

```python
class ResearchAgent(Agent):
    __system_message__ = "I gather and summarize research information."
    __description__ = "Searches and summarizes academic and web sources"

class WritingAgent(Agent):
    __system_message__ = "I create well-structured written content."
    __description__ = "Writes articles, reports, and documentation with proper structure"

class EditingAgent(Agent):
    __system_message__ = "I review and improve written content."
    __description__ = "Reviews content for grammar, style, and clarity"

class ContentCreatorAgent(Agent):
    __system_message__ = "I coordinate research, writing, and editing to create high-quality content."
    
    researcher: ResearchAgent
    writer: WritingAgent
    editor: EditingAgent

# Usage for creating a comprehensive article
content_creator = ContentCreatorAgent(model="gpt-4", api_key=API_KEY, ...)
article = await content_creator.run("Create an article about renewable energy trends")
```

### 2. Multi-Domain Expertise

Combine different areas of expertise:

```python
class TechnicalAgent(Agent):
    __system_message__ = "I provide technical software development expertise."
    __description__ = "Analyzes technical requirements and provides development guidance"

class BusinessAgent(Agent):
    __system_message__ = "I provide business strategy and market analysis."
    __description__ = "Evaluates business impact and strategic considerations"

class ProductManagerAgent(Agent):
    __system_message__ = "I make product decisions by considering both technical and business factors."
    
    tech_expert: TechnicalAgent
    business_expert: BusinessAgent

# The PM agent can consult both experts for balanced decisions
pm = ProductManagerAgent(model="gpt-4", api_key=API_KEY, ...)
decision = await pm.run("Should we implement real-time collaboration features?")
```

### 3. Hierarchical Processing

Create processing pipelines with clear stages:

```python
class ValidationAgent(Agent):
    __system_message__ = "I validate and clean input data."
    __description__ = "Validates data format, checks for errors, and cleans input"

class ProcessingAgent(Agent):
    __system_message__ = "I perform core data processing operations."
    __description__ = "Transforms and processes validated data"

class OutputAgent(Agent):
    __system_message__ = "I format and present processed results."
    __description__ = "Formats results and generates user-friendly output"

class DataPipelineAgent(Agent):
    __system_message__ = "I orchestrate a complete data processing pipeline."
    
    validator: ValidationAgent
    processor: ProcessingAgent
    formatter: OutputAgent

# Creates a structured processing flow
pipeline = DataPipelineAgent(model="gpt-4", api_key=API_KEY, ...)
result = await pipeline.run("Process this customer data: [raw_data]")
```

## Best Practices

### 1. Clear Agent Responsibilities

Each agent should have a clear, focused responsibility:

```python
# Good: Focused, single-responsibility agents
class EmailAgent(Agent):
    __system_message__ = "I handle email communication."
    __description__ = "Sends, formats, and manages email communications"

class CalendarAgent(Agent):
    __system_message__ = "I manage calendar and scheduling."
    __description__ = "Schedules meetings, checks availability, and manages calendar events"

class AssistantAgent(Agent):
    __system_message__ = "I help with daily tasks using email and calendar capabilities."
    
    email: EmailAgent
    calendar: CalendarAgent
```

### 2. Meaningful Agent Descriptions

Always provide clear `__description__` attributes - they become the tool descriptions:

```python
class DatabaseAgent(Agent):
    __system_message__ = "I interact with databases."
    # This description helps the parent agent understand when to use this agent
    __description__ = "Queries databases, performs CRUD operations, and manages data persistence"
```

### 3. Optional vs Required Linking

Agents can be optionally linked by not providing them during initialization:

```python
class FlexibleAgent(Agent):
    __system_message__ = "I can work with or without a helper."
    
    helper: HelperAgent  # Optional - can be None
    
    @tool("Process with optional help")
    def process(self, data: str) -> str:
        if self.helper is not None:
            # Use helper if available
            return f"Processed with help: {data}"
        else:
            # Fallback behavior
            return f"Processed independently: {data}"

# Can create with or without the helper
agent = FlexibleAgent(model="gpt-4", api_key=API_KEY)  # helper will be None
```

### 4. Agent Composition Over Deep Nesting

Prefer composition over deeply nested agent hierarchies:

```python
# Good: Flat composition
class ServiceAgent(Agent):
    __system_message__ = "I coordinate multiple service operations."
    
    auth: AuthAgent
    database: DatabaseAgent
    cache: CacheAgent
    logger: LoggerAgent

# Avoid: Deep nesting
class DeepAgent(Agent):
    level1: Level1Agent  # which has level2: Level2Agent, etc.
```

## What Not to Do

### 1. Avoid Circular Dependencies

Don't create circular references between agents:

```python
# Bad: Circular dependency
class AgentA(Agent):
    __system_message__ = "I am Agent A"
    agent_b: 'AgentB'  # AgentA depends on AgentB

class AgentB(Agent):
    __system_message__ = "I am Agent B" 
    agent_a: AgentA  # AgentB depends on AgentA - CIRCULAR!

# This can lead to infinite loops and stack overflows
```

### 2. Don't Over-Link

Avoid linking too many agents to a single coordinator:

```python
# Bad: Too many linked agents makes the coordinator unfocused
class OverComplexAgent(Agent):
    __system_message__ = "I do everything with many helpers."
    
    agent1: Agent1
    agent2: Agent2
    agent3: Agent3
    agent4: Agent4
    agent5: Agent5
    agent6: Agent6  # Too many!
    # ... this becomes hard to manage and reason about

# Better: Group related functionality
class DatabaseGroup(Agent):
    __system_message__ = "I handle all database operations."
    reader: DatabaseReaderAgent
    writer: DatabaseWriterAgent

class APIGroup(Agent):
    __system_message__ = "I handle all API operations."
    client: APIClientAgent
    validator: APIValidatorAgent

class MainAgent(Agent):
    __system_message__ = "I coordinate database and API operations."
    database: DatabaseGroup
    api: APIGroup
```

### 3. Don't Link Stateful Agents Carelessly

Be careful when linking agents that maintain important state:

```python
# Potentially problematic: Shared stateful agent
shared_counter = CounterAgent(model="gpt-4", api_key=API_KEY)

agent1 = WorkerAgent(model="gpt-4", api_key=API_KEY, counter=shared_counter)
agent2 = WorkerAgent(model="gpt-4", api_key=API_KEY, counter=shared_counter)

# Both agents modify the same counter - this might cause unexpected behavior
```

### 4. Don't Neglect Error Handling

Consider what happens when linked agents fail:

```python
class RobustAgent(Agent):
    __system_message__ = "I handle failures gracefully."
    
    unreliable_helper: UnreliableAgent
    
    @tool("Process with fallback")
    def process_safely(self, data: str) -> str:
        # The framework handles agent failures, but you can add logic
        # to check responses and implement fallback strategies
        return f"Processing {data} with robust error handling"
```

## Advanced Patterns

### 1. Agent Pools

Create pools of similar agents for load balancing:

```python
class WorkerAgent(Agent):
    __system_message__ = "I am a worker that processes tasks."
    __description__ = "Processes individual work items"

class LoadBalancerAgent(Agent):
    __system_message__ = "I distribute work across multiple workers."
    
    worker1: WorkerAgent
    worker2: WorkerAgent  
    worker3: WorkerAgent
    
    # The LLM can choose which worker to use based on load or task type
```

### 2. Hierarchical Decision Making

Create decision trees with specialized agents:

```python
class TriageAgent(Agent):
    __system_message__ = "I categorize and route requests."
    __description__ = "Analyzes requests and determines the appropriate handler"

class SimpleRequestAgent(Agent):
    __system_message__ = "I handle simple, straightforward requests."
    __description__ = "Processes routine requests that don't require specialized expertise"

class ComplexRequestAgent(Agent):
    __system_message__ = "I handle complex, multi-step requests."
    __description__ = "Processes complex requests requiring multiple steps and analysis"

class RouterAgent(Agent):
    __system_message__ = "I route requests to the appropriate handler based on complexity."
    
    triage: TriageAgent
    simple_handler: SimpleRequestAgent
    complex_handler: ComplexRequestAgent

# The router can intelligently decide which path to take
```

### 3. Inheritance with Linking

Combine inheritance and linking for maximum flexibility:

```python
class BaseServiceAgent(Agent):
    __system_message__ = "I provide base service functionality."
    
    logger: LoggerAgent  # All services need logging

class WebServiceAgent(BaseServiceAgent):
    __system_message__ = "I provide web service functionality."
    
    auth: AuthAgent      # Web services need authentication
    # Inherits logger from BaseServiceAgent

class DatabaseServiceAgent(BaseServiceAgent):
    __system_message__ = "I provide database service functionality."
    
    validator: ValidatorAgent  # Database services need validation
    # Inherits logger from BaseServiceAgent
```

Agent linking in PyAgentic provides a powerful way to create sophisticated multi-agent systems. By following these patterns and best practices, you can build robust, maintainable, and scalable agent architectures that leverage the strengths of specialized agents while maintaining clear separation of concerns.
