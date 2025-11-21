# Architecture

This page documents the internal architecture of PyAgentic, showing how agents are constructed from declaration through instantiation to runtime execution.

## Overview

PyAgentic's architecture is built around three distinct phases:

1. **Declaration Phase** - User writes agent class code, metaclass processes it
2. **Instantiation Phase** - Agent class is instantiated with dynamic state and initialization
3. **Runtime Phase** - Agent executes, calling tools and linked agents in an agentic loop

Each phase builds upon the previous one, transforming user-friendly declarations into sophisticated runtime behavior.

## Declaration Phase

The declaration phase occurs when you define an agent class. The `AgentMetaclass` intercepts the class definition and generates all the internal structures needed for the agent to function.

![Declaration Phase Architecture](../diagrams/declaration.svg)

### Key Components

**User Declarations:**
- **BaseAgent** - Your agent class inherits from this
- **System Message** - The agent's core instructions
- **State Fields** - `State[T]` annotations for persistent state
- **Linked Agents** - Type annotations for other agents
- **Tools** - Methods decorated with `@tool`

**Metaclass Processing:**
1. **Extract Attributes** - Pulls state, tools, and linked agents from the class
2. **C3 Linearize** - Resolves inheritance from parent classes and mixins
3. **Validate Definitions** - Ensures all definitions are valid
4. **Generate Definitions** - Creates internal definition objects:
   - `_StateDefinition` - Type and configuration for each state field
   - `_ToolDefinition` - Schema and metadata for each tool
   - `_LinkedAgentDefinition` - References to linked agents
5. **Build Init** - Dynamically generates `__init__` signature and function
6. **Build Response Model** - Creates Pydantic response model from tool definitions

**Generated Class Structure:**
- `__tool_defs__` - Registry of all tool definitions
- `__state_defs__` - Registry of all state definitions
- `__linked_agents__` - Registry of all linked agent types
- `__response_model__` - Pydantic model for agent responses
- `__init__()` - Dynamically generated constructor

### Supporting Utilities

The `spec` object provides configuration helpers:
- `spec.State()` - Configure state fields (persist, default, access control)
- `spec.Param()` - Configure tool parameters (description, default, values)
- `spec.AgentLink()` - Configure linked agent behavior

The `ref` object creates lazy references to state for use in tool parameters:
- `ref.field.subfield` creates a `RefNode` that resolves at runtime
- Used to constrain parameters to valid state values

## Instantiation Phase

The instantiation phase occurs when you create an instance of your agent class (e.g., `agent = MyAgent(...)`). The dynamically generated `__init__` method creates the agent's runtime state and configuration.

![Instantiation Phase Architecture](../diagrams/instantiation.svg)

### Initialization Flow

1. **Make State Model**
   - Creates a dynamic Pydantic model from `__state_defs__`
   - Each state field becomes a validated model field
   - Computed fields are included automatically

2. **Compile State Values**
   - Processes initialization arguments
   - Applies default values from `spec.State()`
   - Type-checks all values

3. **Create State Instance**
   - Instantiates the dynamic state model
   - Stores as `agent.state`
   - Includes system message and templates

4. **Set Linked Agents**
   - Attaches provided agent instances to the parent
   - Creates tool definitions from linked agents
   - Validates linked agent types

5. **Set Attributes**
   - Attaches any additional instance attributes
   - Binds tools as instance methods

6. **Post Initialization (`__post_init__`)**
   - **Check LLM Provider** - Validates model string or provider instance
   - **Setup Tracer** - Initializes observability tracer (defaults to BasicTracer)

### Instance Attributes

After initialization, the agent instance has:
- `state` - The `AgentState` instance with all state fields
- Linked agents - References to other agent instances
- `provider` - The configured LLM provider
- `tracer` - The observability tracer
- `model`, `api_key` - Provider configuration
- `max_call_depth` - Maximum depth for the agentic loop

## Runtime Phase

The runtime phase occurs when you call `agent.run(input)` or `agent(input)`. The agent enters an agentic loop where it can call tools and linked agents multiple times before producing a final response.

![Runtime Phase Architecture](../diagrams/runtime.svg)

### Execution Flow

1. **Add User Message**
   - Input is added to `agent.state._messages`
   - State is now primed for inference

2. **Get Tool Definitions**
   - Collects all `@tool` methods from `__tool_defs__`
   - Generates tool definitions for linked agents via `agent.get_tool_definition()`
   - Creates list of available tools for the LLM

3. **Process LLM Inference**
   - Builds prompt with system message and user input
   - Sends to provider with tool schemas
   - Returns LLM response (text and/or tool calls)

4. **Tool Call Routing**
   - If no tool calls → Build final response
   - If tool calls → Route to appropriate processor:

   **Process Tool Call:**
   - Looks up tool in `__tool_defs__`
   - Compiles arguments (resolves refs, validates types)
   - Executes tool method
   - Returns `ToolResponse` with result

   **Process Agent Call:**
   - Looks up linked agent
   - Calls `linked_agent.run()`
   - Returns `AgentResponse` from linked agent

5. **Increment Depth**
   - Increases loop counter
   - Checks against `max_call_depth`
   - If under limit → Loop back to inference
   - If at limit → Build final response

6. **Build Response**
   - Combines final LLM output with all tool/agent responses
   - Creates `AgentResponse` instance using `__response_model__`
   - Returns to caller

### Response Object

The `AgentResponse` contains:
- `final_output` - The LLM's final text response
- `tool_responses` - List of `ToolResponse` objects (one per tool call)
- `agent_responses` - List of nested `AgentResponse` objects (one per linked agent call)
- `provider_info` - Metadata about the LLM provider and usage

Each `ToolResponse` contains:
- `output` - The string result from the tool
- `call_depth` - Which loop iteration this was called in
- `raw_kwargs` - Original arguments from the LLM
- Compiled parameters specific to that tool

## Key Design Patterns

### Metaclass-Based Construction

Using a metaclass allows PyAgentic to inspect and transform agent classes at definition time, generating optimal runtime structures before any instances are created. This enables:
- Compile-time validation of agent definitions
- Pre-generated response models for type safety
- Efficient tool schema generation
- Inheritance and mixin support via C3 linearization

### Dynamic State Management

State is defined declaratively at the class level but instantiated dynamically per agent instance. This provides:
- Type-safe state access via Pydantic
- Computed fields that update automatically
- Access control (read/write/hidden)
- Serialization support

### Reference Resolution

The `ref` system creates lazy references at declaration time that resolve at runtime:
1. **Declaration**: `ref.field.subfield` creates `RefNode(['field', 'subfield'])`
2. **Storage**: `RefNode` stored in tool parameter definition
3. **Runtime**: When generating tool schema, `RefNode.resolve(agent_reference)` walks the path to get current value

This keeps tool parameters synchronized with live state values.

### Tool as Universal Interface

Both custom `@tool` methods and linked agents use the same `_ToolDefinition` interface. This allows:
- Uniform handling by the LLM
- Consistent parameter validation
- Seamless composition of agents

## Source Diagrams

These architecture diagrams were created using [D2](https://d2lang.com/). The source `.d2` files are available in the `diagrams/` directory:

- `diagrams/declaration.d2` - Declaration phase diagram
- `diagrams/instantiation.d2` - Instantiation phase diagram
- `diagrams/runtime.d2` - Runtime phase diagram

To regenerate the diagrams:
```bash
d2 diagrams/declaration.d2 docs/diagrams/declaration.svg
d2 diagrams/instantiation.d2 docs/diagrams/instantiation.svg
d2 diagrams/runtime.d2 docs/diagrams/runtime.svg
```

## Next Steps

- Learn about the [public API](modules.md) you should use
- Read the [user guide](../getting-started.md) for practical examples
- Explore [state management](../states.md) for persistent agents
- See [tools](../tools.md) for extending agent capabilities
