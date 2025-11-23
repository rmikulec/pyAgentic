# PyAgentic Documentation

Build sophisticated AI agents with declarative Python syntax. PyAgentic provides a type-safe, extensible framework for creating LLM agents with persistent context, powerful tools, and seamless integration with multiple LLM providers including OpenAI, Anthropic, and others.

## Quick Start

New to PyAgentic? Start here to build your first agent in minutes:

<div class="grid cards" markdown>

- **[Getting Started Guide](getting-started.md)**

    ---

    Complete tutorial building a research assistant agent from scratch. Learn core concepts through practical examples.

    [:octicons-arrow-right-24: Start building](getting-started.md)

</div>

## Core Documentation

Dive deeper into PyAgentic's powerful features:

<div class="grid cards" markdown>

- :material-tools: **[Tools](tools.md)**

    ---

    Give agents capabilities with the @tool decorator, parameter validation, and dynamic constraints.

    [:octicons-arrow-right-24: Learn about tools](tools.md)

- :material-database: **[State Management](states.md)**

    ---

    Persistent, type-safe state fields with Pydantic models, computed fields, and access control.

    [:octicons-arrow-right-24: Learn about states](states.md)

- :material-shield-check: **[Policies](policies.md)**

    ---

    React to state changes with validation, history tracking, persistence, and custom behaviors.

    [:octicons-arrow-right-24: Learn about policies](policies.md)

- :material-chat-processing: **[Agent Responses](responses.md)**

    ---

    Understanding structured response objects with tool execution details and type safety.

    [:octicons-arrow-right-24: Learn about responses](responses.md)

- :material-play-circle: **[Execution Modes](execution-modes.md)**

    ---

    Three ways to run agents: simple calls, run(), and step() for streaming responses.

    [:octicons-arrow-right-24: Learn about execution modes](execution-modes.md)

- :material-format-list-numbered: **[Structured Outputs](structured-output.md)**

    ---

    Using Pydantic models to enforce structured output schemas for your agents.

    [:octicons-arrow-right-24: Learn about structured outputs](structured-output.md)

- :material-link-variant: **[Agent Linking](agent-linking.md)**

    ---

    Build complex multi-agent workflows where agents call other agents as specialized tools.

    [:octicons-arrow-right-24: Explore linking](agent-linking.md)

- :material-family-tree: **[Inheritance & Extensions](Inheritance.md)**

    ---

    Create agent hierarchies and add cross-cutting capabilities with extensions.

    [:octicons-arrow-right-24: Build hierarchies](Inheritance.md)

- :material-search: **[Observability](observability.md)**

    ---

    Observe and trace all steps and interactions of an agent.

    [:octicons-arrow-right-24: Trace behavior](observability.md)

</div>


- **GitHub**: [rmikulec/pyagentic](https://github.com/rmikulec/pyagentic) - Source code, issues, and contributions
- **Installation**: `pip install pyagentic-core`
- **Python Support**: 3.13+