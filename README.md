# PyAgentic

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/rmikulec/pyagentic/workflows/Tests/badge.svg?branch=main)](https://github.com/rmikulec/pyAgentic/actions/workflows/testing.yml?query=branch%3Amain)

A declarative, type-safe framework for building AI agents with OpenAI integration. PyAgentic lets you create sophisticated LLM agents using clean Python class syntax, with full support for tools, persistent context, agent linking, and agent inheritance.

## Documentation

Read the **[Getting Started](https://rmikulec.github.io/pyAgentic/getting-started/)**!

Complete documentation, tutorials, and examples can be found **[here](https://rmikulec.github.io/pyagentic)**

## Features

- **Declarative Design** - Define agents with simple class-based syntax and decorators
- **Powerful Tools** - Easy `@tool` decoration with automatic OpenAI schema generation
- **Persistent Context** - Stateful agents with `ContextItem` fields that persist across conversations
- **Agent Linking** - Compose complex workflows by linking agents together
- **Structured Responses** - Type-safe Pydantic responses with full tool execution details
- **Dynamic Constraints** - Use `ContextRef` to create smart parameter validation
- **Inheritance & Extensions** - Build agent hierarchies and mix in cross-cutting capabilities
- **Native Async Support** - Built for scalable applications with async/await throughout
- **Type Safety** - Complete typing support with validation and IDE autocompletion

## Quick Start

### Installation

```bash
pip install pyagentic-core
```

### Simple Agent Example

```python
from pyagentic import Agent, tool, ContextItem

class ResearchAgent(Agent):
    __system_message__ = "I help with research tasks and maintain a collection of papers"
    
    # Persistent context that survives across conversations
    paper_count: int = ContextItem(default=0)
    current_topic: str = ContextItem(default="general")
    
    @tool("Search for academic papers")
    def search_papers(self, query: str, max_results: int = 5) -> str:
        # Your search logic here
        self.context.paper_count += max_results
        return f"Found {max_results} papers about '{query}'"
    
    @tool("Set research focus")
    def set_topic(self, topic: str) -> str:
        self.context.current_topic = topic
        return f"Research focus set to: {topic}"

# Create and use the agent
agent = ResearchAgent(model="openai::gpt-4", api_key="your-key")
response = await agent("Help me research machine learning papers")

# Access structured response data
print(response.final_output)  # Natural language response
print(len(response.tool_responses))  # Number of tools called
print(agent.context.paper_count)  # Persistent state
```

### Advanced Features

#### Agent Linking
Create multi-agent workflows where agents can call other agents as tools:

```python
class DatabaseAgent(Agent):
    __system_message__ = "I query databases"
    __description__ = "Retrieves data from SQL databases"
    
    @tool("Execute SQL query")
    def query(self, sql: str) -> str: ...

class WebAgent(Agent):
    __system_message__ = "I search the web"
    __description__ = "Searches the internet"

    @tool("search")
    def search(self, terms: list[str]) -> str: ...

class ReportAgent(Agent):
    __system_message__ = "I generate business reports"
    database: DatabaseAgent  # Linked agent appears as a tool
    searcher: WebAgent
    
    @tool("Create visualization")
    def create_chart(self, data: str) -> str: ...

# The report agent can automatically coordinate with the database agent
report_agent = ReportAgent(database=DatabaseAgent(...), searcher=WebAgent(...))
response = await report_agent("Generate a plot of the latest financial data")
```

#### Dynamic Parameter Constraints
Use `ContextRef` to create intelligent parameter validation:

```python
from pyagentic import computed_context, ContextRef, ParamInfo

class DataAgent(Agent):
    __system_message__ = "I manage datasets"
    
    available_datasets: list = ContextItem(default_factory=list)
    
    @computed_context
    def dataset_names(self):
        return [ds['name'] for ds in self.available_datasets]
    
    @tool("Analyze specific dataset")
    def analyze(
        self, 
        dataset: str = ParamInfo(
            description="Dataset to analyze",
            values=ContextRef("dataset_names")  # LLM can only pick from available datasets
    )) -> str: ...
```

## Contributing

Contribution is welcome! Whether it's bug fixes, new features, documentation improvements, or examples, help is appreciated.

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pyagentic.git
   cd pyagentic
   ```

2. **Install dependencies with uv**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Install all dependencies including dev tools
   uv sync --group dev
   ```

### Code Quality

Several tools are used to maintain code quality:

#### Formatting with Black
```bash
# Format all Python files
uv run black -l99 pyagentic tests

# Check formatting without making changes
uv run black -l99 --check pyagentic tests
```

#### Linting with Flake8
```bash
# Run linting checks
uv run flake8 --max-line-length=99 pyagentic tests 
```


### Testing

```bash
# Run all tests
uv run pytest tests

# Run tests with coverage
uv run coverage run -m pytest tests
```

### Documentation

Documentation is built with MkDocs and deployed automatically:

#### Local Development
```bash
# Install docs dependencies
uv sync --group docs

# Serve docs locally (auto-reloads on changes)
uv run task serve-docs

# Build docs to ./site/
uv run task build-docs
```

Docs are automatically deployed on pushes to main via GitHub Actions.

### Submitting Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Run the full test suite**
   ```bash
   uv run black pyagentic tests
   uv run flake8 pyagentic tests --max-line-length=99
   uv run pytest
   ```

4. **Commit with conventional commit messages**
   ```bash
   git commit -m "feat: add new agent linking feature"
   git commit -m "fix: resolve context persistence issue"
   git commit -m "docs: improve getting started guide"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Conventional Commits

Conventional commits are used for automatic versioning:

- `feat:` - New features (minor version bump)
- `fix:` - Bug fixes (patch version bump)
- `docs:` - Documentation changes
- `test:` - Test additions or improvements
- `refactor:` - Code refactoring
- `style:` - Code style changes (formatting, etc.)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
