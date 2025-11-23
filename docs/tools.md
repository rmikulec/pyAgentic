# Tools

Tools are the primary way agents interact with the outside world and perform actions. By decorating methods with `@tool`, you give your agents capabilities like searching databases, calling APIs, processing files, or any other Python function you can write.

## Why Use Tools?

LLMs are excellent at reasoning and generating text, but they can't directly interact with external systems. Tools bridge this gap by:

- **Extending Capabilities**: Give agents access to APIs, databases, file systems, and more
- **Type Safety**: Full parameter validation and type checking via Pydantic
- **Dynamic Constraints**: Use `ref` to constrain parameters to valid state values
- **Automatic Schema Generation**: Tool definitions are automatically converted to LLM-compatible schemas
- **Error Handling**: Built-in error handling and result tracking

## Basic Tool Declaration

The simplest tool is a method decorated with `@tool` that returns a string:

```python
from pyagentic import BaseAgent, tool

class ResearchAgent(BaseAgent):
    __system_message__ = "I help with research tasks"

    @tool("Search for academic papers on a topic")
    def search_papers(self, query: str) -> str:
        # Your search logic here
        results = search_database(query)
        return f"Found {len(results)} papers about '{query}'"
```

Every tool must:
- Have a `@tool` decorator with a description
- Have type-annotated parameters
- Return a `str` (the result passed back to the LLM)

When the LLM decides to use this tool, PyAgentic will:
1. Extract the parameters from the LLM's tool call
2. Validate the parameter types
3. Execute your tool method
4. Return the string result to the LLM

## Tool Parameters

Tools can accept any number of typed parameters. PyAgentic automatically generates the proper schema for the LLM:

```python
class DataAgent(BaseAgent):
    __system_message__ = "I analyze data"

    @tool("Query database with filters")
    def query_data(
        self,
        table: str,
        limit: int,
        filters: dict,
        include_metadata: bool
    ) -> str:
        results = db.query(table, limit=limit, filters=filters)
        return f"Retrieved {len(results)} records from {table}"
```

### Supported Parameter Types

PyAgentic supports a variety of parameter types:

| Type | Example | Description |
|------|---------|-------------|
| `str` | `query: str` | String values |
| `int` | `count: int` | Integer numbers |
| `float` | `threshold: float` | Floating point numbers |
| `bool` | `include_all: bool` | Boolean true/false |
| `list[T]` | `tags: list[str]` | Lists of primitives |
| `dict` | `metadata: dict` | Dictionary/object |
| `BaseModel` | `options: SearchOptions` | Custom Pydantic model |
| `list[BaseModel]` | `items: list[Item]` | List of custom models |

## Parameter Configuration with `spec.Param`

Use `spec.Param()` to add descriptions, defaults, and constraints to parameters:

```python
from pyagentic import spec

class FileAgent(BaseAgent):
    __system_message__ = "I manage files"

    @tool("Read a file from the system")
    def read_file(
        self,
        path: str = spec.Param(
            description="Path to the file to read",
            required=True
        ),
        encoding: str = spec.Param(
            description="Text encoding to use",
            default="utf-8"
        ),
        max_lines: int = spec.Param(
            description="Maximum number of lines to read",
            default=100
        )
    ) -> str:
        with open(path, encoding=encoding) as f:
            lines = [f.readline() for _ in range(max_lines)]
        return f"Read {len(lines)} lines from {path}"
```

### `spec.Param()` Options

```python
spec.Param(
    description="Human-readable parameter description",
    required=True,           # Must be provided by LLM
    default="value",          # Default if not provided
    default_factory=list,     # Factory function for mutable defaults
    values=["opt1", "opt2"]  # Constrain to specific values (enum)
)
```

## Custom Parameter Models

For complex structured input, use Pydantic `BaseModel` classes:

```python
from pydantic import BaseModel, Field

class SearchOptions(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=10, description="Maximum results")
    filters: dict = Field(default_factory=dict, description="Additional filters")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")

class SearchAgent(BaseAgent):
    __system_message__ = "I search databases"

    @tool("Search with advanced options")
    def search(self, options: SearchOptions) -> str:
        # Access structured parameters
        results = db.search(
            options.query,
            max_results=options.max_results,
            filters=options.filters,
            case_sensitive=options.case_sensitive
        )
        return f"Found {len(results)} results"
```

Benefits of custom parameter models:
- **Grouped Parameters**: Organize related parameters logically
- **Validation**: Pydantic validates all fields automatically
- **Reusability**: Use the same model across multiple tools
- **Documentation**: Field descriptions help the LLM understand parameters

### Lists of Custom Models

Tools can accept lists of custom models for batch operations:

```python
class Task(BaseModel):
    title: str = Field(..., description="Task title")
    priority: int = Field(default=1, description="Priority level 1-5")
    tags: list[str] = Field(default_factory=list, description="Task tags")

class TaskAgent(BaseAgent):
    __system_message__ = "I manage tasks"

    @tool("Create multiple tasks at once")
    def create_tasks(self, tasks: list[Task]) -> str:
        for task in tasks:
            db.create(task.title, task.priority, task.tags)
        return f"Created {len(tasks)} tasks"
```

## Dynamic Constraints with `ref`

The `ref` system allows you to constrain tool parameters to valid state values, preventing the LLM from hallucinating invalid options:

```python
from pydantic import BaseModel, computed_field
from pyagentic import State, ref, spec

class FileSystemState(BaseModel):
    current_directory: str = "/home"
    available_files: list[str] = []

    @computed_field
    @property
    def file_names(self) -> list[str]:
        """List of just the file names"""
        return [f.split('/')[-1] for f in self.available_files]

class FileAgent(BaseAgent):
    __system_message__ = "I manage files in the current directory"

    filesystem: State[FileSystemState] = spec.State(default_factory=FileSystemState)

    @tool("Open a file from the current directory")
    def open_file(
        self,
        filename: str = spec.Param(
            description="File to open",
            values=ref.filesystem.file_names  # LLM can ONLY choose from this list!
        )
    ) -> str:
        return f"Opened {filename}"
```

### How `ref` Works

When you use `ref.filesystem.file_names`:
1. **Declaration Time**: A `RefNode` is created storing the path `['filesystem', 'file_names']`
2. **Instantiation Time**: The tool definition stores this reference
3. **Runtime**: When generating the tool schema for the LLM:
   - PyAgentic builds an agent reference dictionary from the current state
   - The `RefNode` resolves by walking the path to get the actual value
   - The resolved value is injected into the tool schema as the `enum` constraint

This means the LLM always sees the current, up-to-date list of valid options, and it's impossible for it to hallucinate an invalid filename.

### Common `ref` Patterns

**Constrain to state values:**
```python
@tool("Select a dataset")
def select(
    self,
    dataset: str = spec.Param(values=ref.datasets.available_names)
) -> str: ...
```

**Use computed fields for dynamic lists:**
```python
class ProjectState(BaseModel):
    projects: list[dict] = []

    @computed_field
    @property
    def active_project_ids(self) -> list[str]:
        return [p['id'] for p in self.projects if p['active']]

@tool("Archive a project")
def archive(
    self,
    project_id: str = spec.Param(values=ref.projects.active_project_ids)
) -> str: ...
```

**Reference nested state:**
```python
@tool("Set user preference")
def set_preference(
    self,
    key: str = spec.Param(values=ref.config.user.valid_preference_keys)
) -> str: ...
```

## Accessing Agent State in Tools

Tools have full access to agent state via `self`:

```python
class ResearchAgent(BaseAgent):
    __system_message__ = "I research papers"

    paper_count: State[int] = spec.State(default=0)
    current_topic: State[str] = spec.State(default="general")

    @tool("Add a paper to the collection")
    def add_paper(self, title: str, authors: str) -> str:
        # Read state
        topic = self.current_topic

        # Modify state
        self.paper_count += 1

        # Use state in logic
        db.save_paper(title, authors, topic)

        return f"Added paper #{self.paper_count} about {topic}"
```

State modifications in tools persist across agent calls, enabling stateful workflows.

## Async Tools

Tools can be async functions for I/O operations:

```python
class APIAgent(BaseAgent):
    __system_message__ = "I call external APIs"

    @tool("Fetch data from external API")
    async def fetch_data(self, endpoint: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://api.example.com/{endpoint}")
            data = response.json()
        return f"Retrieved {len(data)} items from {endpoint}"

    @tool("Search the web")
    async def web_search(self, query: str, max_results: int = 5) -> str:
        results = await async_web_search(query, limit=max_results)
        return f"Found {len(results)} results for '{query}'"
```

PyAgentic automatically detects and handles async tools, awaiting them during execution.

## Conditional Tools

Tools can be conditionally included based on state using the `condition` parameter:

```python
class WorkflowAgent(BaseAgent):
    __system_message__ = "I manage workflows"

    workflow_stage: State[str] = spec.State(default="initial")

    @tool(
        "Start data processing",
        condition=lambda self: self.workflow_stage == "ready"
    )
    def start_processing(self) -> str:
        self.workflow_stage = "processing"
        return "Started processing"

    @tool(
        "Finalize results",
        condition=lambda self: self.workflow_stage == "processing"
    )
    def finalize(self) -> str:
        self.workflow_stage = "complete"
        return "Finalized results"
```

### How `condition` Works

The `condition` parameter accepts a callable (typically a lambda function) that receives the agent instance (`self`) and returns a boolean:

- **Evaluation Time**: Conditions are evaluated each time the tool schema is generated for the LLM, which happens before every agent interaction
- **Access to State**: The condition function has full access to the agent's state via `self`, allowing dynamic decisions based on current values
- **Dynamic Toolset**: Only tools whose conditions evaluate to `True` are included in the LLM's available toolset for that interaction

This mechanism allows you to create state-machine-like workflows where available tools change as the agent progresses through different states.

Tools can also be restricted to specific [phases](phases.md) using the `phases` parameter for structured multi-stage workflows. See the phases documentation for more details.

## Error Handling

When tools raise exceptions, PyAgentic catches them and returns an error message to the LLM:

```python
@tool("Divide two numbers")
def divide(self, a: float, b: float) -> str:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    return f"{a} / {b} = {result}"
```

If the LLM calls this tool with `b=0`, PyAgentic will catch the exception and send a message like:
```
Tool `divide` failed: Cannot divide by zero. Please kindly state to the user that it failed, provide state, and ask if they want to try again.
```

The LLM can then inform the user and potentially retry with different parameters.

### Custom Error Handling

For more control, catch exceptions yourself:

```python
@tool("Process data file")
def process_file(self, path: str) -> str:
    try:
        with open(path) as f:
            data = json.load(f)
        results = process_data(data)
        return f"Processed {len(results)} items from {path}"
    except FileNotFoundError:
        return f"Error: File '{path}' not found. Available files: {', '.join(self.available_files)}"
    except json.JSONDecodeError:
        return f"Error: File '{path}' is not valid JSON"
    except Exception as e:
        return f"Error processing file: {str(e)}"
```

Returning error messages as strings allows the LLM to handle errors gracefully and provide helpful feedback to users.