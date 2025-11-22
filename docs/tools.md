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

Tools are only included in the LLM's toolset when their condition returns `True`, allowing you to create state-machine-like workflows.

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

## Best Practices

### 1. Clear, Concise Descriptions

The tool description is crucial - it's what the LLM uses to decide when to use the tool:

```python
# ❌ Too vague
@tool("Does stuff with data")
def process(self, data: str) -> str: ...

# ✅ Clear and specific
@tool("Analyze sentiment of text and return positive/negative/neutral classification")
def analyze_sentiment(self, text: str) -> str: ...
```

### 2. Use `ref` to Prevent Hallucination

Whenever parameters should be constrained to known values, use `ref`:

```python
# ❌ LLM might hallucinate file names
@tool("Open a file")
def open_file(self, filename: str) -> str: ...

# ✅ LLM can only choose from actual files
@tool("Open a file")
def open_file(
    self,
    filename: str = spec.Param(values=ref.filesystem.available_files)
) -> str: ...
```

### 3. Return Rich, Structured Information

Help the LLM make good decisions by returning detailed results:

```python
# ❌ Minimal information
@tool("Search database")
def search(self, query: str) -> str:
    results = db.search(query)
    return str(len(results))

# ✅ Rich information for LLM context
@tool("Search database")
def search(self, query: str) -> str:
    results = db.search(query)
    summary = {
        'count': len(results),
        'top_results': [r['title'] for r in results[:3]],
        'categories': list(set(r['category'] for r in results))
    }
    return json.dumps(summary, indent=2)
```

### 4. Keep Tools Focused

Each tool should do one thing well:

```python
# ❌ Too many responsibilities
@tool("Manage files")
def manage_files(self, action: str, path: str, content: str = None) -> str: ...

# ✅ Focused, single-purpose tools
@tool("Read a file")
def read_file(self, path: str) -> str: ...

@tool("Write to a file")
def write_file(self, path: str, content: str) -> str: ...

@tool("Delete a file")
def delete_file(self, path: str) -> str: ...
```

### 5. Use Pydantic Models for Complex Parameters

Group related parameters into models:

```python
# ❌ Long parameter list
@tool("Create user")
def create_user(
    self,
    name: str,
    email: str,
    age: int,
    department: str,
    role: str,
    permissions: list[str]
) -> str: ...

# ✅ Organized with Pydantic model
from pydantic import BaseModel

class UserData(BaseModel):
    name: str
    email: str
    age: int
    department: str
    role: str
    permissions: list[str] = []

@tool("Create user")
def create_user(self, user: UserData) -> str: ...
```

### 6. Provide Helpful Error Messages

When operations fail, guide the LLM toward solutions:

```python
@tool("Delete project")
def delete_project(self, project_id: str) -> str:
    if project_id not in self.projects:
        available = ', '.join(self.projects.keys())
        return f"Error: Project '{project_id}' not found. Available projects: {available}"

    if self.projects[project_id]['has_active_tasks']:
        return f"Error: Cannot delete project '{project_id}' - it has active tasks. Complete or reassign tasks first."

    del self.projects[project_id]
    return f"Successfully deleted project '{project_id}'"
```

## Common Patterns

### Builder Pattern

Accumulate data over multiple tool calls:

```python
class QueryBuilder(BaseAgent):
    query_parts: State[list[str]] = spec.State(default_factory=list)

    @tool("Add WHERE clause to query")
    def add_where(self, condition: str) -> str:
        self.query_parts.append(f"WHERE {condition}")
        return f"Added condition. Query has {len(self.query_parts)} parts"

    @tool("Add ORDER BY clause")
    def add_order(self, field: str) -> str:
        self.query_parts.append(f"ORDER BY {field}")
        return f"Added ordering. Query has {len(self.query_parts)} parts"

    @tool("Execute built query")
    def execute(self) -> str:
        query = "SELECT * FROM users " + " ".join(self.query_parts)
        results = db.execute(query)
        self.query_parts = []  # Reset
        return f"Executed query, got {len(results)} results"
```

### Validation Pattern

Check state before performing operations:

```python
@tool("Deploy application")
def deploy(self, environment: str) -> str:
    # Validate state first
    if not self.tests_passed:
        return "Cannot deploy: tests have not passed"

    if not self.build_complete:
        return "Cannot deploy: build is not complete"

    if environment == "production" and not self.approval_received:
        return "Cannot deploy to production: approval required"

    # Perform deployment
    deploy_to_environment(environment)
    return f"Successfully deployed to {environment}"
```

### Batch Operation Pattern

Accept lists for efficient bulk operations:

```python
class EmailAgent(BaseAgent):
    @tool("Send emails to multiple recipients")
    def send_bulk_email(
        self,
        recipients: list[str],
        subject: str,
        body: str
    ) -> str:
        sent = []
        failed = []

        for recipient in recipients:
            try:
                send_email(recipient, subject, body)
                sent.append(recipient)
            except Exception as e:
                failed.append(f"{recipient}: {str(e)}")

        result = f"Sent to {len(sent)} recipients"
        if failed:
            result += f"\nFailed: {', '.join(failed)}"
        return result
```

### Progressive Refinement Pattern

Let the LLM refine operations through multiple tool calls:

```python
class ImageAgent(BaseAgent):
    current_image: State[str] = spec.State(default=None)

    @tool("Load an image")
    def load_image(self, path: str) -> str:
        self.current_image = path
        return f"Loaded {path}"

    @tool("Apply filter to current image")
    def apply_filter(self, filter_type: str) -> str:
        if not self.current_image:
            return "Error: No image loaded"
        apply_filter(self.current_image, filter_type)
        return f"Applied {filter_type} filter to {self.current_image}"

    @tool("Save current image")
    def save_image(self, output_path: str) -> str:
        if not self.current_image:
            return "Error: No image loaded"
        save_image(self.current_image, output_path)
        return f"Saved to {output_path}"
```

## Next Steps

- Learn about [State Management](states.md) to build stateful tools
- Explore [Agent Linking](agent-linking.md) to compose tools across agents
- See [Responses](responses.md) to understand tool execution tracking
- Read [Policies](policies.md) to react to state changes from tools
