# State Management

PyAgentic's state system provides persistent, type-safe data storage for agents. States are fields that maintain their values across multiple conversations, allowing agents to build context, remember user preferences, and track complex workflows over time.

## Why Use States?

Without state, agents are stateless - they start fresh with each interaction and can't remember anything from previous conversations. States solve this by:

- **Persisting Data**: Values survive across multiple agent calls
- **Type Safety**: Full Pydantic validation and type checking
- **Dynamic Behavior**: Computed fields that update automatically
- **Smart References**: Link tool parameters to live state values
- **Access Control**: Control how the LLM can interact with state

## Basic State Fields

The simplest way to add state is using `State[T]` with `spec.State()`:

```python
from pyagentic import BaseAgent, State, spec

class ChatAgent(BaseAgent):
    __system_message__ = "I'm a helpful assistant that remembers our conversations"

    user_name: State[str] = spec.State(default="User")
    message_count: State[int] = spec.State(default=0)
    conversation_topic: State[str] = spec.State(default="general")
```

Each state field is:
- **Typed**: The `State[T]` annotation ensures type safety
- **Persistent**: Values are maintained between agent calls
- **Accessible**: Reference directly via `self.field_name` in tools

### Accessing State in Tools

Access state fields directly on `self` - the framework handles state management automatically:

```python
class ChatAgent(BaseAgent):
    __system_message__ = "I'm a helpful assistant"

    message_count: State[int] = spec.State(default=0)

    @tool("Record a new message")
    def record_message(self, content: str) -> str:
        self.message_count += 1
        return f"Message {self.message_count} recorded: {content}"
```

### State Configuration Options

`spec.State()` accepts several configuration options:

```python
class ConfiguredAgent(BaseAgent):
    # Simple default value
    username: State[str] = spec.State(default="anonymous")

    # Default factory for mutable defaults
    tags: State[list[str]] = spec.State(default_factory=list)

    # Description (used in system messages and auto-generated tools)
    bio: State[str] = spec.State(
        default="",
        description="User biography and preferences"
    )

    # Access control (covered below)
    api_key: State[str] = spec.State(
        default="",
        access="write"  # LLM can only write, not read
    )
```

## Pydantic Models as State

For complex state with multiple related fields and computed values, use Pydantic models:

```python
from pydantic import BaseModel, computed_field
from pyagentic import BaseAgent, State, spec

class UserProfile(BaseModel):
    name: str = "Guest"
    email: str = ""
    preferences: dict = {}
    login_count: int = 0

    @computed_field
    @property
    def is_registered(self) -> bool:
        """Computed field automatically updates based on other fields"""
        return bool(self.email)

    @computed_field
    @property
    def greeting(self) -> str:
        """Dynamic greeting based on login count"""
        if self.login_count == 0:
            return f"Welcome, {self.name}!"
        return f"Welcome back, {self.name}! Visit #{self.login_count}"

class UserAgent(BaseAgent):
    __system_message__ = "I manage user profiles. Greeting: {profile.greeting}"

    profile: State[UserProfile] = spec.State(default_factory=UserProfile)

    @tool("Update user email")
    def set_email(self, email: str) -> str:
        self.profile.email = email
        # is_registered automatically updates!
        return f"Email set. Registered: {self.profile.is_registered}"
```

### Benefits of Pydantic Models

1. **Computed Fields**: Use `@computed_field` for values derived from other state
2. **Validation**: Pydantic validates all field types and constraints
3. **Nested Structure**: Organize related data logically
4. **Serialization**: Easy JSON export/import of state
5. **IDE Support**: Full autocomplete for all fields

## Referencing State with `ref`

The `ref` system creates dynamic links between tool parameters and state values:

```python
from pyagentic import ref

class TopicState(BaseModel):
    available_topics: list[str] = ["general", "tech", "science"]
    current_topic: str = "general"

    @computed_field
    @property
    def topic_count(self) -> int:
        return len(self.available_topics)

class TopicAgent(BaseAgent):
    __system_message__ = "I manage topics. Current: {current_topic}"

    topics: State[TopicState] = spec.State(default_factory=TopicState)

    @tool("Switch to a different topic")
    def switch_topic(
        self,
        topic: str = spec.Param(
            description="Topic to switch to",
            values=ref.available_topics  # LLM can only pick from this list!
        )
    ) -> str:
        self.current_topic = topic
        return f"Switched to {topic}"
```

### How `ref` Works

- **Direct References**: `ref.available_topics` creates a reference to state fields
- **Runtime Resolution**: Values are resolved when the tool schema is generated
- **Always Current**: References always point to the latest state values
- **Type Safe**: Full typing support through the reference chain

### Using State in System Messages

You can also reference state in system messages using template syntax:

```python
class Agent(BaseAgent):
    __system_message__ = """
    Current topic: {current_topic}
    Available topics: {available_topics}
    Total topics: {topic_count}
    """

    topics: State[TopicState] = spec.State(default_factory=TopicState)
```

When using a Pydantic model for state, the model's fields (including computed fields) are directly accessible in templates without needing to prefix them with the state field name.

## State Access Control

Control how the LLM can interact with state using the `access` parameter:

```python
class SecureAgent(BaseAgent):
    __system_message__ = "I'm a secure agent with controlled state access"

    # Default: LLM can see value but not modify it
    user_id: State[str] = spec.State(
        default="",
        access="read"
    )

    # LLM can modify but not see the value
    api_key: State[str] = spec.State(
        default="",
        access="write",
        set_description="Store the user's API key securely"
    )

    # LLM can both read and write
    preferences: State[dict] = spec.State(
        default_factory=dict,
        access="readwrite",
        get_description="Get user preferences",
        set_description="Update user preferences"
    )

    # LLM cannot interact with this at all
    internal_cache: State[dict] = spec.State(
        default_factory=dict,
        access="hidden"
    )
```

### Access Levels

| Access | LLM Can Read | LLM Can Write | Auto-Generated Tools |
|--------|--------------|---------------|---------------------|
| `"read"` (default) | ✅ Yes | ❌ No | Getter only |
| `"write"` | ❌ No | ✅ Yes | Setter only |
| `"readwrite"` | ✅ Yes | ✅ Yes | Getter + Setter |
| `"hidden"` | ❌ No | ❌ No | None |

### Auto-Generated State Tools

PyAgentic automatically generates tools for state access based on the `access` level:

```python
class DataAgent(BaseAgent):
    __system_message__ = "I manage data"

    current_data: State[str] = spec.State(
        default="",
        access="readwrite",
        get_description="Retrieve the current stored data",
        set_description="Update the stored data with new content"
    )
```

This automatically creates two tools:
1. **`get_current_data()`** - Returns the current value
2. **`set_current_data(value: str)`** - Updates the value

The LLM can call these tools just like your custom `@tool` methods.

## State in System Messages and Templates

Reference state fields in templates directly by their field names:

```python
class ResearchAgent(BaseAgent):
    __system_message__ = """
    You are a research assistant.
    Current focus: {current_topic}
    Papers collected: {paper_count}
    """

    __input_template__ = """
    Research Topic: {current_topic}
    User Message: {user_message}
    """

    research: State[ResearchState] = spec.State(default_factory=ResearchState)
```

Templates support:
- Direct field access: `{current_topic}`
- Computed fields: `{topic_count}`
- All fields from your state model are automatically available

## State Policies

States can have policies attached that react to state changes, validate values, persist data, and more. Policies provide powerful hooks for implementing cross-cutting concerns without cluttering your tool implementations.

Quick preview:

```python
# Define custom policies
class ScoreValidationPolicy:
    def on_set(self, event, value):
        if not 0 <= value <= 100:
            raise ValueError("Score must be between 0 and 100")
        return None

    async def background_set(self, event, value):
        return None

    def on_get(self, event, value):
        return None

    async def background_get(self, event, value):
        return None

class ChangeHistoryPolicy:
    def __init__(self, max_length=100):
        self.max_length = max_length
        self.history = []

    def on_set(self, event, value):
        self.history.append({
            "old": event.previous,
            "new": value,
            "timestamp": event.timestamp
        })
        if len(self.history) > self.max_length:
            self.history.pop(0)
        return None

    # ... other methods

# Use them
class TrackedAgent(BaseAgent):
    __system_message__ = "I track changes to my state"

    score: State[int] = spec.State(
        default=0,
        policies=[
            ScoreValidationPolicy(),
            ChangeHistoryPolicy(max_length=50)
        ]
    )
```

Policies execute in order when state changes:
1. **ScoreValidationPolicy** - Ensures score is between 0-100 (raises error if not)
2. **ChangeHistoryPolicy** - Records the change in an in-memory history

See the **[Policies documentation](policies.md)** for:
- Complete guide to the Policy protocol
- 7+ patterns for building policies (validation, transformation, history, persistence, etc.)
- Async vs sync handlers with execution flow diagrams
- Best practices and common use cases

## Best Practices

### 1. Use Pydantic Models for Complex State

Instead of many individual state fields:

```python
# ❌ Not ideal - many scattered fields
class Agent(BaseAgent):
    user_name: State[str] = spec.State(default="")
    user_email: State[str] = spec.State(default="")
    user_age: State[int] = spec.State(default=0)
    user_prefs: State[dict] = spec.State(default_factory=dict)
```

Group related data into models:

```python
# ✅ Better - organized and type-safe
class UserData(BaseModel):
    name: str = ""
    email: str = ""
    age: int = 0
    preferences: dict = {}

class Agent(BaseAgent):
    user: State[UserData] = spec.State(default_factory=UserData)
```

### 2. Leverage Computed Fields

Use `@computed_field` for derived values instead of manually updating them:

```python
class TaskState(BaseModel):
    total_tasks: int = 0
    completed_tasks: int = 0

    @computed_field
    @property
    def completion_percentage(self) -> float:
        """Automatically stays in sync"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
```

### 3. Use `ref` for Dynamic Constraints

Prevent hallucination by constraining parameters to valid state values:

```python
@tool("Select a file")
def select_file(
    self,
    filename: str = spec.Param(
        description="File to select",
        values=ref.files.available_files  # Only allow existing files!
    )
) -> str:
    ...
```

### 4. Choose Appropriate Access Levels

- Use `"read"` for data the LLM needs to see but shouldn't modify
- Use `"write"` for sensitive data the LLM should store but not read
- Use `"hidden"` for internal state the LLM shouldn't know about
- Use `"readwrite"` sparingly, only when the LLM truly needs both

### 5. Provide Clear Descriptions

Help the LLM understand your state:

```python
conversation_history: State[list[str]] = spec.State(
    default_factory=list,
    description="Complete history of user messages in this session",
    access="read"
)
```

## Common Patterns

### Accumulator Pattern

Build up data over multiple interactions:

```python
class ResearchAgent(BaseAgent):
    papers: State[list[dict]] = spec.State(default_factory=list)

    @tool("Add paper to collection")
    def add_paper(self, title: str, authors: str) -> str:
        self.papers.append({"title": title, "authors": authors})
        return f"Added paper. Total: {len(self.papers)}"
```

### State Machine Pattern

Track agent state through a workflow:

```python
class WorkflowState(BaseModel):
    stage: str = "initial"
    data: dict = {}

    @computed_field
    @property
    def valid_next_stages(self) -> list[str]:
        stages = {
            "initial": ["gathering"],
            "gathering": ["processing"],
            "processing": ["complete"],
            "complete": []
        }
        return stages.get(self.stage, [])

class WorkflowAgent(BaseAgent):
    workflow: State[WorkflowState] = spec.State(default_factory=WorkflowState)

    @tool("Advance to next stage")
    def advance(
        self,
        stage: str = spec.Param(values=ref.valid_next_stages)
    ) -> str:
        self.stage = stage
        return f"Advanced to {stage}"
```

### Configuration Pattern

Store user preferences and settings:

```python
class Settings(BaseModel):
    language: str = "en"
    theme: str = "light"
    notifications: bool = True

    @computed_field
    @property
    def formatted_display(self) -> str:
        return f"{self.language.upper()} | {self.theme} theme | {'🔔' if self.notifications else '🔕'}"

class Agent(BaseAgent):
    __system_message__ = "Settings: {settings.formatted_display}"

    settings: State[Settings] = spec.State(
        default_factory=Settings,
        access="readwrite"
    )
```

## Next Steps

- Learn about [Policies](policies.md) for advanced state management
- Explore [Agent Linking](agent-linking.md) to share state between agents
- See [Structured Outputs](structured-output.md) for validating agent responses
