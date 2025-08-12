# Agent Inheritance

PyAgentic supports standard Python inheritance, allowing you to build agent hierarchies that share tools, context, and functionality. You can also use extensions to add cross-cutting capabilities to multiple agents without deep inheritance chains.

## How Inheritance Works

Agent inheritance follows Python's standard rules with some PyAgentic-specific behaviors. When you inherit from an agent, you get all its tools, context items, and linked agents. However, each agent must define its own system message to maintain clear identity and purpose.

PyAgentic builds the complete agent schema at class definition time, combining inherited elements with new ones to create a fully-typed, predictable agent interface.

### Inheritance Rules

PyAgentic follows specific rules about what gets inherited and what must be redefined:

#### What Gets Inherited ✅

- **Tools** - All `@tool` methods are inherited and can be overridden
- **Context Items** - `ContextItem` fields inherit with their defaults and types
- **Linked Agents** - Agent references are inherited, child classes can add more
- **Computed Context** - Dynamic context methods are inherited and overridable

#### What Must Be Redefined ❌

- **System Messages** - Each agent must define its own `__system_message__` to maintain clear identity
- **Input Templates** - `__input_template__` is not inherited, allowing agent-specific formatting

This design ensures that while agents can share functionality, each maintains its own distinct purpose and behavior.

## Basic Inheritance

Extend agents using normal Python inheritance to build specialized capabilities on top of base functionality:

```python
class BaseAssistantAgent(Agent):
    __system_message__ = "I am a helpful AI assistant"
    
    user_name: str = ContextItem(default="User")
    session_id: str = ContextItem(default="")
    conversation_context: str = ContextItem(default="")
    
    @tool("Get current timestamp")
    def get_timestamp(self) -> str: ...
    
    @tool("Read a file in the current directory")
    def count_words(self, file_name: str) -> int: ...
    
    @tool("Update conversation context")
    def update_context(self, new_context: str) -> str: ...

class CodeAssistantAgent(BaseAssistantAgent):
    __system_message__ = "I help with programming tasks and code review"
    
    # Inherit user_name, session_id, conversation_context, and basic tools
    # Add coding-specific context
    preferred_language: str = ContextItem(default="python")
    current_project: str = ContextItem(default="")
    debug_mode: bool = ContextItem(default=False)
    
    @tool("Format code snippet")
    def format_code(self, code: str, language: str = None) -> str: ...
    
    @tool("Generate code documentation")
    def document_code(self, code: str) -> str: ...
```

The `CodeAssistantAgent` inherits all the basic functionality from `BaseAssistantAgent` (user tracking, session management, conversation context) while adding its own specialized tools for code formatting and documentation. This creates a clean hierarchy where common assistant behaviors are shared but specific domains add their own capabilities.

Inheritance also works with tool overriding to enhance parent functionality:

```python
class AdvancedCodeAssistantAgent(CodeAssistantAgent):
    __system_message__ = "I provide advanced programming assistance with security analysis"
    
    # Override parent tool with enhanced functionality
    @tool("Format and validate code snippet")
    def format_code(self, code: str, language: str = None, formatting_style: str = None) -> str: ...
```

## Agent Extensions

Extensions allow you to add cross-cutting functionality to multiple agents without creating deep inheritance hierarchies. They're perfect for capabilities like logging, authentication, or caching that many different agents might need.

### Creating Extensions

Extensions inherit from `AgentExtension` and can include tools, context items, and computed context:

```python
class FileOperationsExtension(AgentExtension):
    base_directory: str = ContextItem(default="./workspace")
    
    @tool("Read file contents")
    def read_file(self, filename: str) -> str: ...
    
    @tool("Write content to file")
    def write_file(self, filename: str, content: str) -> str: ...

class WebSearchExtension(AgentExtension):
    max_results: int = ContextItem(default=5)
    
    @tool("Search the web for information")
    def web_search(self, query: str) -> str: ...
    
    @tool("Get webpage content")
    def get_webpage(self, url: str) -> str: ...
```

### Using Extensions

Simply include extensions in your agent's inheritance list:

```python
class ResearchAgent(Agent, FileOperationsExtension, WebSearchExtension):
    __system_message__ = "I help with research by searching the web and managing files"
    
    research_topic: str = ContextItem(default="")
    
    @tool("Conduct comprehensive research")
    def research_topic(self, topic: str) -> str: ...
```

The agent automatically gets all tools and context from both extensions, creating a powerful composition pattern.

## Method Resolution Order (MRO)

When using multiple extensions or inheritance, Python's Method Resolution Order (MRO) determines which implementation is used when there are conflicts. PyAgentic follows Python's C3 linearization:

```python
class MemoryExtension(AgentExtension):
    @tool("Remember information")
    def remember(self, info: str) -> str: ...

class ConversationExtension(AgentExtension):
    @tool("Remember information")
    def remember(self, info: str) -> str: ...

class ChatbotAgent(Agent, MemoryExtension, ConversationExtension):
    __system_message__ = "I am a conversational AI"
    
# MRO: ChatbotAgent -> MemoryExtension -> ConversationExtension -> Agent
# The MemoryExtension.remember() method will be used
```

You can check the MRO with `ChatbotAgent.__mro__` to understand the resolution order. Extensions listed first in the inheritance list take precedence.