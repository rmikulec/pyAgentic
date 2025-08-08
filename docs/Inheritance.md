# Inheritance in PyAgentic

PyAgentic provides powerful inheritance capabilities that allow you to build complex agent hierarchies through class inheritance and extensions. This document covers how to extend agents, use agent extensions, and understand what can and cannot be inherited.

## Table of Contents

- [Extending Agents](#extending-agents)
- [Using Agent Extensions](#using-agent-extensions)
- [What Can Be Inherited](#what-can-be-inherited)
- [What Cannot Be Inherited](#what-cannot-be-inherited)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Extending Agents

### Basic Agent Extension

To extend an existing agent, simply inherit from it using standard Python class inheritance:

```python
from pyagentic import Agent
from pyagentic._base._context import ContextItem
from pyagentic._base._tool import tool

class BaseCalculatorAgent(Agent):
    __system_message__ = "You are a calculator agent that can perform basic math operations."
    
    precision: int = ContextItem(default=2)
    
    @tool("Add two numbers together")
    def add(self, a: float, b: float) -> float:
        return round(a + b, self.context.precision)
    
    @tool("Subtract second number from first")
    def subtract(self, a: float, b: float) -> float:
        return round(a - b, self.context.precision)

class AdvancedCalculatorAgent(BaseCalculatorAgent):
    __system_message__ = "You are an advanced calculator agent with additional mathematical functions."
    
    use_radians: bool = ContextItem(default=True)
    
    @tool("Calculate sine of a number")
    def sine(self, x: float) -> float:
        import math
        result = math.sin(x) if self.context.use_radians else math.sin(math.radians(x))
        return round(result, self.context.precision)
    
    @tool("Calculate natural logarithm")
    def log(self, x: float) -> float:
        import math
        return round(math.log(x), self.context.precision)
```

### Multiple Inheritance and MRO

PyAgentic supports multiple inheritance with proper Method Resolution Order (MRO) using C3 linearization:

```python
class StatisticsAgent(Agent):
    __system_message__ = "I provide statistical functions."
    
    sample_size: int = ContextItem(default=100)
    
    @tool("Calculate mean of a list of numbers")
    def mean(self, numbers: list[float]) -> float:
        return sum(numbers) / len(numbers)

class ScientificCalculatorAgent(AdvancedCalculatorAgent, StatisticsAgent):
    __system_message__ = "I am a scientific calculator with advanced math and statistics."
    
    # Inherits tools from both parent classes
    # MRO: ScientificCalculatorAgent -> AdvancedCalculatorAgent -> BaseCalculatorAgent -> StatisticsAgent -> Agent
```

## Using Agent Extensions

### Creating Agent Extensions

Agent extensions are mixins that provide reusable functionality without directly inheriting from `Agent`. They inherit from `AgentExtension`:

```python
from pyagentic._base._agent import AgentExtension
from pyagentic._base._context import ContextItem, computed_context
from pyagentic._base._tool import tool

class LoggingExtension(AgentExtension):
    """Extension that adds logging capabilities to any agent."""
    
    enable_logging: bool = ContextItem(default=True)
    log_level: str = ContextItem(default="INFO")
    
    @computed_context
    def logger_name(self):
        return f"{self.__class__.__name__}_Logger"
    
    @tool("Log a message with the specified level")
    def log_message(self, message: str, level: str = "INFO") -> str:
        if not self.context.enable_logging:
            return "Logging is disabled"
        
        import logging
        logger = logging.getLogger(self.context.logger_name)
        getattr(logger, level.lower())(message)
        return f"Logged message: {message}"

class DatabaseExtension(AgentExtension):
    """Extension that adds database connectivity."""
    
    db_connection_string: str = ContextItem(default="sqlite:///default.db")
    max_connections: int = ContextItem(default=10)
    
    @tool("Execute a database query")
    def query_database(self, query: str) -> str:
        # Simulate database query
        return f"Executed query: {query} on {self.context.db_connection_string}"
```

### Using Extensions in Agents

```python
class DataAnalysisAgent(Agent, LoggingExtension, DatabaseExtension):
    __system_message__ = "I analyze data using database queries and provide detailed logging."
    
    analysis_depth: str = ContextItem(default="detailed")
    
    @tool("Perform comprehensive data analysis")
    def analyze_data(self, table_name: str) -> str:
        # Use inherited tools from extensions
        self.log_message(f"Starting analysis of {table_name}", "INFO")
        
        query_result = self.query_database(f"SELECT * FROM {table_name}")
        
        self.log_message(f"Completed analysis of {table_name}", "INFO")
        
        return f"Analysis complete: {query_result}"
```

## What Can Be Inherited

The following elements are inherited following MRO (Method Resolution Order):

### ✅ Tools (`@tool` decorated methods)
- All tools from parent classes and extensions are inherited
- Child classes can override parent tools by defining a tool with the same name
- Tool definitions are merged across the inheritance hierarchy

### ✅ Context Attributes (`ContextItem` fields)
- Context items are inherited and can be overridden
- Default values can be changed in child classes
- Type annotations are preserved

### ✅ Computed Context (`@computed_context` methods)
- Computed context methods are inherited
- Can be overridden in child classes
- Maintain their dynamic behavior

### ✅ Linked Agents
- Agent references in type annotations are inherited
- Child classes can add additional linked agents

## What Cannot Be Inherited

### ❌ System Messages (`__system_message__`)
- Each agent **must** define its own `__system_message__`
- System messages are not inherited from parent classes
- This ensures each agent has a clear, specific purpose

### ❌ Input Templates (`__input_template__`)
- Input templates are not inherited
- Each agent must define its own if needed
- Allows for agent-specific input formatting

## Best Practices

### 1. Use Extensions for Cross-Cutting Concerns
```python
# Good: Use extensions for reusable functionality
class MetricsExtension(AgentExtension):
    track_performance: bool = ContextItem(default=True)
    
    @tool("Record performance metric")
    def record_metric(self, metric_name: str, value: float) -> str:
        # Implementation here
        pass

# Use the extension in multiple agents
class ChatAgent(Agent, MetricsExtension):
    __system_message__ = "I am a chat agent with performance tracking."

class AnalysisAgent(Agent, MetricsExtension):
    __system_message__ = "I analyze data with performance tracking."
```

### 2. Create Agent Hierarchies
```python
# Base functionality
class BaseAgent(Agent):
    __system_message__ = "I am a base agent with common functionality."
    
    timeout: int = ContextItem(default=30)
    
    @tool("Check system status")
    def health_check(self) -> str:
        return "System healthy"

# Specialized agents
class WebScrapingAgent(BaseAgent):
    __system_message__ = "I scrape web content."
    
    user_agent: str = ContextItem(default="PyAgentic/1.0")

class APIAgent(BaseAgent):
    __system_message__ = "I interact with APIs."
    
    api_key: str = ContextItem(default="")
```

### 3. Override Thoughtfully
```python
class TimerAgent(Agent):
    __system_message__ = "I provide timing functionality."
    
    @tool("Start a timer")
    def start_timer(self) -> str:
        return "Timer started"

class PreciseTimerAgent(TimerAgent):
    __system_message__ = "I provide high-precision timing."
    
    # Override parent tool with enhanced functionality
    @tool("Start a high-precision timer")
    def start_timer(self) -> str:
        import time
        # Enhanced implementation
        return f"High-precision timer started at {time.time_ns()}"
```

## Examples

### Example 1: Building a Service Agent Hierarchy

```python
class BaseServiceAgent(Agent):
    __system_message__ = "I am a base service agent."
    
    service_name: str = ContextItem(default="unknown")
    retry_count: int = ContextItem(default=3)
    
    @tool("Get service status")
    def get_status(self) -> str:
        return f"Service {self.context.service_name} is running"

class WebServiceAgent(BaseServiceAgent):
    __system_message__ = "I manage web services."
    
    port: int = ContextItem(default=8080)
    ssl_enabled: bool = ContextItem(default=False)
    
    @tool("Start web service")
    def start_service(self) -> str:
        protocol = "https" if self.context.ssl_enabled else "http"
        return f"Started web service on {protocol}://localhost:{self.context.port}"

class DatabaseServiceAgent(BaseServiceAgent):
    __system_message__ = "I manage database services."
    
    db_name: str = ContextItem(default="app_db")
    connection_pool_size: int = ContextItem(default=10)
    
    @tool("Connect to database")
    def connect(self) -> str:
        return f"Connected to database {self.context.db_name} with pool size {self.context.connection_pool_size}"
```

### Example 2: Using Multiple Extensions

```python
class AuthenticationExtension(AgentExtension):
    require_auth: bool = ContextItem(default=True)
    auth_method: str = ContextItem(default="bearer")
    
    @tool("Authenticate user")
    def authenticate(self, token: str) -> str:
        if not self.context.require_auth:
            return "Authentication disabled"
        return f"Authenticated using {self.context.auth_method} token"

class CachingExtension(AgentExtension):
    cache_enabled: bool = ContextItem(default=True)
    cache_ttl: int = ContextItem(default=300)  # seconds
    
    @tool("Cache data")
    def cache_data(self, key: str, data: str) -> str:
        if not self.context.cache_enabled:
            return "Caching disabled"
        return f"Cached data with key {key} for {self.context.cache_ttl}s"

class SecureAPIAgent(Agent, AuthenticationExtension, CachingExtension):
    __system_message__ = "I am a secure API agent with authentication and caching."
    
    api_version: str = ContextItem(default="v1")
    
    @tool("Make secure API call")
    def api_call(self, endpoint: str, token: str) -> str:
        # Use inherited functionality
        auth_result = self.authenticate(token)
        if "disabled" in auth_result:
            return "Authentication failed"
        
        # Simulate API call
        result = f"API call to {endpoint} successful"
        
        # Cache the result
        self.cache_data(f"api_{endpoint}", result)
        
        return result
```

This inheritance system provides flexibility while maintaining clear boundaries about what can and cannot be shared between agents, ensuring that each agent maintains its own identity while benefiting from shared functionality.
