import pytest
import asyncio
import random

from pyagentic._base._agent import Agent, AgentExtension
from pyagentic._base._context import ContextItem, computed_context, _AgentContext
from pyagentic._base._tool import tool, _ToolDefinition
from pyagentic._base._exceptions import SystemMessageNotDeclared
from pyagentic.models.response import ToolResponse, AgentResponse


class TestBasicInheritance:
    """Test basic agent inheritance functionality."""

    def test_inherit_tools_from_parent(self):
        """Test that child agents inherit tools from parent agents."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            @tool("Parent tool")
            def parent_method(self) -> str:
                return "parent"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            @tool("Child tool")
            def child_method(self) -> str:
                return "child"
        
        # Child should have both tools
        assert "parent_method" in ChildAgent.__tool_defs__
        assert "child_method" in ChildAgent.__tool_defs__
        assert len(ChildAgent.__tool_defs__) == 2
        
        # Parent should only have parent tool
        assert "parent_method" in ParentAgent.__tool_defs__
        assert "child_method" not in ParentAgent.__tool_defs__
        assert len(ParentAgent.__tool_defs__) == 1

    def test_inherit_context_items_from_parent(self):
        """Test that child agents inherit context items from parent agents."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            parent_item: str = ContextItem(default="parent_value")
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            child_item: int = ContextItem(default=42)
        
        # Child should have both context items
        assert "parent_item" in ChildAgent.__context_attrs__
        assert "child_item" in ChildAgent.__context_attrs__
        assert len(ChildAgent.__context_attrs__) == 2
        
        # Check types are preserved
        assert ChildAgent.__context_attrs__["parent_item"][0] == str
        assert ChildAgent.__context_attrs__["child_item"][0] == int

    def test_inherit_computed_context_from_parent(self):
        """Test that child agents inherit computed context from parent agents."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            @computed_context
            def parent_computed(self):
                return "computed_parent"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            @computed_context
            def child_computed(self):
                return "computed_child"
        
        # Child should have both computed contexts
        assert "parent_computed" in ChildAgent.__context_attrs__
        assert "child_computed" in ChildAgent.__context_attrs__
        assert len(ChildAgent.__context_attrs__) == 2

    def test_override_parent_tool(self):
        """Test that child agents can override parent tools."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            @tool("Original tool")
            def shared_method(self) -> str:
                return "parent"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            @tool("Overridden tool")
            def shared_method(self) -> str:
                return "child"
        
        # Child should have overridden tool with new description
        assert "shared_method" in ChildAgent.__tool_defs__
        assert ChildAgent.__tool_defs__["shared_method"].description == "Overridden tool"
        assert len(ChildAgent.__tool_defs__) == 1

    def test_override_parent_context_item(self):
        """Test that child agents can override parent context items."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            shared_item: str = ContextItem(default="parent_default")
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            shared_item: str = ContextItem(default="child_default")
        
        # Child should have overridden context item
        assert "shared_item" in ChildAgent.__context_attrs__
        assert ChildAgent.__context_attrs__["shared_item"][1].get_default_value() == "child_default"

    def test_system_message_not_inherited(self):
        """Test that system messages are not inherited and must be defined."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
        
        # Child without system message should raise exception
        with pytest.raises(SystemMessageNotDeclared):
            class ChildAgent(ParentAgent):
                pass

    def test_agent_instance_creation_with_inheritance(self):
        """Test creating instances of agents with inheritance."""
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent agent"
            
            parent_item: str = ContextItem(default="parent_value")
            
            @tool("Parent tool")
            def parent_method(self) -> str:
                return "parent result"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"
            
            child_item: int = ContextItem(default=42)
        
        # Create instance
        child = ChildAgent(model="test", api_key="test")
        
        # Check context has both items
        assert hasattr(child.context, 'parent_item')
        assert hasattr(child.context, 'child_item')
        assert child.context.parent_item == "parent_value"
        assert child.context.child_item == 42


class TestMultipleInheritance:
    """Test multiple inheritance scenarios."""

    def test_multiple_inheritance_tool_merging(self):
        """Test that tools are properly merged from multiple parent classes."""
        
        class MathAgent(Agent):
            __system_message__ = "I do math"
            
            @tool("Add numbers")
            def add(self, a: float, b: float) -> str:
                return str(a + b)
        
        class StringAgent(Agent):
            __system_message__ = "I handle strings"
            
            @tool("Concatenate strings")
            def concat(self, a: str, b: str) -> str:
                return a + b
        
        class CombinedAgent(MathAgent, StringAgent):
            __system_message__ = "I do math and strings"
        
        # Should have tools from both parents
        assert "add" in CombinedAgent.__tool_defs__
        assert "concat" in CombinedAgent.__tool_defs__
        assert len(CombinedAgent.__tool_defs__) == 2

    def test_multiple_inheritance_context_merging(self):
        """Test that context items are properly merged from multiple parents."""
        
        class ConfigAgent(Agent):
            __system_message__ = "I handle config"
            
            timeout: int = ContextItem(default=30)
        
        class SecurityAgent(Agent):
            __system_message__ = "I handle security"
            
            api_key: str = ContextItem(default="secret")
        
        class SecureConfigAgent(ConfigAgent, SecurityAgent):
            __system_message__ = "I handle secure config"
        
        # Should have context items from both parents
        assert "timeout" in SecureConfigAgent.__context_attrs__
        assert "api_key" in SecureConfigAgent.__context_attrs__
        assert len(SecureConfigAgent.__context_attrs__) == 2

    def test_multiple_inheritance_mro_tool_precedence(self):
        """Test that MRO determines tool precedence in multiple inheritance."""
        
        class FirstAgent(Agent):
            __system_message__ = "First agent"
            
            @tool("First version")
            def shared_tool(self) -> str:
                return "first"
        
        class SecondAgent(Agent):
            __system_message__ = "Second agent"
            
            @tool("Second version")
            def shared_tool(self) -> str:
                return "second"
        
        class CombinedAgent(FirstAgent, SecondAgent):
            __system_message__ = "Combined agent"
        
        # First parent should win due to MRO
        assert CombinedAgent.__tool_defs__["shared_tool"].description == "First version"

    def test_diamond_inheritance(self):
        """Test diamond inheritance pattern works correctly."""
        
        class BaseAgent(Agent):
            __system_message__ = "Base agent"
            
            base_item: str = ContextItem(default="base")
            
            @tool("Base tool")
            def base_method(self) -> str:
                return "base"
        
        class LeftAgent(BaseAgent):
            __system_message__ = "Left agent"
            
            left_item: str = ContextItem(default="left")
        
        class RightAgent(BaseAgent):
            __system_message__ = "Right agent"
            
            right_item: str = ContextItem(default="right")
        
        class DiamondAgent(LeftAgent, RightAgent):
            __system_message__ = "Diamond agent"
        
        # Should have all context items
        assert "base_item" in DiamondAgent.__context_attrs__
        assert "left_item" in DiamondAgent.__context_attrs__
        assert "right_item" in DiamondAgent.__context_attrs__
        assert len(DiamondAgent.__context_attrs__) == 3
        
        # Should have base tool
        assert "base_method" in DiamondAgent.__tool_defs__
        assert len(DiamondAgent.__tool_defs__) == 1


class TestAgentExtensions:
    """Test AgentExtension functionality."""

    def test_agent_extension_basic(self):
        """Test basic agent extension functionality."""
        
        class LoggingExtension(AgentExtension):
            enable_logging: bool = ContextItem(default=True)
            
            @tool("Log a message")
            def log(self, message: str) -> str:
                return f"Logged: {message}" if self.context.enable_logging else "Logging disabled"
        
        class MyAgent(Agent, LoggingExtension):
            __system_message__ = "I am an agent with logging"
        
        # Should have extension's context item and tool
        assert "enable_logging" in MyAgent.__context_attrs__
        assert "log" in MyAgent.__tool_defs__

    def test_multiple_extensions(self):
        """Test using multiple extensions."""
        
        class DatabaseExtension(AgentExtension):
            db_url: str = ContextItem(default="sqlite:///test.db")
            
            @tool("Query database")
            def query(self, sql: str) -> str:
                return f"Query result for: {sql}"
        
        class CacheExtension(AgentExtension):
            cache_enabled: bool = ContextItem(default=True)
            
            @tool("Cache data")
            def cache(self, key: str, value: str) -> str:
                return f"Cached {key}={value}" if self.context.cache_enabled else "Caching disabled"
        
        class DataAgent(Agent, DatabaseExtension, CacheExtension):
            __system_message__ = "I handle data with caching"
        
        # Should have all extension items
        assert "db_url" in DataAgent.__context_attrs__
        assert "cache_enabled" in DataAgent.__context_attrs__
        assert "query" in DataAgent.__tool_defs__
        assert "cache" in DataAgent.__tool_defs__

    def test_extension_with_computed_context(self):
        """Test extensions with computed context."""
        
        class MetricsExtension(AgentExtension):
            metrics_enabled: bool = ContextItem(default=True)
            
            @computed_context
            def metric_prefix(self):
                return f"{self.__class__.__name__}_metrics"
            
            @tool("Record metric")
            def record_metric(self, name: str, value: float) -> str:
                if not self.context.metrics_enabled:
                    return "Metrics disabled"
                return f"{self.context.metric_prefix}.{name}: {value}"
        
        class AnalyticsAgent(Agent, MetricsExtension):
            __system_message__ = "I do analytics with metrics"
        
        # Should have computed context
        assert "metric_prefix" in AnalyticsAgent.__context_attrs__
        
        # Test instance creation
        agent = AnalyticsAgent(model="test", api_key="test")
        assert agent.context.metric_prefix == "AnalyticsAgentContext_metrics"

    def test_extension_inheritance_chain(self):
        """Test extensions in inheritance chains."""
        
        class BaseExtension(AgentExtension):
            base_config: str = ContextItem(default="base")
            
            @tool("Base extension tool")
            def base_ext_method(self) -> str:
                return "base extension"
        
        class SpecializedExtension(BaseExtension):
            specialized_config: str = ContextItem(default="specialized")
            
            @tool("Specialized extension tool")
            def specialized_ext_method(self) -> str:
                return "specialized extension"
        
        class MyAgent(Agent, SpecializedExtension):
            __system_message__ = "I use specialized extensions"
        
        # Should inherit from extension hierarchy
        assert "base_config" in MyAgent.__context_attrs__
        assert "specialized_config" in MyAgent.__context_attrs__
        assert "base_ext_method" in MyAgent.__tool_defs__
        assert "specialized_ext_method" in MyAgent.__tool_defs__


class TestInheritanceWithLinkedAgents:
    """Test inheritance with linked agents."""

    def test_inherit_linked_agents(self):
        """Test that linked agents are inherited."""
        
        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent with helper"
            
            helper: HelperAgent
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with inherited helper"
        
        # Child should inherit linked agent
        assert "helper" in ChildAgent.__linked_agents__
        assert ChildAgent.__linked_agents__["helper"] == HelperAgent

    def test_add_linked_agents_to_inheritance(self):
        """Test adding linked agents in child classes."""
        
        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
        
        class WorkerAgent(Agent):
            __system_message__ = "I am a worker"
        
        class ParentAgent(Agent):
            __system_message__ = "I am a parent with helper"
            
            helper: HelperAgent
        
        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with helper and worker"
            
            worker: WorkerAgent
        
        # Child should have both linked agents
        assert "helper" in ChildAgent.__linked_agents__
        assert "worker" in ChildAgent.__linked_agents__
        assert len(ChildAgent.__linked_agents__) == 2


class TestResponseModelInheritance:
    """Test that response models are properly constructed with inheritance."""

    def test_tool_response_models_inherited(self):
        """Test that tool response models include inherited tools."""
        
        class ParentAgent(Agent):
            __system_message__ = "Parent agent"
            
            @tool("Parent tool")
            def parent_tool(self, value: str) -> str:
                return f"parent: {value}"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "Child agent"
            
            @tool("Child tool")
            def child_tool(self, number: int) -> str:
                return str(number * 2)
        
        # Should have response models for both tools
        assert "parent_tool" in ChildAgent.__tool_response_models__
        assert "child_tool" in ChildAgent.__tool_response_models__
        assert len(ChildAgent.__tool_response_models__) == 2
        
        # Each should be a ToolResponse subclass
        for response_model in ChildAgent.__tool_response_models__.values():
            assert issubclass(response_model, ToolResponse)

    def test_agent_response_model_inheritance(self):
        """Test that agent response models include inherited elements."""
        
        class ParentAgent(Agent):
            __system_message__ = "Parent agent"
            
            @tool("Parent tool")
            def parent_tool(self) -> str:
                return "parent"
        
        class ChildAgent(ParentAgent):
            __system_message__ = "Child agent"
            
            @tool("Child tool")
            def child_tool(self) -> str:
                return "child"
        
        # Response model should exist and be AgentResponse subclass
        assert ChildAgent.__response_model__ is not None
        assert issubclass(ChildAgent.__response_model__, AgentResponse)


class TestInheritanceEdgeCases:
    """Test edge cases and error conditions in inheritance."""

    def test_empty_inheritance_chain(self):
        """Test agent with no additional functionality."""
        
        class EmptyAgent(Agent):
            __system_message__ = "I am empty"
        
        # Should have base attributes but empty collections
        assert len(EmptyAgent.__tool_defs__) == 0
        assert len(EmptyAgent.__context_attrs__) == 0
        assert len(EmptyAgent.__linked_agents__) == 0

    def test_deep_inheritance_chain(self):
        """Test deep inheritance chains work correctly."""
        
        class Level1Agent(Agent):
            __system_message__ = "Level 1"
            
            level1_item: str = ContextItem(default="level1")
            
            @tool("Level 1 tool")
            def level1_method(self) -> str:
                return "level1"
        
        class Level2Agent(Level1Agent):
            __system_message__ = "Level 2"
            
            level2_item: str = ContextItem(default="level2")
            
            @tool("Level 2 tool")
            def level2_method(self) -> str:
                return "level2"
        
        class Level3Agent(Level2Agent):
            __system_message__ = "Level 3"
            
            level3_item: str = ContextItem(default="level3")
            
            @tool("Level 3 tool")
            def level3_method(self) -> str:
                return "level3"
        
        # Should have all items from inheritance chain
        assert len(Level3Agent.__context_attrs__) == 3
        assert len(Level3Agent.__tool_defs__) == 3
        
        # Check all levels are present
        assert "level1_item" in Level3Agent.__context_attrs__
        assert "level2_item" in Level3Agent.__context_attrs__
        assert "level3_item" in Level3Agent.__context_attrs__
        
        assert "level1_method" in Level3Agent.__tool_defs__
        assert "level2_method" in Level3Agent.__tool_defs__
        assert "level3_method" in Level3Agent.__tool_defs__

    def test_complex_multiple_inheritance(self):
        """Test complex multiple inheritance scenario."""
        
        class DatabaseMixin(AgentExtension):
            db_timeout: int = ContextItem(default=30)
            
            @tool("Database operation")
            def db_operation(self) -> str:
                return "db result"
        
        class CacheMixin(AgentExtension):
            cache_ttl: int = ContextItem(default=300)
            
            @tool("Cache operation")
            def cache_operation(self) -> str:
                return "cache result"
        
        class BaseServiceAgent(Agent):
            __system_message__ = "Base service"
            
            service_name: str = ContextItem(default="base")
            
            @tool("Base service operation")
            def service_operation(self) -> str:
                return "service result"
        
        class AdvancedServiceAgent(BaseServiceAgent, DatabaseMixin, CacheMixin):
            __system_message__ = "Advanced service with database and cache"
            
            advanced_config: str = ContextItem(default="advanced")
            
            @tool("Advanced operation")
            def advanced_operation(self) -> str:
                return "advanced result"
        
        # Should have all context items
        expected_context_items = {
            "service_name", "db_timeout", "cache_ttl", "advanced_config"
        }
        assert set(AdvancedServiceAgent.__context_attrs__.keys()) == expected_context_items
        
        # Should have all tools
        expected_tools = {
            "service_operation", "db_operation", "cache_operation", "advanced_operation"
        }
        assert set(AdvancedServiceAgent.__tool_defs__.keys()) == expected_tools

    def test_inheritance_preserves_tool_metadata(self):
        """Test that tool metadata is preserved through inheritance."""
        
        class ParentAgent(Agent):
            __system_message__ = "Parent agent"
            
            @tool("Parent tool with specific description")
            def parent_tool(self, param: str) -> str:
                return param
        
        class ChildAgent(ParentAgent):
            __system_message__ = "Child agent"
        
        # Tool metadata should be preserved
        parent_tool_def = ChildAgent.__tool_defs__["parent_tool"]
        assert parent_tool_def.description == "Parent tool with specific description"
        assert parent_tool_def.name == "parent_tool"
