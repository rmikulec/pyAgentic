import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

from pyagentic._base._agent import Agent
from pyagentic._base._context import ContextItem
from pyagentic._base._tool import tool
from pyagentic.models.response import AgentResponse


class TestAgentLinkingBasics:
    """Test basic agent linking functionality."""

    def test_linked_agent_declaration(self):
        """Test that linked agents are properly declared in class attributes."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper agent"
            __description__ = "Provides assistance with tasks"

        class MainAgent(Agent):
            __system_message__ = "I am a main agent with a helper"

            helper: HelperAgent

        # Should have the linked agent in class attributes
        assert "helper" in MainAgent.__linked_agents__
        assert MainAgent.__linked_agents__["helper"] == HelperAgent
        assert len(MainAgent.__linked_agents__) == 1

    def test_multiple_linked_agents(self):
        """Test declaring multiple linked agents."""

        class DatabaseAgent(Agent):
            __system_message__ = "I handle database operations"
            __description__ = "Database interaction specialist"

        class CacheAgent(Agent):
            __system_message__ = "I handle caching operations"
            __description__ = "Cache management specialist"

        class FileAgent(Agent):
            __system_message__ = "I handle file operations"
            __description__ = "File management specialist"

        class MainAgent(Agent):
            __system_message__ = "I coordinate multiple agents"

            database: DatabaseAgent
            cache: CacheAgent
            file_handler: FileAgent

        # Should have all three linked agents
        assert len(MainAgent.__linked_agents__) == 3
        assert "database" in MainAgent.__linked_agents__
        assert "cache" in MainAgent.__linked_agents__
        assert "file_handler" in MainAgent.__linked_agents__

        # Check types are correct
        assert MainAgent.__linked_agents__["database"] == DatabaseAgent
        assert MainAgent.__linked_agents__["cache"] == CacheAgent
        assert MainAgent.__linked_agents__["file_handler"] == FileAgent

    def test_linked_agent_instance_creation(self):
        """Test creating instances with linked agents."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper agent"
            __description__ = "Provides assistance"

            @tool("Help with a task")
            def help_task(self, task: str) -> str:
                return f"Helping with: {task}"

        class MainAgent(Agent):
            __system_message__ = "I am a main agent with a helper"

            helper: HelperAgent

        # Create instances
        helper_instance = HelperAgent(model="_mock::test-model", api_key="test-key")
        main_instance = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper_instance)

        # Verify the linked agent is properly set
        assert hasattr(main_instance, "helper")
        assert main_instance.helper == helper_instance
        assert isinstance(main_instance.helper, HelperAgent)

    def test_linked_agent_optional_initialization(self):
        """Test that linked agents are optional during initialization."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper agent"
            __description__ = "Provides assistance"

        class MainAgent(Agent):
            __system_message__ = "I am a main agent with optional helper"

            helper: HelperAgent

        # Should be able to create without the linked agent
        main_instance = MainAgent(model="_mock::test-model", api_key="test-key")

        # helper should be None when not provided
        assert main_instance.helper is None


class TestAgentLinkingInheritance:
    """Test inheritance behavior with linked agents."""

    def test_inherit_linked_agents(self):
        """Test that linked agents are inherited from parent classes."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class ParentAgent(Agent):
            __system_message__ = "I am a parent with helper"

            helper: HelperAgent

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with inherited helper"

        # Child should inherit linked agent
        assert "helper" in ChildAgent.__linked_agents__
        assert ChildAgent.__linked_agents__["helper"] == HelperAgent
        assert len(ChildAgent.__linked_agents__) == 1

    def test_add_linked_agents_to_inheritance(self):
        """Test adding additional linked agents in child classes."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class WorkerAgent(Agent):
            __system_message__ = "I am a worker"
            __description__ = "Does work"

        class ParentAgent(Agent):
            __system_message__ = "I am a parent with helper"

            helper: HelperAgent

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with helper and worker"

            worker: WorkerAgent

        # Child should have both linked agents
        assert len(ChildAgent.__linked_agents__) == 2
        assert "helper" in ChildAgent.__linked_agents__
        assert "worker" in ChildAgent.__linked_agents__
        assert ChildAgent.__linked_agents__["helper"] == HelperAgent
        assert ChildAgent.__linked_agents__["worker"] == WorkerAgent

    def test_override_linked_agent_type(self):
        """Test overriding a linked agent with a different type."""

        class BaseHelperAgent(Agent):
            __system_message__ = "I am a base helper"
            __description__ = "Basic help"

        class AdvancedHelperAgent(BaseHelperAgent):
            __system_message__ = "I am an advanced helper"
            __description__ = "Advanced help"

        class ParentAgent(Agent):
            __system_message__ = "I am a parent with base helper"

            helper: BaseHelperAgent

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with advanced helper"

            helper: AdvancedHelperAgent  # Override with more specific type

        # Child should use the overridden type
        assert "helper" in ChildAgent.__linked_agents__
        assert ChildAgent.__linked_agents__["helper"] == AdvancedHelperAgent
        assert len(ChildAgent.__linked_agents__) == 1


class TestAgentLinkingToolGeneration:
    """Test that linked agents appear as tools in the parent agent."""

    @pytest.mark.asyncio
    async def test_linked_agent_becomes_tool(self):
        """Test that linked agents are included in the tool definitions."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper agent"
            __description__ = "Provides assistance with various tasks"

        class MainAgent(Agent):
            __system_message__ = "I am a main agent with a helper"

            helper: HelperAgent

        # Create instances
        helper = HelperAgent(model="_mock::test-model", api_key="test-key")
        main = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper)

        # Build tool definitions
        tool_defs = await main._get_tool_defs()

        # Should include the linked agent as a tool
        tool_names = [tool["name"] for tool in tool_defs]
        assert "helper" in tool_names

        # Find the helper tool definition
        helper_tool = next(tool for tool in tool_defs if tool["name"] == "helper")
        assert helper_tool["description"] == "Provides assistance with various tasks"

    @pytest.mark.asyncio
    async def test_multiple_linked_agents_as_tools(self):
        """Test that multiple linked agents all become tools."""

        class DatabaseAgent(Agent):
            __system_message__ = "I handle database operations"
            __description__ = "Database specialist"

        class CacheAgent(Agent):
            __system_message__ = "I handle caching"
            __description__ = "Cache specialist"

        class MainAgent(Agent):
            __system_message__ = "I coordinate multiple specialists"

            database: DatabaseAgent
            cache: CacheAgent

        # Create instances
        db = DatabaseAgent(model="_mock::test-model", api_key="test-key")
        cache = CacheAgent(model="_mock::test-model", api_key="test-key")
        main = MainAgent(model="_mock::test-model", api_key="test-key", database=db, cache=cache)

        # Build tool definitions
        tool_defs = await main._get_tool_defs()
        tool_names = [tool["name"] for tool in tool_defs]

        # Both agents should be available as tools
        assert "database" in tool_names
        assert "cache" in tool_names

    @pytest.mark.asyncio
    async def test_mixed_tools_and_linked_agents(self):
        """Test that regular tools and linked agents coexist."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(Agent):
            __system_message__ = "I have both tools and linked agents"

            helper: HelperAgent

            @tool("Perform a direct calculation")
            def calculate(self, a: int, b: int) -> str:
                return str(a + b)

        # Create instances
        helper = HelperAgent(model="_mock::test-model", api_key="test-key")
        main = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper)

        # Build tool definitions
        tool_defs = await main._get_tool_defs()
        tool_names = [tool["name"] for tool in tool_defs]

        # Should have both regular tools and linked agents
        assert "calculate" in tool_names
        assert "helper" in tool_names
        assert len(tool_names) == 2


class TestAgentLinkingExecution:
    """Test execution of linked agents during agent runs."""

    @pytest.mark.asyncio
    async def test_linked_agent_call_processing(self):
        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

            @tool("Help with a task")
            def help(self, task: str) -> str:
                return f"Helped with: {task}"

        class MainAgent(Agent):
            __system_message__ = "I coordinate with helper"
            helper: HelperAgent

        mock_response = AgentResponse(final_output="Helper completed the task")

        # Patch the class-level __call__ so special-method lookup hits the mock
        with patch.object(
            HelperAgent, "__call__", new=AsyncMock(return_value=mock_response)
        ) as mocked_call:
            helper = HelperAgent(model="_mock::test-model", api_key="test-key")
            main = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper)

            tool_call = MagicMock()
            tool_call.name = "helper"
            tool_call.arguments = '{"user_input": "test task"}'
            tool_call.call_id = "test-call-id"

            response = await main._process_agent_call(tool_call)

            # IMPORTANT: self + keyword
            HelperAgent.__call__.assert_awaited_once_with(user_input="test task")
            assert (
                response == mock_response
            )  # or: assert response.final_output == "Helper completed the task"

            # Messages
            assert len(main.context._messages) == 2
            assert main.context._messages[0] is tool_call
            assert main.context._messages[1]["type"] == "function_call_output"
            assert main.context._messages[1]["call_id"] == "test-call-id"
            assert "Helper completed the task" in main.context._messages[1]["output"]

    @pytest.mark.asyncio
    async def test_linked_agent_call_error_handling(self):
        """Test error handling when linked agent calls fail."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper that might fail"
            __description__ = "Provides help but may error"

        class MainAgent(Agent):
            __system_message__ = "I coordinate with helper"

            helper: HelperAgent

        # Create instances
        helper = HelperAgent(model="_mock::test-model", api_key="test-key")
        main = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper)

        # Mock tool call
        tool_call = MagicMock()
        tool_call.name = "helper"
        tool_call.arguments = '{"user_input": "test task"}'
        tool_call.call_id = "test-call-id"

        # Mock the helper to raise an exception
        helper.__call__ = AsyncMock(side_effect=Exception("Helper failed"))

        # Process the agent call (should handle the error)
        response = await main._process_agent_call(tool_call)

        # Should still return something even with error
        assert response is not None

        # Error message should be in context
        assert len(main.context._messages) == 2
        output_message = main.context._messages[1]["output"]
        assert "failed" in output_message.lower()
        assert "helper" in output_message


class TestAgentLinkingResponseModels:
    """Test that response models are properly built with linked agents."""

    def test_response_model_includes_linked_agents(self):
        """Test that the agent response model includes linked agent responses."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

            @tool("Help with task")
            def help(self, task: str) -> str:
                return f"Helped with: {task}"

        class MainAgent(Agent):
            __system_message__ = "I coordinate with helper"

            helper: HelperAgent

            @tool("Main agent tool")
            def main_tool(self, data: str) -> str:
                return f"Processed: {data}"

        # Response model should exist and be properly structured
        assert MainAgent.__response_model__ is not None

        # Check the model fields
        model_fields = MainAgent.__response_model__.model_fields
        assert "final_output" in model_fields
        assert "tool_responses" in model_fields  # For regular tools
        assert "agent_responses" in model_fields  # For linked agents

    def test_response_model_inheritance_with_linked_agents(self):
        """Test response models work correctly with inheritance and linked agents."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class WorkerAgent(Agent):
            __system_message__ = "I am a worker"
            __description__ = "Does work"

        class ParentAgent(Agent):
            __system_message__ = "I am a parent"

            helper: HelperAgent

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child"

            worker: WorkerAgent

        # Child should have response model that accounts for both linked agents
        assert ChildAgent.__response_model__ is not None

        # Should have fields for agent responses
        model_fields = ChildAgent.__response_model__.model_fields
        assert "agent_responses" in model_fields


class TestAgentLinkingContextIntegration:
    """Test integration between linked agents and context."""

    def test_linked_agents_with_context_items(self):
        """Test that linked agents work properly with context items."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper with context"
            __description__ = "Context-aware helper"

            helper_mode: str = ContextItem(default="standard")

            @tool("Help based on mode")
            def help(self, task: str) -> str:
                return f"[{self.context.helper_mode}] Helped with: {task}"

        class MainAgent(Agent):
            __system_message__ = "I coordinate with contextual helper"

            helper: HelperAgent
            coordination_level: int = ContextItem(default=1)

        # Create instances with different context values
        helper = HelperAgent(model="_mock::test-model", api_key="test-key", helper_mode="advanced")
        main = MainAgent(
            model="_mock::test-model", api_key="test-key", helper=helper, coordination_level=5
        )

        # Verify context is properly set
        assert main.helper.context.helper_mode == "advanced"
        assert main.context.coordination_level == 5

    def test_linked_agents_none_handling(self):
        """Test proper handling when linked agents are None."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(Agent):
            __system_message__ = "I may or may not have a helper"

            helper: HelperAgent

            @tool("Main tool")
            def main_tool(self, data: str) -> str:
                if self.helper is not None:
                    return f"With helper: {data}"
                return f"Without helper: {data}"

        # Create without helper
        main = MainAgent(model="_mock::test-model", api_key="test-key")

        # Should handle None gracefully
        assert main.helper is None

        # Tool should work correctly
        result = main.main_tool("test")
        assert result == "Without helper: test"


class TestAgentLinkingEdgeCases:
    """Test edge cases and complex scenarios for agent linking."""

    def test_circular_agent_references_forward_ref_not_supported(self):
        """
        Forward references are currently unsupported: we should raise TypeError
        and also emit a clear warning telling the user what's wrong.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # ensure we capture even once-per-location warnings

            with pytest.raises(TypeError):

                class AgentA(Agent):
                    __system_message__ = "I am Agent A"
                    __description__ = "Agent A"
                    # Forward reference — should trigger warning + TypeError
                    agent_b: "AgentB"

                class AgentB(Agent):
                    __system_message__ = "I am Agent B"
                    __description__ = "Agent B"
                    agent_a: AgentA

        # Verify a warning was emitted with the expected text
        msgs = [str(w.message) for w in caught]
        assert any(
            "Forward reference for agents are unsupported:" in m and "agent_b" in m for m in msgs
        ), f"No expected forward-ref warning found. Captured: {msgs}"

    def test_linked_agent_without_description(self):
        """Test linked agents that don't have explicit descriptions."""

        class SimpleAgent(Agent):
            __system_message__ = "I am simple"
            # No __description__ defined

        class MainAgent(Agent):
            __system_message__ = "I have a simple agent"

            simple: SimpleAgent

        # Should still work, using default description behavior
        assert "simple" in MainAgent.__linked_agents__
        assert MainAgent.__linked_agents__["simple"] == SimpleAgent

    def test_deeply_nested_agent_hierarchies(self):
        """Test deeply nested agent inheritance with linking."""

        class BaseAgent(Agent):
            __system_message__ = "I am the base"

        class SpecialistAgent(BaseAgent):
            __system_message__ = "I am a specialist"
            __description__ = "Specialized agent"

        class Level1Agent(Agent):
            __system_message__ = "Level 1 agent"

            specialist: SpecialistAgent

        class Level2Agent(Level1Agent):
            __system_message__ = "Level 2 agent"

        class Level3Agent(Level2Agent):
            __system_message__ = "Level 3 agent"

        # All levels should inherit the linked agent
        assert "specialist" in Level1Agent.__linked_agents__
        assert "specialist" in Level2Agent.__linked_agents__
        assert "specialist" in Level3Agent.__linked_agents__

        # All should reference the same agent type
        assert (
            Level1Agent.__linked_agents__["specialist"]
            == Level2Agent.__linked_agents__["specialist"]
            == Level3Agent.__linked_agents__["specialist"]
            == SpecialistAgent
        )

    def test_linked_agent_with_same_name_as_method(self):
        """Test potential conflicts between linked agent names and methods."""

        class HelperAgent(Agent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(Agent):
            __system_message__ = "I have potential naming conflicts"

            helper: HelperAgent

            def helper_method(self):
                """This method should not conflict with the linked agent."""
                return "method called"

        # Should still work properly
        assert "helper" in MainAgent.__linked_agents__

        # Create instance to verify no conflicts
        helper_instance = HelperAgent(model="_mock::test-model", api_key="test-key")
        main = MainAgent(model="_mock::test-model", api_key="test-key", helper=helper_instance)

        # Both should be accessible
        assert main.helper == helper_instance
        assert main.helper_method() == "method called"
