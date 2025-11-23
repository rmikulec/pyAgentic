import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, Link, tool, spec, State


class TestBasicAgentLinking:
    """Test basic agent linking functionality."""

    def test_linked_agent_declaration(self):
        """Test that linked agents are properly declared."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper agent"
            __description__ = "Provides assistance with tasks"

        class MainAgent(BaseAgent):
            __system_message__ = "I am a main agent with a helper"

            helper: Link[HelperAgent]

        # Should have the linked agent in class attributes
        assert "helper" in MainAgent.__linked_agents__
        # Check it's the correct agent type
        linked_def = MainAgent.__linked_agents__["helper"]
        assert hasattr(linked_def, "agent")
        assert linked_def.agent == HelperAgent

    def test_multiple_linked_agents(self):
        """Test declaring multiple linked agents."""

        class DatabaseAgent(BaseAgent):
            __system_message__ = "I handle database operations"
            __description__ = "Database interaction specialist"

        class CacheAgent(BaseAgent):
            __system_message__ = "I handle caching operations"
            __description__ = "Cache management specialist"

        class MainAgent(BaseAgent):
            __system_message__ = "I coordinate multiple agents"

            database: Link[DatabaseAgent]
            cache: Link[CacheAgent]

        # Should have both linked agents
        assert len(MainAgent.__linked_agents__) == 2
        assert "database" in MainAgent.__linked_agents__
        assert "cache" in MainAgent.__linked_agents__

    def test_linked_agent_with_spec(self):
        """Test linked agent with spec.AgentLink configuration."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(BaseAgent):
            __system_message__ = "I have a conditional helper"

            helper: Link[HelperAgent] = spec.AgentLink(condition=lambda self: True)

        assert "helper" in MainAgent.__linked_agents__


class TestAgentLinkingInheritance:
    """Test inheritance behavior with linked agents."""

    def test_inherit_linked_agents(self):
        """Test that linked agents are inherited from parent classes."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class ParentAgent(BaseAgent):
            __system_message__ = "I am a parent with helper"

            helper: Link[HelperAgent]

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with inherited helper"

        # Child should inherit linked agent
        assert "helper" in ChildAgent.__linked_agents__

    def test_add_linked_agents_to_inheritance(self):
        """Test adding additional linked agents in child classes."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class WorkerAgent(BaseAgent):
            __system_message__ = "I am a worker"
            __description__ = "Does work"

        class ParentAgent(BaseAgent):
            __system_message__ = "I am a parent with helper"

            helper: Link[HelperAgent]

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child with helper and worker"

            worker: Link[WorkerAgent]

        # Child should have both linked agents
        assert len(ChildAgent.__linked_agents__) >= 2
        assert "helper" in ChildAgent.__linked_agents__
        assert "worker" in ChildAgent.__linked_agents__


class TestAgentLinkingConditions:
    """Test conditional linked agent functionality."""

    @pytest.mark.asyncio
    async def test_linked_agent_condition_true(self):
        """Test that linked agents with condition=True are available."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(BaseAgent):
            __system_message__ = "I have a conditional helper"

            enabled: State[bool] = spec.State(default=True)

            helper: Link[HelperAgent] = spec.AgentLink(condition=lambda self: self.enabled)

        agent = MainAgent(
            model="_mock::test-model",
            api_key="test",
            helper=HelperAgent(model="_mock::test-model", api_key="test")
        )

        # Helper should be available when enabled=True
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "helper" in tool_names

    @pytest.mark.asyncio
    async def test_linked_agent_condition_false(self):
        """Test that linked agents with condition=False are not available."""

        class HelperAgent(BaseAgent):
            __system_message__ = "I am a helper"
            __description__ = "Provides help"

        class MainAgent(BaseAgent):
            __system_message__ = "I have a conditional helper"

            enabled: State[bool] = spec.State(default=False)

            helper: Link[HelperAgent] = spec.AgentLink(condition=lambda self: self.enabled)

        agent = MainAgent(
            model="_mock::test-model",
            api_key="test",
            helper=HelperAgent(model="_mock::test-model", api_key="test")
        )

        # Helper should not be available when enabled=False
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "helper" not in tool_names

    @pytest.mark.asyncio
    async def test_linked_agent_condition_state_access(self):
        """Test that linked agent conditions have access to agent state."""

        class ExpertAgent(BaseAgent):
            __system_message__ = "I am an expert"
            __description__ = "Expert consultant"

        class MainAgent(BaseAgent):
            __system_message__ = "I have conditional experts"

            needs_expert: State[bool] = spec.State(default=False)
            complexity_level: State[int] = spec.State(default=1)

            expert: Link[ExpertAgent] = spec.AgentLink(
                condition=lambda self: self.needs_expert
            )
            advanced_expert: Link[ExpertAgent] = spec.AgentLink(
                condition=lambda self: self.complexity_level > 7
            )

        agent = MainAgent(
            model="_mock::test-model",
            api_key="test",
            expert=ExpertAgent(model="_mock::test-model", api_key="test"),
            advanced_expert=ExpertAgent(model="_mock::test-model", api_key="test")
        )

        # Initially, neither expert should be available
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "expert" not in tool_names
        assert "advanced_expert" not in tool_names

        # Enable needs_expert
        agent.state.needs_expert = True
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "expert" in tool_names
        assert "advanced_expert" not in tool_names

        # Increase complexity level
        agent.state.complexity_level = 10
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "expert" in tool_names
        assert "advanced_expert" in tool_names

    @pytest.mark.asyncio
    async def test_linked_agent_condition_complex_logic(self):
        """Test that linked agent conditions can use complex logic."""

        class SpecialistAgent(BaseAgent):
            __system_message__ = "I am a specialist"
            __description__ = "Specialist for complex tasks"

        class MainAgent(BaseAgent):
            __system_message__ = "I have a specialist"

            tasks_completed: State[int] = spec.State(default=1)
            quality_check: State[bool] = spec.State(default=False)

            specialist: Link[SpecialistAgent] = spec.AgentLink(
                condition=lambda self: self.tasks_completed and self.tasks_completed >= 3 and self.quality_check
            )

        agent = MainAgent(
            model="_mock::test-model",
            api_key="test",
            specialist=SpecialistAgent(model="_mock::test-model", api_key="test")
        )

        # Neither condition met
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "specialist" not in tool_names

        # Only tasks_completed condition met
        agent.state.tasks_completed = 5
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "specialist" not in tool_names

        # Both conditions met
        agent.state.quality_check = True
        tool_defs = await agent._get_tool_defs()
        tool_names = [tool.name for tool in tool_defs]
        assert "specialist" in tool_names


class TestAgentLinkingPhases:
    """Test phase-based linked agent functionality."""

    def test_linked_agent_with_phases_declaration(self):
        """Test that linked agents with phases parameter are properly declared."""

        class PlannerAgent(BaseAgent):
            __system_message__ = "I create plans"
            __description__ = "Planning specialist"

        class MainAgent(BaseAgent):
            __system_message__ = "I manage projects"

            planner: Link[PlannerAgent] = spec.AgentLink(phases=["planning"])

        assert "planner" in MainAgent.__linked_agents__
        linked_def = MainAgent.__linked_agents__["planner"]
        assert linked_def.info.phases == ["planning"]

    def test_linked_agent_with_multiple_phases(self):
        """Test that linked agents can be assigned to multiple phases."""

        class ReviewerAgent(BaseAgent):
            __system_message__ = "I review work"
            __description__ = "Review specialist"

        class MainAgent(BaseAgent):
            __system_message__ = "I manage workflows"

            reviewer: Link[ReviewerAgent] = spec.AgentLink(phases=["review", "finalize"])

        linked_def = MainAgent.__linked_agents__["reviewer"]
        assert linked_def.info.phases == ["review", "finalize"]
