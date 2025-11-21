import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, Link, tool, spec, State


class TestBasicAgentLinking:
    """Test basic agent linking functionality."""

    @pytest.mark.skip(reason="Link[] type detection not yet implemented in metaclass")
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

    @pytest.mark.skip(reason="Link[] type detection not yet implemented in metaclass")
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

    @pytest.mark.skip(reason="Link[] type detection not yet implemented in metaclass")
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

    @pytest.mark.skip(reason="Link[] type detection not yet implemented in metaclass")
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

    @pytest.mark.skip(reason="Link[] type detection not yet implemented in metaclass")
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
