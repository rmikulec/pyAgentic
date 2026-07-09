import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, AgentExtension, tool, spec, State
from pyagentic._base._exceptions import InstructionsNotDeclared, SystemMessageNotDeclared

_PROVIDER_KWARGS = {"model": "openai::validation", "api_key": "validation"}


class TestBasicInheritance:
    """Test basic agent inheritance functionality."""

    def test_inherit_tools_from_parent(self):
        """Test that child agents inherit tools from parent agents."""

        class ParentAgent(BaseAgent):
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

    def test_inherit_state_from_parent(self):
        """Test that child agents inherit state from parent agents."""

        class ParentStateModel(BaseModel):
            value: str = "parent"

        class ChildStateModel(BaseModel):
            value: str = "child"

        class ParentAgent(BaseAgent):
            __system_message__ = "I am a parent agent"

            parent_state: State[ParentStateModel] = spec.State(
                default_factory=lambda: ParentStateModel()
            )

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child agent"

            child_state: State[ChildStateModel] = spec.State(
                default_factory=lambda: ChildStateModel()
            )

        # Child should have both state fields
        assert "parent_state" in ChildAgent.__state_defs__
        assert "child_state" in ChildAgent.__state_defs__
        assert len(ChildAgent.__state_defs__) == 2

    def test_override_parent_tool(self):
        """Test that child can override parent tools."""

        class ParentAgent(BaseAgent):
            __system_message__ = "I am a parent"

            @tool("Process data")
            def process(self, data: str) -> str:
                return f"parent: {data}"

        class ChildAgent(ParentAgent):
            __system_message__ = "I am a child"

            @tool("Process data differently")
            def process(self, data: str) -> str:
                return f"child: {data}"

        # Child should override the tool
        assert "process" in ChildAgent.__tool_defs__
        assert ChildAgent.__tool_defs__["process"].description == "Process data differently"


class TestAgentExtension:
    """Test AgentExtension (mixin) functionality."""

    def test_agent_extension_basic(self):
        """Test basic AgentExtension functionality."""

        class LoggingExtension(AgentExtension):
            logs: State[list] = spec.State(default_factory=list)

            @tool("Log a message")
            def log(self, message: str) -> str:
                self.state.logs.append(message)
                return f"Logged: {message}"

        class MyAgent(BaseAgent, LoggingExtension):
            __system_message__ = "I am an agent with logging"

        # Should have the logging tool
        assert "log" in MyAgent.__tool_defs__
        # Should have the logs state
        assert "logs" in MyAgent.__state_defs__

    def test_multiple_extensions(self):
        """Test using multiple AgentExtensions."""

        class LoggingExtension(AgentExtension):
            @tool("Log a message")
            def log(self, message: str) -> str:
                return f"Logged: {message}"

        class CachingExtension(AgentExtension):
            @tool("Cache data")
            def cache(self, key: str, value: str) -> str:
                return f"Cached {key}: {value}"

        class MyAgent(BaseAgent, LoggingExtension, CachingExtension):
            __system_message__ = "I am an agent with extensions"

        # Should have tools from both extensions
        assert "log" in MyAgent.__tool_defs__
        assert "cache" in MyAgent.__tool_defs__


class TestInstructionInheritance:
    """Test that __instructions__ are inherited and composable via {{ super }}."""

    def test_child_inherits_parent_instructions(self):
        """Test that a child without __instructions__ uses the parent's."""

        class ParentAgent(BaseAgent):
            __instructions__ = "I am a parent agent"

        class ChildAgent(ParentAgent):
            pass

        assert ChildAgent.__instructions__ == "I am a parent agent"
        child = ChildAgent(**_PROVIDER_KWARGS)
        assert child.state.system_message == "I am a parent agent"

    def test_no_instructions_anywhere_raises(self):
        """Test that an agent hierarchy with no instructions at all still raises."""

        with pytest.raises(InstructionsNotDeclared):

            class NoInstructionsAgent(BaseAgent):
                pass

    def test_override_injects_parent_via_super(self):
        """Test that an overriding child can embed the parent's instructions."""

        class ParentAgent(BaseAgent):
            __instructions__ = "I am a parent agent."

        class ChildAgent(ParentAgent):
            __instructions__ = "{{ super }} I also review code."

        child = ChildAgent(**_PROVIDER_KWARGS)
        assert child.state.system_message == "I am a parent agent. I also review code."

    def test_override_without_super_replaces(self):
        """Test that overriding without {{ super }} fully replaces the parent's."""

        class ParentAgent(BaseAgent):
            __instructions__ = "I am a parent agent."

        class ChildAgent(ParentAgent):
            __instructions__ = "I am my own agent."

        child = ChildAgent(**_PROVIDER_KWARGS)
        assert child.state.system_message == "I am my own agent."

    def test_super_renders_parent_with_state(self):
        """Test that state fields interpolate inside the injected parent template."""

        class ParentAgent(BaseAgent):
            __instructions__ = "I help the {{ team }} team."

            team: State[str] = spec.State(default="core")

        class ChildAgent(ParentAgent):
            __instructions__ = "{{ super }} I specialize in {{ language }}."

            language: State[str] = spec.State(default="python")

        child = ChildAgent(team="data", **_PROVIDER_KWARGS)
        assert child.state.system_message == "I help the data team. I specialize in python."

    def test_super_chains_through_grandparent(self):
        """Test that {{ super }} folds through multiple overriding levels."""

        class GrandparentAgent(BaseAgent):
            __instructions__ = "Level 1."

        class ParentAgent(GrandparentAgent):
            __instructions__ = "{{ super }} Level 2."

        class ChildAgent(ParentAgent):
            __instructions__ = "{{ super }} Level 3."

        child = ChildAgent(**_PROVIDER_KWARGS)
        assert child.state.system_message == "Level 1. Level 2. Level 3."

    def test_super_chain_skips_non_declaring_level(self):
        """Test that a non-declaring middle class passes its parent's chain through."""

        class GrandparentAgent(BaseAgent):
            __instructions__ = "Level 1."

        class ParentAgent(GrandparentAgent):
            pass

        class ChildAgent(ParentAgent):
            __instructions__ = "{{ super }} Level 3."

        child = ChildAgent(**_PROVIDER_KWARGS)
        assert child.state.system_message == "Level 1. Level 3."

    def test_super_empty_without_parent(self):
        """Test that {{ super }} renders empty when there is nothing to inherit."""

        class RootAgent(BaseAgent):
            __instructions__ = "{{ super }}Root."

        agent = RootAgent(**_PROVIDER_KWARGS)
        assert agent.state.system_message == "Root."

    def test_fork_preserves_inherited_instructions(self):
        """Test that forked agents render the same composed system message."""

        class ParentAgent(BaseAgent):
            __instructions__ = "I am a parent agent."

        class ChildAgent(ParentAgent):
            __instructions__ = "{{ super }} I also review code."

        child = ChildAgent(**_PROVIDER_KWARGS)
        fork = child.fork()
        assert fork.state.system_message == child.state.system_message

    def test_deprecated_system_message_inherits(self):
        """Test that the deprecated __system_message__ spelling still inherits."""

        class ParentAgent(BaseAgent):
            __system_message__ = "I am a parent agent"

        class ChildAgent(ParentAgent):
            pass

        assert ChildAgent.__instructions__ == "I am a parent agent"


class TestComplexInheritance:
    """Test complex inheritance scenarios."""

    def test_deep_inheritance_chain(self):
        """Test inheritance through multiple levels."""

        class Level1Agent(BaseAgent):
            __system_message__ = "Level 1"

            @tool("Level 1 tool")
            def level1_tool(self) -> str:
                return "level1"

        class Level2Agent(Level1Agent):
            __system_message__ = "Level 2"

            @tool("Level 2 tool")
            def level2_tool(self) -> str:
                return "level2"

        class Level3Agent(Level2Agent):
            __system_message__ = "Level 3"

            @tool("Level 3 tool")
            def level3_tool(self) -> str:
                return "level3"

        # Level 3 should have all tools
        assert "level1_tool" in Level3Agent.__tool_defs__
        assert "level2_tool" in Level3Agent.__tool_defs__
        assert "level3_tool" in Level3Agent.__tool_defs__
        assert len(Level3Agent.__tool_defs__) == 3
