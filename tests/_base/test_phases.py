import pytest
import asyncio
from pydantic import BaseModel

from pyagentic import BaseAgent, tool, spec, State
from pyagentic.models.llm import Message


class TaskModel(BaseModel):
    """Model for task state"""
    name: str = ""
    completed: bool = False


def test_agent_phases_declaration():
    """Test that phases can be declared on an agent"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent with phases"

        task: State[TaskModel] = spec.State(default_factory=lambda: TaskModel())

        phases = [
            ("init", "working", lambda self: bool(self.task.name)),
            ("working", "done", lambda self: self.task.completed),
        ]

    assert PhaseAgent.phases is not None
    assert len(PhaseAgent.phases) == 2
    assert PhaseAgent.phases[0] == ("init", "working", PhaseAgent.phases[0][2])


def test_agent_phases_initialization():
    """Test that phase machine is initialized when phases are defined"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent with phases"

        task: State[TaskModel] = spec.State(default_factory=lambda: TaskModel())

        phases = [
            ("init", "working", lambda self: bool(self.task.name)),
            ("working", "done", lambda self: self.task.completed),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Phase machine should be initialized
    assert agent.state._machine is not None
    # Initial state should be the first source state
    assert agent.state.phase == "init"


def test_agent_no_phases():
    """Test that agents without phases work normally"""

    class NoPhaseAgent(BaseAgent):
        __system_message__ = "Agent without phases"

        @tool("Test tool")
        def test_tool(self) -> str:
            return "test"

    agent = NoPhaseAgent(model="_mock::test-model", api_key="test")

    # Should not have a phase machine
    assert agent.state._machine is None
    assert agent.state.phase is None


def test_phase_property_access():
    """Test that phase can be accessed via state.phase property"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent with phases"

        count: State[int] = spec.State(default=0)

        phases = [
            ("start", "middle", lambda self: self.count > 0),
            ("middle", "end", lambda self: self.count > 5),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Check initial phase
    assert agent.state.phase == "start"

    # Update state to trigger transition
    agent.state.count = 3
    agent.state._update_state_machine(phases=agent.phases)

    # Phase should have transitioned
    assert agent.state.phase == "middle"


def test_phase_transitions_sequential():
    """Test that phases transition correctly based on state"""

    class MultiPhaseAgent(BaseAgent):
        __system_message__ = "Multi-phase agent"

        step: State[int] = spec.State(default=0)

        phases = [
            ("phase1", "phase2", lambda self: self.step == 1),
            ("phase2", "phase3", lambda self: self.step == 2),
            ("phase3", "phase4", lambda self: self.step == 3),
        ]

    agent = MultiPhaseAgent(model="_mock::test-model", api_key="test")

    assert agent.state.phase == "phase1"

    # Transition to phase2
    agent.state.step = 1
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "phase2"

    # Transition to phase3
    agent.state.step = 2
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "phase3"

    # Transition to phase4
    agent.state.step = 3
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "phase4"


def test_phase_in_system_message():
    """Test that phase variable is available in system message template"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Current phase: {{ phase }}"

        step: State[int] = spec.State(default=0)

        phases = [
            ("planning", "executing", lambda self: self.step > 0),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Check system message contains current phase
    system_message = agent.state.system_message
    assert "Current phase: planning" in system_message

    # Transition to next phase
    agent.state.step = 1
    agent.state._update_state_machine(phases=agent.phases)

    # Check system message updates with new phase
    system_message = agent.state.system_message
    assert "Current phase: executing" in system_message


def test_phase_in_system_message_with_conditionals():
    """Test that phase can be used in conditional blocks in system message"""

    class PhaseAgent(BaseAgent):
        __system_message__ = """
        {% if phase == "planning" %}
        You are planning.
        {% elif phase == "executing" %}
        You are executing.
        {% endif %}
        """

        step: State[int] = spec.State(default=0)

        phases = [
            ("planning", "executing", lambda self: self.step > 0),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Check system message for planning phase
    system_message = agent.state.system_message
    assert "You are planning." in system_message
    assert "You are executing." not in system_message

    # Transition to executing phase
    agent.state.step = 1
    agent.state._update_state_machine(phases=agent.phases)

    # Check system message for executing phase
    system_message = agent.state.system_message
    assert "You are executing." in system_message
    assert "You are planning." not in system_message


def test_phase_in_input_template():
    """Test that phase variable is available in input template"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"
        __input_template__ = "Phase: {{ phase }} | Input: {{ user_message }}"

        step: State[int] = spec.State(default=0)

        phases = [
            ("start", "end", lambda self: self.step > 0),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Add a user message
    agent.state.add_user_message("Hello")

    # Check that phase is in the message
    last_message = agent.state._messages[-1]
    assert "Phase: start" in last_message.content
    assert "Input:" in last_message.content


def test_phase_aware_tools_filtering():
    """Test that tools are filtered based on current phase"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent with phase-aware tools"

        step: State[int] = spec.State(default=0)

        phases = [
            ("init", "work", lambda self: self.step == 3),
            ("work", "done", lambda self: self.step == 10),
        ]

        @tool("Tool for init phase", phases=["init"])
        def init_tool(self) -> str:
            return "init"

        @tool("Tool for work phase", phases=["work"])
        def work_tool(self) -> str:
            return "work"

        @tool("Tool for multiple phases", phases=["work", "done"])
        def multi_phase_tool(self) -> str:
            return "multi"

        @tool("Tool for all phases")
        def universal_tool(self) -> str:
            return "universal"

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # In init phase
    assert agent.state.phase == "init"
    tool_defs = asyncio.run(agent._get_tool_defs())
    tool_names = [tool.name for tool in tool_defs]

    # Should have init_tool and universal_tool
    assert "init_tool" in tool_names
    assert "universal_tool" in tool_names
    # Should not have work_tool or multi_phase_tool
    assert "work_tool" not in tool_names
    assert "multi_phase_tool" not in tool_names

    # Transition to work phase
    agent.state.step = 3
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "work"

    tool_defs = asyncio.run(agent._get_tool_defs())
    tool_names = [tool.name for tool in tool_defs]

    # Should have work_tool, multi_phase_tool, and universal_tool
    assert "work_tool" in tool_names
    assert "multi_phase_tool" in tool_names
    assert "universal_tool" in tool_names
    # Should not have init_tool
    assert "init_tool" not in tool_names

    # Transition to done phase
    agent.state.step = 10
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "done"

    tool_defs = asyncio.run(agent._get_tool_defs())
    tool_names = [tool.name for tool in tool_defs]

    # Should have multi_phase_tool and universal_tool
    assert "multi_phase_tool" in tool_names
    assert "universal_tool" in tool_names
    # Should not have init_tool or work_tool
    assert "init_tool" not in tool_names
    assert "work_tool" not in tool_names


def test_tool_phases_attribute():
    """Test that tool definitions store phases correctly"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        @tool("Tool with phases", phases=["phase1", "phase2"])
        def phase_tool(self) -> str:
            return "test"

        @tool("Tool without phases")
        def no_phase_tool(self) -> str:
            return "test"

    # Check tool definitions
    assert PhaseAgent.__tool_defs__["phase_tool"].phases == ["phase1", "phase2"]
    assert PhaseAgent.__tool_defs__["no_phase_tool"].phases == []


def test_phase_conditions_with_complex_logic():
    """Test that phase conditions can use complex logic"""

    class ComplexPhaseAgent(BaseAgent):
        __system_message__ = "Complex phase agent"

        tasks_completed: State[int] = spec.State(default=0)
        quality_check: State[bool] = spec.State(default=False)
        reviewed: State[bool] = spec.State(default=False)

        phases = [
            (
                "working",
                "review",
                lambda self: self.tasks_completed >= 3 and self.quality_check and not self.reviewed,
            ),
            (
                "review",
                "complete",
                lambda self: self.reviewed and self.quality_check,
            ),
        ]

    agent = ComplexPhaseAgent(model="_mock::test-model", api_key="test")

    # Initial phase
    assert agent.state.phase == "working"

    # Set tasks_completed but not quality_check
    agent.state.tasks_completed = 5
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "working"  # Should not transition

    # Now set quality_check
    agent.state.quality_check = True
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "review"  # Should transition

    # Set reviewed to complete
    agent.state.reviewed = True
    agent.state._update_state_machine(phases=agent.phases)
    assert agent.state.phase == "complete"


def test_phase_transitions_only_first_match():
    """Test that only the first matching transition is executed"""

    class MultiTransitionAgent(BaseAgent):
        __system_message__ = "Agent with multiple possible transitions"

        value: State[int] = spec.State(default=0)

        phases = [
            ("start", "option1", lambda self: self.value == 10),
            ("start", "option2", lambda self: self.value == 5),
            ("start", "option3", lambda self: self.value == 1),
        ]

    agent = MultiTransitionAgent(model="_mock::test-model", api_key="test")

    # Set value to 10 (first condition is true)
    agent.state.value = 10
    agent.state._update_state_machine(phases=agent.phases)

    # Should transition to option1 (first matching condition)
    assert agent.state.phase == "option1"


def test_phase_state_extraction():
    """Test that states are correctly extracted from phase tuples"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        step: State[int] = spec.State(default=0)

        phases = [
            ("init", "working", lambda self: self.step > 0),
            ("working", "review", lambda self: self.step > 5),
            ("review", "done", lambda self: self.step > 10),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Check that all states are in the machine
    machine_states = agent.state._machine.states
    # States are stored as a dict mapping name -> State object
    state_names = list(machine_states.keys())

    assert "init" in state_names
    assert "working" in state_names
    assert "review" in state_names
    assert "done" in state_names


def test_phase_no_duplicate_states():
    """Test that duplicate state names in phases are handled correctly"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        step: State[int] = spec.State(default=0)

        phases = [
            ("start", "middle", lambda self: self.step > 0),
            ("middle", "end", lambda self: self.step > 5),
            ("start", "end", lambda self: self.step > 10),  # Reuses start and end
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Check that states are not duplicated
    machine_states = agent.state._machine.states
    state_names = list(machine_states.keys())

    # Each state should appear exactly once
    assert len([s for s in state_names if s == "start"]) == 1
    assert len([s for s in state_names if s == "middle"]) == 1
    assert len([s for s in state_names if s == "end"]) == 1


def test_phases_integration_with_state_updates():
    """Test that phases work correctly with state updates"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent with state and phases"

        counter: State[int] = spec.State(default=0)

        phases = [
            ("counting", "done", lambda self: self.counter >= 5),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    assert agent.state.phase == "counting"

    # Update counter directly
    agent.state.counter = 5

    # Manually trigger phase update
    agent.state._update_state_machine(phases=agent.phases)

    # Phase should have transitioned
    assert agent.state.phase == "done"


def test_phases_with_no_conditions_met():
    """Test that phase remains unchanged when no conditions are met"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        value: State[int] = spec.State(default=0)

        phases = [
            ("start", "end", lambda self: self.value > 100),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    assert agent.state.phase == "start"

    # Set value but don't meet condition
    agent.state.value = 50
    agent.state._update_state_machine(phases=agent.phases)

    # Should still be in start phase
    assert agent.state.phase == "start"


def test_phase_condition_with_state_methods():
    """Test that phase conditions can access state fields directly"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        count: State[int] = spec.State(default=0)
        threshold: State[int] = spec.State(default=3)

        phases = [
            ("waiting", "ready", lambda self: self.count >= self.threshold),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    assert agent.state.phase == "waiting"

    agent.state.count = 3
    agent.state._update_state_machine(phases=agent.phases)

    assert agent.state.phase == "ready"


def test_empty_phases_list():
    """Test that an empty phases list doesn't break the agent"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        phases = []

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    # Should not have a phase machine
    assert agent.state._machine is None
    assert agent.state.phase is None


def test_phases_with_single_transition():
    """Test that a single phase transition works correctly"""

    class PhaseAgent(BaseAgent):
        __system_message__ = "Agent"

        ready: State[bool] = spec.State(default=False)

        phases = [
            ("init", "done", lambda self: self.ready),
        ]

    agent = PhaseAgent(model="_mock::test-model", api_key="test")

    assert agent.state.phase == "init"

    agent.state.ready = True
    agent.state._update_state_machine(phases=agent.phases)

    assert agent.state.phase == "done"
