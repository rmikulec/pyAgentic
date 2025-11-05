"""
D2 Diagram Generator for BaseAgent visualization.

This module provides utilities to generate d2 diagrams that visualize
the structure of a BaseAgent, including its states, tools, and linked agents.
"""

from typing import Type, Optional, Any
from pyagentic._base._agent._agent import BaseAgent
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._state import _StateDefinition


class AgentDiagramGenerator:
    """
    Generates d2 diagrams from BaseAgent classes to visualize agent structure.

    This class inspects a BaseAgent and creates a d2 diagram showing:
    - Agent metadata (system message, description)
    - State fields and their types
    - Tool definitions with descriptions
    - Linked agents and their relationships

    Example:
        ```python
        class MyAgent(BaseAgent):
            __system_message__ = "I am a helpful agent"
            user_name: State[str]

            @tool("Get user info")
            def get_user(self) -> str:
                return self.user_name

        generator = AgentDiagramGenerator(MyAgent)
        d2_code = generator.generate()

        # Save to file
        with open("my_agent.d2", "w") as f:
            f.write(d2_code)
        ```
    """

    def __init__(self, agent_class: Type[BaseAgent], visited: Optional[set] = None):
        """
        Initialize the diagram generator with an agent class.

        Args:
            agent_class: A BaseAgent subclass to visualize
            visited: Set of already visited agent classes (for recursion tracking)
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{agent_class} must be a subclass of BaseAgent")

        self.agent_class = agent_class
        self.agent_name = agent_class.__name__
        self.visited = visited if visited is not None else set()

    def _escape_d2_string(self, s: str) -> str:
        """
        Escape special characters in strings for d2 format.

        Args:
            s: String to escape

        Returns:
            Escaped string safe for d2
        """
        if not s:
            return ""
        # Escape quotes and newlines
        s = s.replace('"', '\\"')
        s = s.replace('\n', '\\n')
        return s

    def _truncate_text(self, text: str, max_length: int = 80) -> str:
        """
        Truncate text to a maximum length with ellipsis.

        Args:
            text: Text to truncate
            max_length: Maximum length before truncation

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

    def _generate_header(self) -> str:
        """Generate the d2 diagram header with configuration."""
        return """vars: {
  d2-config: {
    layout-engine: tala
  }
}
direction: down

"""

    def _generate_agent_info(self) -> str:
        """Generate the agent information section."""
        system_msg = getattr(self.agent_class, "__system_message__", "No system message defined")
        description = getattr(self.agent_class, "__description__", "")

        system_msg_escaped = self._escape_d2_string(self._truncate_text(system_msg, 100))

        info_lines = [
            f"{self.agent_name}: {{",
            f'  label: "{self.agent_name}"',
            "  direction: down",
            "  style.fill: \"#eff6ff\"",
            "  style.stroke: \"#2563eb\"",
            "  style.stroke-width: 3",
            "  style.font-color: \"#1e40af\"",
            "  style.double-border: true",
            "",
            "  Info: {",
            '    label: "Agent Information"',
            "    style.fill: \"white\"",
            "    style.stroke: \"#60a5fa\"",
            "    style.stroke-width: 2",
            "",
            "    SystemMessage: {",
            f'      label: "{system_msg_escaped}"',
            "      style.fill: \"#dbeafe\"",
            "      style.stroke: \"#2563eb\"",
            "      style.stroke-width: 2",
            "    }",
        ]

        if description:
            desc_escaped = self._escape_d2_string(self._truncate_text(description, 100))
            info_lines.extend([
                "    Description: {",
                f'      label: "{desc_escaped}"',
                "      style.fill: \"#dbeafe\"",
                "      style.stroke: \"#2563eb\"",
                "      style.stroke-width: 2",
                "    }",
            ])

        info_lines.extend([
            "  }",
            "",
        ])

        return "\n".join(info_lines)

    def _generate_state_section(self) -> str:
        """Generate the state fields section."""
        state_defs = self.agent_class.__state_defs__

        if not state_defs:
            return ""

        lines = [
            "  State: {",
            '    label: "State Fields"',
            "    style.fill: \"white\"",
            "    style.stroke: \"#f59e0b\"",
            "    style.stroke-width: 2",
            "",
        ]

        for name, state_def in state_defs.items():
            if isinstance(state_def, _StateDefinition):
                type_name = state_def.model.__name__ if hasattr(state_def.model, '__name__') else str(state_def.model)
                label = f"{name}: {type_name}"
            else:
                label = f"{name}"

            label_escaped = self._escape_d2_string(label)

            lines.extend([
                f"    {name}: {{",
                f'      label: "{label_escaped}"',
                "      style.fill: \"#fef3c7\"",
                "      style.stroke: \"#f59e0b\"",
                "      style.stroke-width: 2",
                "    }",
            ])

        lines.extend([
            "  }",
            "",
        ])

        return "\n".join(lines)

    def _generate_tools_section(self) -> str:
        """Generate the tools section."""
        tool_defs = self.agent_class.__tool_defs__

        if not tool_defs:
            return ""

        lines = [
            "  Tools: {",
            '    label: "Available Tools"',
            "    style.fill: \"white\"",
            "    style.stroke: \"#ea580c\"",
            "    style.stroke-width: 2",
            "",
        ]

        for name, tool_def in tool_defs.items():
            desc = self._escape_d2_string(self._truncate_text(tool_def.description, 60))

            # Build parameter info
            param_info = []
            for param_name, (param_type, param_default) in tool_def.parameters.items():
                type_name = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)
                param_info.append(f"{param_name}: {type_name}")

            params_str = ", ".join(param_info) if param_info else ""
            label = f"{name}({params_str})\\n{desc}"
            label_escaped = self._escape_d2_string(label)

            lines.extend([
                f"    {name}: {{",
                f'      label: "{label_escaped}"',
                "      style.fill: \"#fed7aa\"",
                "      style.stroke: \"#ea580c\"",
                "      style.stroke-width: 2",
                "    }",
            ])

        lines.extend([
            "  }",
            "",
        ])

        return "\n".join(lines)

    def _generate_linked_agents_recursive(self) -> str:
        """Generate full agent structures for all linked agents recursively."""
        linked_agents = self.agent_class.__linked_agents__

        if not linked_agents:
            return ""

        lines = []

        for name, agent_class in linked_agents.items():
            # Skip if we've already processed this agent class
            if agent_class in self.visited:
                continue

            # Mark this agent as visited
            self.visited.add(agent_class)

            # Create a new generator for the linked agent with the shared visited set
            linked_generator = AgentDiagramGenerator(agent_class, self.visited)

            # Generate the agent info (without header)
            lines.append(linked_generator._generate_agent_info())
            lines.append(linked_generator._generate_state_section())
            lines.append(linked_generator._generate_tools_section())
            lines.append("}")  # Close the linked agent container
            lines.append("")

            # Recursively generate any linked agents of this agent
            lines.append(linked_generator._generate_linked_agents_recursive())

        return "\n".join(lines)

    def _generate_relationships(self, relationships_visited: Optional[set] = None) -> str:
        """Generate relationships between components recursively."""
        if relationships_visited is None:
            relationships_visited = set()

        # Skip if we've already generated relationships for this agent
        if self.agent_class in relationships_visited:
            return ""

        relationships_visited.add(self.agent_class)
        lines = []

        # Relationship: Agent uses State
        if self.agent_class.__state_defs__:
            lines.append(f"{self.agent_name}.Info -> {self.agent_name}.State: {{")
            lines.append('  label: "manages"')
            lines.append("  style.stroke: \"#f59e0b\"")
            lines.append("  style.stroke-width: 2")
            lines.append("  style.stroke-dash: 5")
            lines.append("}")
            lines.append("")

        # Relationship: Agent uses Tools
        if self.agent_class.__tool_defs__:
            lines.append(f"{self.agent_name}.Info -> {self.agent_name}.Tools: {{")
            lines.append('  label: "uses"')
            lines.append("  style.stroke: \"#ea580c\"")
            lines.append("  style.stroke-width: 2")
            lines.append("  style.stroke-dash: 5")
            lines.append("}")
            lines.append("")

        # Relationship: Agent links to other agents (direct connections)
        if self.agent_class.__linked_agents__:
            for name, agent_class in self.agent_class.__linked_agents__.items():
                linked_agent_name = agent_class.__name__
                lines.append(f"{self.agent_name}.Info -> {linked_agent_name}.Info: {{")
                lines.append(f'  label: "links to {name}"')
                lines.append("  style.stroke: \"#10b981\"")
                lines.append("  style.stroke-width: 2")
                lines.append("  style.stroke-dash: 5")
                lines.append("}")
                lines.append("")

        # Tools can access State
        if self.agent_class.__tool_defs__ and self.agent_class.__state_defs__:
            lines.append(f"{self.agent_name}.Tools -> {self.agent_name}.State: {{")
            lines.append('  label: "accesses"')
            lines.append("  style.stroke: \"#f59e0b\"")
            lines.append("  style.stroke-width: 1")
            lines.append("  style.stroke-dash: 3")
            lines.append("}")
            lines.append("")

        # Recursively generate relationships for linked agents
        if self.agent_class.__linked_agents__:
            for name, agent_class in self.agent_class.__linked_agents__.items():
                linked_generator = AgentDiagramGenerator(agent_class, self.visited)
                lines.append(linked_generator._generate_relationships(relationships_visited))

        return "\n".join(lines)

    def generate(self) -> str:
        """
        Generate the complete d2 diagram code.

        Returns:
            Complete d2 diagram as a string
        """
        # Mark this agent as visited
        self.visited.add(self.agent_class)

        sections = [
            self._generate_header(),
            self._generate_agent_info(),
            self._generate_state_section(),
            self._generate_tools_section(),
            "}",  # Close the agent container
            "",
            self._generate_linked_agents_recursive(),
            "",
            self._generate_relationships(),
        ]

        return "\n".join(sections)

    def save(self, filename: str) -> None:
        """
        Generate and save the diagram to a file.

        Args:
            filename: Path to the output .d2 file
        """
        d2_code = self.generate()
        with open(filename, 'w') as f:
            f.write(d2_code)
        print(f"Diagram saved to {filename}")
