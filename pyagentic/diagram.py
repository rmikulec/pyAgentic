"""
Graphviz Diagram Generator for BaseAgent visualization.

This module provides utilities to generate Graphviz diagrams that visualize
the structure of a BaseAgent, including its states, tools, and linked agents.
"""

from typing import Type, Optional, Set, List
import re
import graphviz
from pyagentic._base._agent._agent import BaseAgent
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._state import _StateDefinition


class AgentDiagramGenerator:
    """
    Generates Graphviz diagrams from BaseAgent classes to visualize agent structure.

    This class inspects a BaseAgent and creates a Graphviz diagram showing:
    - Agent metadata (system message, description)
    - State fields and their types
    - Tool definitions with descriptions
    - Linked agents and their relationships (recursive)

    Example:
        ```python
        class MyAgent(BaseAgent):
            __system_message__ = "I am a helpful agent"
            user_name: State[str]

            @tool("Get user info")
            def get_user(self) -> str:
                return self.user_name

        generator = AgentDiagramGenerator(MyAgent)

        # Save to file
        generator.save("my_agent.png")

        # View directly
        generator.view()
        ```
    """

    def __init__(self, agent_class: Type[BaseAgent]):
        """
        Initialize the diagram generator with an agent class.

        Args:
            agent_class: A BaseAgent subclass to visualize
        """
        if not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{agent_class} must be a subclass of BaseAgent")

        self.agent_class = agent_class
        self.agent_name = agent_class.__name__
        self.visited: Set[Type[BaseAgent]] = set()
        self.graph = graphviz.Digraph(
            name=f"Agent_{self.agent_name}",
            comment=f"Agent Architecture: {self.agent_name}",
            format='png'
        )
        # Compact vertical layout
        self.graph.attr(
            rankdir='LR',  # Left to right for better horizontal use
            splines='spline',
            nodesep='0.4',
            ranksep='1.0',
            compound='true',
            bgcolor='white',
            pad='0.3',
            dpi='150'
        )
        self.graph.attr('node', fontname='Arial', fontsize='10')
        self.graph.attr('edge', fontname='Arial', fontsize='8')

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

    def _sanitize_label(self, text: str) -> str:
        """
        Sanitize text for use in Graphviz labels.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text safe for Graphviz
        """
        if not text:
            return ""
        # Replace newlines with literal \n for label formatting
        text = text.replace('\n', '\\n')
        # Escape quotes
        text = text.replace('"', '\\"')
        return text

    def _node_id(self, agent_name: str, component: str) -> str:
        """Generate a unique node ID."""
        return f"{agent_name}_{component}"

    def _extract_state_variables(self, template: str, state_defs: dict) -> List[str]:
        """
        Extract state variable names used in a Jinja2 template string.

        Args:
            template: Jinja2 template string that may contain {{ variable }} references
            state_defs: Dictionary of state definitions

        Returns:
            List of state variable names found in the template
        """
        if not template or not state_defs:
            return []

        # Pattern to match Jinja2 variable references
        # Matches: {{ variable }}, {{ self.variable }}, {% for x in variable %}, etc.
        patterns = [
            r'\{\{\s*(?:self\.)?([a-zA-Z_][a-zA-Z0-9_]*)',  # {{ variable }} or {{ self.variable }}
            r'\{%\s+\w+\s+\w+\s+in\s+(?:self\.)?([a-zA-Z_][a-zA-Z0-9_]*)',  # {% for x in variable %}
            r'\{%\s+if\s+(?:self\.)?([a-zA-Z_][a-zA-Z0-9_]*)',  # {% if variable %}
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, template))

        # Filter to only include actual state variables
        state_vars = [var for var in matches if var in state_defs]

        # Return unique variables in order of appearance
        seen = set()
        result = []
        for var in state_vars:
            if var not in seen:
                seen.add(var)
                result.append(var)

        return result

    def _add_agent_to_graph(self, agent_class: Type[BaseAgent]) -> None:
        """
        Add an agent and its components to the graph recursively.

        Args:
            agent_class: The agent class to add
        """
        # Skip if already visited
        if agent_class in self.visited:
            return

        self.visited.add(agent_class)
        agent_name = agent_class.__name__

        # Get metadata
        state_defs = agent_class.__state_defs__
        tool_defs = agent_class.__tool_defs__
        system_msg = getattr(agent_class, "__system_message__", "")
        input_template = getattr(agent_class, "__input_template__", "")
        description = getattr(agent_class, "__description__", "")

        # Extract state variables used in templates
        system_msg_vars = self._extract_state_variables(system_msg, state_defs) if system_msg else []
        input_template_vars = self._extract_state_variables(input_template, state_defs) if input_template else []

        # Create cluster for this agent
        cluster_label = agent_name
        if description:
            cluster_label += f"\\n{self._sanitize_label(self._truncate_text(description, 50))}"

        with self.graph.subgraph(name=f'cluster_{agent_name}') as cluster:
            cluster.attr(
                label=cluster_label,
                style='rounded,filled',
                fillcolor='#f8fafc',
                color='#475569',
                penwidth='2.5',
                fontsize='13',
                fontname='Arial Bold'
            )

            # System Message node (just label, no text)
            if system_msg and system_msg_vars:
                cluster.node(
                    self._node_id(agent_name, 'SystemMessage'),
                    label='system_message',
                    fillcolor='#ddd6fe',
                    color='#7c3aed',
                    penwidth='1.5',
                    shape='box',
                    style='rounded,filled',
                    fontsize='9'
                )

            # Input Template node (just label, no text)
            if input_template and input_template_vars:
                cluster.node(
                    self._node_id(agent_name, 'InputTemplate'),
                    label='input_template',
                    fillcolor='#fce7f3',
                    color='#db2777',
                    penwidth='1.5',
                    shape='box',
                    style='rounded,filled',
                    fontsize='9'
                )

            # State section - all states in one container
            if state_defs:
                with cluster.subgraph(name=f'cluster_{agent_name}_State') as state_cluster:
                    state_cluster.attr(
                        label='State',
                        style='rounded',
                        color='#d97706',
                        penwidth='1.5',
                        fontsize='11',
                        fontname='Arial'
                    )

                    for name, state_def in state_defs.items():
                        if isinstance(state_def, _StateDefinition):
                            type_name = state_def.model.__name__ if hasattr(state_def.model, '__name__') else str(state_def.model)
                            label = f"{name}: {type_name}"
                        else:
                            label = name

                        # Highlight if used in system_msg or input_template
                        is_used = name in system_msg_vars or name in input_template_vars
                        fillcolor = '#fde68a' if is_used else '#fef3c7'
                        penwidth = '2' if is_used else '1'

                        state_cluster.node(
                            self._node_id(agent_name, f'State_{name}'),
                            label=label,
                            fillcolor=fillcolor,
                            color='#d97706',
                            shape='cylinder',
                            style='filled',
                            penwidth=penwidth,
                            fontsize='9'
                        )

            # Tools section - all tools in one container
            if tool_defs:
                with cluster.subgraph(name=f'cluster_{agent_name}_Tools') as tools_cluster:
                    tools_cluster.attr(
                        label='Tools',
                        style='rounded',
                        color='#dc2626',
                        penwidth='1.5',
                        fontsize='11',
                        fontname='Arial'
                    )

                    for name, tool_def in tool_defs.items():
                        # Build parameter info (truncate if too long)
                        param_info = []
                        for param_name, (param_type, param_default) in tool_def.parameters.items():
                            type_name = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)
                            param_info.append(f"{param_name}: {type_name}")

                        # Truncate params if too many
                        if len(param_info) > 2:
                            params_display = ", ".join(param_info[:2]) + ", ..."
                        else:
                            params_display = ", ".join(param_info) if param_info else ""

                        label = f"{name}({params_display})"

                        tools_cluster.node(
                            self._node_id(agent_name, f'Tool_{name}'),
                            label=label,
                            fillcolor='#fecaca',
                            color='#dc2626',
                            shape='box',
                            style='rounded,filled',
                            penwidth='1',
                            fontsize='9'
                        )

        # Add edges: system_message -> state variables it uses
        if system_msg and system_msg_vars:
            for var in system_msg_vars:
                self.graph.edge(
                    self._node_id(agent_name, 'SystemMessage'),
                    self._node_id(agent_name, f'State_{var}'),
                    color='#7c3aed',
                    penwidth='1.5',
                    arrowsize='0.8'
                )

        # Add edges: input_template -> state variables it uses
        if input_template and input_template_vars:
            for var in input_template_vars:
                self.graph.edge(
                    self._node_id(agent_name, 'InputTemplate'),
                    self._node_id(agent_name, f'State_{var}'),
                    color='#db2777',
                    penwidth='1.5',
                    arrowsize='0.8'
                )

        # Recursively add linked agents
        linked_agents = agent_class.__linked_agents__
        if linked_agents:
            for link_name, linked_class in linked_agents.items():
                # Recursively add the linked agent
                self._add_agent_to_graph(linked_class)

                # Add edge from this agent cluster to linked agent cluster
                # Connect from first state/tool node to first state/tool node of linked agent
                from_node = None
                to_node = None

                # Try to find a node to connect from in this agent
                if state_defs:
                    from_node = self._node_id(agent_name, f'State_{list(state_defs.keys())[0]}')
                elif tool_defs:
                    from_node = self._node_id(agent_name, f'Tool_{list(tool_defs.keys())[0]}')
                elif system_msg and system_msg_vars:
                    from_node = self._node_id(agent_name, 'SystemMessage')
                elif input_template and input_template_vars:
                    from_node = self._node_id(agent_name, 'InputTemplate')

                # Try to find a node to connect to in linked agent
                linked_state_defs = linked_class.__state_defs__
                linked_tool_defs = linked_class.__tool_defs__

                if linked_state_defs:
                    to_node = self._node_id(linked_class.__name__, f'State_{list(linked_state_defs.keys())[0]}')
                elif linked_tool_defs:
                    to_node = self._node_id(linked_class.__name__, f'Tool_{list(linked_tool_defs.keys())[0]}')

                # Only add edge if we found both nodes
                if from_node and to_node:
                    self.graph.edge(
                        from_node,
                        to_node,
                        label=link_name,
                        color='#10b981',
                        penwidth='2',
                        fontsize='9',
                        arrowsize='1.0',
                        ltail=f'cluster_{agent_name}',
                        lhead=f'cluster_{linked_class.__name__}'
                    )

    def generate(self) -> graphviz.Digraph:
        """
        Generate the complete Graphviz diagram.

        Returns:
            Graphviz Digraph object
        """
        self._add_agent_to_graph(self.agent_class)
        return self.graph

    def save(self, filename: str, format: str = 'png') -> str:
        """
        Generate and save the diagram to a file.

        Args:
            filename: Path to the output file (without extension)
            format: Output format (png, pdf, svg, etc.)

        Returns:
            Path to the saved file
        """
        self.graph.format = format
        self.generate()
        output_path = self.graph.render(filename, cleanup=True)
        print(f"Diagram saved to {output_path}")
        return output_path

    def view(self) -> None:
        """
        Generate and display the diagram.

        This will open the diagram in the default viewer for your system.
        """
        self.generate()
        self.graph.view(cleanup=True)
