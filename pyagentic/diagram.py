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

        # Structure diagram (declaration)
        self.structure_graph = graphviz.Digraph(
            name=f"Structure_{self.agent_name}",
            comment=f"Agent Structure: {self.agent_name}",
            format='png'
        )
        self.structure_graph.attr(
            rankdir='LR',
            splines='spline',
            nodesep='0.4',
            ranksep='1.0',
            compound='true',
            bgcolor='white',
            pad='0.3',
            dpi='150'
        )
        self.structure_graph.attr('node', fontname='Arial', fontsize='10')
        self.structure_graph.attr('edge', fontname='Arial', fontsize='8')

        # Execution flow diagram
        self.flow_graph = graphviz.Digraph(
            name=f"Flow_{self.agent_name}",
            comment=f"Execution Flow: {self.agent_name}",
            format='png'
        )
        self.flow_graph.attr(
            rankdir='TB',  # Top to bottom for flow
            splines='spline',
            nodesep='0.6',
            ranksep='0.8',
            compound='true',
            bgcolor='white',
            pad='0.4',
            dpi='150'
        )
        self.flow_graph.attr('node', fontname='Arial', fontsize='9')
        self.flow_graph.attr('edge', fontname='Arial', fontsize='8')

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

    def _add_agent_to_structure(self, agent_class: Type[BaseAgent]) -> None:
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

        with self.structure_graph.subgraph(name=f'cluster_{agent_name}') as cluster:
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

                            # Get access level and policies
                            access = "read"  # default
                            policies = []
                            if state_def.info:
                                access = getattr(state_def.info, 'access', 'read')
                                policies = getattr(state_def.info, 'policies', []) or []

                            # Access level indicator
                            access_icon = {
                                'read': '👁️',
                                'write': '✏️',
                                'readwrite': '🔄',
                                'hidden': '🔒'
                            }.get(access, '')

                            label = f"{access_icon} {name}: {type_name}"

                            # Add policy indicator if policies exist
                            if policies:
                                policy_names = [p.__class__.__name__ for p in policies]
                                label += f"\\n[{', '.join(policy_names)}]"
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
                self.structure_graph.edge(
                    self._node_id(agent_name, 'SystemMessage'),
                    self._node_id(agent_name, f'State_{var}'),
                    color='#7c3aed',
                    penwidth='1.5',
                    arrowsize='0.8'
                )

        # Add edges: input_template -> state variables it uses
        if input_template and input_template_vars:
            for var in input_template_vars:
                self.structure_graph.edge(
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
                self._add_agent_to_structure(linked_class)

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
                    self.structure_graph.edge(
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

    def _add_execution_flow(self, visited_agents: Optional[Set] = None) -> None:
        """Add simplified agent execution flow diagram."""
        if visited_agents is None:
            visited_agents = set()

        agent_class = self.agent_class
        agent_name = agent_class.__name__

        # Skip if already added
        if agent_name in visited_agents:
            return
        visited_agents.add(agent_name)

        # Extract flow components
        state_defs = agent_class.__state_defs__
        tool_defs = agent_class.__tool_defs__
        linked_agents = agent_class.__linked_agents__
        system_msg = getattr(agent_class, "__system_message__", "")
        input_template = getattr(agent_class, "__input_template__", "")
        max_call_depth = getattr(agent_class, "max_call_depth", 1)

        # Get which states are used where
        system_msg_vars = self._extract_state_variables(system_msg, state_defs) if system_msg else []
        input_template_vars = self._extract_state_variables(input_template, state_defs) if input_template else []

        # Count states with policies
        states_with_policies = sum(1 for name, state_def in state_defs.items()
                                   if isinstance(state_def, _StateDefinition) and state_def.info
                                   and getattr(state_def.info, 'policies', None))

        # Build simplified flow
        with self.flow_graph.subgraph(name=f'cluster_flow_{agent_name}') as flow:
            flow.attr(
                label=f'Execution Flow: {agent_name}',
                style='rounded',
                fillcolor='#fafafa',
                color='#64748b',
                penwidth='2',
                fontsize='12',
                fontname='Arial Bold'
            )

            # Step 1: User Input
            flow.node(f'flow_{agent_name}_1', label='1. User Input', shape='box',
                     style='rounded,filled', fillcolor='#e0e7ff', color='#4f46e5',
                     fontsize='10', penwidth='2')

            # Step 2: Format templates with state
            if input_template_vars or system_msg_vars:
                template_info = []
                if input_template_vars:
                    template_info.append(f"input_template: {', '.join(input_template_vars)}")
                if system_msg_vars:
                    template_info.append(f"system_message: {', '.join(system_msg_vars)}")

                flow.node(f'flow_{agent_name}_2',
                         label=f'2. Format Templates\\n{chr(10).join(template_info)}',
                         shape='box', style='rounded,filled', fillcolor='#ddd6fe',
                         color='#7c3aed', fontsize='10', penwidth='2')

            # Step 3: LLM Processing
            flow.node(f'flow_{agent_name}_3', label='3. LLM Processing', shape='ellipse',
                     style='filled', fillcolor='#e0f2fe', color='#0284c7',
                     fontsize='10', penwidth='2', fontname='Arial Bold')

            # Step 4: Decision
            flow.node(f'flow_{agent_name}_4', label='4. Decision', shape='diamond',
                     style='filled', fillcolor='#fef3c7', color='#d97706',
                     fontsize='10', penwidth='2')

            # Step 5: Tool execution
            if tool_defs or linked_agents:
                tool_count = len(tool_defs) + len(linked_agents)
                tool_names = list(tool_defs.keys())[:3]
                if len(tool_defs) > 3:
                    tool_label = f"5. Execute Tool\\n({', '.join(tool_names)}... +{len(tool_defs)-3} more)"
                elif tool_names:
                    tool_label = f"5. Execute Tool\\n({', '.join(tool_names)})"
                else:
                    tool_label = "5. Execute Tool"

                if linked_agents:
                    tool_label += f"\\n(or call linked agent)"

                flow.node(f'flow_{agent_name}_5', label=tool_label, shape='box',
                         style='rounded,filled', fillcolor='#fee2e2', color='#dc2626',
                         fontsize='10', penwidth='2')

            # Step 6: State access with policies
            if state_defs:
                state_label = f"6. Access State\\n({len(state_defs)} variables"
                if states_with_policies > 0:
                    state_label += f", {states_with_policies} with policies)"
                else:
                    state_label += ")"

                flow.node(f'flow_{agent_name}_6', label=state_label, shape='cylinder',
                         style='filled', fillcolor='#fef3c7', color='#d97706',
                         fontsize='10', penwidth='2')

            # Step 7: Loop check
            flow.node(f'flow_{agent_name}_7',
                     label=f'7. Check Loop\\n(max {max_call_depth} iterations)',
                     shape='diamond', style='filled', fillcolor='#f1f5f9',
                     color='#64748b', fontsize='10', penwidth='2')

            # Step 8: Final Response
            flow.node(f'flow_{agent_name}_8', label='8. Return Response', shape='box',
                     style='rounded,filled', fillcolor='#d1fae5', color='#10b981',
                     fontsize='10', penwidth='2', fontname='Arial Bold')

        # Connect the flow - simple linear path
        self.flow_graph.edge(f'flow_{agent_name}_1', f'flow_{agent_name}_2' if (input_template_vars or system_msg_vars) else f'flow_{agent_name}_3',
                           color='#64748b', penwidth='2')

        if input_template_vars or system_msg_vars:
            self.flow_graph.edge(f'flow_{agent_name}_2', f'flow_{agent_name}_3',
                               color='#7c3aed', penwidth='2')

        self.flow_graph.edge(f'flow_{agent_name}_3', f'flow_{agent_name}_4',
                           color='#0284c7', penwidth='2')

        if tool_defs or linked_agents:
            self.flow_graph.edge(f'flow_{agent_name}_4', f'flow_{agent_name}_5',
                               label='call tool', color='#dc2626', penwidth='2',
                               fontsize='9')

            if state_defs:
                self.flow_graph.edge(f'flow_{agent_name}_5', f'flow_{agent_name}_6',
                                   color='#d97706', penwidth='2')
                self.flow_graph.edge(f'flow_{agent_name}_6', f'flow_{agent_name}_7',
                                   color='#64748b', penwidth='2')
            else:
                self.flow_graph.edge(f'flow_{agent_name}_5', f'flow_{agent_name}_7',
                                   color='#64748b', penwidth='2')

            # Loop back
            self.flow_graph.edge(f'flow_{agent_name}_7', f'flow_{agent_name}_3',
                               label='continue', color='#64748b', penwidth='1.5',
                               style='dashed', fontsize='9')

        # Done path
        final_decision = f'flow_{agent_name}_7' if (tool_defs or linked_agents) else f'flow_{agent_name}_4'
        self.flow_graph.edge(final_decision, f'flow_{agent_name}_8',
                           label='done', color='#10b981', penwidth='2',
                           fontsize='9')

        # Recursively add linked agents
        for link_name, linked_class in linked_agents.items():
            linked_gen = AgentDiagramGenerator(linked_class)
            linked_gen.flow_graph = self.flow_graph
            linked_gen._add_execution_flow(visited_agents)

    def generate_structure(self) -> graphviz.Digraph:
        """
        Generate the agent structure (declaration) diagram.

        Shows:
        - Agent components (state, tools, templates)
        - Linked agents
        - Template → state dependencies
        - Access levels and policies

        Returns:
            Graphviz Digraph object for structure
        """
        self.visited.clear()
        self._add_agent_to_structure(self.agent_class)

        # Add legend for access levels
        with self.structure_graph.subgraph(name='cluster_legend') as legend:
            legend.attr(
                label='State Access Levels',
                style='rounded,dashed',
                color='#94a3b8',
                penwidth='1',
                fontsize='10',
                fontname='Arial Bold'
            )

            legend.node('legend_read', label='👁️ read: LLM can see', shape='note',
                       style='filled', fillcolor='#f8fafc', fontsize='8')
            legend.node('legend_write', label='✏️ write: Tools can modify', shape='note',
                       style='filled', fillcolor='#f8fafc', fontsize='8')
            legend.node('legend_readwrite', label='🔄 readwrite: LLM sees + Tools modify', shape='note',
                       style='filled', fillcolor='#f8fafc', fontsize='8')
            legend.node('legend_hidden', label='🔒 hidden: Neither LLM nor tools', shape='note',
                       style='filled', fillcolor='#f8fafc', fontsize='8')

        return self.structure_graph

    def generate_flow(self) -> graphviz.Digraph:
        """
        Generate the execution flow diagram.

        Shows how the agent executes step-by-step with individual nodes
        for states, tools, and policies.

        Returns:
            Graphviz Digraph object for execution flow
        """
        self._add_execution_flow()
        return self.flow_graph

    def save_structure(self, filename: str, format: str = 'png') -> str:
        """
        Generate and save the structure diagram to a file.

        Args:
            filename: Path to output file (without extension)
            format: Output format (png, pdf, svg, etc.)

        Returns:
            Path to the saved file
        """
        self.structure_graph.format = format
        self.generate_structure()
        output_path = self.structure_graph.render(filename, cleanup=True)
        print(f"Structure diagram saved to {output_path}")
        return output_path

    def save_flow(self, filename: str, format: str = 'png') -> str:
        """
        Generate and save the execution flow diagram to a file.

        Args:
            filename: Path to output file (without extension)
            format: Output format (png, pdf, svg, etc.)

        Returns:
            Path to the saved file
        """
        self.flow_graph.format = format
        self.generate_flow()
        output_path = self.flow_graph.render(filename, cleanup=True)
        print(f"Execution flow diagram saved to {output_path}")
        return output_path

    def view_structure(self) -> None:
        """
        Generate and display the structure diagram.

        Opens the diagram in the default system viewer.
        """
        self.generate_structure()
        self.structure_graph.view(cleanup=True)

    def view_flow(self) -> None:
        """
        Generate and display the execution flow diagram.

        Opens the diagram in the default system viewer.
        """
        self.generate_flow()
        self.flow_graph.view(cleanup=True)
