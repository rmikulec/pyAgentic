"""
Agent class discovery and loading from module:ClassName entry points.
"""

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyagentic._base._agent._agent import BaseAgent


def load_agent_class(entry: str) -> "type[BaseAgent]":
    """Load an agent class from a 'module:ClassName' entry point string.

    The current working directory is added to sys.path if not already present,
    so that local agent modules can be imported.

    Args:
        entry (str): Entry point in 'module_path:ClassName' format.
            Example: 'my_agent:ResearchAgent'

    Returns:
        type[BaseAgent]: The agent class (not an instance).

    Raises:
        ValueError: If the entry point format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    if ":" not in entry:
        raise ValueError(
            f"Invalid entry point format: {entry!r}. "
            "Expected 'module_path:ClassName' (e.g. 'my_agent:ResearchAgent')."
        )

    module_path, class_name = entry.rsplit(":", 1)

    # Ensure cwd is importable
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls
