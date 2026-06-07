import pytest
import sys
from pathlib import Path

from pyagentic.serve._discovery import load_agent_class
from pyagentic._base._agent._agent import BaseAgent


def test_load_agent_class_valid():
    """Test loading an agent class from a module:ClassName entry point."""
    cls = load_agent_class("pyagentic._base._agent._agent:BaseAgent")
    assert cls is BaseAgent


def test_load_agent_class_invalid_format():
    """Test that an entry point without ':' raises ValueError."""
    with pytest.raises(ValueError, match="Invalid entry point format"):
        load_agent_class("my_agent.ResearchAgent")


def test_load_agent_class_missing_module():
    """Test that a non-existent module raises ImportError."""
    with pytest.raises(ImportError):
        load_agent_class("nonexistent_module_xyz:Agent")


def test_load_agent_class_missing_class():
    """Test that a missing class in an existing module raises AttributeError."""
    with pytest.raises(AttributeError):
        load_agent_class("pyagentic:NonExistentClassName")


def test_load_agent_class_adds_cwd_to_sys_path():
    """Test that cwd is added to sys.path for local module imports."""
    cwd = str(Path.cwd())
    # Remove cwd from sys.path temporarily
    original_path = sys.path.copy()
    if cwd in sys.path:
        sys.path.remove(cwd)

    try:
        # load_agent_class should add cwd back
        load_agent_class("pyagentic._base._agent._agent:BaseAgent")
        assert cwd in sys.path
    finally:
        sys.path = original_path
