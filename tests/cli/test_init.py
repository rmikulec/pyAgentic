import pytest
from pathlib import Path

from pyagentic.cli._init import _to_module_name, _to_class_name
from pyagentic.cli._templates import (
    MINIMAL_AGENT,
    MINIMAL_TOML,
    FULL_AGENT,
    FULL_TOML,
    REQUIREMENTS_TXT,
    GITIGNORE,
)


def test_to_module_name_simple():
    """Test converting a simple project name to module name."""
    assert _to_module_name("my-agent") == "my_agent"


def test_to_module_name_underscores():
    """Test that underscores are preserved."""
    assert _to_module_name("my_agent") == "my_agent"


def test_to_module_name_special_chars():
    """Test that special characters are replaced with underscores."""
    assert _to_module_name("my.agent!v2") == "my_agent_v2"


def test_to_module_name_leading_trailing():
    """Test that leading/trailing non-alphanumeric chars are stripped."""
    assert _to_module_name("--my-agent--") == "my_agent"


def test_to_class_name_simple():
    """Test converting a simple name to PascalCase."""
    assert _to_class_name("my-agent") == "MyAgent"


def test_to_class_name_underscores():
    """Test converting underscored name to PascalCase."""
    assert _to_class_name("my_cool_agent") == "MyCoolAgent"


def test_to_class_name_mixed():
    """Test converting mixed separators to PascalCase."""
    assert _to_class_name("my-cool_agent") == "MyCoolAgent"


def test_minimal_agent_template():
    """Test that the minimal agent template renders correctly."""
    result = MINIMAL_AGENT.format(
        project_name="test-proj",
        module_name="test_proj",
        class_name="TestProj",
    )
    assert "class TestProj(BaseAgent):" in result
    assert "from pyagentic import BaseAgent" in result
    assert "__system_message__" in result


def test_minimal_toml_template():
    """Test that the minimal TOML template renders correctly."""
    result = MINIMAL_TOML.format(
        project_name="test-proj",
        module_name="test_proj",
        class_name="TestProj",
    )
    assert 'name = "test-proj"' in result
    assert 'entry = "test_proj:TestProj"' in result
    assert "[project]" in result
    assert "[agent]" in result


def test_full_agent_template():
    """Test that the full agent template renders correctly."""
    result = FULL_AGENT.format(
        project_name="research",
        module_name="research",
        class_name="Research",
    )
    assert "class Research(BaseAgent):" in result
    assert "@tool" in result
    assert "State[list]" in result


def test_full_toml_template():
    """Test that the full TOML template renders correctly."""
    result = FULL_TOML.format(
        project_name="research",
        module_name="research",
        class_name="Research",
    )
    assert 'description = "A PyAgentic agent project"' in result


def test_requirements_txt_content():
    """Test that requirements.txt contains the core deploy dependency."""
    assert "pyagentic-core[deploy]" in REQUIREMENTS_TXT


def test_gitignore_content():
    """Test that .gitignore has essential entries."""
    assert ".env" in GITIGNORE
    assert "__pycache__/" in GITIGNORE
    assert ".venv/" in GITIGNORE
