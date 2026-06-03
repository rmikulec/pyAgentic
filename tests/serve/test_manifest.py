import pytest
from pathlib import Path

from pyagentic.serve._manifest import (
    Manifest,
    ProjectConfig,
    AgentConfig,
    ServerConfig,
    BuildConfig,
    EnvConfig,
    load_manifest,
)


def test_manifest_full_parse():
    """Test parsing a complete manifest with all sections."""
    data = {
        "project": {"name": "test-agent", "version": "1.0.0", "description": "A test agent"},
        "agent": {"entry": "my_agent:TestAgent", "model": "openai::gpt-4o"},
        "server": {"host": "127.0.0.1", "port": 9000},
        "build": {"python_version": "3.12", "dependencies": ["requests", "numpy"]},
        "env": {"required": ["OPENAI_API_KEY", "MY_SECRET"]},
    }
    manifest = Manifest(**data)

    assert manifest.project.name == "test-agent"
    assert manifest.project.version == "1.0.0"
    assert manifest.project.description == "A test agent"
    assert manifest.agent.entry == "my_agent:TestAgent"
    assert manifest.agent.model == "openai::gpt-4o"
    assert manifest.server.host == "127.0.0.1"
    assert manifest.server.port == 9000
    assert manifest.build.python_version == "3.12"
    assert manifest.build.dependencies == ["requests", "numpy"]
    assert manifest.env.required == ["OPENAI_API_KEY", "MY_SECRET"]


def test_manifest_defaults():
    """Test that optional sections use sensible defaults."""
    data = {
        "project": {"name": "minimal"},
        "agent": {"entry": "app:Agent"},
    }
    manifest = Manifest(**data)

    assert manifest.project.version == "0.1.0"
    assert manifest.project.description == ""
    assert manifest.agent.model == "openai::gpt-4o"
    assert manifest.server.host == "0.0.0.0"
    assert manifest.server.port == 8000
    assert manifest.build.python_version == "3.13"
    assert manifest.build.dependencies == []
    assert manifest.env.required == []


def test_manifest_missing_project_raises():
    """Test that missing [project] section raises a validation error."""
    with pytest.raises(Exception):
        Manifest(agent={"entry": "app:Agent"})


def test_manifest_missing_agent_raises():
    """Test that missing [agent] section raises a validation error."""
    with pytest.raises(Exception):
        Manifest(project={"name": "test"})


def test_manifest_missing_agent_entry_raises():
    """Test that missing agent.entry raises a validation error."""
    with pytest.raises(Exception):
        Manifest(project={"name": "test"}, agent={"model": "openai::gpt-4o"})


def test_load_manifest_from_file(tmp_path):
    """Test loading a manifest from a real TOML file."""
    toml_content = """\
[project]
name = "file-agent"
version = "2.0.0"

[agent]
entry = "my_mod:MyAgent"
model = "openai::gpt-4o-mini"

[server]
port = 3000
"""
    manifest_path = tmp_path / "pyagentic.toml"
    manifest_path.write_text(toml_content)

    manifest = load_manifest(manifest_path)
    assert manifest.project.name == "file-agent"
    assert manifest.project.version == "2.0.0"
    assert manifest.agent.entry == "my_mod:MyAgent"
    assert manifest.server.port == 3000


def test_load_manifest_file_not_found(tmp_path):
    """Test that loading a non-existent manifest raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does_not_exist.toml")


def test_project_config_model():
    """Test ProjectConfig standalone."""
    config = ProjectConfig(name="hello")
    assert config.name == "hello"
    assert config.version == "0.1.0"
    assert config.description == ""


def test_agent_config_model():
    """Test AgentConfig standalone."""
    config = AgentConfig(entry="mod:Cls")
    assert config.entry == "mod:Cls"
    assert config.model == "openai::gpt-4o"


def test_server_config_defaults():
    """Test ServerConfig defaults."""
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000


def test_build_config_defaults():
    """Test BuildConfig defaults."""
    config = BuildConfig()
    assert config.python_version == "3.13"
    assert config.dependencies == []


def test_env_config_defaults():
    """Test EnvConfig defaults."""
    config = EnvConfig()
    assert config.required == []
