import pytest

from pyagentic.api._config import (
    AgentsConfig,
    AppConfig,
    DeployConfig,
    load_config,
)


def test_full_parse():
    """Test parsing a complete agents.toml with both sections."""
    data = {
        "app": {
            "name": "my-agents",
            "version": "1.0.0",
            "description": "A test app",
            "model": "openai::gpt-4o",
        },
        "deploy": {
            "target": "main:app",
            "python_version": "3.12",
            "dependencies": ["requests", "numpy"],
            "port": 9000,
            "env": ["OPENAI_API_KEY"],
        },
    }
    config = AgentsConfig(**data)

    assert config.app.name == "my-agents"
    assert config.app.version == "1.0.0"
    assert config.app.model == "openai::gpt-4o"
    assert config.deploy.target == "main:app"
    assert config.deploy.python_version == "3.12"
    assert config.deploy.dependencies == ["requests", "numpy"]
    assert config.deploy.port == 9000
    assert config.deploy.env == ["OPENAI_API_KEY"]


def test_defaults():
    """Test that an empty config uses sensible defaults."""
    config = AgentsConfig()
    assert config.app.name == "agents"
    assert config.app.version == "0.1.0"
    assert config.app.model is None
    assert config.deploy.target == "main:app"
    assert config.deploy.python_version == "3.13"
    assert config.deploy.port == 8000
    assert config.deploy.dependencies == []
    assert config.deploy.env == []


def test_invalid_python_version():
    """Test python_version validation."""
    with pytest.raises(ValueError, match="Invalid python_version"):
        DeployConfig(python_version="2.7")


def test_invalid_port():
    """Test port range validation."""
    with pytest.raises(ValueError, match="Port must be between"):
        DeployConfig(port=70000)


def test_invalid_target():
    """Test target must be module:attr."""
    with pytest.raises(ValueError, match="Invalid target"):
        DeployConfig(target="main")


def test_load_config_missing_default_returns_defaults(tmp_path, monkeypatch):
    """A missing ./agents.toml yields defaults rather than an error."""
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert isinstance(config, AgentsConfig)
    assert config.app.name == "agents"


def test_load_config_explicit_missing_raises(tmp_path):
    """An explicitly-given missing path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_load_config_reads_file(tmp_path):
    """load_config parses a real agents.toml file."""
    toml = tmp_path / "agents.toml"
    toml.write_text(
        '[app]\nname = "bot"\nversion = "2.0.0"\n\n'
        '[deploy]\ntarget = "srv:app"\nport = 1234\n'
    )
    config = load_config(toml)
    assert config.app.name == "bot"
    assert config.app.version == "2.0.0"
    assert config.deploy.target == "srv:app"
    assert config.deploy.port == 1234
