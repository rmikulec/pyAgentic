from pyagentic.api._config import AgentsConfig
from pyagentic.api._docker import generate_dockerfile, write_dockerfile


def _make_config(**deploy) -> AgentsConfig:
    return AgentsConfig(app={"name": "test"}, deploy=deploy or {})


def test_generate_dockerfile_basic():
    """Test basic Dockerfile generation."""
    dockerfile = generate_dockerfile(_make_config())

    assert "FROM python:3.13-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "RUN pip install uv" in dockerfile
    assert 'RUN uv pip install --system "pyagentic-core[api]"' in dockerfile
    assert "COPY . ." in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile


def test_generate_dockerfile_custom_python():
    """Test Dockerfile with custom Python version."""
    dockerfile = generate_dockerfile(_make_config(python_version="3.11"))
    assert "FROM python:3.11-slim" in dockerfile


def test_generate_dockerfile_custom_target_and_port():
    """Test Dockerfile with custom target and port."""
    dockerfile = generate_dockerfile(_make_config(target="srv:application", port=9000))
    assert "EXPOSE 9000" in dockerfile
    assert 'CMD ["uvicorn", "srv:application", "--host", "0.0.0.0", "--port", "9000"]' in dockerfile


def test_generate_dockerfile_with_dependencies():
    """Test Dockerfile with extra dependencies."""
    dockerfile = generate_dockerfile(_make_config(dependencies=["requests>=2.0", "numpy"]))
    assert 'RUN uv pip install --system "requests>=2.0" "numpy"' in dockerfile


def test_generate_dockerfile_no_dependencies():
    """Test that only the base install line appears when there are no extra deps."""
    dockerfile = generate_dockerfile(_make_config(dependencies=[]))
    pip_lines = [l for l in dockerfile.split("\n") if "uv pip install --system" in l]
    assert len(pip_lines) == 1


def test_write_dockerfile(tmp_path):
    """write_dockerfile writes the generated content to disk."""
    out = write_dockerfile(_make_config(), path=tmp_path / "Dockerfile")
    assert out.exists()
    assert "FROM python:3.13-slim" in out.read_text()
