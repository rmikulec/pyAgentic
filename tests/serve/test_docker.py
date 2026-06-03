import pytest

from pyagentic.serve._manifest import Manifest
from pyagentic.serve._docker import generate_dockerfile, build_image


def _make_manifest(**overrides) -> Manifest:
    data = {
        "project": {"name": "test-agent", "version": "1.0.0"},
        "agent": {"entry": "app:Agent"},
    }
    data.update(overrides)
    return Manifest(**data)


def test_generate_dockerfile_basic():
    """Test basic Dockerfile generation."""
    manifest = _make_manifest()
    dockerfile = generate_dockerfile(manifest)

    assert "FROM python:3.13-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "RUN pip install uv" in dockerfile
    assert "COPY requirements.txt ." in dockerfile
    assert "RUN uv pip install --system -r requirements.txt" in dockerfile
    assert "RUN uv pip install --system pyagentic-core[deploy]" in dockerfile
    assert "COPY . ." in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'CMD ["python", "-m", "pyagentic.cli", "run", "--host", "0.0.0.0"]' in dockerfile


def test_generate_dockerfile_custom_python():
    """Test Dockerfile with custom Python version."""
    manifest = _make_manifest(build={"python_version": "3.11"})
    dockerfile = generate_dockerfile(manifest)

    assert "FROM python:3.11-slim" in dockerfile


def test_generate_dockerfile_custom_port():
    """Test Dockerfile with custom port."""
    manifest = _make_manifest(server={"port": 9000})
    dockerfile = generate_dockerfile(manifest)

    assert "EXPOSE 9000" in dockerfile


def test_generate_dockerfile_with_dependencies():
    """Test Dockerfile with extra dependencies."""
    manifest = _make_manifest(build={"dependencies": ["requests>=2.0", "numpy"]})
    dockerfile = generate_dockerfile(manifest)

    assert 'RUN uv pip install --system "requests>=2.0" "numpy"' in dockerfile


def test_generate_dockerfile_no_dependencies():
    """Test that no extra pip install line is added when dependencies list is empty."""
    manifest = _make_manifest(build={"dependencies": []})
    dockerfile = generate_dockerfile(manifest)

    lines = dockerfile.split("\n")
    # Should not have a deps install line beyond the base ones
    pip_install_lines = [l for l in lines if "uv pip install --system" in l]
    # Only the two base installs: requirements.txt and pyagentic-core
    assert len(pip_install_lines) == 2


def test_build_image_no_docker(tmp_path, monkeypatch):
    """Test that build_image raises FileNotFoundError when docker is not installed."""
    monkeypatch.setattr("shutil.which", lambda x: None)

    manifest = _make_manifest()
    with pytest.raises(FileNotFoundError, match="Docker is not installed"):
        build_image(manifest, tmp_path)
