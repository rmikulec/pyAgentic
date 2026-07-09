import pytest
from pydantic import BaseModel

from pyagentic import BaseAgent, LocalPromptEngine, PromptRef, PromptSource, State, spec
from pyagentic._base._exceptions import InstructionsNotDeclared, PromptNotFound


@pytest.fixture
def prompt_dir(tmp_path):
    """Creates a prompt directory with a few prompt files for testing"""
    (tmp_path / "researcher.md").write_text("You research {{ topic.value }}")
    (tmp_path / "plain.md").write_text("You are a plain assistant")
    return tmp_path


@pytest.fixture
def versioned_prompt_dir(tmp_path):
    """Creates a prompt directory laid out as {key}/{version}.md"""
    researcher = tmp_path / "researcher"
    researcher.mkdir()
    (researcher / "v1.md").write_text("Researcher v1")
    (researcher / "v2.md").write_text("Researcher v2")
    (researcher / "v10.md").write_text("Researcher v10")
    return tmp_path


# ---- LocalPromptEngine: default (unversioned) pattern ----


def test_local_engine_default_pattern_loads_key_md(prompt_dir):
    """Test that the default '{key}.md' pattern maps keys to markdown files"""
    engine = LocalPromptEngine(prompt_dir)
    source = engine.load("researcher")

    assert source.text == "You research {{ topic.value }}"
    assert source.source_type == "local"
    assert source.source == str(prompt_dir / "researcher.md")


def test_local_engine_custom_flat_pattern(prompt_dir):
    """Test that a custom pattern controls how keys map to files"""
    (prompt_dir / "helper.txt").write_text("You are a helper")
    engine = LocalPromptEngine(prompt_dir, pattern="{key}.txt")

    assert engine.load("helper").text == "You are a helper"


def test_local_engine_pattern_requires_key_placeholder(prompt_dir):
    """Test that a pattern without {key} is rejected"""
    with pytest.raises(ValueError, match=r"\{key\}"):
        LocalPromptEngine(prompt_dir, pattern="prompts/latest.md")


def test_local_engine_missing_key_raises(prompt_dir):
    """Test that loading a non-existent key raises PromptNotFound"""
    engine = LocalPromptEngine(prompt_dir)

    with pytest.raises(PromptNotFound) as e:
        engine.load("nope")

    assert "nope" in str(e.value)
    assert str(prompt_dir) in str(e.value)


def test_local_engine_version_is_content_hash(prompt_dir):
    """Test that unversioned patterns hash the content, changing when the file changes"""
    engine = LocalPromptEngine(prompt_dir)

    first = engine.load("plain")
    (prompt_dir / "plain.md").write_text("You are an edited assistant")
    second = engine.load("plain")

    assert first.version != second.version
    assert second.text == "You are an edited assistant"


def test_local_engine_unversioned_pattern_rejects_version_arg(prompt_dir):
    """Test that requesting a version without a {version} placeholder raises"""
    engine = LocalPromptEngine(prompt_dir)

    with pytest.raises(ValueError, match=r"\{version\}"):
        engine.load("researcher", version="v1")


def test_engine_ref_is_deferred(prompt_dir):
    """Test that ref() returns a PromptRef that only reads the file on resolve()"""
    engine = LocalPromptEngine(prompt_dir)
    ref_ = engine.ref("missing-for-now")

    assert isinstance(ref_, PromptRef)

    (prompt_dir / "missing-for-now.md").write_text("Created after ref")
    source = ref_.resolve()

    assert isinstance(source, PromptSource)
    assert source.text == "Created after ref"


# ---- LocalPromptEngine: versioned pattern ----


def test_versioned_pattern_loads_latest(versioned_prompt_dir):
    """Test that an unpinned load picks the latest version with natural sort (v10 > v2)"""
    engine = LocalPromptEngine(versioned_prompt_dir, pattern="{key}/{version}.md")
    source = engine.load("researcher")

    assert source.text == "Researcher v10"
    assert source.version == "v10"
    assert source.source == str(versioned_prompt_dir / "researcher" / "v10.md")


def test_versioned_pattern_loads_pinned_version(versioned_prompt_dir):
    """Test that an explicit version loads that file and reports its version"""
    engine = LocalPromptEngine(versioned_prompt_dir, pattern="{key}/{version}.md")
    source = engine.load("researcher", version="v1")

    assert source.text == "Researcher v1"
    assert source.version == "v1"


def test_versioned_pattern_missing_version_raises(versioned_prompt_dir):
    """Test that pinning a non-existent version raises PromptNotFound"""
    engine = LocalPromptEngine(versioned_prompt_dir, pattern="{key}/{version}.md")

    with pytest.raises(PromptNotFound) as e:
        engine.load("researcher", version="v99")

    assert "v99" in str(e.value)


def test_versioned_pattern_missing_key_raises(versioned_prompt_dir):
    """Test that a key with no versions raises PromptNotFound"""
    engine = LocalPromptEngine(versioned_prompt_dir, pattern="{key}/{version}.md")

    with pytest.raises(PromptNotFound):
        engine.load("nope")


def test_versioned_pattern_with_version_in_filename(tmp_path):
    """Test a flat versioned layout like researcher_v1.md"""
    (tmp_path / "researcher_v1.md").write_text("Flat v1")
    (tmp_path / "researcher_v2.md").write_text("Flat v2")
    engine = LocalPromptEngine(tmp_path, pattern="{key}_{version}.md")

    assert engine.load("researcher").version == "v2"
    assert engine.load("researcher", version="v1").text == "Flat v1"


def test_ref_pins_version(versioned_prompt_dir):
    """Test that ref(key, version=...) resolves the pinned version"""
    engine = LocalPromptEngine(versioned_prompt_dir, pattern="{key}/{version}.md")
    source = engine.ref("researcher", version="v2").resolve()

    assert source.text == "Researcher v2"
    assert source.version == "v2"


# ---- Agent integration ----


class TopicModel(BaseModel):
    value: str = "transformers"


def test_agent_instructions_from_prompt_engine(prompt_dir):
    """Test that a PromptRef in __instructions__ resolves at instantiation and renders state"""
    engine = LocalPromptEngine(prompt_dir)

    class ResearchAgent(BaseAgent):
        __instructions__ = engine.ref("researcher")

        topic: State[TopicModel] = spec.State(default_factory=TopicModel)

    agent = ResearchAgent(model="_mock::test-model", api_key="key")

    assert agent.state.instructions == "You research {{ topic.value }}"
    assert agent.state.system_message == "You research transformers"
    assert agent.state.prompt_source.source_type == "local"
    assert agent.state.prompt_source.source == str(prompt_dir / "researcher.md")


def test_agent_instructions_plain_string_gets_inline_source():
    """Test that plain string instructions produce an inline PromptSource named after the agent"""

    class PlainAgent(BaseAgent):
        __instructions__ = "You are plain"

    agent = PlainAgent(model="_mock::test-model", api_key="key")

    assert agent.state.instructions == "You are plain"
    assert agent.state.system_message == "You are plain"
    assert agent.state.prompt_source.source_type == "inline"
    assert agent.state.prompt_source.source == "PlainAgent"
    assert agent.state.prompt_source.text == "You are plain"
    assert agent.state.prompt_source.version


def test_agent_loads_prompt_per_instantiation(prompt_dir):
    """Test that each instantiation re-reads the prompt file"""
    engine = LocalPromptEngine(prompt_dir)

    class PlainAgent(BaseAgent):
        __instructions__ = engine.ref("plain")

    first = PlainAgent(model="_mock::test-model", api_key="key")
    (prompt_dir / "plain.md").write_text("You are an updated assistant")
    second = PlainAgent(model="_mock::test-model", api_key="key")

    assert first.state.instructions == "You are a plain assistant"
    assert second.state.instructions == "You are an updated assistant"


def test_agent_fork_reloads_prompt(prompt_dir):
    """Test that fork() picks up prompt edits made after the original was built"""
    engine = LocalPromptEngine(prompt_dir)

    class PlainAgent(BaseAgent):
        __instructions__ = engine.ref("plain")

    agent = PlainAgent(model="_mock::test-model", api_key="key")
    (prompt_dir / "plain.md").write_text("You are a forked assistant")
    fork = agent.fork()

    assert agent.state.instructions == "You are a plain assistant"
    assert fork.state.instructions == "You are a forked assistant"


@pytest.mark.asyncio
async def test_agent_response_includes_prompt_source(prompt_dir):
    """Test that AgentResponse carries the PromptSource the instructions were loaded from"""
    engine = LocalPromptEngine(prompt_dir)

    class PlainAgent(BaseAgent):
        __instructions__ = engine.ref("plain")

    agent = PlainAgent(model="_mock::test-model", api_key="key")
    response = await agent.run("hello")

    assert response.prompt == agent.state.prompt_source
    assert response.prompt.source == str(prompt_dir / "plain.md")
    assert response.prompt.source_type == "local"


@pytest.mark.asyncio
async def test_agent_response_inline_prompt_for_plain_string():
    """Test that AgentResponse.prompt is an inline source named after the agent"""

    class PlainAgent(BaseAgent):
        __instructions__ = "You are plain"

    agent = PlainAgent(model="_mock::test-model", api_key="key")
    response = await agent.run("hello")

    assert response.prompt.source_type == "inline"
    assert response.prompt.source == "PlainAgent"
    assert response.prompt.text == "You are plain"


# ---- Naming: __instructions__ vs deprecated __system_message__ ----


def test_system_message_dunder_warns_and_still_works():
    """Test that __system_message__ emits a DeprecationWarning but keeps working"""
    with pytest.warns(DeprecationWarning, match="__system_message__"):

        class LegacyAgent(BaseAgent):
            __system_message__ = "You are legacy"

    agent = LegacyAgent(model="_mock::test-model", api_key="key")

    assert LegacyAgent.__instructions__ == "You are legacy"
    assert agent.state.system_message == "You are legacy"


def test_instructions_dunder_sets_deprecated_alias():
    """Test that __system_message__ stays readable when __instructions__ is declared"""

    class ModernAgent(BaseAgent):
        __instructions__ = "You are modern"

    assert ModernAgent.__system_message__ == "You are modern"


def test_declaring_neither_raises():
    """Test that declaring neither dunder raises InstructionsNotDeclared"""
    with pytest.raises(InstructionsNotDeclared):

        class NamelessAgent(BaseAgent):
            pass
