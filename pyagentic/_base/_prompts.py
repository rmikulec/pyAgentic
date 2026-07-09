"""
Prompt management for agents.

A `PromptEngine` is a source agents can pull managed instructions from (local files,
a database, a prompt-management service, ...). Engines load prompt text by key and
return it as a `PromptSource` carrying version metadata. `PromptEngine.ref(key)`
produces a deferred `PromptRef` that can be assigned to `__instructions__`; the agent
resolves it at instantiation time, so every new instance (and every fork) picks up
the latest version of the prompt.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from pyagentic._base._exceptions import PromptNotFound


class PromptSource(BaseModel):
    """
    A loaded prompt along with metadata about where and when it was loaded.

    Attributes:
        text (str): The prompt text itself.
        source (str): Identifier of where the prompt came from (e.g. a file path).
        source_type (str): The kind of engine that loaded it (e.g. "local").
        version (str): Version identifier for the loaded text (e.g. a content hash).
        loaded_at (datetime): When the prompt was loaded.
    """

    text: str
    source: str
    source_type: str
    version: str
    loaded_at: datetime


def _version_hash(text: str) -> str:
    """Content hash used as the version identifier for prompt text."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _inline_source(text: str, source: str) -> PromptSource:
    """Build a PromptSource for instructions declared inline as a plain string."""
    return PromptSource(
        text=text,
        source=source,
        source_type="inline",
        version=_version_hash(text),
        loaded_at=datetime.now(timezone.utc),
    )


class PromptRef:
    """
    A deferred pointer to a prompt in a `PromptEngine`.

    Created via `PromptEngine.ref(key)` and assigned to `__instructions__`; resolved
    at agent instantiation, so each new instance (and each fork) reloads the prompt.
    """

    def __init__(self, engine: "PromptEngine", key: str, version: str | None = None):
        self.engine = engine
        self.key = key
        self.version = version

    def resolve(self) -> PromptSource:
        """
        Load the referenced prompt from its engine.

        Returns:
            PromptSource: The loaded prompt text plus metadata.

        Raises:
            PromptNotFound: If the engine has no prompt for the key.
        """
        return self.engine.load(self.key, version=self.version)

    def __repr__(self) -> str:
        return f"PromptRef(engine={self.engine!r}, key={self.key!r}, version={self.version!r})"


class PromptEngine(ABC):
    """
    Base class for prompt engines, designed to be extended to support multiple
    different source types (local files, databases, managed prompt services, ...).
    """

    @abstractmethod
    def load(self, key: str, version: str | None = None) -> PromptSource:
        """
        Load a prompt by key.

        Args:
            key (str): Engine-specific identifier for the prompt.
            version (str | None): Specific version to load; the engine's latest
                when omitted.

        Returns:
            PromptSource: The loaded prompt text plus metadata.

        Raises:
            PromptNotFound: If no prompt exists for the key (or version).
        """

    def ref(self, key: str, version: str | None = None) -> PromptRef:
        """
        Create a deferred reference to a prompt, resolved at agent instantiation.

        Args:
            key (str): Engine-specific identifier for the prompt.
            version (str | None): Pin a specific version; the engine's latest
                when omitted.

        Returns:
            PromptRef: A reference that loads the prompt when resolved.
        """
        return PromptRef(self, key, version=version)


def _natural_sort_key(version: str) -> list:
    """Sort key treating digit runs numerically, so e.g. v10 orders after v2."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", version)]


class LocalPromptEngine(PromptEngine):
    """
    Prompt engine backed by the local file system.

    A `pattern` defines how prompt files are laid out under the root directory
    (default `.prompts/` in the current working directory). The pattern is a
    template with a required `{key}` placeholder and an optional `{version}`
    placeholder:

      - Without `{version}` (default `"{key}.md"`): the key maps to a single file
        and the prompt is versioned by content hash.
      - With `{version}` (e.g. `"{key}/{version}.md"`): the directory structure
        holds the versions. `load(key)` picks the latest version (natural sort,
        so `v10` > `v2`); `load(key, version="v1")` pins one. The path's version
        segment becomes `PromptSource.version`.

    Example:
        ```python
        # .prompts/researcher/v1.md, .prompts/researcher/v2.md, ...
        prompts = LocalPromptEngine(".prompts", pattern="{key}/{version}.md")

        class ResearchAgent(BaseAgent):
            __instructions__ = prompts.ref("researcher")  # latest version
        ```
    """

    def __init__(self, path: str | Path | None = None, pattern: str = "{key}.md"):
        """
        Args:
            path (str | Path | None): Root directory holding prompt files. Defaults
                to `.prompts/` under the current working directory.
            pattern (str): File layout template relative to the root. Must contain
                `{key}`; may contain `{version}` to derive versions from the file
                structure. Defaults to `"{key}.md"`.

        Raises:
            ValueError: If the pattern does not contain a `{key}` placeholder.
        """
        if "{key}" not in pattern:
            raise ValueError(
                f"LocalPromptEngine pattern {pattern!r} must contain a '{{key}}' placeholder"
            )
        self.root = Path(path) if path else Path.cwd() / ".prompts"
        self.pattern = pattern
        self.versioned = "{version}" in pattern

    def _resolve_versioned(self, key: str, version: str | None) -> tuple[Path, str]:
        """Resolve path and version for a versioned pattern; latest version when unpinned."""
        if version is not None:
            path = self.root / self.pattern.format(key=key, version=version)
            if not path.is_file():
                raise PromptNotFound(key=f"{key} (version {version})", source=str(self.root))
            return path, version

        # Glob for every version of the key, then parse the version segment back
        # out of each path with a regex built from the pattern
        regex = re.compile(
            re.escape(self.pattern)
            .replace(re.escape("{key}"), re.escape(key))
            .replace(re.escape("{version}"), r"(?P<version>[^/]+)")
        )
        found: dict[str, Path] = {}
        for path in self.root.glob(self.pattern.format(key=key, version="*")):
            match = regex.fullmatch(path.relative_to(self.root).as_posix())
            if match:
                found[match.group("version")] = path

        if not found:
            raise PromptNotFound(key=key, source=str(self.root))
        latest = max(found, key=_natural_sort_key)
        return found[latest], latest

    def load(self, key: str, version: str | None = None) -> PromptSource:
        """
        Load a prompt from a file laid out per the engine's pattern.

        Args:
            key (str): Prompt identifier substituted into the pattern's `{key}`.
            version (str | None): Specific version to load. Only valid for
                versioned patterns; the latest version is used when omitted.

        Returns:
            PromptSource: The file contents. Versioned patterns take the version
                from the file structure; unversioned patterns use a content hash.

        Raises:
            PromptNotFound: If no matching file exists under the root.
            ValueError: If a version is requested but the pattern has no
                `{version}` placeholder.
        """
        if self.versioned:
            path, resolved_version = self._resolve_versioned(key, version)
        else:
            if version is not None:
                raise ValueError(
                    f"Engine pattern {self.pattern!r} has no '{{version}}' placeholder; "
                    "a specific version cannot be requested"
                )
            path = self.root / self.pattern.format(key=key)
            if not path.is_file():
                raise PromptNotFound(key=key, source=str(self.root))
            resolved_version = None

        text = path.read_text()
        return PromptSource(
            text=text,
            source=str(path),
            source_type="local",
            version=resolved_version or _version_hash(text),
            loaded_at=datetime.now(timezone.utc),
        )

    def __repr__(self) -> str:
        return f"LocalPromptEngine(root={str(self.root)!r}, pattern={self.pattern!r})"
