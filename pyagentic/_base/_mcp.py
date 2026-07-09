"""
MCP (Model Context Protocol) integration types and helpers.

Provides:
  - ``MCPLink``: Type annotation marker for MCP server connections
  - ``_MCPDefinition``: Pairs a field name with its ``MCPInfo`` config
  - ``_MCPToolDefinition``: A ``_ToolDefinition`` subclass for MCP-sourced tools
  - ``mcp_tool_to_tool_def()``: Converts a fastmcp ``Tool`` into ``_MCPToolDefinition``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from pyagentic._base._info import MCPInfo, ParamInfo
from pyagentic._base._tool import _ToolDefinition


class MCPLink:
    """Type annotation marker for MCP server connections.

    Used as a type annotation on agent class fields so the metaclass can
    detect MCP server configurations.  Paired with ``spec.MCPLink()`` which
    returns an ``MCPInfo`` descriptor.

    Example:
        ```python
        class MyAgent(BaseAgent):
            __instructions__ = "You are helpful"

            fs: MCPLink = spec.MCPLink(
                "npx",
                args=["@modelcontextprotocol/server-filesystem", "/tmp"],
                tools=["read_file", "write_file"],
                prefix=True,
            )
        ```
    """

    pass


@dataclass
class _MCPDefinition:
    """Pairs an agent field name with its MCP configuration."""

    field_name: str
    info: MCPInfo


@dataclass
class _MCPToolDefinition(_ToolDefinition):
    """A tool definition sourced from an MCP server.

    Overrides the base ``_ToolDefinition`` to work with raw JSON Schema
    from MCP instead of Python type introspection.

    Attributes:
        mcp_field_name: The agent field name (e.g. ``"fs"``).
        mcp_original_name: The original tool name on the MCP server.
        json_schema: Raw JSON Schema for the tool's input parameters.
    """

    mcp_field_name: str = ""
    mcp_original_name: str = ""
    json_schema: dict = field(default_factory=dict)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        json_schema: dict,
        mcp_field_name: str,
        mcp_original_name: str,
    ):
        super().__init__(
            name=name,
            description=description,
            parameters={},
            return_type=str,
        )
        self.mcp_field_name = mcp_field_name
        self.mcp_original_name = mcp_original_name
        self.json_schema = json_schema

    # JSON Schema keywords not supported by Anthropic's strict tool mode
    _UNSUPPORTED_SCHEMA_KEYS = frozenset({
        "$schema", "exclusiveMaximum", "exclusiveMinimum",
        "maxLength", "minLength", "maxItems", "minItems",
        "uniqueItems", "pattern", "format",
        "maximum", "minimum", "multipleOf",
    })

    @classmethod
    def _strip_unsupported(cls, obj: dict) -> dict:
        """Recursively strip unsupported JSON Schema keys and enforce strict mode.

        Also injects ``additionalProperties: false`` on every ``object``
        type node, as required by Anthropic's strict tool mode.
        """
        cleaned = {}
        for key, value in obj.items():
            if key in cls._UNSUPPORTED_SCHEMA_KEYS:
                continue
            if isinstance(value, dict):
                cleaned[key] = cls._strip_unsupported(value)
            else:
                cleaned[key] = value
        # Anthropic strict mode requires additionalProperties: false
        # on every object-typed node in the schema
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
        return cleaned

    def _clean_schema(self) -> dict:
        """Return a cleaned copy of the MCP JSON Schema.

        Strips meta-keys and unsupported validation keywords that LLM
        APIs don't accept, and ensures ``type`` and ``properties`` are
        present.
        """
        schema = self._strip_unsupported(
            dict(self.json_schema) if self.json_schema else {}
        )
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        return schema

    def to_openai_spec(self) -> dict:
        """Emit the raw JSON Schema from the MCP server.

        Returns:
            dict: An OpenAI-compliant tool specification dictionary.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._clean_schema(),
        }

    def to_anthropic_spec(self) -> dict:
        """Emit Anthropic-formatted tool spec with strict mode from MCP JSON Schema.

        Uses ``strict: true`` and ``additionalProperties: false`` to guarantee
        schema conformance via grammar-constrained sampling.

        Returns:
            dict: An Anthropic-compliant tool specification dictionary.
        """
        schema = self._clean_schema()
        schema["additionalProperties"] = False

        return {
            "name": self.name,
            "description": self.description,
            "strict": True,
            "input_schema": schema,
        }

    def compile_args(self, **kwargs) -> dict[str, Any]:
        """Pass-through: MCP handles its own validation.

        Args:
            **kwargs: Raw keyword arguments from the LLM tool call.

        Returns:
            dict[str, Any]: The same kwargs, unmodified.
        """
        return kwargs

    def resolve(self, agent_reference: dict) -> Self:
        """MCP tools have no StateRefs to resolve.

        Args:
            agent_reference (dict): The agent reference dict (unused).

        Returns:
            Self: Returns self unchanged.
        """
        return self


def _json_schema_to_parameters(
    schema: dict,
) -> dict[str, tuple[type, ParamInfo]]:
    """Convert a JSON Schema properties dict to ``(type, ParamInfo)`` pairs.

    This is a simple mapping used for response model compatibility. Complex
    schemas fall back to ``str``.

    Args:
        schema (dict): JSON Schema with ``properties`` and optionally ``required``.

    Returns:
        dict[str, tuple[type, ParamInfo]]: Mapping of parameter names to
            ``(python_type, ParamInfo)`` tuples.
    """
    _JSON_TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    params: dict[str, tuple[type, ParamInfo]] = {}

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        py_type = _JSON_TYPE_MAP.get(json_type, str)
        is_required = prop_name in required_fields
        description = prop_schema.get("description")
        params[prop_name] = (
            py_type,
            ParamInfo(required=is_required, description=description),
        )

    return params


def mcp_tool_to_tool_def(
    mcp_tool: Any,
    field_name: str,
    prefix: bool | str,
) -> _MCPToolDefinition:
    """Convert a fastmcp ``Tool`` object to an ``_MCPToolDefinition``.

    Applies prefix logic: when *prefix* is ``True``, the tool name becomes
    ``{field_name}__{original_name}``.  When *prefix* is a string, it is
    used instead of *field_name*.

    Args:
        mcp_tool (Any): A fastmcp ``Tool`` object with ``name``,
            ``description``, and ``inputSchema`` attributes.
        field_name (str): The agent field name (e.g. ``"fs"``).
        prefix (bool | str): Prefix mode — ``True`` uses *field_name*,
            a string uses that value, ``False`` uses no prefix.

    Returns:
        _MCPToolDefinition: The converted tool definition.
    """
    original_name = mcp_tool.name
    description = mcp_tool.description or ""
    input_schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}

    if prefix is True:
        prefixed_name = f"{field_name}__{original_name}"
    elif isinstance(prefix, str) and prefix:
        prefixed_name = f"{prefix}__{original_name}"
    else:
        prefixed_name = original_name

    return _MCPToolDefinition(
        name=prefixed_name,
        description=description,
        json_schema=input_schema,
        mcp_field_name=field_name,
        mcp_original_name=original_name,
    )
