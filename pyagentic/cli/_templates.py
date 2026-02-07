"""
Scaffold templates for `pyagentic init`. Stored as string constants to
avoid data_files packaging complexity.
"""

# ---------- minimal template ----------

MINIMAL_AGENT = """\
from pyagentic import BaseAgent


class {class_name}(BaseAgent):
    __system_message__ = "You are a helpful assistant."

    # Add tools with the @tool decorator:
    #
    # from pyagentic import tool
    #
    # @tool("Describe what this tool does")
    # def my_tool(self, query: str) -> str:
    #     return f"Result for {{query}}"
"""

MINIMAL_TOML = """\
[project]
name = "{project_name}"
version = "0.1.0"
description = ""

[agent]
entry = "{module_name}:{class_name}"
model = "openai::gpt-4o"

[server]
host = "0.0.0.0"
port = 8000

[build]
python_version = "3.13"
dependencies = []

[env]
required = ["OPENAI_API_KEY"]
"""

# ---------- full template ----------

FULL_AGENT = '''\
from pyagentic import BaseAgent, tool, State, spec


class {class_name}(BaseAgent):
    """A research agent that can search and summarize information."""

    __system_message__ = """You are a helpful research assistant.
Use your tools to find and summarize information for the user.
Always cite your sources when providing information."""

    notes: State[list] = spec.State(
        default_factory=list,
        access="readwrite",
        get_description="Retrieve all saved research notes",
        set_description="Update the research notes",
    )

    @tool("Search for information on a topic and return a summary")
    def search(self, query: str) -> str:
        """Replace this with a real search implementation."""
        return f"[Placeholder] Search results for: {{query}}"

    @tool("Save a research note for later reference")
    def save_note(self, note: str) -> str:
        self.notes.append(note)
        return f"Note saved. Total notes: {{len(self.notes)}}"
'''

FULL_TOML = """\
[project]
name = "{project_name}"
version = "0.1.0"
description = "A PyAgentic agent project"

[agent]
entry = "{module_name}:{class_name}"
model = "openai::gpt-4o"

[server]
host = "0.0.0.0"
port = 8000

[build]
python_version = "3.13"
dependencies = []

[env]
required = ["OPENAI_API_KEY"]
"""

# ---------- shared files ----------

REQUIREMENTS_TXT = """\
pyagentic-core[deploy]
"""

ENV_EXAMPLE = """\
# Required environment variables for this agent.
# Copy this file to .env and fill in the values.
#
# NEVER commit your .env file to version control.
{env_lines}
"""

GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/

# Environment variables
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
