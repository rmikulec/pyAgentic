"""
PyAgentic CLI: Command-line interface for scaffolding, running, building,
and publishing PyAgentic agents.
"""

try:
    import typer
except ImportError:
    raise ImportError(
        "CLI dependencies are not installed. "
        "Install them with: pip install pyagentic-core[deploy]"
    )

from pyagentic.cli._init import init
from pyagentic.cli._run import run
from pyagentic.cli._build import build
from pyagentic.cli._publish import publish

app = typer.Typer(
    name="pyagentic",
    help="PyAgentic CLI - scaffold, run, build, and publish LLM agents.",
    no_args_is_help=True,
)

app.command()(init)
app.command()(run)
app.command()(build)
app.command()(publish)
