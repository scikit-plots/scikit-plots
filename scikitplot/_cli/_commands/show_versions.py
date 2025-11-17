import json

import click
from rich.console import Console
from rich.syntax import Syntax

from ... import show_versions
from .. import _cmdoptions_click

module_name = __name__.split(".")[-1]  # Command name derived from filename
console = Console()


@click.command(name=module_name)
@click.help_option("-h", "--help")
@_cmdoptions_click.apply_groups("logging:level")
@click.option(
    "--json/--no-json",
    "-j/-nj",
    "as_json",
    is_flag=True,
    help="Output version info as JSON instead of plaintext.",
)
@click.option(
    "--yaml/--no-yaml",
    "-y/-ny",
    "as_yaml",
    is_flag=True,
    help="Output version info as YAML (requires PyYAML).",
)
@click.option(
    "--rich/--no-rich",
    "-r/-nr",
    "as_rich",
    is_flag=True,
    help="Output version info using rich formatting (requires rich).",
)
def cli(**kwargs: dict[str, any]) -> None:
    """
    Display scikit-plot version and environment info for debugging.

    Outputs can be formatted as JSON, YAML, or rich console output.
    Otherwise, plain text is shown.
    """
    as_json = kwargs.pop("as_json", False)
    as_yaml = kwargs.pop("as_yaml", False)
    as_rich = kwargs.pop("as_rich", False)

    # Get the raw version info (printed to console or dict depending on mode)
    if as_json:
        version_info = show_versions(mode="dict")
        console.print(Syntax(json.dumps(version_info, indent=2), "json"))
    elif as_yaml:
        version_info = show_versions(mode="yaml")
        if version_info:
            console.print(Syntax(version_info, "yaml"))
    elif as_rich:
        show_versions(mode="rich")
    else:
        console.print("[bold cyan]Scikit-plot Version Information:[/bold cyan]\n")
        show_versions()  # Default behavior prints to stdout
