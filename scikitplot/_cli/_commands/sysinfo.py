import json
import os
import platform
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ... import logger as _logger
from .. import _cmdoptions_click

# os.path.splitext(os.path.basename(__file__))[0]
module_name = __name__.split(".")[-1]
console = Console()


@click.command(name=module_name)
# ←→ optional if not using _cmdoptions_click.apply_groups for help
@click.help_option("-h", "--help")
@_cmdoptions_click.apply_groups(
    "logging:level"
)  # ←→ Logging applied last (proper order)
@click.option(
    "--json/--no-json",
    "-j/-nj",
    "as_json",  # ←→ parameter name → what you use inside the function.
    is_flag=True,
    help="Display output in JSON format instead of a table.",
)
def cli(**kwargs):
    """Show system information."""
    # import sys
    # sys.stdout.write("Direct stdout\n")
    # sys.stderr.write("Error output\n")
    as_json = kwargs.pop("as_json", False)

    info = {
        "os": platform.system(),
        "version": platform.version(),
        "python": platform.python_version(),
        "exe": sys.executable,
        "cwd": os.getcwd(),
    }

    if as_json:
        # For structured programmatic use
        console.print(json.dumps(info, indent=2))
    else:
        # Nicely formatted table for humans
        table = Table(
            title="System Info",
            title_style="bold cyan",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Property", style="dim")
        table.add_column("Value")
        for key, value in info.items():
            table.add_row(key.capitalize(), value)
        console.print("\n")
        # console.print(table)
        # console.print("[bold green]Hello, World![/bold green]")
        # console.print("Error occurred!", style="bold red")
        console.print(
            Panel(
                table,
                title="CLI Environment",
                title_align="center",
                border_style="green",
            )
        )

    # Example logs at various levels
    _logger.debug("Debug message for diagnostics.")
    _logger.info("System info displayed successfully.")

    # click.echo(f"Changed logging level: {_logger.getLevelName(level)}")
    # click.secho("Success!", fg="green", bold=True)
    # click.secho("Warning!", fg="yellow", underline=True)
    # click.secho("Error!", fg="red", bold=True)
