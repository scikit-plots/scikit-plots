# import os

import click
from rich.console import Console

# from .. import _cmdoptions_click

# os.path.splitext(os.path.basename(__file__))[0]
module_name = __name__.split(".")[-1]
console = Console()


@click.command(name=module_name)
# ←→ optional if not using _cmdoptions_click.apply_groups for help
@click.help_option("-h", "--help")
# https://click.palletsprojects.com/en/stable/api/#click.argument
# https://click.palletsprojects.com/en/stable/testing/#file-system-isolation
# @click.argument('f', type=click.File())
@click.argument("name", default="World", required=False)
@click.option(
    "-e/-ne",
    "--emoji/--no-emoji",
    default=True,
    is_flag=True,
    show_default=True,
    required=False,
    help="Add an emoji to the greeting.",
)
# @_cmdoptions_click.apply_groups("logging:level")  # ←→ must come after all options
def cli(name, emoji, **kwargs):
    """
    Greet someone by name `greet World -e`.
    """
    # import sys
    # sys.stdout.write("Direct stdout\n")
    # sys.stderr.write("Error output\n")
    message = f"Hello, {name}!"
    if emoji:
        message += " 👋"

    # console.print("[bold green]Hello, World![/bold green]")
    # console.print("Error occurred!", style="bold red")
    console.print(f"[bold green]{message}[/bold green]")

    # click.echo(f"Changed logging level: {_logger.getLevelName(level)}")
    # click.secho("Success!", fg="green", bold=True)
    # click.secho("Warning!", fg="yellow", underline=True)
    # click.secho("Error!", fg="red", bold=True)
