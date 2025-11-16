import json

import click
from rich.console import Console
from rich.syntax import Syntax

from ... import show_config
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
    help="Output config as JSON instead of text.",
)
@click.option(
    "--yaml/--no-yaml",
    "-y/-ny",
    "as_yaml",  # ←→ parameter name → what you use inside the function.
    is_flag=True,
    help="Output config as YAML (requires PyYAML).",
)
def cli(**kwargs):
    """
    Display Scikit-plot build & runtime configuration information.
    """
    # import sys
    # sys.stdout.write("Direct stdout\n")
    # sys.stderr.write("Error output\n")
    as_json = kwargs.pop("as_json", False)
    as_yaml = kwargs.pop("as_yaml", False)

    if as_json:
        config = show_config(mode="dicts")
        console.print(Syntax(json.dumps(config, indent=2), "json"))
    elif as_yaml:
        try:
            import yaml  # noqa: PLC0415

            config = show_config(mode="dicts")
            yaml_dump = yaml.safe_dump(config, sort_keys=False)
            console.print(Syntax(yaml_dump, "yaml"))
        except ImportError:
            console.print("[bold red]PyYAML is not installed![/bold red]")
    else:
        # Default mode 'stdout' prints the nice rich-format config
        console.print("[bold cyan]Scikit-plot Build Information:[/bold cyan]\n")
        show_config(mode="stdout")
