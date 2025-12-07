# ruff: noqa: PLC0415
# ruff: noqa: F401
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=undefined-variable
# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# pylint: disable=unreachable

# This module was copied from the mlflow project.
# https://github.com/mlflow/mlflow/blob/master/mlflow/cli.py

r"""
cli.py.

.. seealso::
    * https://pocoo-click.readthedocs.io/en/latest/commands/
    * https://github.com/mlflow/mlflow/blob/master/mlflow/cli.py
    * https://github.com/python/cpython/blob/main/Lib/getopt.py

Examples
--------
!python -VV

# Python prints -VV, by os
os.system("python -VV")
with os.popen("python -VV") as f: output=f.read().strip()

# Python prints -VV, by subprocess
output = subprocess.Popen(["python", "-VV"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).communicate()[0].strip()
output = subprocess.run(["python", "-VV"], capture_output=True, text=True).stdout.strip()
output = subprocess.check_output("python -VV", shell=True, text=True).strip()
print(output)

# Python prints
import os, platform, subprocess, sys
print("Implementation\t:", platform.python_implementation(), sys.implementation.name)
print("Version\t\t:", sys.version)
print("Compiler\t:", platform.python_compiler())

print(platform.system())      # Same as platform_system
print(platform.platform())    # Shows musl / glibc details
print(platform.libc_ver())    # Helps detect musllinux (musl vs glibc)
"""

from __future__ import annotations

# https://docs.python.org/3/library/optparse.html
# getopt -> optparse -> argparse
import argparse
import contextlib
import errno
import importlib
import itertools
import json
import os
import pathlib
import pkgutil
import re
import sys
import tempfile
import textwrap
import traceback
import warnings
from datetime import timedelta
from functools import partial

import click

# import logging
from .. import __version__
from .. import logger as _logger
from .._utils import cli_args
from .._utils.logging_utils import eprint
from .._utils.os import is_windows
from .._utils.process import ShellCommandException
from ..environment_variables import SKPLT_EXPERIMENT_ID, SKPLT_EXPERIMENT_NAME
from ..exceptions import InvalidUrlException, ScikitplotException
from . import _cmdoptions_click

INVALID_PARAMETER_VALUE = 0

__all__ = [
    "cli",
]


######################################################################
## ←→ Accept aliased command.
######################################################################
class AliasedGroup(click.Group):
    """Accept aliased command."""

    def get_command(self, ctx, name):
        """get_command."""
        # `scikitplot ui` is an alias for `scikitplot server`
        name = "server" if name == "ui" else name
        try:
            return super().get_command(ctx, name)
        except Exception:
            # Load only when the command is invoked
            # mod = importlib.import_module(f"scikitplot._cli._commands.{name}")
            # print(mod)
            # return getattr(mod, "cli", None)
            pass


######################################################################
## ←→ Main Entry-point: cli()
######################################################################
@click.group(
    # cls=click.Group,
    cls=AliasedGroup,
    # flag on the main CLI entrypoint and handle the case when no subcommand is provided.
    invoke_without_command=True,
    # help="Scikit-plots main CLI entrypoint helper."
)
# https://click.palletsprojects.com/en/stable/api/#click.help_option
@click.help_option(
    # param_decls (only positional)
    "--help",
    "-h",
    # "help",  # ←→ parameter name → what you use inside the function.
)
# https://click.palletsprojects.com/en/stable/api/#click.version_option
# Automatically provides --version (or -v) as a command-line flag
# to print the program version and exit.
@click.version_option(
    __version__,  # version
    # param_decls (only positional)
    "--version",
    "-V",
    # "version",  # ←→ parameter name → what you use inside the function.
    # package_name="scikitplot",  # optional, use version from installed `scikitplot`
    # prog_name="scikit-plots",  # optional, default display name in the version message
    # The message to show.
    # The values %(prog)s, %(package)s, and %(version)s are available.
    # Defaults to "%(prog)s, version %(version)s".
    # message="%(prog)s, %(package)s, version %(version)s",
)
# https://click.palletsprojects.com/en/stable/api/#click.option
# https://click.palletsprojects.com/en/stable/api/#click.Option
# ← prevent redefining option
# ⚠️ Redefining is a name conflict — click.version_option() reserves (--version, -V),
# so you cannot reuse --version or -V for another purpose.
# @click.help_option("-h", "--help")  # optional if not using apply_groups for help
@_cmdoptions_click.apply_groups(  # <-- must come after all options
    # "help",
    "logging:level"
)
# No default needed for ctx, since Click injects it at runtime.
@click.pass_context
def cli(ctx: click.Context | None = None, **kwargs: dict) -> any:
    """Scikit-plots main CLI entrypoint helper."""
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    # ctx and ctx.ensure_object(dict)
    # ctx.obj["DEBUG"] = debug

    # Optionally handle logging or version stuff here
    _cmdoptions_click.set_log_level(ctx)

    # if kwargs.pop('version'):
    #     click.echo("scikitplot version 1.0.0")  # Replace with dynamic version lookup
    #     ctx.exit()

    # Optional: Handle --debug, -v, -q without requiring command
    # if kwargs.pop('debug'):
    #     click.echo("Debug logging enabled.")

    # Important: Handle no subcommand
    # if ctx.invoked_subcommand is None:
    #     click.echo("No command provided. Showing system info by default...\n")
    #     ctx.invoke(sysinfo)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


######################################################################
## ←→ _load_commands
######################################################################
# @cli.command("command-name")
# --- Auto-discover and register subcommands dynamically ---
def _load_commands():
    from . import _commands  # noqa: PLC0415

    # Discover all modules in _commands
    for _, mod_name, _ in pkgutil.iter_modules(_commands.__path__):
        module = importlib.import_module(f"scikitplot._cli._commands.{mod_name}")
        if not hasattr(module, "cli"):
            continue

        # Wrap each command with apply_groups
        sub_cli = module.cli
        # Wrap all commands and subcommands with logging options before registering
        # sub_cli = _cmdoptions_click._wrap_command(
        #     sub_cli, groups=("logging:level",), list_order=list, override=False
        # )
        # Add the sub_cli (command or group) to the main CLI
        # def register_command(group, command):
        #     group.add_command(command)
        cli.add_command(sub_cli)


_load_commands()


######################################################################
## ←→ Add a COMMAND to Entry-point: doctor()
######################################################################
@cli.command(
    short_help=textwrap.dedent(
        """
        Prints out useful information for debugging issues
        with Scikit-plots."""
    )
)
# @click.help_option("-h", "--help")  # optional if not using apply_groups for help
@_cmdoptions_click.apply_groups(
    "help", "logging:level"
)  # <-- must come after all options
@click.option(
    "--mask-envs",
    is_flag=True,
    help=(
        "If set (the default behavior without setting this flag is not to obfuscate information), "
        'mask the Scikit-plots environment variable values (e.g. `"SKPLT_ENV_VAR": "***"`) '
        "in the output to prevent leaking sensitive information."
    ),
)
@click.pass_context
def doctor(ctx, **kwargs):
    """
    Doctor.
    """
    _cmdoptions_click.set_log_level(ctx)
    mask_envs = kwargs.pop("mask_envs")
    # raise NotImplementedError
    click.echo("Currently Not Implemented. Just test log level...")
    click.echo(
        f"{mask_envs}",
    )
    # scikitplot.doctor(mask_envs)

    # Example log messages
    _logger.debug("This is a DEBUG log.")
    _logger.info("This is an INFO log.")
    _logger.warning("This is a WARNING log.")
    _logger.error("This is an ERROR log.")
    _logger.critical("This is a CRITICAL log.")


######################################################################
## ←→ Add a COMMAND to Entry-point: st()
######################################################################
@cli.command(
    short_help="Launch the Streamlit app with the provided configuration options.",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def st2(argv: list[str] | None = None) -> None:
    """
    Launch the Streamlit app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.streamlit.run_ui_app_st import run_ui_app_st

    run_ui_app_st(argv)


@cli.command()
@click.option(
    "--file_path",
    default="template_ui_app_st.py",
    help="Streamlit app file, default 'template_ui_app_st.py'.",
)
@click.option(
    "--address",
    "-a",
    default="localhost",
    help="Streamlit Host address, default 'localhost'.",
)
@click.option("--port", "-p", default="8501", help="Streamlit Port, default '8501'.")
@click.option(
    "--dark_theme",
    "-d",
    is_flag=True,
    default=False,
    help="Streamlit Enable dark theme",
)
@click.option(
    "--lib_sample",
    "-ls",
    is_flag=True,
    default=True,
    help="Use lib sample app, default True.",
)
def st(file_path, address, port, dark_theme, lib_sample):
    """
    Launch the Streamlit app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.streamlit.run_ui_app_st import launch_streamlit  # run_ui_app_st

    try:
        click.echo(f"Launching {file_path} on port {port}")
        launch_streamlit(
            file_path=file_path,
            address=address,
            port=port,
            dark_theme=dark_theme,
            lib_sample=lib_sample,
        )
    except ShellCommandException:
        eprint(
            "Running the scikitplot streamlit UI app failed. "
            "Please see the logs above for details."
        )
        sys.exit(1)


######################################################################
## ←→ Add a COMMAND to Entry-point: gr()
######################################################################
@cli.command()
@click.option(
    "--share",
    "-s",
    is_flag=True,
    default=True,
    help="Gradio public link serve.",
)
def gr(share):
    """
    Launch the gradio app with the provided configuration options.

    For Docker or WSL 2 environments::

        # Inside Docker: set address to "0.0.0.0" instead of "localhost"

        python -m scikitplot.streamlit.run_app st --address 0.0.0.0

        # (optionally persist) ~/.wslconfig

        localhostforwarding=true
    """
    from ._ui_app.gradio.template_ui_app_gr import ui_app_gr

    try:
        ui_app_gr.launch(share=True)
    except ShellCommandException:
        eprint(
            "Running the scikitplot gradio UI app failed. "
            "Please see the logs above for details."
        )
        sys.exit(1)


######################################################################
## ←→ Add a COMMAND to Entry-point: from defined py
######################################################################
with contextlib.suppress(
    AttributeError,
    ModuleNotFoundError,
    NameError,
):
    import scikitplot

    cli.add_command(scikitplot.runs.commands)  # noqa: F821
    cli.add_command(scikitplot.db.commands)  # noqa: F821

# We are conditional loading these commands since the skinny client does
# not support them due to the pandas and numpy dependencies of Scikit-plots Models
with contextlib.suppress(
    ImportError,
):
    import scikitplot

    from .gateway import cli  # type: ignore  # noqa: PGH003

    cli.add_command(scikitplot.gateway.cli.commands)


if __name__ == "__main__":
    cli()
