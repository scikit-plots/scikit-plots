# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
shared options and groups.

The principle here is to define options once, but *not* instantiate them
globally. One reason being that options with action='append' can carry state
between parses. pip parses general options twice internally, and shouldn't
pass on state. To be consistent, all options will follow this design.

.. seealso::
    * https://github.com/pypa/pip/blob/main/src/pip/_internal/cli/cmdoptions.py
"""

from __future__ import annotations

# import logging
from functools import partial

import click

from .. import __version__
from .. import logger as _logger  # ←→ unified logging, logger
from ..externals import _appdirs

# VERSION = lambda: getattr(__import__("scikitplot.version", fromlist=[""]), "__version__", None)

# Application Directories
USER_CACHE_DIR = _appdirs.user_cache_dir("scikitplot")

# Base level is WARNING.
# Each -v increases verbosity ←→ (toward DEBUG).
LOG_LEVELS = {
    0: _logger.WARNING,  # default
    1: _logger.INFO,  # -v
    2: _logger.DEBUG,  # -vv
}
# Each -q decreases verbosity ←→ (toward CRITICAL).
QUIET_LEVELS = {
    1: _logger.ERROR,  # -q
    2: _logger.CRITICAL,  # -qq
}


def set_log_level(ctx, param=None, value=None):
    """
    Set logging level based on -v/-q/-d flags in ctx.params.

    -d (debug) always sets DEBUG
    -v increases verbosity
    -q decreases verbosity

    Quiet overrides verbose if both are set.
    Default is WARNING if no flags provided.
    """
    # import sys
    # sys.stdout.write("Direct stdout\n")
    # sys.stderr.write("Error output\n")
    # Retrieve the flags
    debug = ctx.params.get("debug", False)
    quiet = ctx.params.get("quiet", 0)
    verbose = ctx.params.get("verbose", 0)

    # Base level WARNING
    level = _logger.WARNING

    # Adjust level based on flags
    if debug:
        level = _logger.DEBUG
    elif verbose or quiet:
        # Adjust by verbosity/quiet
        level += quiet * 10  # more quiet → higher number (ERROR > WARNING)
        level -= verbose * 10  # more verbose → lower number (DEBUG < INFO < WARNING)
        # Clamp to valid logging levels
        level = max(_logger.DEBUG, min(_logger.CRITICAL, level))
    else:
        pass

    # Apply to root logger
    if not getattr(_logger, "hasHandlers", True):
        # No handlers yet → configure default logging
        _logger.basicConfig(level=level)
    else:
        # Already configured → just set level
        _logger.setLevel(level)

    # click.echo(f"Changed logging level: {_logger.getLevelName(level)}")
    # click.secho("Success!", fg="green", bold=True)
    # click.secho("Warning!", fg="yellow", underline=True)
    # click.secho("Error!", fg="red", bold=True)
    click.secho(
        f"Changed logging level: {_logger.getLevelName(level)}", fg="green", bold=True
    )

    # console.print("[bold green]Hello, World![/bold green]")
    # console.print("Error occurred!", style="bold red")


# -----------------------------
# ←→ Custom validators and callbacks
# -----------------------------
def validate_python_version(ctx, param, value):
    if not value:
        return None
    parts = value.split(".")
    if len(parts) > 3:  # noqa: PLR2004
        raise click.BadParameter("At most three version parts allowed.")
    try:
        version_info = tuple(int(x) for x in parts)
    except ValueError as e:
        raise click.BadParameter("Each version part must be an integer.") from e
    return version_info


# -----------------------------
# ←→ Reusable Click Option Constants
# -----------------------------
# is_eager=True ensures it's processed first runs before other options.
# expose_value=False avoids passing it to command args; ctx.params['log_level'] can be used internally.
OPTION_HELP = partial(
    click.help_option, "-h", "--help"
)  # help="Show this message and exit."
OPTION_VERSION = partial(click.version_option, __version__, "-V", "--version")

OPTION_DEBUG = partial(
    click.option,
    "-d",
    "--debug",
    is_flag=True,
    help="Enable-Force debug logging mode (sets logging to DEBUG).",
)
OPTION_QUIET = partial(
    click.option,
    "-q",
    "--quiet",
    count=True,
    help="Decrease verbosity to\n(e.g., -q, -qq, -qqq) ←→ (WARNING → ERROR → CRITICAL).",
)
OPTION_VERBOSE = partial(
    click.option,
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity to\n(e.g., -v, -vv, -vvv) ←→ (WARNING → INFO → DEBUG).",
)

OPTION_LOG_FILE = partial(
    click.option, "-l", "--log", type=click.Path(), help="Path to log file."
)
OPTION_NO_COLOR = partial(
    click.option, "-nc", "--no-color", is_flag=True, help="Suppress colored output."
)

PYTHON_VERSION = partial(
    click.option,
    "-py",
    "--python-version",
    callback=validate_python_version,
    help="Specify Python version like 3, 3.7, or 3.7.3.",
)

# -----------------------------
# ←→ Option Group Definitions
# -----------------------------
# ← dict[str, Union[list[callable], dict[str, list[callable]]]]
option_groups = {
    # using the reusable option from above
    "help": [OPTION_HELP],
    "version": [OPTION_VERSION],
    "logging": {
        "level": [OPTION_DEBUG, OPTION_QUIET, OPTION_VERBOSE],
        "file": [OPTION_LOG_FILE],
        "color": [OPTION_NO_COLOR],
    },
    "execution": [
        partial(
            click.option,
            "--timeout",
            type=float,
            default=15,
            metavar="sec",
            help="Set the socket timeout (default 15 seconds).",
        ),
        partial(click.option, "--retries", type=int, default=5, help="Retry attempts."),
        partial(
            click.option,
            "--resume-retries",
            type=int,
            default=5,
            help="Retry count for resuming downloads.",
        ),
    ],
    "environment": [
        partial(
            click.option,
            "--no-input",
            is_flag=True,
            help="Disable prompting for input.",
        ),
        partial(
            click.option,
            "--cache-dir",
            type=click.Path(),
            default="/tmp/cache",  # noqa: S108
            help="Cache directory.",
        ),
        partial(click.option, "--no-cache-dir", is_flag=True, help="Disable cache."),
    ],
    "index_options": [
        partial(
            click.option,
            "-i",
            "--index-url",
            default="https://pypi.org/simple",
            help="Base URL of Python Package Index.",
        ),
        partial(
            click.option,
            "--extra-index-url",
            multiple=True,
            help="Additional index URLs.",
        ),
        partial(click.option, "--no-index", is_flag=True, help="Ignore package index."),
        partial(
            click.option,
            "-f",
            "--find-links",
            multiple=True,
            help="Parse local or HTML file links.",
        ),
    ],
}


# -----------------------------
# ←→ Fully reusable @apply_groups decorator
# -----------------------------
# flatten_group("logging")
def flatten_group(group_name: str) -> list[callable]:
    g = option_groups.get(group_name, [])
    if isinstance(g, dict):
        # Flatten all subgroups
        return [opt for sublist in g.values() for opt in sublist]
    return g


# @apply_options(flatten_group("logging"))
# @apply_options(all_logging_opts)
def apply_options(options: list):
    def decorator(command):
        # Apply options in reverse order to maintain original order
        for opt in reversed(options):
            command = opt()(command)
        return command

    return decorator


# @apply_groups("logging:level", "logging:file", "logging", "execution", "environment")
def apply_groups(*groups: str, list=reversed):
    """
    Apply selectable groups and subgroups to a Click command.

    Usage as decorator:
        @apply_groups("logging:level", "environment")
        def cmd(...): ...

    Usage as direct function call:
        cmd = apply_groups("logging:level", list=list)(cmd)

    Parameters
    ----------
    groups : str
        Names of groups or subgroups to apply. Subgroups separated by ':'.
    list : callable
        Function to iterate over options, e.g., `reversed` or `list`.
    """

    def decorator(command):
        # Apply options in reverse order to maintain original order
        for group in list(groups):
            # Split subgroup by ':', e.g., "logging:level"
            parts = group.split(":")
            g = option_groups.get(parts[0], [])
            if g is None:
                raise ValueError(f"Option group '{parts[0]}' does not exist.")
            if isinstance(g, dict):
                # Apply only the selected subgroup(s)
                # support default subgroup (all if none specified)
                subs = parts[1:] if len(parts) > 1 else g.keys()
                for sub in subs:
                    subgroup_opts = g.get(sub, [])
                    if subgroup_opts is None:
                        raise ValueError(
                            f"Subgroup '{sub}' does not exist in group '{parts[0]}'."
                        )
                    # Apply options in reverse order to maintain original order
                    for opt in list(subgroup_opts):
                        command = opt()(command)
            else:
                # Apply options in reverse order to maintain original order
                for opt in list(g):
                    command = opt()(command)
        return command

    return decorator


# @apply_options(all_logging_opts)
# Flatten all logging options
all_logging_opts = flatten_group("logging")
# Flatten all execution options
all_execution_opts = flatten_group("execution")


######################################################################
## ←→ _wrap_command
######################################################################
def _remove_existing_opts(cmd, names: list[str]):
    """Remove options from a command by name to allow re-adding."""
    cmd.params = [p for p in cmd.params if p.name not in names]
    return cmd


# Recursively apply logging to all subcommands if it's a Group
def _wrap_command(
    cmd,
    groups=("logging:level",),
    list_order=list,
    override=False,
):
    """
    Recursively apply option groups to a Click command or group.

    Avoid duplicates unless override=True.
    """
    # Only wrap commands, not groups again
    if isinstance(cmd, click.Command) and not isinstance(cmd, click.Group):
        # Gather existing option names to avoid re-applying if already exists
        existing_params = {p.name for p in cmd.params if isinstance(p, click.Option)}
        for group_name in groups:
            # Collect option names for potential override
            parts = group_name.split(":")
            g = option_groups.get(parts[0], [])
            g = g.get(parts[1], []) if isinstance(g, dict) else g
            # Check if option already exists by name
            param_name = [getattr(opt, "keywords", {}).get("name") for opt in g]
            if override:
                cmd = _remove_existing_opts(cmd, [n for n in param_name if n])
            if not any(opt in existing_params for opt in param_name):
                cmd = apply_groups(
                    "logging:level",
                    list=list_order,
                )(cmd)
        return cmd
    if isinstance(cmd, click.Group):
        # Recurse into subcommands
        for name, subcmd in cmd.commands.items():
            cmd.commands[name] = _wrap_command(
                subcmd, groups=groups, list_order=list_order, override=override
            )
        return cmd
    return cmd


# -----------------------------
# ←→ CLI Definition
# -----------------------------
# @click.group()
# def cli():
#     """Main command-line interface."""
#     pass

# @cli.command()
# @apply_options(general_options)
# @apply_options(index_options)
# @python_version()
# def install(debug, verbose, version, quiet, log, no_input, retries, timeout, cache_dir, no_cache_dir, no_color, resume_retries,
#             index_url, extra_index_url, no_index, find_links, python_version):
#     """Install packages."""
#     click.echo(f"Debug: {debug}")
#     click.echo(f"Verbose: {verbose}")
#     click.echo(f"Index URL: {index_url}")
#     click.echo(f"Python Version: {python_version}")
#     click.echo("Installation logic goes here.")

# @cli.command()
# @apply_options(general_options)
# def uninstall(debug, verbose, quiet, **kwargs):
#     """Uninstall packages."""
#     click.echo(f"Debug: {debug}")
#     click.echo(f"Verbose: {verbose}")
#     click.echo("Uninstallation logic goes here.")


if __name__ == "__main__":
    # cli()
    pass
