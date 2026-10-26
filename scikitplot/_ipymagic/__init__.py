# scikitplot/_ipymagic/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Notebook frontend of scikit-plots: every ``scikitplot`` command and a profiler as IPython magics.

Load it in Jupyter, JupyterLab, VS Code, Colab or terminal IPython::

    %load_ext scikitplot            # or: %load_ext scikitplot._ipymagic

======================================  ========================================
Magic                                   Does
======================================  ========================================
``%scikitplot ARGS``                    run ``scikitplot ARGS``; returns a
                                        :py:class:`CommandResult`
``%%scikitplot ARGS`` + cell            the same, the cell on standard input
``%scikitplot_profile STMT`` /          profile code; returns a table
``%%scikitplot_profile`` + cell         (:py:class:`ProfileResult`)
``%scikitplot_config key=value``        session settings
                                        (:py:class:`MagicSettings`)
======================================  ========================================

The engine works without IPython, for scripts and agents:

>>> import io
>>> from scikitplot._ipymagic import run_command
>>> run_command("info --format json", out=io.StringIO()).data["scikitplot"][
...     "version"
... ] != ""
True

Notes
-----
**User notes.** ``%scikitplot`` is the command line, in a cell: the same
commands, options, output and exit status as ``scikitplot`` in a terminal
(``%scikitplot --help``), with ``<TAB>`` completion, parsed JSON, errors in
two lines instead of a traceback, and nothing left changed in the kernel. A
mistyped command is refused with the closest names; nothing is guessed.

**Developer notes.** One engine (:py:mod:`._bridge`), one model (the CLI's
registry), thin magics (:py:mod:`._magics`, the only IPython import). This
package imports nothing until a name is used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type checkers and linters (not imported)
    from ._bridge import Route, route, run_command, split_line, wants_json
    from ._completion import complete
    from ._errors import (
        CapabilityMissingError,
        CommandFailedError,
        IpyMagicError,
        MagicUsageError,
        ProfilerBusyError,
    )
    from ._profiling import profile_code, profiler_busy_reason
    from ._results import CommandResult, ProfileResult, ProfileRow
    from ._settings import MagicSettings

__all__ = [
    "CapabilityMissingError",
    "CommandFailedError",
    "CommandResult",
    "IpyMagicError",
    "MagicSettings",
    "MagicUsageError",
    "ProfileResult",
    "ProfileRow",
    "ProfilerBusyError",
    "Route",
    "complete",
    "load_ipython_extension",
    "profile_code",
    "profiler_busy_reason",
    "route",
    "run_command",
    "split_line",
    "unload_ipython_extension",
    "wants_json",
]

_LAZY = {
    **dict.fromkeys(
        ("Route", "route", "run_command", "split_line", "wants_json"), "._bridge"
    ),
    "complete": "._completion",
    **dict.fromkeys(
        (
            "CapabilityMissingError",
            "CommandFailedError",
            "IpyMagicError",
            "MagicUsageError",
            "ProfilerBusyError",
        ),
        "._errors",
    ),
    **dict.fromkeys(("profile_code", "profiler_busy_reason"), "._profiling"),
    **dict.fromkeys(("CommandResult", "ProfileResult", "ProfileRow"), "._results"),
    "MagicSettings": "._settings",
}


def load_ipython_extension(ipython: Any) -> None:
    """
    Register the scikitplot magics (``%load_ext scikitplot._ipymagic``).

    Parameters
    ----------
    ipython : InteractiveShell
        The shell IPython passes.
    """
    from ._magics import load  # noqa: PLC0415 - IPython only when loading

    load(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """
    Remove the magics, stop snakeviz servers, delete temporary profiles.

    Parameters
    ----------
    ipython : InteractiveShell
        The shell IPython passes.
    """
    from ._magics import unload  # noqa: PLC0415

    unload(ipython)


def __getattr__(name: str) -> Any:
    """
    Import a public name on first use.

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    object
        The attribute, cached in the module namespace.

    Raises
    ------
    AttributeError
        For a name the package does not export.
    """
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
