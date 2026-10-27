# scikitplot/_ipymagic/_magics.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The IPython magics: a thin layer over :py:mod:`._bridge` and :py:mod:`._profiling`.

Notes
-----
**Developer notes.** This is the only module that imports IPython. Each
magic parses its line, calls the engine, and returns the engine's result;
behaviour belongs in the engine, where it is usable and testable without
IPython.
"""

from __future__ import annotations

import html
import os
import shutil
import tempfile
from typing import Any

from IPython.core.error import UsageError
from IPython.core.magic import (
    Magics,
    line_cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)
from IPython.display import HTML, IFrame, display

from ._bridge import run_command
from ._completion import complete
from ._errors import IpyMagicError
from ._profiling import SORT_KEYS, profile_code
from ._settings import MagicSettings
from ._snakeviz import require_snakeviz, start_snakeviz, stop_all

__all__ = ["ScikitplotMagics", "load", "unload"]

#: Names ``%load_ext`` accepts for this extension (all load the same magics).
EXTENSION_NAMES = ("scikitplot", "scikitplot._ipymagic", "scikitplot._magic")

#: The magics this extension adds, by kind.
MAGIC_NAMES = {
    "line": ("scikitplot", "scikitplot_config", "scikitplot_profile"),
    "cell": ("scikitplot", "scikitplot_profile"),
}


@magics_class
class ScikitplotMagics(Magics):
    """
    ``%scikitplot``, ``%scikitplot_profile`` and ``%scikitplot_config``.

    Parameters
    ----------
    shell : InteractiveShell or None, default=None
        The IPython shell.
    **kwargs
        Passed to :py:class:`IPython.core.magic.Magics`.
    """

    def __init__(self, shell: Any = None, **kwargs: Any) -> None:  # noqa: D107
        super().__init__(shell=shell, **kwargs)
        self.settings = MagicSettings()
        self._profile_dir: str | None = None

    @line_cell_magic
    def scikitplot(self, line: str, cell: str | None = None) -> Any:
        """
        Run a ``scikitplot`` command; the cell, if any, is its standard input.

        Usage::

            %scikitplot COMMAND [ARGS ...]
            %%scikitplot COMMAND [ARGS ...]
            <text given to the command on standard input>

        Native commands (``info``, ``doctor``, ``show-versions`` ...) run in
        the kernel, isolated; delegated ones (``env``, ``mcp``,
        ``cleanprompt``) and any command with cell input run in a subprocess
        of the kernel's Python, with output shown as it comes, stopped by
        the interrupt button or the ``timeout`` setting. ``--format json``
        returns parsed data. ``{var}`` and ``$var`` expand to Python values.

        Returns a ``CommandResult`` (``.exit_code``, ``.stdout``, ``.data``
        ...); assign it to keep it: ``r = %scikitplot info --format json``.
        See ``%scikitplot_config`` for ``raise_on_error`` and ``timeout``.

        Examples::

            %scikitplot doctor
            r = %scikitplot info --format json
            %scikitplot env plan -n analysis --package "pandas>=2"
            %%scikitplot cleanprompt redact
            My key is sk-live-0123456789abcdef and my email is a@b.example

        Parameters
        ----------
        line : str
            The command and its arguments, as on the command line.
        cell : str or None, default=None
            The cell body (``%%scikitplot``), given to the command on
            standard input; None or blank for no input.

        Returns
        -------
        CommandResult
            Exit status, output, parsed ``--format json`` data, duration.
        """
        stdin = cell if cell is not None and cell.strip() else None
        return run_command(line, stdin=stdin, settings=self.settings)

    @line_magic
    def scikitplot_config(self, line: str) -> MagicSettings:
        """
        Show or change the settings of the scikitplot magics in this session.

        Usage::

            %scikitplot_config
            %scikitplot_config raise_on_error=true timeout=600 isolate=false stream=true

        ``raise_on_error``: a non-zero exit raises (stops "Run all").
        ``timeout``: seconds a subprocess command may run (``none``: no limit).
        ``isolate``: run native commands in a subprocess too.
        ``stream``: show subprocess output as it comes.

        Parameters
        ----------
        line : str
            ``key=value`` pairs; empty to show the settings. All are applied
            or none (a bad pair raises ``MagicUsageError``).

        Returns
        -------
        MagicSettings
            The settings now in force.
        """
        if line.strip():
            self.settings = self.settings.updated(line)
        return self.settings

    @needs_local_scope
    @line_cell_magic
    def scikitplot_profile(
        self, line: str, cell: str | None = None, local_ns: Any = None
    ) -> Any:
        """
        Profile a statement or a cell and show the slowest functions.

        Usage::

            %scikitplot_profile [-s KEY] [-l N] [-D FILE] [--snakeviz] STATEMENT
            %%scikitplot_profile [-s KEY] [-l N] [-D FILE] [--snakeviz]
            <code>
            %scikitplot_profile --stop-snakeviz

        ``-s/--sort``: cumulative (default), tottime or ncalls.
        ``-l/--limit``: rows of the table (default 25).
        ``-D/--dump``: also write the pstats data to FILE.
        ``--snakeviz``: open the profile in snakeviz (``pip install snakeviz``),
        served on 127.0.0.1 for a browser on the kernel's machine.
        ``--stop-snakeviz``: stop the snakeviz servers this session started.

        Returns a ``ProfileResult`` (shown as a table; ``.rows``,
        ``.to_dict()``). Code that raises is profiled up to the error, the
        table is shown, then the error is raised. Refused while another
        profiler records the kernel.

        Parameters
        ----------
        line : str
            Options and, for the line magic, the statement.
        cell : str or None, default=None
            The code to profile (``%%scikitplot_profile``).
        local_ns : dict or None, default=None
            The caller's local namespace (IPython passes it).

        Returns
        -------
        ProfileResult or None
            The profile; None for ``--stop-snakeviz``.
        """
        opts, statement = self.parse_options(
            line,
            "s:l:D:",
            "sort=",
            "limit=",
            "dump=",
            "snakeviz",
            "stop-snakeviz",
            posix=False,
            strict=False,
        )
        if "stop-snakeviz" in opts:
            stopped = stop_all()
            print(f"stopped {stopped} snakeviz server(s)")  # noqa: T201 - asked for
            return None
        sort, limit, stats_file = _profile_options(opts)
        source = statement
        if cell is not None:
            if statement.strip():
                raise UsageError(
                    "in a cell, write the code in the cell, not after the options"
                )
            source = cell
        if not source.strip():
            raise UsageError("nothing to profile")
        if "snakeviz" in opts:
            require_snakeviz()  # before the code runs: its effects are not repeated
        if stats_file is None and "snakeviz" in opts:
            if self._profile_dir is None:
                self._profile_dir = tempfile.mkdtemp(prefix="skplt-profile-")
            fd, stats_file = tempfile.mkstemp(suffix=".prof", dir=self._profile_dir)
            os.close(fd)
        code = compile(
            self.shell.transform_cell(source), "<scikitplot_profile>", "exec"
        )
        result, error = profile_code(
            code,
            self.shell.user_ns,
            local_ns,
            sort=sort,
            limit=limit,
            stats_file=stats_file,
        )
        if error is not None:
            display(result)
            raise error
        if "snakeviz" in opts:
            try:
                server = start_snakeviz(stats_file)
            except IpyMagicError:
                display(result)  # the profile is not lost with the viewer
                raise
            link = html.escape(server.url, quote=True)
            display(HTML(f"<a href='{link}' target='_blank'>open in snakeviz</a>"))
            display(IFrame(server.url, width="100%", height=600))
        return result

    def cleanup(self) -> None:
        """Stop snakeviz servers and remove this session's temporary profiles."""
        stop_all()
        if self._profile_dir is not None:
            shutil.rmtree(self._profile_dir, ignore_errors=True)
            self._profile_dir = None


def _profile_options(opts: Any) -> tuple[str, int, str | None]:
    """
    Read the sort key, row limit and stats file of ``%scikitplot_profile``.

    Parameters
    ----------
    opts : IPython.utils.struct.Struct
        Options from ``Magics.parse_options``.

    Returns
    -------
    sort : str
        A key of ``SORT_KEYS``.
    limit : int
        Rows of the table.
    stats_file : str or None
        The ``-D`` path, unquoted.

    Raises
    ------
    UsageError
        For an unknown sort key or a limit that is not an integer.
    """
    sort = opts.get("s", opts.get("sort", "cumulative"))
    if sort not in SORT_KEYS:
        raise UsageError(f"-s {sort!r}: use one of {sorted(SORT_KEYS)}")
    try:
        limit = int(opts.get("l", opts.get("limit", 25)))
    except ValueError:
        raise UsageError("-l needs an integer") from None
    stats_file = opts.get("D", opts.get("dump"))
    return sort, limit, None if stats_file is None else _unquote(stats_file)


def _unquote(value: str) -> str:
    """
    Remove one pair of matching quotes around an option value.

    Parameters
    ----------
    value : str
        A value as ``parse_options(..., posix=False)`` returns it (quotes kept,
        backslashes untouched, so Windows paths survive).

    Returns
    -------
    str
        ``value`` without its surrounding quotes.
    """
    if value[:1] in ("'", '"') and value[1:].endswith(value[0]):
        return value[1:-1]
    return value


def _complete_hook(shell: Any, event: Any) -> list[str]:
    return complete(getattr(event, "text_until_cursor", None) or event.line)


def load(ipython: Any) -> None:
    """
    Register the magics and the ``%scikitplot`` completer with ``ipython``.

    Parameters
    ----------
    ipython : InteractiveShell
        The shell (``%load_ext`` passes it).

    Notes
    -----
    Idempotent across the extension's names (``scikitplot``,
    ``scikitplot._ipymagic``, ``scikitplot._magic``): a second load keeps
    the registered instance, its settings and its temporary profiles.
    """
    if ScikitplotMagics.__name__ in ipython.magics_manager.registry:
        return
    ipython.register_magics(ScikitplotMagics)
    for key in ("%scikitplot", "%%scikitplot"):
        ipython.set_hook("complete_command", _complete_hook, str_key=key)


def unload(ipython: Any) -> None:
    """
    Remove the magics, stop servers and delete temporary profiles.

    Parameters
    ----------
    ipython : InteractiveShell
        The shell (``%unload_ext`` passes it).
    """
    manager = ipython.magics_manager
    instance = manager.registry.pop(ScikitplotMagics.__name__, None)
    if instance is not None:
        instance.cleanup()
    for kind, names in MAGIC_NAMES.items():
        for name in names:
            manager.magics[kind].pop(name, None)
    dispatcher = getattr(ipython, "strdispatchers", {}).get("complete_command")
    if dispatcher is not None:
        for key in ("%scikitplot", "%%scikitplot"):
            dispatcher.strs.pop(key, None)
    stop_all()
    # Every name is unloaded with the magics, so any name loads them again
    # (IPython itself forgets only the name that was unloaded).
    loaded = getattr(getattr(ipython, "extension_manager", None), "loaded", None)
    if isinstance(loaded, set):
        loaded.difference_update(EXTENSION_NAMES)
