# scikitplot/_ipymagic/_results.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Results of the magics: plain value objects that display themselves.

Notes
-----
**User notes.** ``%scikitplot`` returns a :py:class:`CommandResult`; assign
it (``r = %scikitplot info --format json``) to keep the exit status, the
captured output and, for ``--format json``, the parsed data in ``r.data``.
Shown on its own it is one status line, or the JSON data. ``%scikitplot_profile``
returns a :py:class:`ProfileResult`, shown as a table.

**Developer notes.** Both are frozen dataclasses; ``data`` (any JSON value)
is excluded from equality and hashing, so the objects stay hashable
(Rule 112). Every string put into HTML goes through :py:func:`html.escape`.
"""

from __future__ import annotations

import dataclasses
import html
import json
from typing import Any

__all__ = ["CommandResult", "ProfileResult", "ProfileRow"]


@dataclasses.dataclass(frozen=True)
class CommandResult:  # numpydoc ignore=PR02
    """
    The outcome of one ``scikitplot`` command run from a notebook or Python.

    Parameters
    ----------
    argv : tuple of str
        Arguments given to ``scikitplot`` (without the program name).
    command : str or None
        Registered command that ran (canonical name), None for the root
        (``--help``, ``--version``).
    mode : {"in-process", "subprocess"}
        Where it ran.
    exit_code : int
        Exit status (``124`` when the timeout stopped it; an interrupt
        raises ``KeyboardInterrupt`` instead of returning a result).
    stdout, stderr : str
        Captured output (bounded; see ``truncated``).
    duration : float
        Seconds.
    data : object, default=None
        Parsed stdout of a ``--format json`` run; None otherwise or when
        stdout was not JSON.
    timed_out : bool, default=False
        Stopped because the timeout elapsed.
    truncated : bool, default=False
        Captured output was cut to the capture limit; output shown while
        the command ran was not, output shown after it ends with a note.
    """

    argv: tuple[str, ...]
    command: str | None
    mode: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    data: Any = dataclasses.field(default=None, compare=False, hash=False)
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:  # numpydoc ignore=RT01
        """Whether the command exited with status 0 (bool)."""
        return self.exit_code == 0

    def summary(self) -> str:
        """
        Return the one-line status of the run.

        Returns
        -------
        str
            ``"scikitplot <command>: exit <n> in <t> s (<mode>)"`` plus the
            reason it was stopped, if it was.
        """
        name = " ".join(("scikitplot", self.command or "")).strip()
        why = " (timed out)" if self.timed_out else ""
        return (
            f"{name}: exit {self.exit_code}{why} in {self.duration:.2f} s ({self.mode})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return the result as plain data (JSON-serialisable).

        Returns
        -------
        dict
            Every field; ``argv`` as a list.
        """
        out = dataclasses.asdict(self)
        out["argv"] = list(self.argv)
        out["ok"] = self.ok
        return out

    def __repr__(self) -> str:  # noqa: D105
        return f"<CommandResult {self.summary()}>"

    def _repr_mimebundle_(
        self, include: Any = None, exclude: Any = None
    ) -> dict[str, Any]:
        bundle: dict[str, Any] = {"text/plain": self.summary()}
        if self.data is not None:
            bundle["text/plain"] = json.dumps(
                self.data, indent=2, ensure_ascii=False, default=str
            )
            if isinstance(self.data, (dict, list)):
                bundle["application/json"] = self.data
        return bundle


@dataclasses.dataclass(frozen=True)
class ProfileRow:  # numpydoc ignore=PR02
    """
    One function of a profile.

    Parameters
    ----------
    ncalls : int
        Calls, recursive ones included.
    primitive_calls : int
        Calls that were not recursive.
    tottime : float
        Seconds in the function itself.
    cumtime : float
        Seconds in the function and everything it called.
    file : str
        Source file (``"~"`` for built-ins).
    line : int
        First line.
    function : str
        Function name.
    """

    ncalls: int
    primitive_calls: int
    tottime: float
    cumtime: float
    file: str
    line: int
    function: str

    @property
    def location(self) -> str:  # numpydoc ignore=RT01
        """Where the function is: ``file:line(function)``, as :py:mod:`pstats` writes it."""
        if self.file == "~":
            return self.function
        return f"{self.file}:{self.line}({self.function})"

    def to_dict(self) -> dict[str, Any]:
        """
        Return the row as plain data.

        Returns
        -------
        dict
            Every field.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProfileResult:  # numpydoc ignore=PR02
    """
    A profile of a statement or cell, displayed as a table.

    Parameters
    ----------
    rows : tuple of ProfileRow
        The top functions, sorted by ``sort``.
    total_calls : int
        Function calls recorded.
    total_time : float
        Seconds recorded (the profiler's own clock).
    sort : str
        Sort key (``"cumulative"``, ``"tottime"``, ``"ncalls"``).
    limit : int
        Rows kept.
    functions : int
        Distinct functions recorded (rows beyond ``limit`` included).
    stats_file : str or None, default=None
        Where the :py:mod:`pstats` data was written, if anywhere.
    """

    rows: tuple[ProfileRow, ...]
    total_calls: int
    total_time: float
    sort: str
    limit: int
    functions: int
    stats_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the profile as plain data (JSON-serialisable).

        Returns
        -------
        dict
            Every field; rows as mappings.
        """
        out = dataclasses.asdict(self)
        out["rows"] = [r.to_dict() for r in self.rows]
        return out

    def _header(self) -> str:
        shown = min(self.limit, self.functions)
        text = (
            f"{self.total_calls} calls in {self.total_time:.4f} s;"
            f" {shown} of {self.functions} functions by {self.sort}"
        )
        if self.stats_file:
            text += f"; stats in {self.stats_file}"
        return text

    def to_text(self) -> str:
        """
        Return the profile as a fixed-width table.

        Returns
        -------
        str
            A header line, column titles, one line per row.
        """
        lines = [self._header(), ""]
        lines.append(f"{'ncalls':>12} {'tottime':>10} {'cumtime':>10}  function")
        for r in self.rows:
            calls = (
                f"{r.ncalls}/{r.primitive_calls}"
                if r.ncalls != r.primitive_calls
                else str(r.ncalls)
            )
            lines.append(
                f"{calls:>12} {r.tottime:>10.6f} {r.cumtime:>10.6f}  {r.location}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return f"<ProfileResult {self._header()}>"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(self.to_text())

    def _repr_html_(self) -> str:
        cells = []
        for r in self.rows:
            calls = (
                f"{r.ncalls}/{r.primitive_calls}"
                if r.ncalls != r.primitive_calls
                else str(r.ncalls)
            )
            cells.append(
                "<tr>"
                f"<td style='text-align:right'>{calls}</td>"
                f"<td style='text-align:right'>{r.tottime:.6f}</td>"
                f"<td style='text-align:right'>{r.cumtime:.6f}</td>"
                f"<td style='text-align:left'><code>{html.escape(r.location)}</code></td>"
                "</tr>"
            )
        return (
            f"<div><p>{html.escape(self._header())}</p>"
            "<table><thead><tr><th>ncalls</th><th>tottime (s)</th>"
            "<th>cumtime (s)</th><th style='text-align:left'>function</th></tr>"
            f"</thead><tbody>{''.join(cells)}</tbody></table></div>"
        )
