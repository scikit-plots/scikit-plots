# scikitplot/_ipymagic/_profiling.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Profile code with :py:mod:`cProfile`, safely, and summarise it as rows.

Notes
-----
**User notes.** Only one profiler can record a Python process at a time.
When another one is active (``python -m cProfile``, a profiling IDE, an
outer ``%%scikitplot_profile``) the profile is refused instead of silently
stopping the other one. The table counts the code you ran; the profiler's
own ``disable`` call is left out.

**Developer notes.** Rule 133: on Python 3.11 a second
``cProfile.Profile().enable()`` replaces the first without an error; on
3.12+ it raises ``ValueError``. The slot is therefore checked before
enabling, never inferred from an error: ``sys.getprofile()`` (3.11 and
older, and any pure-Python profiler) and ``sys.monitoring``'s profiler tool
(3.12+).
"""

from __future__ import annotations

import cProfile
import logging
import os
import pstats
import sys
from typing import Any, Mapping

from ._errors import IpyMagicError, MagicUsageError, ProfilerBusyError
from ._results import ProfileResult, ProfileRow

__all__ = ["SORT_KEYS", "profile_code", "profiler_busy_reason"]

logger = logging.getLogger(__name__)

#: Sort keys and the row field each sorts by (largest first).
SORT_KEYS = {"cumulative": "cumtime", "tottime": "tottime", "ncalls": "ncalls"}

#: Largest number of rows a table may hold.
MAX_LIMIT = 100_000

_OWN = ("~", 0, "<method 'disable' of '_lsprof.Profiler' objects>")


def profiler_busy_reason() -> str | None:
    """
    Return why a profiler cannot be enabled now, or None when it can.

    Returns
    -------
    str or None
        The active profiler, described; None when the slot is free.
    """
    current = sys.getprofile()
    if current is not None:
        return f"a profile function is installed ({current!r})"
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        tool = monitoring.get_tool(monitoring.PROFILER_ID)
        if tool is not None:
            return f"the sys.monitoring profiler slot is taken by {tool!r}"
    return None


def _check(sort: str, limit: int, stats_file: str | None) -> None:
    if sort not in SORT_KEYS:
        raise MagicUsageError(
            f"sort={sort!r} is not a sort key", hint=f"one of {sorted(SORT_KEYS)}"
        )
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not (1 <= limit <= MAX_LIMIT)
    ):
        raise MagicUsageError(f"limit={limit!r} must be an integer 1..{MAX_LIMIT}")
    if stats_file is not None:
        parent = os.path.dirname(os.path.abspath(stats_file))
        if not os.path.isdir(parent):
            raise MagicUsageError(
                f"cannot write the stats to {stats_file}: {parent} does not exist"
            )


def rows_of(stats: pstats.Stats, sort: str, limit: int) -> list[ProfileRow]:
    """
    Return the rows of ``stats``, the largest ``sort`` values first.

    Parameters
    ----------
    stats : pstats.Stats
        Recorded statistics.
    sort : {"cumulative", "tottime", "ncalls"}
        Sort key.
    limit : int
        Rows to keep.

    Returns
    -------
    list of ProfileRow
        Sorted rows; equal values are ordered by location, so the table is
        the same for the same statistics.
    """
    rows = [
        ProfileRow(
            ncalls=int(nc),
            primitive_calls=int(cc),
            tottime=float(tt),
            cumtime=float(ct),
            file=str(file),
            line=int(line),
            function=str(func),
        )
        for (file, line, func), (cc, nc, tt, ct, _callers) in stats.stats.items()
        if (file, line, func) != _OWN
    ]
    field = SORT_KEYS[sort]
    rows.sort(key=lambda r: (-getattr(r, field), r.file, r.line, r.function))
    return rows[:limit]


def profile_code(
    code: Any,
    namespace: Mapping[str, Any],
    local_namespace: Mapping[str, Any] | None = None,
    *,
    sort: str = "cumulative",
    limit: int = 25,
    stats_file: str | None = None,
) -> tuple[ProfileResult, BaseException | None]:
    """
    Run ``code`` under :py:mod:`cProfile` and summarise the profile.

    Parameters
    ----------
    code : code object or str
        What to run (``exec``).
    namespace : mapping
        Global namespace (the notebook's).
    local_namespace : mapping or None, default=None
        Local namespace; ``namespace`` when None.
    sort : {"cumulative", "tottime", "ncalls"}, default="cumulative"
        Sort key of the table.
    limit : int, default=25
        Rows of the table.
    stats_file : str or None, default=None
        Write the :py:mod:`pstats` data there (readable by ``pstats``,
        snakeviz, ``python -m pstats``).

    Returns
    -------
    ProfileResult
        The profile (also of a run that raised, up to the error).
    BaseException or None
        What the code raised, for the caller to re-raise after showing the
        profile; else an ``IpyMagicError`` (code ``"stats_file"``) when the
        stats could not be written; None when all went well.

    Raises
    ------
    ProfilerBusyError
        When another profiler is active (nothing is run).
    MagicUsageError
        For a bad ``sort``, ``limit`` or ``stats_file`` (nothing is run).

    Examples
    --------
    >>> from scikitplot._ipymagic import profile_code
    >>> result, error = profile_code("sum(range(10))", {})
    >>> error is None and result.total_calls >= 1
    True
    """
    _check(sort, limit, stats_file)
    busy = profiler_busy_reason()
    if busy is not None:
        raise ProfilerBusyError(
            f"cannot profile: {busy}",
            hint="stop the other profiler first; two cannot record one process",
        )
    profiler = cProfile.Profile()
    error: BaseException | None = None
    profiler.enable()
    try:
        exec(  # noqa: S102 - running the caller's code is the purpose
            code, namespace, namespace if local_namespace is None else local_namespace
        )
    except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
        error = exc
    finally:
        profiler.disable()
    written = stats_file
    if stats_file is not None:
        try:
            profiler.dump_stats(stats_file)
        except OSError as exc:
            written = None
            if error is None:
                error = IpyMagicError(
                    f"cannot write the stats to {stats_file}: {exc}",
                    code="stats_file",
                    hint="choose a writable path; the table is above",
                )
                error.__cause__ = exc
            else:  # the code's own error is the one to raise
                logger.warning("cannot write the stats to %s: %s", stats_file, exc)
    stats = pstats.Stats(profiler)
    rows = rows_of(stats, sort, limit)
    functions = sum(1 for key in stats.stats if key != _OWN)
    own = stats.stats.get(_OWN)
    total_calls = int(stats.total_calls) - (int(own[1]) if own else 0)
    result = ProfileResult(
        rows=tuple(rows),
        total_calls=total_calls,
        total_time=float(stats.total_tt),
        sort=sort,
        limit=limit,
        functions=functions,
        stats_file=None if written is None else os.path.abspath(written),
    )
    return result, error
