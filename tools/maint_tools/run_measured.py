#!/usr/bin/env python3
"""
Run a command and report its wall time and peak memory, once or several times.

Why this exists
---------------
A benchmark of a submodule (a reader over a folder, an index build, a test
shard) needs the same two numbers every time: how long it took and how much
memory it held at its peak. ``/usr/bin/time`` is not installed everywhere,
reports in different units per platform and cannot repeat a run; this tool
does the same on every POSIX system, repeats, and writes JSON that two runs
can be compared with.

Usage
-----
::

    python -B tools/maint_tools/run_measured.py [--repeat N] [--timeout SECONDS]
        [--json OUT] [--label TEXT] -- COMMAND [ARG ...]

Each run prints ``run i/N: status S, T s, peak M MiB``; the summary line
gives the median time and the largest peak. The exit status is the first
non-zero status of a run (a run that timed out counts as 124, as
``timeout(1)``), else 0.

Notes
-----
**What the peak is.** Per run, the peak resident set size of the command's
process, or of a descendant it waited for if that one was larger (the
kernel's ``ru_maxrss`` for that child, from ``os.wait4``). It is the largest
single process, not the sum of processes alive at the same time; a command
that runs workers in parallel can hold more in total. ``ru_maxrss`` is in
kibibytes on Linux and in bytes on macOS; it is reported in mebibytes here.
On a platform without ``os.wait4`` (Windows) the peak is ``null``.

**Timeout.** With ``--timeout`` the command runs in its own session and the
whole process group is ended (``SIGKILL``) when the time passes, so a stuck
helper process (an OCR engine, a compiler) does not outlive the run.

**Developer notes.** Standard library only; no import of ``scikitplot``
(lesson CI-12). Python 3.8 and newer.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

#: Version of the JSON this tool writes.
FORMAT = 1

#: Exit status recorded for a run that passed ``--timeout``, as timeout(1).
TIMEOUT_STATUS = 124

#: Seconds between checks of a running command when a timeout is set.
_POLL = 0.02


def _exit_status(status: int) -> int:
    """Return the exit status of a ``wait`` status (a signal as ``-signal``)."""
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


def _mib(maxrss: int) -> float:
    """Return ``ru_maxrss`` in mebibytes (kibibytes on Linux, bytes on macOS)."""
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(maxrss / divisor, 1)


def run_once(command: Sequence[str], *, timeout: float | None = None) -> dict[str, Any]:
    """
    Run *command* once; return its status, wall time and peak memory.

    Parameters
    ----------
    command : sequence of str
        Program and arguments (no shell).
    timeout : float or None, optional
        Seconds before the run is ended; ``None`` waits without limit.

    Returns
    -------
    dict
        ``status`` (``TIMEOUT_STATUS`` after a timeout, ``-N`` when ended by
        signal N), ``seconds`` and ``peak_rss_mib`` (``None`` where it
        cannot be measured), ``timed_out``.

    Raises
    ------
    ValueError
        If *command* is empty or *timeout* is not ``None`` or a number > 0.
    FileNotFoundError
        If the program does not exist.
    """
    command = [str(part) for part in command]
    if not command:
        raise ValueError("run_measured: no command given.")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
    ):
        raise ValueError(f"run_measured: timeout must be a number > 0; got {timeout!r}.")
    posix = hasattr(os, "wait4")
    start = time.perf_counter()
    proc = subprocess.Popen(command, start_new_session=posix and timeout is not None)  # noqa: S603
    timed_out = False
    if not posix:
        try:
            status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status, timed_out = TIMEOUT_STATUS, True
        return {
            "status": status,
            "seconds": round(time.perf_counter() - start, 3),
            "peak_rss_mib": None,
            "timed_out": timed_out,
        }
    deadline = None if timeout is None else start + timeout
    while True:
        pid, wait_status, usage = os.wait4(proc.pid, 0 if deadline is None else os.WNOHANG)
        if pid:
            break
        if time.perf_counter() >= deadline:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            pid, wait_status, usage = os.wait4(proc.pid, 0)
            timed_out = True
            break
        time.sleep(_POLL)
    seconds = time.perf_counter() - start
    proc.returncode = _exit_status(wait_status)  # reaped here, not by Popen
    return {
        "status": TIMEOUT_STATUS if timed_out else proc.returncode,
        "seconds": round(seconds, 3),
        "peak_rss_mib": _mib(usage.ru_maxrss),
        "timed_out": timed_out,
    }


def measure(
    command: Sequence[str],
    *,
    repeat: int = 1,
    timeout: float | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """
    Run *command* *repeat* times in a row and summarise.

    Parameters
    ----------
    command : sequence of str
        Program and arguments.
    repeat : int, optional
        Runs, at least 1. Default: 1.
    timeout : float or None, optional
        Per run, as :func:`run_once`. Default: none.
    label : str or None, optional
        Free text stored with the result (what was measured). Default: none.

    Returns
    -------
    dict
        ``format``, ``command``, ``label``, ``runs`` (one :func:`run_once`
        result each), ``median_seconds``, ``max_peak_rss_mib`` (``None``
        when not measured), ``status`` (the first non-zero, else 0),
        ``python`` and ``platform``.

    Raises
    ------
    ValueError
        If *repeat* is not an int of at least 1, or as :func:`run_once`.
    """
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"run_measured: repeat must be an int >= 1; got {repeat!r}.")
    runs = [run_once(command, timeout=timeout) for _ in range(repeat)]
    peaks = [r["peak_rss_mib"] for r in runs if r["peak_rss_mib"] is not None]
    return {
        "format": FORMAT,
        "command": [str(part) for part in command],
        "label": label,
        "runs": runs,
        "median_seconds": round(statistics.median(r["seconds"] for r in runs), 3),
        "max_peak_rss_mib": max(peaks) if peaks else None,
        "status": next((r["status"] for r in runs if r["status"]), 0),
        "python": platform.python_version(),
        "platform": sys.platform,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s [--repeat N] [--timeout S] [--json OUT] [--label TEXT] -- COMMAND ...",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--json", type=Path, default=None, metavar="OUT")
    parser.add_argument("--label", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = measure(command, repeat=args.repeat, timeout=args.timeout, label=args.label)
    except ValueError as exc:
        parser.error(str(exc))
    except FileNotFoundError as exc:
        parser.error(f"cannot run {command[0]!r}: {exc.strerror}")
    total = len(result["runs"])
    for number, run in enumerate(result["runs"], 1):
        peak = "n/a" if run["peak_rss_mib"] is None else f"{run['peak_rss_mib']:g} MiB"
        note = " (timed out)" if run["timed_out"] else ""
        print(  # noqa: T201
            f"run {number}/{total}: status {run['status']}{note},"
            f" {run['seconds']:g} s, peak {peak}",
            file=sys.stderr,
        )
    peak = result["max_peak_rss_mib"]
    print(  # noqa: T201
        f"median {result['median_seconds']:g} s, largest peak"
        f" {'n/a' if peak is None else f'{peak:g} MiB'}",
        file=sys.stderr,
    )
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return result["status"] if 0 <= result["status"] < 256 else 1


if __name__ == "__main__":
    sys.exit(main())
