# scikitplot/cython/_budget.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build resource budgets: deadlines and bounded output (CYTHON-RES-001).

This module supplies the *in-process* portion of the resource-budget contract:
a build deadline enforced by a watchdog thread, and a bounded output buffer that
caps captured compiler logs.  These prevent an unbounded compile from blocking a
caller indefinitely or accumulating unbounded log memory.

Scope and limitations
---------------------
Compilation runs in-process (``cythonize`` + ``build_ext``), which spawns the C
compiler as a subprocess deep inside distutils.  A watchdog can therefore
enforce a *deadline observed by the caller* and stop waiting, but it cannot, by
itself, guarantee OS-level termination of a compiler subprocess or its
grandchildren.  Hard process-group termination, disk quotas, and a cleanup
journal require a subprocess-based ``BuildExecutor`` and are intentionally left
as a documented follow-up rather than being faked here.

Notes
-----
- User-focused: pass ``build_timeout_s`` to bound how long a build may run.
- Developer-focused: :func:`run_with_deadline` runs a callable on a worker
  thread and raises :class:`BuildTimeoutError` if the deadline elapses; the
  worker is left as a daemon so it cannot keep the interpreter alive.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, TypeVar

__all__ = [
    "BoundedBuffer",
    "BuildBudget",
    "BuildDiagnostic",
    "BuildTimeoutError",
    "run_with_deadline",
]

_T = TypeVar("_T")


class BuildTimeoutError(TimeoutError):
    """Raised when a build exceeds its :class:`BuildBudget` deadline."""


@dataclass(frozen=True)
class BuildBudget:
    """Enforceable resource budget for a single build.

    Parameters
    ----------
    compile_timeout_s : float or None, default=None
        Maximum wall-clock seconds the compile step may run before a
        :class:`BuildTimeoutError` is raised to the caller.  ``None`` disables
        the deadline.
    max_output_bytes : int, default=1_048_576
        Maximum number of bytes of captured compiler output retained.  Output
        beyond this cap is dropped (oldest-first) so logs cannot grow without
        bound.

    Raises
    ------
    ValueError
        If ``compile_timeout_s`` is not positive, or ``max_output_bytes`` <= 0.
    """

    compile_timeout_s: float | None = None
    max_output_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if self.compile_timeout_s is not None and self.compile_timeout_s <= 0:
            raise ValueError("compile_timeout_s must be > 0 or None")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be > 0")


class BoundedBuffer:
    """A write-only text buffer that retains at most ``max_bytes`` bytes.

    Keeps the most recent output (tail), which is what matters for diagnosing a
    failure.  Thread-safe for concurrent writes.

    Parameters
    ----------
    max_bytes : int
        Maximum retained size in bytes; must be > 0.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._max = max_bytes
        self._chunks: list[str] = []
        self._size = 0
        self._lock = threading.Lock()
        self.truncated = False

    def write(self, s: str) -> int:
        n = len(s)
        with self._lock:
            self._chunks.append(s)
            self._size += n
            # Drop from the front until within budget (retain the tail).
            while self._size > self._max and len(self._chunks) > 1:
                dropped = self._chunks.pop(0)
                self._size -= len(dropped)
                self.truncated = True
            # A single oversized chunk is tail-trimmed in place.
            if self._size > self._max and self._chunks:
                only = self._chunks[0]
                self._chunks[0] = only[-self._max :]
                self._size = len(self._chunks[0])
                self.truncated = True
        return n

    def flush(self) -> None:  # pragma: no cover - stream protocol
        pass

    def getvalue(self) -> str:
        with self._lock:
            text = "".join(self._chunks)
        if self.truncated:
            return "[...output truncated to budget...]\n" + text
        return text


def run_with_deadline(
    func: Callable[[], _T],
    *,
    timeout_s: float | None,
    what: str = "build",
) -> _T:
    """Run ``func`` on a worker thread, enforcing a wall-clock deadline.

    Parameters
    ----------
    func : callable
        Zero-argument callable performing the work.
    timeout_s : float or None
        Deadline in seconds; ``None`` runs ``func`` inline with no watchdog.
    what : str, default="build"
        Label used in the timeout message.

    Returns
    -------
    Any
        Whatever ``func`` returns.

    Raises
    ------
    BuildTimeoutError
        If ``func`` does not complete within ``timeout_s``.
    BaseException
        Any exception raised by ``func`` is re-raised in the calling thread.

    Notes
    -----
    The worker is a daemon thread: if the deadline elapses it is abandoned (it
    cannot block interpreter shutdown).  In-process compilation cannot be force
    -killed from Python, so on timeout the caller stops waiting and surfaces the
    error; hard subprocess termination is a documented follow-up (see module
    docstring).
    """
    if timeout_s is None:
        return func()

    result: list[_T] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(func())
        except BaseException as e:  # noqa: BLE001 - propagate to caller thread
            error.append(e)

    worker = threading.Thread(target=_target, name=f"cython-{what}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise BuildTimeoutError(
            f"{what} exceeded its {timeout_s:g}s deadline and was abandoned"
        )
    if error:
        raise error[0]
    return result[0]


@dataclass(frozen=True)
class BuildDiagnostic:
    """Typed diagnostic record for a build phase (CYTHON-OBS-001).

    Instead of a bare ``RuntimeError`` string, build failures carry a structured
    record: which phase failed, for which module, the tool versions in play, an
    exit status when known, and a bounded tail of the captured log (plus a path
    to the retained full log when one was written).

    Parameters
    ----------
    phase : str
        The build phase: ``"cythonize"``, ``"build_ext"``, ``"link"``, etc.
    module : str
        The module (or package) name being built.
    status : int or None, default=None
        Process/exit status when known (e.g. a ``build_ext`` exit code).
    command : tuple of str, default=()
        Command tokens for the failing step, when available.
    tool_versions : Mapping[str, str], default empty
        Relevant tool versions (Python, Cython, compiler), when available.
    log_tail : str, default=""
        A bounded tail of the captured compiler/Cython output.
    log_path : str or None, default=None
        Path to the retained full log on disk, if one was written.

    Notes
    -----
    Attached to the raised exception as ``exc.diagnostic`` so callers can branch
    on structured fields; the exception message is preserved for compatibility.
    """

    phase: str
    module: str
    status: int | None = None
    command: tuple[str, ...] = ()
    tool_versions: Mapping[str, str] = field(default_factory=dict)
    log_tail: str = ""
    log_path: str | None = None
