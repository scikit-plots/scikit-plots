# scikitplot/_testing/_threads.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Run one function in many threads at once, for tests of thread safety.

Notes
-----
**User notes.** :func:`run_threads_together` starts every thread, waits until
all of them are ready, and only then lets them run, optionally with a short
interpreter switch interval so they interleave inside the code under test.
The target reports its own failures (collect them in a list); an exception
that escapes it is reported by pytest as an unhandled thread exception.

**Developer notes.** Threads are started one at a time: thread ``k + 1`` is
started only after thread ``k`` has entered its target. Starting a thread
runs ``sys.settrace`` in it when a tracer is set with
:func:`threading.settrace`, as coverage does. On CPython 3.10 and 3.11 that
call holds a process-wide "being installed" flag while it calls the audit
hooks, and a Python audit hook (this suite installs one,
``_perf_telemetry.py``) can give up the GIL; a second thread starting at that
moment fails with ``RuntimeError: Cannot install a trace function while
another trace function is being installed`` (python/cpython#98257, fixed in
3.12). Serial starts make that overlap impossible; the switch interval is
shortened only after every thread has started.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional

__all__ = ["run_threads_together"]

#: How often a start waits before checking that the new thread is still alive.
#: It bounds the time to detect a thread that died while starting, nothing else.
_POLL_SECONDS = 0.05


def run_threads_together(
    target: Callable[[int], None],
    count: int,
    *,
    switch_interval: Optional[float] = None,
    name: str = "run-together",
) -> None:
    """
    Call ``target(i)`` for ``i`` in ``range(count)``, each in its own thread, at once.

    Parameters
    ----------
    target : callable
        Called once per thread with the thread's index. It should catch and
        record its own failures.
    count : int
        Number of threads, at least 1.
    switch_interval : float or None, default None
        When set, :func:`sys.setswitchinterval` while the threads run, and the
        previous interval restored afterwards; ``None`` leaves it unchanged.
    name : str, default "run-together"
        Thread names are ``f"{name}-{i}"``.

    Returns
    -------
    None
        After every thread has finished.

    Raises
    ------
    TypeError
        When *target* is not callable, *count* is not an int, *switch_interval*
        is not a real number or *name* is not a string.
    ValueError
        When *count* is below 1 or *switch_interval* is not positive.
    RuntimeError
        When a thread ends before entering *target* (it failed while starting);
        the threads already started are released without calling *target*
        and joined first.

    See Also
    --------
    threading.Barrier : Releases threads together, but does not order their starts.

    Notes
    -----
    Thread ``k + 1`` is started only once thread ``k`` is inside its target,
    so no two of these threads start up at the same time (see the module
    notes for why that matters under coverage on CPython 3.10 and 3.11).

    Examples
    --------
    >>> seen = []
    >>> run_threads_together(seen.append, 4)
    >>> sorted(seen)
    [0, 1, 2, 3]
    """
    if not callable(target):
        raise TypeError(f"target must be callable, not {type(target).__name__}")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"count must be an int, not {type(count).__name__}")
    if count < 1:
        raise ValueError(f"count must be at least 1, not {count}")
    if switch_interval is not None:
        if isinstance(switch_interval, bool) or not isinstance(
            switch_interval, (int, float)
        ):
            raise TypeError(
                f"switch_interval must be a number or None, not {type(switch_interval).__name__}"
            )
        if not switch_interval > 0:
            raise ValueError(
                f"switch_interval must be positive, not {switch_interval!r}"
            )
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, not {type(name).__name__}")

    entered = threading.Semaphore(0)
    go = threading.Event()
    cancelled = []  # non-empty once a start failed: released threads return

    def run(index: int) -> None:
        entered.release()
        go.wait()
        if not cancelled:
            target(index)

    threads = []
    try:
        for index in range(count):
            thread = threading.Thread(target=run, args=(index,), name=f"{name}-{index}")
            thread.start()
            threads.append(thread)
            while not entered.acquire(timeout=_POLL_SECONDS):
                if not thread.is_alive() and not entered.acquire(blocking=False):
                    raise RuntimeError(
                        f"thread {thread.name!r} ended before entering its target;"
                        " it failed while starting (see the thread exception above)"
                    )
    except BaseException:
        cancelled.append(True)
        go.set()
        for thread in threads:
            thread.join()
        raise
    previous = sys.getswitchinterval()
    if switch_interval is not None:
        sys.setswitchinterval(switch_interval)
    try:
        go.set()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous)
