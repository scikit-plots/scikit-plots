# scikitplot/annoy/_threads.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
How many threads build an index: one rule, decided at run time.

An index is built by ``build``, ``fit``, ``fit_transform`` or ``rebuild``, each
of which takes ``n_jobs``. Whether ``n_jobs`` can have any effect depends on
how the compiled module was built, which the caller cannot see. This module
is the one place that decides, from three facts:

* what the caller asked for (``n_jobs``, or the index's stored parameter);
* whether the compiled module contains the multithreaded code
  (:func:`threads_compiled`);
* the mode, from the environment variable ``SKPLT_ANNOY_THREADS``
  (:func:`threads_mode`).

Modes
-----
``auto`` (the default)
    Threads are used when the caller names a number above one *and* the
    module has them. Otherwise the build runs on one thread. Nothing is
    turned on that was not asked for: a build without ``n_jobs``, or with
    ``n_jobs=-1``, runs on one thread, exactly as it does in a module without
    threads.
``single``
    Every build runs on one thread, whatever ``n_jobs`` says. For results
    that are the same on every machine and from every wheel.
``multi``
    ``n_jobs`` is honoured, and ``-1`` (or nothing) means every CPU. A module
    without threads refuses with an error instead of quietly running on one.

Notes
-----
**User.** One wheel serves all three modes when it was compiled with threads
(the nightly wheels are; see the package description for how to get or build
one). ``threads_info()`` says what the installed one can do::

    >>> from scikitplot.annoy import threads_info
    >>> sorted(threads_info())
    ['compiled', 'cpu_count', 'env', 'mode', 'modes']

Each thread seeds its own trees, so an index built on two threads holds other
trees than one built on one thread, and answers can differ in the approximate
tail. With one thread the index is the same in every wheel: a module with
threads builds on the calling thread then, by the same statements as a module
without them.

**Developer.** Measured on Linux x86_64, 2 CPUs, 60 000 vectors of 64
dimensions, 24 trees: 4.2 to 4.7 s on one thread in either kind of module,
2.15 s with ``n_jobs=2`` in a module with threads.

The compiled ``build`` reads ``n_jobs=-1`` as "use the index's stored
parameter", whose own default is ``-1``, and that ran on one thread. The
documented meaning "all CPUs" is therefore given here, in ``multi`` mode only,
by passing the CPU count explicitly; in ``auto`` an explicit ``1`` is passed,
so the outcome does not depend on that reading. The compiled methods store
the ``n_jobs`` they are called with; after a call in which another number was
passed than the caller's, the caller's is put back, so parameters and metadata
keep saying what was asked for, not what this machine resolved it to.
"""

from __future__ import annotations

import functools
import logging
import operator
import os
from typing import Any, Callable, Mapping

from ..cexternals._annoy import annoylib
from ..environment_variables import EnvironmentVariable

logger = logging.getLogger(__name__)

__all__ = [
    "MODES",
    "SKPLT_ANNOY_THREADS",
    "resolve_n_jobs",
    "threads_compiled",
    "threads_info",
    "threads_mode",
]

#: The modes, in the order they are documented. The first is the default.
MODES: tuple[str, ...] = ("auto", "single", "multi")

#: Environment variable that selects the mode for the whole process.
SKPLT_ANNOY_THREADS = EnvironmentVariable("SKPLT_ANNOY_THREADS", str, MODES[0])

#: ``n_jobs`` value that means "not a number of threads; decide for me".
_ALL = -1

#: Whether the notice about a module without threads has been logged. It is
#: logged once per process: the fact does not change while the process runs.
_notice = {"logged": False}


def threads_compiled() -> bool:
    """
    Return whether the compiled module contains the multithreaded build.

    Returns
    -------
    bool
        ``True`` when ``annoylib.MULTITHREADED_BUILD`` is 1. ``False`` when it
        is 0, and also when the module has no such constant (it was compiled
        before the constant existed, and then nothing is known about it).

    See Also
    --------
    threads_info : This fact together with the mode and the CPU count.
    """
    return bool(getattr(annoylib, "MULTITHREADED_BUILD", 0))


def threads_mode(environ: Mapping[str, str] | None = None) -> str:
    """
    Return the mode named by ``SKPLT_ANNOY_THREADS``.

    Parameters
    ----------
    environ : mapping, optional
        The environment; ``os.environ`` when omitted.

    Returns
    -------
    {"auto", "single", "multi"}
        ``"auto"`` when the variable is unset or empty.

    Raises
    ------
    ValueError
        If the variable is set to anything else. A misspelled mode is not
        read as the default: the caller asked for something specific.

    Examples
    --------
    >>> threads_mode({})
    'auto'
    >>> threads_mode({"SKPLT_ANNOY_THREADS": " Single "})
    'single'
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(SKPLT_ANNOY_THREADS.name, "")
    value = raw.strip().lower()
    if not value:
        return MODES[0]
    if value not in MODES:
        raise ValueError(
            f"{SKPLT_ANNOY_THREADS.name}={raw!r} is not a mode. "
            f"Set it to one of {', '.join(MODES)}, or unset it for {MODES[0]!r}."
        )
    return value


def _cpu_count() -> int:
    """Return the number of CPUs this process may use; at least 1."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def resolve_n_jobs(
    n_jobs: int | None,
    *,
    mode: str | None = None,
    compiled: bool | None = None,
    cpu_count: int | None = None,
) -> int:
    """
    Return the number of threads a build will run on.

    Parameters
    ----------
    n_jobs : int or None
        What was asked for: a positive number of threads, or ``-1`` or
        ``None`` for "no particular number".
    mode : {"auto", "single", "multi"}, optional
        The mode; :func:`threads_mode` when omitted.
    compiled : bool, optional
        Whether the module has threads; :func:`threads_compiled` when omitted.
    cpu_count : int, optional
        The number of CPUs to use for "all"; the CPUs this process may run on
        when omitted.

    Returns
    -------
    int
        At least 1.

    Raises
    ------
    TypeError
        If ``n_jobs`` is not an integer or ``None``.
    ValueError
        If ``n_jobs`` is zero or below ``-1``, or ``mode`` is not a mode.
    RuntimeError
        In ``multi`` mode when the module has no threads. The message says
        how to get a module that has them, and how to leave the mode.

    Notes
    -----
    **User.** The whole rule:

    ========  =====================  ===========================  =============
    mode      asked for              module with threads          without
    ========  =====================  ===========================  =============
    auto      ``N`` above 1          ``N``                        1, logged once
    auto      1, -1, None            1                            1
    single    anything               1                            1
    multi     ``N``                  ``N``                        error
    multi     -1, None               every CPU                    error
    ========  =====================  ===========================  =============

    Examples
    --------
    >>> resolve_n_jobs(4, mode="auto", compiled=True)
    4
    >>> resolve_n_jobs(-1, mode="auto", compiled=True)
    1
    >>> resolve_n_jobs(4, mode="single", compiled=True)
    1
    >>> resolve_n_jobs(None, mode="multi", compiled=True, cpu_count=8)
    8
    >>> resolve_n_jobs(4, mode="auto", compiled=False)
    1
    """
    if n_jobs is None:
        n_jobs = _ALL
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError(
            f"n_jobs must be an integer or None; got {type(n_jobs).__name__}"
        )
    if n_jobs == 0 or n_jobs < _ALL:
        raise ValueError(
            f"n_jobs must be a positive number of threads, or -1; got {n_jobs}"
        )
    mode = threads_mode() if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    compiled = threads_compiled() if compiled is None else bool(compiled)

    if mode == "single":
        return 1
    if mode == "multi":
        if not compiled:
            raise RuntimeError(
                f"{SKPLT_ANNOY_THREADS.name}=multi asks for a multithreaded "
                "build, and the installed scikitplot.annoy was compiled "
                "without threads. Install a wheel that has them (the nightly "
                "wheels do), or compile one with SKPLT_BUILD_THREADS=1; or "
                f"set {SKPLT_ANNOY_THREADS.name} to auto or single to build "
                "on one thread."
            )
        if n_jobs == _ALL:
            return _cpu_count() if cpu_count is None else max(1, int(cpu_count))
        return n_jobs
    # auto
    if n_jobs > 1 and compiled:
        return n_jobs
    if n_jobs > 1 and not _notice["logged"]:
        _notice["logged"] = True
        logger.warning(
            "n_jobs=%d was asked for, and the installed scikitplot.annoy was "
            "compiled without threads: the index is built on one thread. "
            "Install a wheel with threads (the nightly wheels have them) or "
            "compile one with SKPLT_BUILD_THREADS=1 to use more. This is "
            "logged once; scikitplot.annoy.threads_info() shows the state.",
            n_jobs,
        )
    return 1


def threads_info() -> dict[str, Any]:
    """
    Describe what decides the number of threads in this process.

    Returns
    -------
    dict
        ``compiled``
            Whether the compiled module has the multithreaded build.
        ``mode``
            The mode in effect (:func:`threads_mode`).
        ``modes``
            Every mode, the default first.
        ``env``
            The name of the environment variable that selects the mode.
        ``cpu_count``
            The number of CPUs ``multi`` uses for ``n_jobs=-1``.

    Raises
    ------
    ValueError
        If the environment variable holds something that is not a mode.

    Examples
    --------
    >>> info = threads_info()
    >>> info["env"], info["modes"]
    ('SKPLT_ANNOY_THREADS', ('auto', 'single', 'multi'))
    """
    return {
        "compiled": threads_compiled(),
        "mode": threads_mode(),
        "modes": MODES,
        "env": SKPLT_ANNOY_THREADS.name,
        "cpu_count": _cpu_count(),
    }


def threaded(
    backend_method: Callable[..., Any], *, position: int | None = None
) -> Callable[..., Any]:
    """
    Wrap a compiled method that takes ``n_jobs`` so that the rule applies.

    Parameters
    ----------
    backend_method : callable
        The unbound compiled method (``Annoy.build`` and the like).
    position : int, optional
        Index of ``n_jobs`` among the positional arguments after ``self``,
        for a method that accepts it by position; ``None`` when it is
        keyword-only.

    Returns
    -------
    callable
        A method with the compiled method's name and documentation.

    Notes
    -----
    **Developer.** In a module without threads the call is passed on exactly
    as it came (after the rule has had the chance to refuse ``multi``), so
    nothing changes there. In a module with threads the resolved number is
    passed, and the index's stored ``n_jobs`` is put back to what the caller
    asked for when the two differ (see the module notes).
    """

    @functools.wraps(backend_method)
    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        by_position = position is not None and len(args) > position
        given = args[position] if by_position else kwargs.get("n_jobs")
        stored = self.get_params().get("n_jobs", _ALL)
        asked = stored if given is None else given
        try:
            asked = int(operator.index(asked))
        except TypeError:
            # Not a number: the compiled method says so, in its own words.
            return backend_method(self, *args, **kwargs)
        if asked == 0 or asked < _ALL:
            return backend_method(self, *args, **kwargs)
        compiled = threads_compiled()
        threads = resolve_n_jobs(asked, compiled=compiled)
        if not compiled:
            return backend_method(self, *args, **kwargs)
        if by_position:
            args = (*args[:position], threads, *args[position + 1 :])
        else:
            kwargs = dict(kwargs, n_jobs=threads)
        logger.debug(
            "%s: n_jobs asked=%s, mode=%s, threads=%d",
            backend_method.__name__,
            asked,
            threads_mode(),
            threads,
        )
        try:
            return backend_method(self, *args, **kwargs)
        finally:
            if threads != asked:
                self.set_params(n_jobs=asked)

    return method
