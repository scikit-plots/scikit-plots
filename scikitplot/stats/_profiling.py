# scikitplot/stats/_profiling.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Python profiling tools.

This package provides two types of profilers:

- profiling.tracing: Deterministic tracing profiler that instruments every
  function call and return. Higher overhead but provides exact call counts
  and timing.

- profiling.sampling: Statistical sampling profiler that periodically samples
  the call stack. Low overhead and suitable for production use.
"""

from __future__ import annotations

from contextlib import suppress

# https://docs.python.org/3/library/concurrent.futures.html
# import concurrent.futures
# from concurrent.futures import InterpreterPoolExecutor, ProcessPoolExecutor, ThreadPoolExecutor

## https://docs.python.org/3/library/intro.html
## https://docs.python.org/3/library/multiprocessing.html
## The multiprocessing package mostly replicates the API of the threading module.
## This module is not supported on mobile platforms or WebAssembly platforms.
# import multiprocessing as mp
# from multiprocessing import Lock, RLock, Pool

## https://docs.python.org/3/library/threading.html
## This module does not work or is not available on WebAssembly.
# import threading
# from threading import Lock, RLock, Thread

with suppress(ImportError, ModuleNotFoundError):
    # Also works (backward compatible as an alias to profiling.tracing — Deterministic profiler)

    # https://docs.python.org/3/library/profiling.html

    # https://docs.python.org/3/library/profiling.sampling.html
    # The statistical profiler (profiling.sampling — Statistical profiler) is recommended for most performance analysis tasks.

    # Note The cProfile module remains available as a backward-compatible alias to profiling.tracing.
    # Existing code using import cProfile will continue to work without modification.
    # Preferred (new style profiling.tracing — Deterministic profiler) (or its alias cProfile)
    pass

with suppress(ImportError, ModuleNotFoundError):
    # https://docs.python.org/3/library/profile.html
    # The pure Python profile module is deprecated and will be removed in Python 3.17.
    # Use profiling.tracing (or its alias cProfile) instead.
    # Deprecated since version 3.15, will be removed in version 3.17.
    pass

with suppress(ImportError, ModuleNotFoundError):
    # https://docs.python.org/3/library/pstats.html
    pass

    ## https://github.com/jiffyclub/snakeviz
    ## https://jiffyclub.github.io/snakeviz/
    ## pip install snakeviz
    ## SnakeViz is a browser based graphical viewer for the output of Python's cProfile module
    ## and an alternative to using the standard library pstats module.
    ## It was originally inspired by RunSnakeRun.
