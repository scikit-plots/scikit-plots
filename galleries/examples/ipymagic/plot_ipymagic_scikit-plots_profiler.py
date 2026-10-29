"""
Profile a Notebook Cell with ``%%scikitplot_profile``
=====================================================

.. currentmodule:: scikitplot._ipymagic

``%scikitplot_profile`` (a line) and ``%%scikitplot_profile`` (a cell) run
code under :mod:`cProfile` and show the functions that took the time, as a
table that is also data. The steps:

1. profile a line, then a cell, and read the table;
2. sort and limit (``-s``, ``-l``) and use the rows as data;
3. save the statistics (``-D``) for :mod:`pstats` or snakeviz;
4. code that raises: the profile up to the error, then the error;
5. one profiler per process: a second one is refused, not stacked;
6. the interactive viewer (snakeviz, opt in).
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Some code worth profiling
# -------------------------
#
# A small, deterministic workload: a slow pure-Python function and a fast
# built-in doing the same job.

import cProfile
import os
import pstats
import shutil
import tempfile
from pathlib import Path

from scikitplot._ipymagic import ProfilerBusyError, profile_code

# In a notebook (this page's download), load the magics so the ``%%`` cells
# below run; a plain script has no IPython and skips this.
_shell = globals().get("get_ipython", lambda: None)()
if _shell is not None:
    _shell.run_line_magic("load_ext", "scikitplot")


def slow_sum_of_squares(n):
    """Sum the squares below ``n`` in a Python loop."""
    total = 0
    for value in range(n):
        total += value * value
    return total


def fast_sum_of_squares(n):
    """Sum the squares below ``n`` with the built-in ``sum``."""
    return sum(value * value for value in range(n))


namespace = {
    "slow_sum_of_squares": slow_sum_of_squares,
    "fast_sum_of_squares": fast_sum_of_squares,
}

# %%
# Profile a line
# --------------
#
# .. code-block:: ipython
#
#     %scikitplot_profile slow_sum_of_squares(200_000)
#
# The magic runs the line in the notebook's namespace; here
# :func:`profile_code` gets the namespace explicitly. It returns the
# :class:`ProfileResult` and what the code raised (None).

result, error = profile_code("slow_sum_of_squares(200_000)", namespace, limit=5)
print(result.to_text())
assert error is None

# %%
# Profile a cell, sorted by call count
# ------------------------------------
#
# .. code-block:: ipython
#
#     %%scikitplot_profile -s ncalls -l 4
#     slow_sum_of_squares(50_000)
#     fast_sum_of_squares(50_000)
#
# ``-s`` is ``cumulative`` (default), ``tottime`` or ``ncalls``; ``-l`` the
# number of rows. Equal values are ordered by location, so the same run
# gives the same table.

cell = "slow_sum_of_squares(50_000)\nfast_sum_of_squares(50_000)\n"
result, _ = profile_code(cell, namespace, sort="ncalls", limit=4)
print(result.to_text())

# %%
# The rows are data: the generator expression of the fast version is called
# once per value, which is where its calls go.

top = result.rows[0]
print(f"most calls: {top.function}, {top.ncalls} calls, line {top.line}")
print("as a dict:", sorted(result.to_dict()))

# %%
# Save the statistics
# -------------------
#
# .. code-block:: ipython
#
#     %%scikitplot_profile -D run.prof
#     slow_sum_of_squares(100_000)
#
# The file is ordinary :mod:`pstats` data: ``python -m pstats run.prof``,
# snakeviz, or any tool that reads ``cProfile`` output. Quote a path with
# spaces: ``-D "my run.prof"``.

work = Path(tempfile.mkdtemp(prefix="skplt-profile-"))
stats_file = work / "my run.prof"
result, _ = profile_code(
    "slow_sum_of_squares(100_000)", namespace, stats_file=str(stats_file)
)
print("written:", Path(result.stats_file).name)
pstats.Stats(str(stats_file)).sort_stats("cumulative").print_stats(2)

# %%
# Code that raises
# ----------------
#
# The profile is kept up to the error; the magic shows the table, then
# raises the error as the cell's own:

result, error = profile_code("slow_sum_of_squares(10_000); 1 / 0", namespace)
print("raised:", type(error).__name__, "| calls profiled:", result.total_calls)

# %%
# One profiler per process
# ------------------------
#
# Python allows one profiler at a time. When another one records the
# process (``python -m cProfile``, an IDE profiler, an outer profile), the
# magic refuses, instead of silently replacing it:

outer = cProfile.Profile()
outer.enable()
try:
    profile_code("1 + 1", {})
except ProfilerBusyError as exc:
    refused = f"{type(exc).__name__} [{exc.code}]: {exc}"
finally:
    outer.disable()
print(refused)

# %%
# The interactive viewer (opt in)
# -------------------------------
#
# .. code-block:: ipython
#
#     %%scikitplot_profile --snakeviz
#     slow_sum_of_squares(100_000)
#
# and, when done:
#
# .. code-block:: ipython
#
#     %scikitplot_profile --stop-snakeviz
#
# ``--snakeviz`` (``pip install snakeviz``) opens the profile as an
# interactive icicle chart below the cell. The server listens on
# ``127.0.0.1``, so it works where the browser and the kernel share a
# machine; on a remote kernel, keep the table or download the ``-D`` file.
# Servers stop with ``--stop-snakeviz``, ``%unload_ext`` or when the kernel
# exits. A documentation build starts one only when
# ``SCIKITPLOT_GALLERY_RUN_SNAKEVIZ=1``.

if os.environ.get("SCIKITPLOT_GALLERY_RUN_SNAKEVIZ", "0").strip() == "1":
    from scikitplot._ipymagic._snakeviz import start_snakeviz, stop_all

    server = start_snakeviz(str(stats_file))
    print("serving on loopback:", server.url.startswith("http://127.0.0.1:"))
    print("stopped:", stop_all())
else:
    print("SKIP: set SCIKITPLOT_GALLERY_RUN_SNAKEVIZ=1 to start snakeviz")

shutil.rmtree(work)

# %%
# .. tags::
#
#    model-workflow: ipymagic
#    plot-type: text
#    level: intermediate
