"""
A fresh interpreter for the claims that the test process cannot make.

Notes
-----
**User notes.** Two kinds of test use this module: the import-isolation tests,
which need a process that has imported nothing yet, and the doctest tests,
which need a process whose ``sys.stdout`` nobody else replaces.

**Developer notes.** ``import scikitplot.cleanprompt`` first runs
``scikitplot/__init__.py``, and that file imports NumPy unconditionally
(``CP-085``). What the parent loads is not a property of this package, so the
default prelude registers an empty stand-in for the parent whose ``__path__``
is the real package directory: every submodule resolves from the real files
and nothing outside ``cleanprompt`` runs.

``doctest`` judges an example by what reaches ``sys.stdout`` while it runs, and
replaces ``sys.stdout`` to see it. Under pytest that replacement does not
survive a log record (``CP-086``): with live logging on (``log_cli_level``),
pytest suspends and resumes its capture around each record it prints, and
resuming assigns pytest's own stream back to ``sys.stdout``. Every example
after the first such record prints to pytest and ``doctest`` reports "Got
nothing". A fresh interpreter has no capture to resume, so the result depends
on the examples alone. No test in this package calls ``doctest.testmod`` in
the pytest process; an architecture test holds that.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

__all__ = [
    "ISOLATED_PARENT",
    "REAL_PARENT",
    "assert_doctests_pass",
    "doctest_counts",
    "in_subprocess",
]

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent
_ROOT = _PACKAGE.parents[1]

#: Registers an empty stand-in for the parent package before anything imports it.
ISOLATED_PARENT = (
    "import sys, types\n"
    "_parent = types.ModuleType('scikitplot')\n"
    "_parent.__path__ = [{0!r}]\n"
    "sys.modules['scikitplot'] = _parent\n"
).format(str(_PACKAGE.parent))

#: Imports the real parent package from the checkout the tests run in.
REAL_PARENT = "import sys;sys.path.insert(0, {0!r})\n".format(str(_ROOT))

_COUNTS_MARKER = "DOCTEST-COUNTS "

_RUN_DOCTESTS = (
    "import doctest, importlib, json\n"
    "counts = {{}}\n"
    "for name in {0!r}:\n"
    "    module = importlib.import_module('scikitplot.cleanprompt.' + name)\n"
    "    result = doctest.testmod(module, verbose=False, report=False)\n"
    "    counts[name] = [result.failed, result.attempted]\n"
    "print({1!r} + json.dumps(counts, sort_keys=True))\n"
)


def in_subprocess(body, *, prelude=ISOLATED_PARENT):
    """
    Run ``body`` in a fresh interpreter and return its stdout.

    Parameters
    ----------
    body : str
        Source to execute after ``prelude``.
    prelude : str, default=ISOLATED_PARENT
        Source executed first. ``ISOLATED_PARENT`` measures this package
        alone; ``REAL_PARENT`` measures it under the real ``scikitplot``.

    Returns
    -------
    str
        Captured standard output.

    Raises
    ------
    AssertionError
        If the interpreter exits with a non-zero status.
    """
    completed = subprocess.run(
        [sys.executable, "-c", prelude + body], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def doctest_counts(modules):
    """
    Run the doctests of ``modules`` in a fresh interpreter.

    Parameters
    ----------
    modules : sequence of str
        Module names relative to the package, for example ``("_api",)``.

    Returns
    -------
    counts : dict
        ``{module: [failed, attempted]}``.
    output : str
        Everything the interpreter printed, including doctest's own failure
        reports, for an assertion message.

    Raises
    ------
    ValueError
        If ``modules`` is empty.
    AssertionError
        If the interpreter fails or does not report its counts exactly once.
    """
    modules = tuple(modules)
    if not modules:
        raise ValueError("doctest_counts needs at least one module name")
    output = in_subprocess(_RUN_DOCTESTS.format(modules, _COUNTS_MARKER))
    reported = [
        line for line in output.splitlines() if line.startswith(_COUNTS_MARKER)
    ]
    assert len(reported) == 1, output
    return json.loads(reported[0][len(_COUNTS_MARKER) :]), output


def assert_doctests_pass(*modules):
    """
    Assert that every example in ``modules`` ran and none failed.

    Parameters
    ----------
    *modules : str
        Module names relative to the package.

    Returns
    -------
    dict
        ``{module: [failed, attempted]}``, for a caller that asserts more.

    Raises
    ------
    AssertionError
        If any example failed, or a module's examples were not attempted. A
        run that attempted nothing reports zero failures too.
    """
    counts, output = doctest_counts(modules)
    assert sorted(counts) == sorted(modules), output
    failed = {name: pair[0] for name, pair in counts.items() if pair[0]}
    assert failed == {}, output
    empty = [name for name, pair in counts.items() if pair[1] == 0]
    assert empty == [], "no example was attempted in {0}".format(empty)
    return counts
