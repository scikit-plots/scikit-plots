# libs/_tools/pytest_partial.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Pytest plugin: run a part's test suite inside a partial distribution.

The test suites under ``scikitplot/<part>/tests`` were written for the full
distribution. A few of their tests reach for another part of the package: a
CLI test that runs ``show-config`` (which needs ``scikitplot.config``), an MCP
test that builds a corpus index (which needs ``scikitplot.corpus``). In a
partial installation those parts are deliberately absent, and such a test
cannot pass there, by construction and not by fault.

This plugin applies one rule and nothing else:

    A test that fails with ``ModuleNotFoundError`` for a module of the
    ``scikitplot`` package that is **not installed** is reported as skipped,
    with the name of the missing part as the reason.

Everything else is left alone. A test that fails for any other reason fails,
including one that cannot import a third-party package: test tooling is
declared in ``libs/_tools/registry.py``, so an undeclared one is a finding.

Notes
-----
**User.** ``python -m libs._tools verify`` loads this plugin for you. To use it
by hand in an environment that has only partial distributions installed::

    PYTHONPATH=libs/_tools python -m pytest -p pytest_partial --pyargs scikitplot.mcp

**Developer.** The plugin is opt-in (``-p pytest_partial``) and lives with the
verification tooling, not in the package: it must never be active in a run
that someone did not ask it to be active in, because turning a failure into a
skip is only safe where the absence it explains is intended. It refuses to
load when the full distribution is installed, where no part is ever absent on
purpose. It imports nothing from ``libs._tools`` and is loaded as a plain
top-level module.
"""

from __future__ import annotations

from collections import Counter
from importlib import util

import pytest

__all__ = ["missing_part"]

_PACKAGE = "scikitplot"
_REASON = "needs {name}, which is not installed in this partial distribution"
_skipped: Counter[str] = Counter()


def _is_installed(module: str) -> bool:
    """Return whether the top-level part a ``scikitplot`` module belongs to exists."""
    package, _, rest = module.partition(".")
    if not rest:
        return True  # the root package itself; its absence is not this rule's case
    try:
        return util.find_spec(f"{package}.{rest.split('.', 1)[0]}") is not None
    except (ImportError, ValueError):
        return False


def missing_part(exc: BaseException | None) -> str | None:
    """
    Return the absent ``scikitplot`` module an exception is ultimately about.

    Parameters
    ----------
    exc : BaseException or None
        The exception a test or a collection step ended with.

    Returns
    -------
    str or None
        The module name when ``exc``, or an exception it was raised from, is a
        ``ModuleNotFoundError`` naming a module of ``scikitplot`` whose part is
        not installed. ``None`` otherwise.

    Notes
    -----
    **Developer.** The chain is followed through both ``__cause__`` and
    ``__context__``, because the code under test converts the import failure
    into its own error type (``CapabilityMissingError`` in the CLI) and the
    original is then only reachable that way. Each exception is visited once,
    so a cyclic chain terminates.
    """
    seen = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError):
            name = current.name or ""
            if (
                name == _PACKAGE or name.startswith(_PACKAGE + ".")
            ) and not _is_installed(name):
                return name
        stack.append(current.__cause__)
        stack.append(current.__context__)
    return None


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run where a missing part cannot be intentional."""
    from scikitplot import _distributions  # noqa: PLC0415

    kind = _distributions.flavor()
    if kind != "partial":
        raise pytest.UsageError(
            f"pytest_partial is for partial distributions only; this environment "
            f"is {kind!r}. Run the tests without `-p pytest_partial`."
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Report a test that needs an absent part as skipped."""
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or not report.failed:
        return
    name = missing_part(call.excinfo.value)
    if name is None:
        return
    _skipped[name.split(".")[1] if "." in name else name] += 1
    report.outcome = "skipped"
    report.longrepr = (
        str(item.path),
        item.location[1] or 0,
        "Skipped: " + _REASON.format(name=name),
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    """Report a test module that cannot be imported without an absent part as skipped."""
    outcome = yield
    report = outcome.get_result()
    if not report.failed:
        return
    # The collection error is kept by pytest as text only; the exception that
    # caused it is recovered by importing the module again, which fails the
    # same way.
    path = getattr(collector, "path", None)
    if path is None or path.suffix != ".py":
        return
    try:
        collector.obj  # noqa: B018 - evaluated for its side effect: the import
    except BaseException as exc:  # noqa: BLE001 - classified, not swallowed
        name = missing_part(exc)
    else:
        return
    if name is None:
        return
    _skipped[name.split(".")[1] if "." in name else name] += 1
    report.outcome = "skipped"
    report.longrepr = (str(path), 0, "Skipped: " + _REASON.format(name=name))


def pytest_terminal_summary(terminalreporter) -> None:
    """Say how many tests were skipped by this plugin's rule, and for which part."""
    if not _skipped:
        return
    detail = ", ".join(f"{part}: {count}" for part, count in sorted(_skipped.items()))
    terminalreporter.write_line(
        f"pytest_partial: {sum(_skipped.values())} skipped for parts not installed ({detail})"
    )
