"""
Shared pytest helpers for :mod:`scikitplot.corpus` tests.

Notes
-----
Developer note: ``captured_logger`` is the package's one log capture,
:py:func:`~.logging._capture.captured_logger`, registered here because the
``scikit-plots-corpus`` distribution ships no root conftest (Rule 139). It exists
because the obvious way to capture a named logger with ``caplog`` double-counts.

``caplog.handler`` is already installed by pytest's logging plugin on the root
logger. Attaching it to a module logger as well means one ``logger.warning()``
is handled twice -- once directly, once after propagating to root -- and both
appends land in the same ``caplog.records`` list. A test asserting
``len(records) == 1`` then passes or fails depending on whether ``propagate``
happens to be True, which varies with import order and with whether the root
logger already had handlers when ``scikitplot.logging`` configured itself.

That is why such a test can pass locally and fail in CI with no code change.
"""

from __future__ import annotations

import contextlib
import logging

import pytest

from ..logging import _capture

# How the fixture reaches every test below this directory:

#                    ┌─────────────┐
#                    │ same caplog │
#                    └──────┬──────┘
#                           │
#                ┌──────────┴──────────┐
#                ↓                     ↓
# test_x(caplog,            captured_logger(level, logger)
#     captured_logger)                   │
#                                        ↓
#                                     _capture()
#
#                   scikitplot.corpus
#                          │
#                   conftest.py
#                          │
#                captured_logger
#                     fixture
#                          │
#        ┌─────────────────┼──────────────────┐
#        │                 │                  │
#        ↓                 ↓                  ↓
#  corpus/tests      _normalizers/tests   _readers/tests
#        │                 │                  │
#        ↓                 ↓                  ↓
#   request it         request it          request it
#   by logger name     by logger name      by logger name
#
#               NO IMPORTS
#
# test
# │
# ├── requests caplog
# │        │
# │        └──────────────────┐
# │                           │
# └── requests captured_logger│
#              │              │
#              └── requires caplog
#                     │
#                     ↓
#               SAME caplog instance


@pytest.fixture(autouse=True)
def _library_default_log_level():
    """
    Run every test at the level the library has when nobody configures it.

    Notes
    -----
    **Developer notes.** The library logs its audit trail at ``INFO`` and is
    silent by default, because an unconfigured logger inherits ``WARNING``
    from the root. The project's pytest configuration lowers the root to
    ``INFO`` and prints records live, so every ``encode`` and ``decode`` in
    this suite became a line of output: 4 413 of them, between and across the
    result lines (``CP-091``). Holding the namespace at ``WARNING`` restores
    the condition a user starts from. A test about the audit trail asks for
    the level it needs, as an application would, and gets it back here.
    """
    logger = logging.getLogger(__package__)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


# The shared capture (Rule 139): the scikit-plots-corpus distribution ships
# no root conftest, so it registers the function itself.
captured_logger = pytest.fixture(_capture.captured_logger, name="captured_logger")


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers corpus tests use, for runs without the root conftest.

    Notes
    -----
    Developer note: the root ``scikitplot/conftest.py`` registers
    ``thread_unsafe`` only for full-package runs. The ``scikit-plots-corpus``
    partial distribution ships this tree alone, so without this registration
    ``pytest --pyargs scikitplot.corpus`` warns ``PytestUnknownMarkWarning``
    (and fails under ``--strict-markers``). Registering a marker twice is
    harmless.
    """
    config.addinivalue_line(
        "markers", "thread_unsafe: mark the test function as single-threaded"
    )


class SQLiteLeakGuard:
    """Record the SQLite connections opened while active (R3-8).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches ``sqlite3.connect`` for the guard's lifetime.

    Notes
    -----
    Developer note: Python 3.13 reports an unclosed ``sqlite3.Connection`` as a
    ``ResourceWarning`` only when it is garbage-collected, usually during a
    later, unrelated test, or not at all (the suite left 80 open at exit). The
    victim is then the wrong test; the autouse fixture below fails the test that
    opened the connection, and names where.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sqlite3  # noqa: PLC0415
        import traceback  # noqa: PLC0415

        self._sqlite = sqlite3
        self.opened: list[tuple[object, list[str]]] = []
        real_connect = sqlite3.connect

        def _connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            site = [
                f"{frame.filename}:{frame.lineno}"
                for frame in traceback.extract_stack()[:-1]
                if "scikitplot" in frame.filename
            ]
            self.opened.append((conn, site[-4:]))
            return conn

        monkeypatch.setattr(sqlite3, "connect", _connect)

    def close_leaks(self) -> list[list[str]]:
        """Close every connection still open; return where each was opened."""
        leaked = []
        for conn, site in self.opened:
            try:
                conn.total_changes  # noqa: B018 (raises once closed)
            except self._sqlite.ProgrammingError:
                continue
            # Opened in another thread (check_same_thread), it cannot be closed
            # from here, but it is still a leak to report.
            with contextlib.suppress(self._sqlite.ProgrammingError):
                conn.close()
            leaked.append(site)
        return leaked


@pytest.fixture(autouse=True)
def _sqlite_connections_are_closed(monkeypatch: pytest.MonkeyPatch):
    """Fail the test that leaves a SQLite connection open (R3-8).

    The leaked connections are closed first, so one failure does not cascade.
    """
    guard = SQLiteLeakGuard(monkeypatch)
    yield
    leaked = guard.close_leaks()
    if leaked:
        pytest.fail(
            f"{len(leaked)} SQLite connection(s) left open; opened at: {leaked}",
            pytrace=False,
        )
