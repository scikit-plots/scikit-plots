"""
Shared pytest helpers for :mod:`scikitplot.corpus` tests.

Notes
-----
Developer note: the ``captured_logger`` helper here exists because the obvious
way to capture a named logger with ``caplog`` double-counts.

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

import logging

# Update the return type annotation from Iterator[logging.Logger] to Generator[logging.Logger, None, None]
# Change: Replace Iterator[logging.Logger] with Generator[logging.Logger, None, None]
# Pylance is warning you that using Iterator[Foo] as the return type annotation for a function decorated
# with @contextmanager is deprecated. You should use Generator[Foo, None, None] instead.
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generator

import pytest

# → dedicated _helpers.py / testing module rather than relying on conftest import
# Make captured_logger a pytest fixture factory.
# Then every descendant test can request it automatically by name;
# no `from ..conftest import captured_logger` imports are needed.
#
# @contextmanager
# def captured_logger(caplog, name: str, level: int = logging.WARNING):
#     """
#     Capture one named logger through exactly one handler path.
#
#     Parameters
#     ----------
#     caplog : pytest.LogCaptureFixture
#         The ``caplog`` fixture from the calling test.
#     name : str
#         Dotted name of the logger to capture.
#     level : int, optional
#         Level to capture at. Default :data:`logging.WARNING`.
#
#     Yields
#     ------
#     logging.Logger
#         The logger being captured.
#
#     Notes
#     -----
#     Propagation is disabled for the duration and restored afterwards, so the
#     record count is the number of ``logger.<level>()`` calls made -- which is
#     what a test asserting "warned exactly once" means to assert. Without that,
#     the count also measures how many handler paths were live, which is a
#     property of the environment rather than of the code under test.
#
#     Examples
#     --------
#     >>> with captured_logger(caplog, "pkg.mod") as logger:  # doctest: +SKIP
#     ...     do_something_that_warns()
#     >>> [r.getMessage() for r in caplog.records]  # doctest: +SKIP
#     ['fell back to the default']
#     """
#     logger = logging.getLogger(name)
#     previous_propagate = logger.propagate
#     logger.addHandler(caplog.handler)
#     logger.propagate = False
#     try:
#         caplog.clear()
#         with caplog.at_level(level, logger=name):
#             yield logger
#     finally:
#         logger.propagate = previous_propagate
#         logger.removeHandler(caplog.handler)
#
#                    ┌─────────────┐
#                    │ same caplog │
#                    └──────┬──────┘
#                           │
#                ┌──────────┴──────────┐
#                ↓                     ↓
# test_x(caplog,            captured_logger(name, level)
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
#   by name            by name             by name
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
    import logging

    logger = logging.getLogger("scikitplot.corpus")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


@pytest.fixture
def captured_logger(
    caplog: pytest.LogCaptureFixture,
):
    """Return a context-manager factory for capturing one named logger."""

    @contextmanager
    def _capture(
        name: str,
        level: int = logging.WARNING,
    ) -> Generator[logging.Logger, None, None]:  # Iterator[logging.Logger]
        logger = logging.getLogger(name)
        previous_propagate = logger.propagate

        logger.addHandler(caplog.handler)
        logger.propagate = False

        try:
            caplog.clear()
            with caplog.at_level(level, logger=name):
                yield logger
        finally:
            logger.propagate = previous_propagate
            logger.removeHandler(caplog.handler)

    return _capture
