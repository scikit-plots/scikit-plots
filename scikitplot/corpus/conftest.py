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
from contextlib import contextmanager


@contextmanager
def captured_logger(caplog, name: str, level: int = logging.WARNING):
    """
    Capture one named logger through exactly one handler path.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        The ``caplog`` fixture from the calling test.
    name : str
        Dotted name of the logger to capture.
    level : int, optional
        Level to capture at. Default :data:`logging.WARNING`.

    Yields
    ------
    logging.Logger
        The logger being captured.

    Notes
    -----
    Propagation is disabled for the duration and restored afterwards, so the
    record count is the number of ``logger.<level>()`` calls made -- which is
    what a test asserting "warned exactly once" means to assert. Without that,
    the count also measures how many handler paths were live, which is a
    property of the environment rather than of the code under test.

    Examples
    --------
    >>> with captured_logger(caplog, "pkg.mod") as logger:  # doctest: +SKIP
    ...     do_something_that_warns()
    >>> [r.getMessage() for r in caplog.records]  # doctest: +SKIP
    ['fell back to the default']
    """
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
