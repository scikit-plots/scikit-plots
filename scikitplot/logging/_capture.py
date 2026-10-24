# scikitplot/logging/_capture.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Capture the records of one logger into pytest's ``caplog``, through one path.

The function here is the body of the ``captured_logger`` fixture (Rule 139).
It takes the ``caplog`` fixture as an argument and never imports pytest, so
it ships in ``scikit-plots-skinny`` with the rest of :mod:`~.logging`, and the
test suite of every partial distribution can register it.

Notes
-----
**User notes.** Request the fixture and capture inside the block::

    def test_warns_once(captured_logger, caplog):
        with captured_logger(logging.WARNING, _io.logger):
            index.save_bundle(path)
        (record,) = caplog.records

**Developer notes.** The ``"scikitplot"`` logger has ``propagate = False``
(:mod:`~.logging._logging`), so a record of a ``scikitplot.*`` module never
reaches the root logger, where pytest's capture handler is. pytest up to 9.0
attaches that handler to the root logger only, pytest 9.1 also to every
logger that does not propagate when capture starts; ``caplog.at_level`` sets
a level and routes nothing. A test that relied on either saw records under
one pytest release and none under another. This capture attaches
``caplog.handler`` to the logger under test itself and turns its propagation
off for the duration, so each call of ``logger.<level>()`` is one record in
``caplog.records`` whatever the release, and nothing is counted twice.
Registration, in a ``conftest.py``::

    from ..logging import _capture

    captured_logger = pytest.fixture(_capture.captured_logger, name="captured_logger")
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator, Union

__all__ = ["captured_logger"]

#: What a level argument may be: a number, or a registered level name.
LevelLike = Union[int, str]


def _resolve_level(level: LevelLike) -> int:
    """
    Return the number of a logging level given as a number or a name.

    Parameters
    ----------
    level : int or str
        A level number (``logging.WARNING``) or a level name registered with
        :mod:`logging` (``"WARNING"``, any case).

    Returns
    -------
    int
        The level number.

    Raises
    ------
    TypeError
        If ``level`` is neither an ``int`` nor a ``str``; ``bool`` is refused
        although it is an ``int``.
    ValueError
        If ``level`` is a negative number or a name :mod:`logging` does not
        know.
    """
    if isinstance(level, bool) or not isinstance(level, (int, str)):
        raise TypeError(
            f"level must be an int or a level name, not {type(level).__name__}"
        )
    if isinstance(level, int):
        if level < 0:
            raise ValueError(f"level must be 0 or more, got {level}")
        return level
    number = logging.getLevelName(level.upper())
    if isinstance(number, int):
        return number
    raise ValueError(f"unknown logging level name {level!r}")


def _resolve_logger(logger: logging.Logger | str) -> logging.Logger:
    """
    Return the logger to capture, given as a logger or by its name.

    Parameters
    ----------
    logger : logging.Logger or str
        The logger of the module under test (``module.logger``) or its
        dotted name (``module.__name__``).

    Returns
    -------
    logging.Logger
        The logger, never the root logger.

    Raises
    ------
    TypeError
        If ``logger`` is neither a :class:`logging.Logger` nor a ``str``.
    ValueError
        If it names or is the root logger: pytest already captures the root
        logger, so a second handler there would count every record twice.
    """
    if isinstance(logger, str):
        if not logger:
            raise ValueError("logger name must not be empty (that is the root logger)")
        resolved = logging.getLogger(logger)
    elif isinstance(logger, logging.Logger):
        resolved = logger
    else:
        raise TypeError(
            f"logger must be a logging.Logger or its name, not {type(logger).__name__}"
        )
    if resolved is logging.getLogger():
        raise ValueError(
            "the root logger is captured by caplog already; "
            "name the logger of the module under test"
        )
    return resolved


def captured_logger(caplog: Any) -> Any:
    """
    Return a context-manager factory that captures one logger into ``caplog``.

    Parameters
    ----------
    caplog : _pytest.logging.LogCaptureFixture
        The ``caplog`` fixture of the calling test. Only its ``handler``
        attribute and ``clear()`` method are used.

    Returns
    -------
    callable
        ``capture(level, logger)``, a context manager. ``level`` is a level
        number or a registered level name; ``logger`` is a
        :class:`logging.Logger` or its dotted name, never the root logger.
        Entering it clears ``caplog``, attaches ``caplog.handler`` to the
        logger, turns the logger's propagation off, enables it, and sets the
        logger and the handler to ``level``; it yields the logger. Leaving
        it, also by an exception, restores each of these as it was.

    See Also
    --------
    logging.Logger.addHandler

    Notes
    -----
    **User notes.** Read the records from ``caplog.records`` after the
    block. An assertion that nothing was logged belongs inside the same
    capture as a test that shows the capture sees records (Rule 153).
    Entering ``capture`` raises :py:class:`TypeError` for a ``level`` or
    ``logger`` of the wrong type, and :py:class:`ValueError` for an unknown
    or negative level, an empty logger name, or the root logger.

    **Developer notes.** Arguments are checked before any state changes, so
    a refused call leaves the logger as it was. A handler already attached to
    the logger (pytest 9.1 attaches ``caplog.handler`` to a logger that does
    not propagate, such as ``"scikitplot"`` itself) is neither attached
    twice nor removed on exit. ``logging.disable`` and ``Logger.disabled``
    would hide every record and make a negative assertion pass without
    looking, so both are lifted for the duration and put back afterwards.

    Examples
    --------
    >>> import logging
    >>> class FakeCaplog:
    ...     def __init__(self):
    ...         self.records = []
    ...         self.handler = logging.Handler()
    ...         self.handler.emit = self.records.append
    ...
    ...     def clear(self):
    ...         self.records.clear()
    >>> caplog = FakeCaplog()
    >>> capture = captured_logger(caplog)
    >>> with capture(logging.WARNING, "skplt.doctest.capture") as logger:
    ...     logger.warning("seen")
    ...     logger.info("below the level")
    >>> [record.getMessage() for record in caplog.records]
    ['seen']
    >>> logger.propagate, logger.handlers
    (True, [])
    """

    @contextlib.contextmanager
    def capture(
        level: LevelLike, logger: logging.Logger | str
    ) -> Iterator[logging.Logger]:
        number = _resolve_level(level)
        target = _resolve_logger(logger)
        handler = caplog.handler
        attach = handler not in target.handlers
        saved = (target.level, target.propagate, target.disabled, handler.level)
        manager = target.manager
        saved_disable = manager.disable
        caplog.clear()
        if attach:
            target.addHandler(handler)
        try:
            target.propagate = False
            target.disabled = False
            if saved_disable >= number:
                logging.disable(logging.NOTSET)
            target.setLevel(number)
            handler.setLevel(number)
            yield target
        finally:
            if manager.disable != saved_disable:
                logging.disable(saved_disable)
            handler.setLevel(saved[3])
            target.setLevel(saved[0])
            target.disabled = saved[2]
            target.propagate = saved[1]
            if attach:
                target.removeHandler(handler)

    return capture
