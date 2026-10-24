"""Run a block of code with a wall-clock limit (POSIX, main thread)."""

import math as _math
import numbers as _numbers
import signal as _signal
from contextlib import contextmanager

from ..exceptions import ScikitplotException
from .os import is_windows


class ScikitplotTimeoutError(Exception):
    """Raised by :py:func:`~.run_with_timeout` when its block runs too long."""


def _check_seconds(seconds):
    """
    Return ``seconds`` as a float after checking it is a usable limit.

    Parameters
    ----------
    seconds : int or float
        The limit to check.

    Returns
    -------
    float
        The limit, in seconds.

    Raises
    ------
    TypeError
        If ``seconds`` is not a real number; ``bool`` is refused.
    ValueError
        If ``seconds`` is not finite or not greater than zero.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, _numbers.Real):
        raise TypeError(
            f"seconds must be an int or a float, not {type(seconds).__name__}"
        )
    value = float(seconds)
    if not _math.isfinite(value) or value <= 0:
        raise ValueError(f"seconds must be a finite number above 0, got {seconds!r}")
    return value


@contextmanager
def run_with_timeout(seconds):
    """
    Run a block of code, and stop it if it runs longer than ``seconds``.

    Parameters
    ----------
    seconds : int or float
        The wall-clock limit in seconds, above 0; fractions are allowed
        (``0.5``).

    Yields
    ------
    None
        The block runs with the timer armed.

    Raises
    ------
    TypeError
        If ``seconds`` is not a real number (``bool`` is refused).
    ValueError
        If ``seconds`` is not finite or not above 0.
    RuntimeError
        If a real-time interval timer is already armed in the process (an
        outer ``run_with_timeout`` or :py:func:`signal.alarm`): one process
        has one such timer, and arming it here would cancel the other.
    ScikitplotTimeoutError
        If the block is still running after ``seconds``.
    ScikitplotException
        On Windows, which has no ``SIGALRM``.

    Notes
    -----
    **User notes.** The block is interrupted by ``SIGALRM``: a call blocked
    in the operating system (``time.sleep``, a read) returns early, and the
    error is raised where the block was. The limit works only in the main
    thread of the main interpreter, as every Python signal handler does.

    **Developer notes.** The timer is ``signal.setitimer(ITIMER_REAL)``,
    which takes fractional seconds; ``signal.alarm`` takes whole seconds and
    raised ``TypeError`` for ``0.1``. On exit, also by an exception, the
    timer is disarmed and the ``SIGALRM`` handler that was installed before
    is put back.

    Examples
    --------
    >>> import time
    >>> try:
    ...     with run_with_timeout(0.05):
    ...         time.sleep(10)
    ... except ScikitplotTimeoutError as error:
    ...     print(error)
    Operation timed out after 0.05 seconds
    """
    limit = _check_seconds(seconds)
    if is_windows():
        raise ScikitplotException(
            "Timeouts are not implemented yet for non-Unix platforms",
            error_code=0,
        )
    if _signal.getitimer(_signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError(
            "a real-time timer is already armed in this process "
            "(an outer run_with_timeout or signal.alarm); it would be cancelled"
        )

    def signal_handler(signum, frame):
        raise ScikitplotTimeoutError(f"Operation timed out after {seconds} seconds")

    previous_handler = _signal.signal(_signal.SIGALRM, signal_handler)
    try:
        _signal.setitimer(_signal.ITIMER_REAL, limit)
        yield
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, previous_handler)
