"""timeout.py."""

import signal as _signal
from contextlib import contextmanager

from ..exceptions import ScikitplotException
from ..utils.os import is_windows


class ScikitplotTimeoutError(Exception):
    """ScikitplotTimeoutError."""


@contextmanager
def run_with_timeout(seconds):
    """
    Context manager to runs a block of code with a timeout.

    If the block of code takes longer than `seconds` to execute, a `TimeoutError` is raised.
    NB: This function uses Unix signals to implement the timeout, so it is not thread-safe.
    Also it does not work on non-Unix platforms such as Windows.

    Examples
    --------
    >>> with run_with_timeout(5):
    ...     model.predict(data)
    """
    if is_windows():
        raise ScikitplotException(
            "Timeouts are not implemented yet for non-Unix platforms",
            error_code=0,
        )

    def signal_handler(signum, frame):
        raise ScikitplotTimeoutError(f"Operation timed out after {seconds} seconds")

    _signal.signal(_signal.SIGALRM, signal_handler)
    _signal.alarm(seconds)

    try:
        yield
    finally:
        _signal.alarm(0)  # Disable the alarm after the operation completes or times out
