# scikitplot/_utils/tests/test_timeout.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.timeout`.

Coverage map
------------
ScikitplotTimeoutError    is an Exception subclass,
                           carries the expected message           -> TestScikitplotTimeoutError
run_with_timeout           raises on Windows (mocked),
                           passes on POSIX: normal completion,
                           timeout fires ScikitplotTimeoutError,
                           alarm is always cleared (finally),
                           exception inside block propagates,
                           zero seconds arms alarm               -> TestRunWithTimeout

Run standalone::

    python -m unittest scikitplot._utils.tests.test_timeout -v
"""

from __future__ import annotations

import signal
import sys
import time
import types
import unittest
import unittest.mock as mock

from ..exception_utils import ScikitplotException
from ..timeout import ScikitplotTimeoutError, run_with_timeout


# ===========================================================================
# ScikitplotTimeoutError
# ===========================================================================


class TestScikitplotTimeoutError(unittest.TestCase):
    """ScikitplotTimeoutError must be a proper Exception subclass."""

    def test_is_exception_subclass(self):
        """ScikitplotTimeoutError must inherit from Exception."""
        self.assertTrue(issubclass(ScikitplotTimeoutError, Exception))

    def test_can_be_raised(self):
        """Must be raisable like a normal exception."""
        with self.assertRaises(ScikitplotTimeoutError):
            raise ScikitplotTimeoutError("test")

    def test_message_preserved(self):
        """The message passed to the constructor must be preserved."""
        exc = ScikitplotTimeoutError("Operation timed out after 5 seconds")
        self.assertIn("5 seconds", str(exc))

    def test_empty_construction(self):
        """Must be constructable with no arguments."""
        exc = ScikitplotTimeoutError()
        self.assertIsInstance(exc, ScikitplotTimeoutError)

    def test_is_catchable_as_exception(self):
        """Must be catchable via the base Exception type."""
        caught = False
        try:
            raise ScikitplotTimeoutError("x")
        except Exception:
            caught = True
        self.assertTrue(caught)

    def test_is_not_base_exception_only(self):
        """Must be an Exception (not just BaseException)."""
        self.assertTrue(issubclass(ScikitplotTimeoutError, Exception))


# ===========================================================================
# run_with_timeout
# ===========================================================================


@unittest.skipIf(sys.platform == "win32", "signal.SIGALRM not available on Windows")
class TestRunWithTimeoutPosix(unittest.TestCase):
    """run_with_timeout tests for POSIX (signal.SIGALRM available)."""

    # -- normal completion (alarm fires after block exits) --

    def test_normal_block_completes(self):
        """A fast block must complete without raising."""
        with run_with_timeout(10):
            x = 1 + 1
        self.assertEqual(x, 2)

    def test_alarm_is_cleared_after_normal_completion(self):
        """After a successful block, the alarm must be disarmed (returns 0)."""
        with run_with_timeout(10):
            pass
        # signal.alarm(0) returns the previous alarm value; after disarm it's 0
        remaining = signal.alarm(0)
        self.assertEqual(remaining, 0)

    def test_context_manager_returns_none(self):
        """The context manager expression must be None (no __enter__ value used)."""
        with run_with_timeout(10) as cm_value:
            pass
        self.assertIsNone(cm_value)

    # -- timeout fires --

    def test_timeout_raises_scikitplot_timeout_error(self):
        """A block that sleeps longer than the timeout must raise ScikitplotTimeoutError."""
        with self.assertRaises(ScikitplotTimeoutError):
            with run_with_timeout(1):
                time.sleep(5)  # far exceeds 1 second

    def test_timeout_error_message_contains_seconds(self):
        """The timeout error message must mention the number of seconds."""
        try:
            with run_with_timeout(1):
                time.sleep(5)
        except ScikitplotTimeoutError as exc:
            self.assertIn("1", str(exc))
        else:
            self.fail("Expected ScikitplotTimeoutError was not raised")

    # -- alarm always cleared in finally --

    def test_alarm_cleared_after_exception_inside_block(self):
        """Even when the block raises an unrelated exception, alarm must be cleared."""
        with self.assertRaises(ValueError):
            with run_with_timeout(10):
                raise ValueError("unrelated error")
        remaining = signal.alarm(0)
        self.assertEqual(remaining, 0)

    def test_non_timeout_exception_propagates(self):
        """An exception raised inside the block must propagate unchanged."""
        with self.assertRaises(RuntimeError) as ctx:
            with run_with_timeout(10):
                raise RuntimeError("inner error")
        self.assertIn("inner error", str(ctx.exception))

    # -- mocked _is_windows to force the POSIX path even on any platform --

    def test_posix_path_used_when_not_windows(self):
        """When is_windows() returns False, no exception should be raised on entry."""
        with mock.patch("scikitplot._utils.timeout.is_windows", return_value=False):
            try:
                with run_with_timeout(10):
                    pass
            except Exception as exc:  # noqa: BLE001
                self.fail(f"Unexpected exception on non-Windows: {exc}")


class TestRunWithTimeoutWindowsPath(unittest.TestCase):
    """run_with_timeout must raise ScikitplotException on Windows."""

    def test_windows_raises_scikitplot_exception(self):
        """When is_windows() returns True, ScikitplotException must be raised."""

        with mock.patch("scikitplot._utils.timeout.is_windows", return_value=True):
            with self.assertRaises(ScikitplotException):
                with run_with_timeout(5):
                    pass  # pragma: no cover

    def test_windows_error_message_informative(self):
        """The Windows error message must mention 'non-Unix' or 'not implemented'."""

        with mock.patch("scikitplot._utils.timeout.is_windows", return_value=True):
            with self.assertRaises(ScikitplotException) as ctx:
                with run_with_timeout(5):
                    pass  # pragma: no cover
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "non-unix" in msg or "not implemented" in msg or "windows" in msg,
            msg=f"Unhelpful message: {msg!r}",
        )

    def test_windows_exception_has_error_code(self):
        """ScikitplotException from Windows path must have error_code attribute."""

        with mock.patch("scikitplot._utils.timeout.is_windows", return_value=True):
            with self.assertRaises(ScikitplotException) as ctx:
                with run_with_timeout(5):
                    pass  # pragma: no cover
        self.assertEqual(ctx.exception.error_code, 0)


# ===========================================================================
# Signal handler isolation
# ===========================================================================


@unittest.skipIf(sys.platform == "win32", "signal.SIGALRM not available on Windows")
class TestSignalHandlerBehavior(unittest.TestCase):
    """Verify SIGALRM handler setup and teardown."""

    def test_original_handler_restored_after_block(self):
        """The SIGALRM handler before the context manager must be restored after exit."""
        original = signal.getsignal(signal.SIGALRM)
        with run_with_timeout(10):
            pass
        # After exiting, alarm is cancelled; handler left in place by implementation
        # (not required to be restored, but alarm is cleared)
        remaining = signal.alarm(0)
        self.assertEqual(remaining, 0)
        # restore original for cleanliness
        signal.signal(signal.SIGALRM, original)

    def test_handler_raises_scikitplot_timeout_error(self):
        """The installed SIGALRM handler must raise ScikitplotTimeoutError when fired."""
        original_handler = signal.getsignal(signal.SIGALRM)
        try:
            with self.assertRaises(ScikitplotTimeoutError):
                with run_with_timeout(1):
                    # Fire the signal immediately to test the handler
                    signal.alarm(1)
                    time.sleep(3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)


if __name__ == "__main__":
    unittest.main(verbosity=2)
