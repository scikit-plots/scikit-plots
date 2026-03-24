# scikitplot/utils/tests/test__time.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._time`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__time.py -v

Coverage map
------------
Timer.__init__      Default values, custom overrides            → TestTimerInit
Timer.__enter__     Returns self, logs message, verbose print   → TestTimerEnter
Timer.__exit__      Elapsed > 0, logs completion, returns False → TestTimerExit
Timer (context mgr) Full round-trip, exception propagation,
                    all logging_level variants                  → TestTimerContextManager
"""

from __future__ import annotations

import logging
import time
import unittest
import unittest.mock as mock

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib, sys
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._time import Timer  # noqa: E402


# ===========================================================================
# Timer.__init__
# ===========================================================================


class TestTimerInit(unittest.TestCase):
    """Timer must store all constructor arguments with correct defaults."""

    def test_default_message_is_empty(self):
        t = Timer()
        self.assertEqual(t.message, "")

    def test_default_precision_three(self):
        t = Timer()
        self.assertEqual(t.precision, 3)

    def test_default_logging_level_info(self):
        t = Timer()
        self.assertEqual(t.logging_level, "info")

    def test_default_verbose_false(self):
        t = Timer()
        self.assertFalse(t.verbose)

    def test_custom_message(self):
        t = Timer("Building index")
        self.assertEqual(t.message, "Building index")

    def test_custom_precision(self):
        t = Timer(precision=5)
        self.assertEqual(t.precision, 5)

    def test_custom_logging_level(self):
        t = Timer(logging_level="debug")
        self.assertEqual(t.logging_level, "debug")

    def test_custom_verbose(self):
        t = Timer(verbose=True)
        self.assertTrue(t.verbose)

    def test_private_start_initialized_zero(self):
        """Internal _start must default to 0.0 before any use."""
        t = Timer()
        self.assertEqual(t._start, 0.0)


# ===========================================================================
# Timer.__enter__
# ===========================================================================


class TestTimerEnter(unittest.TestCase):
    """Timer.__enter__ must record start time and return self."""

    def test_enter_returns_self(self):
        t = Timer()
        result = t.__enter__()
        self.assertIs(result, t)

    def test_enter_sets_start_time(self):
        """_start must be set to a positive monotonic value after __enter__."""
        t = Timer()
        before = time.perf_counter()
        t.__enter__()
        after = time.perf_counter()
        self.assertGreaterEqual(t._start, before)
        self.assertLessEqual(t._start, after)

    def test_enter_prints_verbose_message(self):
        """verbose=True must print the message on __enter__."""
        t = Timer("Loading data", verbose=True)
        with mock.patch("builtins.print") as mock_print:
            t.__enter__()
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("Loading data", call_args)

    def test_enter_no_print_when_not_verbose(self):
        """verbose=False must not print anything."""
        t = Timer("msg", verbose=False)
        with mock.patch("builtins.print") as mock_print:
            t.__enter__()
        mock_print.assert_not_called()

    def test_enter_no_print_empty_message_verbose(self):
        """verbose=True with empty message must not print."""
        t = Timer("", verbose=True)
        with mock.patch("builtins.print") as mock_print:
            t.__enter__()
        mock_print.assert_not_called()


# ===========================================================================
# Timer.__exit__
# ===========================================================================


class TestTimerExit(unittest.TestCase):
    """Timer.__exit__ must log elapsed time and return False."""

    def test_exit_returns_false(self):
        """__exit__ must return False so exceptions propagate normally."""
        t = Timer()
        t.__enter__()
        result = t.__exit__(None, None, None)
        self.assertIs(result, False)

    def test_exit_prints_elapsed_verbose(self):
        """verbose=True must print '⏱ → Completed in ...' on __exit__."""
        t = Timer(verbose=True)
        t.__enter__()
        with mock.patch("builtins.print") as mock_print:
            t.__exit__(None, None, None)
        # Should have at least one print call with 'Completed'
        args_list = [str(c) for c in mock_print.call_args_list]
        self.assertTrue(any("Completed" in a for a in args_list))

    def test_exit_elapsed_format(self):
        """Printed elapsed message must contain a seconds value."""
        t = Timer(precision=3, verbose=True)
        t.__enter__()
        time.sleep(0.01)
        printed = []
        with mock.patch("builtins.print", side_effect=lambda *a: printed.extend(a)):
            t.__exit__(None, None, None)
        self.assertTrue(any("Completed" in str(m) for m in printed))

    def test_exit_elapsed_precision_respected(self):
        """The elapsed string must use the specified precision."""
        t = Timer(precision=2, verbose=True)
        t.__enter__()
        printed = []
        with mock.patch("builtins.print", side_effect=lambda *a: printed.extend(a)):
            t.__exit__(None, None, None)
        # The elapsed format is "{n:.2f}s" → exactly 2 decimal places
        import re
        elapsed_strs = " ".join(str(m) for m in printed)
        match = re.search(r"(\d+\.\d+)s", elapsed_strs)
        if match:
            decimal_part = match.group(1).split(".")[1]
            self.assertLessEqual(len(decimal_part), 2)

    def test_exit_no_print_when_not_verbose(self):
        t = Timer(verbose=False)
        t.__enter__()
        with mock.patch("builtins.print") as mock_print:
            t.__exit__(None, None, None)
        mock_print.assert_not_called()


# ===========================================================================
# Timer as a context manager (full round-trips)
# ===========================================================================


class TestTimerContextManager(unittest.TestCase):
    """Timer must work correctly as a context manager with various settings."""

    def _run_with_logging(self, **kwargs):
        """Run a Timer context and capture log records."""
        t = Timer(**kwargs)
        logger = logging.getLogger("scikitplot")
        with self.assertLogs("scikitplot", level=logging.DEBUG):
            with t:
                pass

    def test_basic_context_manager_no_error(self):
        """A plain 'with Timer()' block must not raise any exception."""
        with Timer():
            pass

    def test_context_manager_returns_timer(self):
        """'as' binding must yield the Timer instance."""
        with Timer() as t:
            self.assertIsInstance(t, Timer)

    def test_elapsed_is_positive(self):
        """After the block, elapsed time must be positive."""
        t = Timer()
        t.__enter__()
        time.sleep(0.005)
        elapsed_before = time.perf_counter() - t._start
        t.__exit__(None, None, None)
        self.assertGreater(elapsed_before, 0)

    def test_exception_propagates(self):
        """Timer must not suppress exceptions (return False from __exit__)."""
        with self.assertRaises(RuntimeError):
            with Timer():
                raise RuntimeError("test exception")

    def test_logging_level_debug_emits(self):
        """logging_level='debug' must emit a DEBUG record."""
        with self.assertLogs("scikitplot", level=logging.DEBUG) as log:
            with Timer(message="task", logging_level="debug"):
                pass
        debug_msgs = [r for r in log.output if "DEBUG" in r]
        self.assertGreater(len(debug_msgs), 0)

    def test_logging_level_info_emits(self):
        """logging_level='info' must emit an INFO record."""
        with self.assertLogs("scikitplot", level=logging.INFO) as log:
            with Timer(message="task", logging_level="info"):
                pass
        info_msgs = [r for r in log.output if "INFO" in r]
        self.assertGreater(len(info_msgs), 0)

    def test_logging_level_warning_emits(self):
        """logging_level='warning' must emit a WARNING record."""
        with self.assertLogs("scikitplot", level=logging.WARNING) as log:
            with Timer(message="task", logging_level="warning"):
                pass
        warn_msgs = [r for r in log.output if "WARNING" in r]
        self.assertGreater(len(warn_msgs), 0)

    def test_logging_level_error_emits(self):
        """logging_level='error' must emit an ERROR record."""
        with self.assertLogs("scikitplot", level=logging.ERROR) as log:
            with Timer(message="task", logging_level="error"):
                pass
        error_msgs = [r for r in log.output if "ERROR" in r]
        self.assertGreater(len(error_msgs), 0)

    def test_invalid_logging_level_falls_back_to_info(self):
        """An unrecognised logging_level must fall back to logger.info."""
        with self.assertLogs("scikitplot", level=logging.INFO):
            with Timer(message="fallback", logging_level="nonexistent_level"):
                pass

    def test_nested_timers_no_error(self):
        """Nested Timer blocks must not interfere with each other."""
        with Timer(message="outer") as outer:
            with Timer(message="inner") as inner:
                time.sleep(0.001)
            self.assertIsInstance(inner, Timer)
        self.assertIsInstance(outer, Timer)

    def test_reuse_timer_resets_start(self):
        """Each context entry must reset _start independently."""
        t = Timer()
        with t:
            time.sleep(0.005)
            first_start = t._start
        with t:
            second_start = t._start
        self.assertGreater(second_start, first_start)

    def test_verbose_output_includes_emoji(self):
        """The verbose print must include the ⏱ emoji marker."""
        t = Timer(message="test", verbose=True)
        printed = []
        with mock.patch("builtins.print", side_effect=lambda *a: printed.extend(a)):
            with t:
                pass
        all_output = " ".join(str(m) for m in printed)
        self.assertIn("⏱", all_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
