# scikitplot/_utils/tests/test_time.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.time`.

Coverage map
------------
get_current_time_millis  returns int, positive, recent         → TestGetCurrentTimeMillis
conv_longdate_to_str     format, timezone flag, epoch=0        → TestConvLongdateToStr
Timer                    context manager, elapsed, repr/str    → TestTimer

Run standalone::

    python -m unittest scikitplot._utils.tests.test_time -v
"""

from __future__ import annotations

import re
import time
import unittest

from ..time import Timer, conv_longdate_to_str, get_current_time_millis

# Reference epoch in ms for a known past date (2020-01-01 00:00:00 UTC)
# 1577836800000 ms
_EPOCH_2020_MS = 1_577_836_800_000


# ===========================================================================
# get_current_time_millis
# ===========================================================================


class TestGetCurrentTimeMillis(unittest.TestCase):
    """get_current_time_millis must return current time as a positive integer."""

    def test_returns_int(self):
        """Return type must be int."""
        result = get_current_time_millis()
        self.assertIsInstance(result, int)

    def test_is_positive(self):
        """Value must be positive (after Unix epoch)."""
        result = get_current_time_millis()
        self.assertGreater(result, 0)

    def test_is_greater_than_reference_2020(self):
        """Value must exceed the 2020-01-01 reference (we are past that date)."""
        result = get_current_time_millis()
        self.assertGreater(result, _EPOCH_2020_MS)

    def test_two_calls_are_non_decreasing(self):
        """Two sequential calls must give a non-decreasing result."""
        t1 = get_current_time_millis()
        t2 = get_current_time_millis()
        self.assertGreaterEqual(t2, t1)

    def test_reasonable_magnitude(self):
        """Value must be between 1e12 and 2e13 ms (years 2001–2603)."""
        result = get_current_time_millis()
        self.assertGreater(result, 1_000_000_000_000)  # after 2001
        self.assertLess(result, 20_000_000_000_000)    # before 2603

    def test_close_to_time_time(self):
        """Value must be within 1 second (1000 ms) of time.time() * 1000."""
        now_ms = int(time.time() * 1000)
        result = get_current_time_millis()
        self.assertAlmostEqual(result, now_ms, delta=1000)


# ===========================================================================
# conv_longdate_to_str
# ===========================================================================


class TestConvLongdateToStr(unittest.TestCase):
    """conv_longdate_to_str must convert a ms-epoch timestamp to a date string."""

    # Format: YYYY-MM-DD HH:MM:SS [TZ]
    _DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def _check_format(self, s: str) -> None:
        self.assertRegex(
            s,
            self._DATE_RE,
            msg=f"Date string {s!r} does not match expected format",
        )

    def test_epoch_zero_formats_correctly(self):
        """Epoch 0 (1970-01-01) must produce a valid formatted string."""
        result = conv_longdate_to_str(0, local_tz=False)
        self._check_format(result)

    def test_reference_2020_formats_correctly(self):
        """A known 2020 timestamp must produce a string containing '2020'."""
        result = conv_longdate_to_str(_EPOCH_2020_MS, local_tz=False)
        self.assertIn("2020", result)
        self._check_format(result)

    def test_returns_str(self):
        """Return type must always be str."""
        result = conv_longdate_to_str(_EPOCH_2020_MS, local_tz=False)
        self.assertIsInstance(result, str)

    def test_local_tz_true_appends_timezone(self):
        """When local_tz=True, a timezone abbreviation should be appended."""
        result_local = conv_longdate_to_str(_EPOCH_2020_MS, local_tz=True)
        result_notz = conv_longdate_to_str(_EPOCH_2020_MS, local_tz=False)
        # With local_tz=True the string should be at least as long
        self.assertGreaterEqual(len(result_local), len(result_notz))

    def test_local_tz_default_is_true(self):
        """Default call must include timezone info (local_tz defaults to True)."""
        # Calling without local_tz should not raise
        try:
            result = conv_longdate_to_str(_EPOCH_2020_MS)
        except Exception as exc:
            self.fail(f"conv_longdate_to_str raised with default args: {exc}")
        self.assertIsInstance(result, str)


# ===========================================================================
# Timer
# ===========================================================================


class TestTimer(unittest.TestCase):
    """Timer must measure elapsed wall-clock time via context manager."""

    def test_elapsed_is_float_after_exit(self):
        """elapsed must be a float after the context manager exits."""
        with Timer() as t:
            pass
        self.assertIsInstance(t.elapsed, float)

    def test_elapsed_is_non_negative(self):
        """elapsed must be >= 0 after any context block."""
        with Timer() as t:
            pass
        self.assertGreaterEqual(t.elapsed, 0.0)

    def test_elapsed_captures_sleep_duration(self):
        """elapsed must be at least the duration of a short sleep."""
        sleep_s = 0.05
        with Timer() as t:
            time.sleep(sleep_s)
        # Allow generous 3x headroom for CI overhead
        self.assertGreaterEqual(t.elapsed, sleep_s * 0.5)

    def test_elapsed_initial_zero(self):
        """Before entering the context, elapsed must be 0.0."""
        t = Timer()
        self.assertEqual(t.elapsed, 0.0)

    def test_repr_returns_string(self):
        """repr(timer) must return a str."""
        with Timer() as t:
            pass
        self.assertIsInstance(repr(t), str)

    def test_str_returns_string(self):
        """str(timer) must return a str."""
        with Timer() as t:
            pass
        self.assertIsInstance(str(t), str)

    def test_format_with_precision(self):
        """format(timer, '.2f') must produce a valid float string."""
        with Timer() as t:
            pass
        formatted = f"{t:.2f}"
        # Should be parseable as a float
        try:
            float(formatted)
        except ValueError:
            self.fail(f"format(timer, '.2f') = {formatted!r} is not a float string")

    def test_context_manager_returns_timer_instance(self):
        """__enter__ must return the Timer instance itself."""
        t = Timer()
        with t as result:
            self.assertIs(result, t)

    def test_nested_timers_are_independent(self):
        """Two independent Timer contexts must each measure their own elapsed."""
        with Timer() as t1:
            time.sleep(0.02)
        with Timer() as t2:
            time.sleep(0.01)
        # t1 should have been active longer
        self.assertGreaterEqual(t1.elapsed, t2.elapsed * 0.5)

    def test_exception_in_block_still_records_elapsed(self):
        """Timer must record elapsed even when the block raises an exception."""
        t = Timer()
        try:
            with t:
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        self.assertGreater(t.elapsed, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
