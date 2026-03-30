# scikitplot/_utils/tests/test_logging_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.logging_utils`.

Coverage map
------------
ScikitplotLoggingStream   write/flush forwarding, enabled property,
                          write suppressed when disabled           -> TestScikitplotLoggingStream
disable_logging           disables global stream                   -> TestDisableLogging
enable_logging            re-enables global stream                 -> TestEnableLogging
ScikitplotFormatter       format with known color, unknown color,
                          no color attr, win32 suppression,
                          _escape helper                           -> TestScikitplotFormatter
LoggerMessageFilter       filter passes non-matching, blocks match -> TestLoggerMessageFilter
suppress_logs             context manager removes filter after     -> TestSuppressLogs
eprint                    writes to MLFLOW_LOGGING_STREAM          -> TestEprint
LOGGING constants         format strings defined and non-empty     -> TestLoggingConstants

Run standalone::

    python -m unittest scikitplot._utils.tests.test_logging_utils -v
"""

from __future__ import annotations

import io
import logging
import re
import sys
import types
import unittest
import unittest.mock as mock

from ... import environment_variables
from ... import logging
from ..logging_utils import (  # noqa: E402
    LOGGING_DATETIME_FORMAT,
    LOGGING_LINE_FORMAT,
    MLFLOW_LOGGING_STREAM,
    LoggerMessageFilter,
    ScikitplotFormatter,
    ScikitplotLoggingStream,
    disable_logging,
    enable_logging,
    eprint,
    suppress_logs,
)


# ===========================================================================
# Constants
# ===========================================================================


class TestLoggingConstants(unittest.TestCase):
    """Module-level format strings must be non-empty strings."""

    def test_line_format_is_str(self):
        self.assertIsInstance(LOGGING_LINE_FORMAT, str)

    def test_line_format_non_empty(self):
        self.assertGreater(len(LOGGING_LINE_FORMAT), 0)

    def test_datetime_format_is_str(self):
        self.assertIsInstance(LOGGING_DATETIME_FORMAT, str)

    def test_datetime_format_non_empty(self):
        self.assertGreater(len(LOGGING_DATETIME_FORMAT), 0)

    def test_line_format_contains_levelname(self):
        """Format must include %(levelname)s."""
        self.assertIn("levelname", LOGGING_LINE_FORMAT)

    def test_line_format_contains_message(self):
        """Format must include %(message)s."""
        self.assertIn("message", LOGGING_LINE_FORMAT)


# ===========================================================================
# ScikitplotLoggingStream
# ===========================================================================


class TestScikitplotLoggingStream(unittest.TestCase):
    """ScikitplotLoggingStream must forward writes to sys.stderr when enabled."""

    def setUp(self):
        self.stream = ScikitplotLoggingStream()

    def tearDown(self):
        # Ensure stream left in enabled state
        self.stream.enabled = True

    # -- initial state --

    def test_enabled_by_default(self):
        """Stream must be enabled immediately after construction."""
        self.assertTrue(self.stream.enabled)

    # -- write when enabled --

    def test_write_forwards_to_stderr(self):
        """write() must forward text to sys.stderr when enabled."""
        fake_stderr = io.StringIO()
        with mock.patch("sys.stderr", fake_stderr):
            self.stream.write("hello")
        self.assertEqual(fake_stderr.getvalue(), "hello")

    def test_write_suppressed_when_disabled(self):
        """write() must be a no-op when stream is disabled."""
        self.stream.enabled = False
        fake_stderr = io.StringIO()
        with mock.patch("sys.stderr", fake_stderr):
            self.stream.write("should not appear")
        self.assertEqual(fake_stderr.getvalue(), "")

    # -- flush when enabled --

    def test_flush_calls_stderr_flush(self):
        """flush() must call sys.stderr.flush() when enabled."""
        fake_stderr = mock.MagicMock()
        with mock.patch("sys.stderr", fake_stderr):
            self.stream.flush()
        fake_stderr.flush.assert_called_once()

    def test_flush_suppressed_when_disabled(self):
        """flush() must be a no-op when disabled."""
        self.stream.enabled = False
        fake_stderr = mock.MagicMock()
        with mock.patch("sys.stderr", fake_stderr):
            self.stream.flush()
        fake_stderr.flush.assert_not_called()

    # -- enabled property setter --

    def test_set_enabled_false(self):
        """Setting enabled = False must reflect immediately."""
        self.stream.enabled = False
        self.assertFalse(self.stream.enabled)

    def test_set_enabled_true(self):
        """Re-enabling the stream must restore forwarding."""
        self.stream.enabled = False
        self.stream.enabled = True
        self.assertTrue(self.stream.enabled)


# ===========================================================================
# disable_logging / enable_logging
# ===========================================================================


class TestDisableEnableLogging(unittest.TestCase):
    """disable_logging and enable_logging must toggle MLFLOW_LOGGING_STREAM."""

    def tearDown(self):
        # Always restore to enabled after each test
        enable_logging()

    def test_disable_sets_stream_disabled(self):
        """disable_logging must set MLFLOW_LOGGING_STREAM.enabled to False."""
        enable_logging()
        disable_logging()
        self.assertFalse(MLFLOW_LOGGING_STREAM.enabled)

    def test_enable_sets_stream_enabled(self):
        """enable_logging must set MLFLOW_LOGGING_STREAM.enabled to True."""
        disable_logging()
        enable_logging()
        self.assertTrue(MLFLOW_LOGGING_STREAM.enabled)

    def test_disable_then_enable_roundtrip(self):
        """Disabling then enabling must restore the original state."""
        disable_logging()
        enable_logging()
        self.assertTrue(MLFLOW_LOGGING_STREAM.enabled)

    def test_double_disable_idempotent(self):
        """Calling disable twice must leave the stream disabled."""
        disable_logging()
        disable_logging()
        self.assertFalse(MLFLOW_LOGGING_STREAM.enabled)

    def test_double_enable_idempotent(self):
        """Calling enable twice must leave the stream enabled."""
        enable_logging()
        enable_logging()
        self.assertTrue(MLFLOW_LOGGING_STREAM.enabled)


# ===========================================================================
# ScikitplotFormatter
# ===========================================================================


class TestScikitplotFormatter(unittest.TestCase):
    """ScikitplotFormatter must apply ANSI color codes on non-Win32 platforms."""

    def _make_record(self, color=None, msg="test message"):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        if color is not None:
            record.color = color
        return record

    def _make_formatter(self):
        return ScikitplotFormatter(fmt="%(message)s")

    # -- no color attr -> plain format --

    def test_format_without_color_returns_plain(self):
        """Records with no color attribute must be formatted without ANSI codes."""
        formatter = self._make_formatter()
        record = self._make_record()  # no color attr
        result = formatter.format(record)
        self.assertIn("test message", result)
        self.assertNotIn("\033[", result)

    # -- known color on non-win32 --

    @unittest.skipIf(sys.platform == "win32", "ANSI codes disabled on win32")
    def test_format_with_known_color_adds_ansi(self):
        """Records with a known color must have ANSI escape codes prepended."""
        formatter = self._make_formatter()
        record = self._make_record(color="red")
        result = formatter.format(record)
        self.assertIn("\033[", result)
        self.assertIn("test message", result)

    @unittest.skipIf(sys.platform == "win32", "ANSI codes disabled on win32")
    def test_format_with_known_color_ends_with_reset(self):
        """Colored output must end with the ANSI reset sequence."""
        formatter = self._make_formatter()
        record = self._make_record(color="green")
        result = formatter.format(record)
        self.assertTrue(result.endswith("\033[0m"))

    # -- unknown color -> plain format --

    def test_format_with_unknown_color_returns_plain(self):
        """An unrecognised color must fall back to plain format."""
        formatter = self._make_formatter()
        record = self._make_record(color="neonpurple")
        result = formatter.format(record)
        self.assertIn("test message", result)
        # No escape code for unknown color
        self.assertNotIn("\033[3", result)  # no colour-specific code

    # -- win32 suppression --

    def test_format_suppresses_color_on_win32(self):
        """On win32, color must be suppressed regardless of record attribute."""
        formatter = self._make_formatter()
        record = self._make_record(color="red")
        with mock.patch("sys.platform", "win32"):
            result = formatter.format(record)
        self.assertNotIn("\033[3", result)

    # -- _escape helper --

    def test_escape_returns_correct_ansi_sequence(self):
        """_escape must return '\\033[{code}m'."""
        formatter = self._make_formatter()
        self.assertEqual(formatter._escape(31), "\033[31m")
        self.assertEqual(formatter._escape(0), "\033[0m")

    # -- COLORS dict --

    def test_colors_dict_contains_red(self):
        self.assertIn("red", ScikitplotFormatter.COLORS)

    def test_colors_dict_contains_green(self):
        self.assertIn("green", ScikitplotFormatter.COLORS)

    def test_all_color_codes_are_ints(self):
        for name, code in ScikitplotFormatter.COLORS.items():
            with self.subTest(name=name):
                self.assertIsInstance(code, int)


# ===========================================================================
# LoggerMessageFilter
# ===========================================================================


class TestLoggerMessageFilter(unittest.TestCase):
    """LoggerMessageFilter must block matching messages from the target module."""

    def _make_record(self, name: str, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_filter_blocks_matching_record(self):
        """A record whose name and message match must be blocked (filter returns False)."""
        f = LoggerMessageFilter(
            module="mymodule",
            filter_regex=re.compile(r"error occurred"),
        )
        record = self._make_record("mymodule", "error occurred at line 42")
        self.assertFalse(f.filter(record))

    def test_filter_passes_non_matching_message(self):
        """A record whose message does NOT match the regex must pass through."""
        f = LoggerMessageFilter(
            module="mymodule",
            filter_regex=re.compile(r"error occurred"),
        )
        record = self._make_record("mymodule", "all good")
        self.assertTrue(f.filter(record))

    def test_filter_passes_different_module(self):
        """A record from a different module must pass even if message matches regex."""
        f = LoggerMessageFilter(
            module="mymodule",
            filter_regex=re.compile(r"error occurred"),
        )
        record = self._make_record("othermodule", "error occurred")
        self.assertTrue(f.filter(record))

    def test_filter_blocks_partial_message_match(self):
        """Regex search (not full match) — partial pattern must block too."""
        f = LoggerMessageFilter(
            module="mymodule",
            filter_regex=re.compile(r"timeout"),
        )
        record = self._make_record("mymodule", "Connection timeout after 30s")
        self.assertFalse(f.filter(record))

    def test_filter_is_logging_filter_subclass(self):
        """LoggerMessageFilter must inherit from logging.Filter."""
        self.assertTrue(issubclass(LoggerMessageFilter, logging.Filter))


# ===========================================================================
# suppress_logs
# ===========================================================================


class TestSuppressLogs(unittest.TestCase):
    """suppress_logs must add/remove the filter around the context block."""

    def test_suppresses_matching_log(self):
        """Log records matching the pattern must be suppressed inside the context."""
        logger = logging.getLogger("suppress_test_module")
        logger.setLevel(logging.DEBUG)

        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            with suppress_logs("suppress_test_module", re.compile(r"suppress me")):
                logger.info("suppress me please")
            self.assertEqual(captured, [])
        finally:
            logger.removeHandler(handler)

    def test_does_not_suppress_non_matching(self):
        """Log records NOT matching the pattern must still be emitted."""
        logger = logging.getLogger("suppress_test_module2")
        logger.setLevel(logging.DEBUG)

        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            with suppress_logs("suppress_test_module2", re.compile(r"suppress me")):
                logger.info("keep this one")
            self.assertEqual(captured, ["keep this one"])
        finally:
            logger.removeHandler(handler)

    def test_filter_removed_after_context_exits(self):
        """After the context exits, logs must no longer be suppressed."""
        logger = logging.getLogger("suppress_test_module3")
        logger.setLevel(logging.DEBUG)

        captured = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = CapturingHandler()
        logger.addHandler(handler)
        try:
            with suppress_logs("suppress_test_module3", re.compile(r"suppress me")):
                pass
            logger.info("suppress me")  # now outside context, must appear
            self.assertIn("suppress me", captured)
        finally:
            logger.removeHandler(handler)

    def test_filter_removed_on_exception(self):
        """Filter must be removed even when an exception exits the context."""
        logger = logging.getLogger("suppress_test_exc")
        filters_before = list(logger.filters)
        try:
            with suppress_logs("suppress_test_exc", re.compile(r"x")):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(logger.filters, filters_before)


# ===========================================================================
# eprint
# ===========================================================================


class TestEprint(unittest.TestCase):
    """eprint must write to MLFLOW_LOGGING_STREAM."""

    def test_eprint_writes_to_stream(self):
        """eprint must produce output on the logging stream when enabled."""
        capture = io.StringIO()
        # with mock.patch.object(MLFLOW_LOGGING_STREAM, "write", side_effect=lambda t: capture.write(t)):
        with mock.patch.object(sys, "stderr", capture):
            enable_logging()
            eprint("hello eprint")
        self.assertIn("hello eprint", capture.getvalue())

    def test_eprint_suppressed_when_disabled(self):
        """eprint must produce no output when the stream is disabled."""
        capture = io.StringIO()
        # with mock.patch.object(MLFLOW_LOGGING_STREAM, "write", side_effect=lambda t: capture.write(t)):
        with mock.patch.object(sys, "stderr", capture):
            disable_logging()
            try:
                eprint("should not appear")
            finally:
                enable_logging()
        self.assertNotIn("should not appear", capture.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
