# scikitplot/_utils/tests/test_exception_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.exception_utils`.

Coverage map
------------
get_stacktrace    with traceback, without traceback,
                  ScikitplotException fallback path, return type  -> TestGetStacktrace

Run standalone::

    python -m unittest scikitplot._utils.tests.test_exception_utils -v

Developer note
--------------
The scikitplot stub in this package's ``__init__.py`` is installed
automatically before any test module is imported, so no per-file stub
injection is needed here.
"""

from __future__ import annotations

import unittest

from ..exception_utils import get_stacktrace, ScikitplotException


# ===========================================================================
# get_stacktrace
# ===========================================================================


class TestGetStacktrace(unittest.TestCase):
    """get_stacktrace must produce a non-empty string for any exception."""

    # -- return type --

    def test_returns_str(self):
        """Return type must be str."""
        try:
            raise ValueError("test error")
        except ValueError as exc:
            result = get_stacktrace(exc)
        self.assertIsInstance(result, str)

    def test_non_empty_result(self):
        """Result must never be an empty string."""
        try:
            raise TypeError("t")
        except TypeError as exc:
            result = get_stacktrace(exc)
        self.assertGreater(len(result.strip()), 0)

    # -- content with live traceback --

    def test_contains_exception_type_name(self):
        """Result must include the exception class name."""
        try:
            raise ValueError("some message")
        except ValueError as exc:
            result = get_stacktrace(exc)
        self.assertIn("ValueError", result)

    def test_contains_exception_message(self):
        """Result must include the exception message string."""
        try:
            raise ValueError("unique_test_message_xyz")
        except ValueError as exc:
            result = get_stacktrace(exc)
        self.assertIn("unique_test_message_xyz", result)

    def test_contains_traceback_info(self):
        """Result must include traceback when one is available."""
        try:
            raise RuntimeError("with traceback")
        except RuntimeError as exc:
            result = get_stacktrace(exc)
        # A real traceback includes either 'Traceback' header or 'File' entries
        self.assertTrue(
            "Traceback" in result or "File" in result,
            msg=f"Expected traceback info in: {result!r}",
        )

    # -- exception with no attached traceback --

    def test_exception_without_traceback_returns_repr(self):
        """An exception constructed without raising must still return a string."""
        exc = ValueError("bare error, no raise")
        result = get_stacktrace(exc)
        self.assertIsInstance(result, str)
        self.assertIn("bare error", result)

    # -- ScikitplotException fallback path --

    def test_scikitplot_exception_returns_repr_only(self):
        """A ScikitplotException must fall back to repr (no format_exception call).

        The implementation catches ScikitplotException inside its try/except
        and returns repr(error) directly, preventing infinite recursion when
        format_exception itself raises a ScikitplotException.
        """
        exc = ScikitplotException("skplt error")
        result = get_stacktrace(exc)
        self.assertIsInstance(result, str)
        self.assertIn("ScikitplotException", result)

    # -- chained exceptions --

    def test_chained_exception_context(self):
        """Chained exceptions (raise ... from ...) must not crash get_stacktrace."""
        try:
            try:
                raise ValueError("original")
            except ValueError as origin:
                raise RuntimeError("wrapped") from origin
        except RuntimeError as exc:
            result = get_stacktrace(exc)
        self.assertIsInstance(result, str)
        self.assertIn("RuntimeError", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
