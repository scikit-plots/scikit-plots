# scikitplot/_utils/tests/test_os_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.os`.

Coverage map
------------
is_windows     returns bool, correct for current platform,
               responds correctly to os.name override          → TestIsWindows

Run standalone::

    python -m unittest scikitplot._utils.tests.test_os_utils -v
"""

from __future__ import annotations

import os
import sys
import unittest
import unittest.mock as mock

from ..os import is_windows


# ===========================================================================
# is_windows
# ===========================================================================


class TestIsWindows(unittest.TestCase):
    """is_windows must return True only when os.name == 'nt'."""

    # -- return type --

    def test_returns_bool(self):
        """is_windows must return a bool (not a truthy/falsy object)."""
        result = is_windows()
        self.assertIsInstance(result, bool)

    # -- current platform --

    def test_correct_for_current_platform(self):
        """Result must match os.name == 'nt' on the current platform."""
        expected = os.name == "nt"
        self.assertEqual(is_windows(), expected)

    def test_false_on_posix_platforms(self):
        """On any POSIX system (os.name == 'posix'), is_windows must be False."""
        if os.name == "posix":
            self.assertFalse(is_windows())
        else:
            self.skipTest("Not running on POSIX")

    # -- behaviour under mocked os.name --

    def test_true_when_os_name_is_nt(self):
        """is_windows must return True when os.name is patched to 'nt'."""
        from .. import os as _utils_os
        with mock.patch.object(_utils_os._os, "name", "nt"):
            self.assertTrue(is_windows())

    def test_false_when_os_name_is_posix(self):
        """is_windows must return False when os.name is patched to 'posix'."""
        from .. import os as _utils_os
        with mock.patch.object(_utils_os._os, "name", "posix"):
            self.assertFalse(is_windows())

    def test_false_when_os_name_is_java(self):
        """is_windows must return False for Jython 'java' os.name."""
        from .. import os as _utils_os
        with mock.patch.object(_utils_os._os, "name", "java"):
            self.assertFalse(is_windows())

    # -- callable --

    def test_is_callable(self):
        """is_windows must be callable."""
        self.assertTrue(callable(is_windows))

    def test_accepts_no_arguments(self):
        """is_windows must be callable with no arguments."""
        try:
            is_windows()
        except TypeError as exc:
            self.fail(f"is_windows raised TypeError with no args: {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
