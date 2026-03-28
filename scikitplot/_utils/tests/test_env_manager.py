# scikitplot/_utils/tests/test_env_manager.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.env_manager`.

Coverage map
------------
constants         LOCAL, CONDA, VIRTUALENV, UV values and types   -> TestEnvManagerConstants
validate          all valid values pass, invalid raises,
                  empty string, None, numeric, case-sensitive      -> TestValidate

Run standalone::

    python -m unittest scikitplot._utils.tests.test_env_manager -v
"""

from __future__ import annotations

import unittest

from ..env_manager import CONDA, LOCAL, UV, VIRTUALENV, validate
from ..exception_utils import ScikitplotException


# ===========================================================================
# Constants
# ===========================================================================


class TestEnvManagerConstants(unittest.TestCase):
    """Module-level constants must have the expected string values."""

    def test_local_value(self):
        """LOCAL must equal 'local'."""
        self.assertEqual(LOCAL, "local")

    def test_conda_value(self):
        """CONDA must equal 'conda'."""
        self.assertEqual(CONDA, "conda")

    def test_virtualenv_value(self):
        """VIRTUALENV must equal 'virtualenv'."""
        self.assertEqual(VIRTUALENV, "virtualenv")

    def test_uv_value(self):
        """UV must equal 'uv'."""
        self.assertEqual(UV, "uv")

    def test_all_constants_are_strings(self):
        """All constants must be str instances."""
        for const in (LOCAL, CONDA, VIRTUALENV, UV):
            with self.subTest(const=const):
                self.assertIsInstance(const, str)

    def test_all_constants_are_non_empty(self):
        """All constants must be non-empty strings."""
        for const in (LOCAL, CONDA, VIRTUALENV, UV):
            with self.subTest(const=const):
                self.assertGreater(len(const), 0)

    def test_constants_are_distinct(self):
        """No two constants may share the same value."""
        values = [LOCAL, CONDA, VIRTUALENV, UV]
        self.assertEqual(len(values), len(set(values)))


# ===========================================================================
# validate
# ===========================================================================


class TestValidate(unittest.TestCase):
    """validate must accept only known env_manager values."""

    # -- valid values return None --

    def test_local_is_valid(self):
        """LOCAL must be accepted without raising."""
        result = validate(LOCAL)
        self.assertIsNone(result)

    def test_conda_is_valid(self):
        """CONDA must be accepted without raising."""
        result = validate(CONDA)
        self.assertIsNone(result)

    def test_virtualenv_is_valid(self):
        """VIRTUALENV must be accepted without raising."""
        result = validate(VIRTUALENV)
        self.assertIsNone(result)

    def test_uv_is_valid(self):
        """UV must be accepted without raising."""
        result = validate(UV)
        self.assertIsNone(result)

    def test_all_valid_values_via_subtest(self):
        """Every known manager string must pass validation."""
        for value in (LOCAL, CONDA, VIRTUALENV, UV):
            with self.subTest(value=value):
                try:
                    validate(value)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"validate({value!r}) raised unexpectedly: {exc}")

    # -- invalid values raise ScikitplotException --

    def test_invalid_string_raises(self):
        """An unknown string must raise ScikitplotException."""
        with self.assertRaises(ScikitplotException):
            validate("pipenv")

    def test_empty_string_raises(self):
        """An empty string must raise ScikitplotException."""
        with self.assertRaises(ScikitplotException):
            validate("")

    def test_none_raises(self):
        """None must raise ScikitplotException (not in allowed list)."""
        with self.assertRaises(ScikitplotException):
            validate(None)

    def test_integer_raises(self):
        """An integer must raise ScikitplotException."""
        with self.assertRaises(ScikitplotException):
            validate(42)

    def test_uppercase_local_raises(self):
        """Validation must be case-sensitive; 'LOCAL' is not 'local'."""
        with self.assertRaises(ScikitplotException):
            validate("LOCAL")

    def test_uppercase_conda_raises(self):
        """'CONDA' must not be accepted when only 'conda' is valid."""
        with self.assertRaises(ScikitplotException):
            validate("CONDA")

    def test_whitespace_padded_value_raises(self):
        """Values with surrounding whitespace must not be accepted."""
        with self.assertRaises(ScikitplotException):
            validate(" local ")

    def test_partial_match_raises(self):
        """Partial matches like 'loc' must not be accepted."""
        with self.assertRaises(ScikitplotException):
            validate("loc")

    def test_error_message_contains_invalid_value(self):
        """The exception message must identify the invalid value."""
        with self.assertRaises(ScikitplotException) as ctx:
            validate("bad_manager")
        self.assertIn("bad_manager", str(ctx.exception))

    def test_error_message_mentions_allowed_values(self):
        """The exception message must list the allowed values."""
        with self.assertRaises(ScikitplotException) as ctx:
            validate("bad_manager")
        msg = str(ctx.exception)
        # at least one of the valid values must be mentioned
        valid_mentioned = any(v in msg for v in (LOCAL, CONDA, VIRTUALENV, UV))
        self.assertTrue(valid_mentioned, msg=f"Allowed values not in message: {msg!r}")

    def test_exception_has_error_code(self):
        """ScikitplotException must carry the expected error_code attribute."""
        with self.assertRaises(ScikitplotException) as ctx:
            validate("invalid")
        self.assertEqual(ctx.exception.error_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
