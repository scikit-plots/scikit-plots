# scikitplot/_compat/tests/test__sklearn.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._compat._sklearn`.

Runs standalone (``python -m unittest scikitplot._compat.tests.test__sklearn``)
or under pytest.

Coverage map
------------
learning_curve_params   Version branch (pre-1.6 / 1.6+),
                        key correctness, value pass-through,
                        None / empty-dict / nested inputs    → TestLearningCurveParams
"""

from __future__ import annotations

import unittest
import unittest.mock as mock

import sklearn

from packaging.version import Version

from .._sklearn import learning_curve_params


# ---------------------------------------------------------------------------
# Runtime version flag — used to assert the correct branch is taken without
# hard-coding the exact sklearn version in every test.
# ---------------------------------------------------------------------------
_SKLEARN_PRE_16 = Version(sklearn.__version__) < Version("1.6.0")
_EXPECTED_KEY = "fit_params" if _SKLEARN_PRE_16 else "params"
_OTHER_KEY = "params" if _SKLEARN_PRE_16 else "fit_params"


# ===========================================================================
# learning_curve_params
# ===========================================================================


class TestLearningCurveParams(unittest.TestCase):
    """learning_curve_params must route to the correct kwarg key for sklearn."""

    # ------------------------------------------------------------------
    # Return-type invariants (always true regardless of sklearn version)
    # ------------------------------------------------------------------

    def test_returns_dict(self):
        """Must always return a dict."""
        result = learning_curve_params({"a": 1})
        self.assertIsInstance(result, dict)

    def test_returns_exactly_one_key(self):
        """Returned dict must have exactly one key."""
        result = learning_curve_params({"a": 1})
        self.assertEqual(len(result), 1)

    def test_value_passed_through_unchanged(self):
        """The supplied value must appear unmodified in the returned dict."""
        val = {"alpha": 0.01, "max_iter": 100}
        result = learning_curve_params(val)
        self.assertIs(list(result.values())[0], val)

    def test_none_value_passed_through(self):
        """None must be accepted and passed through as the dict value."""
        result = learning_curve_params(None)
        self.assertIsNone(list(result.values())[0])

    def test_empty_dict_value_passed_through(self):
        """An empty dict must be passed through as the value."""
        result = learning_curve_params({})
        self.assertEqual(list(result.values())[0], {})

    def test_nested_dict_value_passed_through(self):
        """A nested dict value must be passed through without modification."""
        val = {"outer": {"inner": [1, 2, 3]}}
        result = learning_curve_params(val)
        self.assertIs(list(result.values())[0], val)

    # ------------------------------------------------------------------
    # Version-correct key selection
    # ------------------------------------------------------------------

    def test_correct_key_for_running_sklearn(self):
        """The returned key must match the sklearn version in this environment."""
        result = learning_curve_params({"x": 1})
        self.assertIn(_EXPECTED_KEY, result)

    def test_wrong_key_absent_for_running_sklearn(self):
        """The key for the OTHER sklearn version must not appear."""
        result = learning_curve_params({"x": 1})
        self.assertNotIn(_OTHER_KEY, result)

    # ------------------------------------------------------------------
    # Explicit branch testing via mock — covers BOTH branches regardless
    # of which sklearn version is installed in CI.
    # ------------------------------------------------------------------

    def test_pre_16_branch_uses_fit_params(self):
        """With sklearn < 1.6.0, the key must be 'fit_params'."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=True,
        ):
            result = learning_curve_params({"a": 1})
        self.assertIn("fit_params", result)
        self.assertNotIn("params", result)

    def test_pre_16_branch_value_correct(self):
        """With sklearn < 1.6.0, the value under 'fit_params' must be the input."""
        val = {"alpha": 0.5}
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=True,
        ):
            result = learning_curve_params(val)
        self.assertIs(result["fit_params"], val)

    def test_16_plus_branch_uses_params(self):
        """With sklearn >= 1.6.0, the key must be 'params'."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ):
            result = learning_curve_params({"a": 1})
        self.assertIn("params", result)
        self.assertNotIn("fit_params", result)

    def test_16_plus_branch_value_correct(self):
        """With sklearn >= 1.6.0, the value under 'params' must be the input."""
        val = {"max_iter": 200}
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ):
            result = learning_curve_params(val)
        self.assertIs(result["params"], val)

    # ------------------------------------------------------------------
    # Edge values for the val argument
    # ------------------------------------------------------------------

    def test_none_with_pre_16_mock(self):
        """None value in pre-1.6 mode must map to {'fit_params': None}."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=True,
        ):
            result = learning_curve_params(None)
        self.assertEqual(result, {"fit_params": None})

    def test_none_with_16_plus_mock(self):
        """None value in 1.6+ mode must map to {'params': None}."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ):
            result = learning_curve_params(None)
        self.assertEqual(result, {"params": None})

    def test_empty_dict_pre_16_mock(self):
        """Empty dict in pre-1.6 mode must map to {'fit_params': {}}."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=True,
        ):
            result = learning_curve_params({})
        self.assertEqual(result, {"fit_params": {}})

    def test_empty_dict_16_plus_mock(self):
        """Empty dict in 1.6+ mode must map to {'params': {}}."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ):
            result = learning_curve_params({})
        self.assertEqual(result, {"params": {}})

    # ------------------------------------------------------------------
    # _version_predates is called with the correct arguments
    # ------------------------------------------------------------------

    def test_version_predates_called_with_sklearn_and_version(self):
        """_version_predates must be called with (sklearn, '1.6.0')."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ) as mock_vp:
            learning_curve_params({"x": 1})
        mock_vp.assert_called_once_with(sklearn, "1.6.0")

    def test_version_predates_called_exactly_once(self):
        """_version_predates must be called exactly once per invocation."""
        with mock.patch(
            "scikitplot._compat._sklearn._version_predates",
            return_value=False,
        ) as mock_vp:
            learning_curve_params({"x": 1})
        self.assertEqual(mock_vp.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
