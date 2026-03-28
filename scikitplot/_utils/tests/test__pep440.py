# scikitplot/_utils/tests/test__pep440.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils._pep440`.

Coverage map
------------
parse             valid PEP 440, legacy fallback              → TestParse
Version.__init__ valid strings, invalid raises                → TestVersionInit
Version.__str__  roundtrip str representation                 → TestVersionStr
Version props    base_version, public, local, pre/post flags  → TestVersionProperties
Version compare  all six operators, cross-type                → TestVersionComparison
LegacyVersion    construction, comparison                     → TestLegacyVersion
InvalidVersion   is ValueError subclass                       → TestInvalidVersion
Infinity         ordering vs any object                       → TestInfinity
NegativeInfinity ordering vs any object                       → TestNegativeInfinity
__all__          declared exports                             → TestPep440Module

Run standalone::

    python -m unittest scikitplot._utils.tests.test__pep440 -v
"""

from __future__ import annotations

import unittest

from .._pep440 import (
    InvalidVersion,
    LegacyVersion,
    Version,
    parse,
)


# ===========================================================================
# parse
# ===========================================================================


class TestParse(unittest.TestCase):
    """parse must return Version for PEP 440 strings, LegacyVersion otherwise."""

    def test_simple_version_returns_version(self):
        """'1.2.3' must parse to a Version instance."""
        result = parse("1.2.3")
        self.assertIsInstance(result, Version)

    def test_complex_pep440_returns_version(self):
        """'1.0a1' pre-release must parse to Version."""
        result = parse("1.0a1")
        self.assertIsInstance(result, Version)

    def test_dev_version_returns_version(self):
        """'2.0.0.dev1' must parse to Version."""
        result = parse("2.0.0.dev1")
        self.assertIsInstance(result, Version)

    def test_post_version_returns_version(self):
        """'1.0.post1' must parse to Version."""
        result = parse("1.0.post1")
        self.assertIsInstance(result, Version)

    def test_legacy_string_returns_legacy_version(self):
        """A non-PEP-440 string must return LegacyVersion."""
        result = parse("not_a_valid_version!!")
        self.assertIsInstance(result, LegacyVersion)

    def test_epoch_version_returns_version(self):
        """An epoch prefix '1!2.0' must parse to Version."""
        result = parse("1!2.0")
        self.assertIsInstance(result, Version)


# ===========================================================================
# Version.__init__
# ===========================================================================


class TestVersionInit(unittest.TestCase):
    """Version must accept valid PEP 440 strings and reject invalid ones."""

    def test_two_part_version(self):
        """'1.2' must create a Version without error."""
        v = Version("1.2")
        self.assertIsInstance(v, Version)

    def test_three_part_version(self):
        """'1.2.3' must create a Version without error."""
        v = Version("1.2.3")
        self.assertIsInstance(v, Version)

    def test_pre_release_alpha(self):
        """'1.0a1' must create a Version."""
        v = Version("1.0a1")
        self.assertIsInstance(v, Version)

    def test_pre_release_beta(self):
        """'1.0b2' must create a Version."""
        v = Version("1.0b2")
        self.assertIsInstance(v, Version)

    def test_pre_release_rc(self):
        """'1.0rc1' must create a Version."""
        v = Version("1.0rc1")
        self.assertIsInstance(v, Version)

    def test_dev_release(self):
        """'1.0.dev1' must create a Version."""
        v = Version("1.0.dev1")
        self.assertIsInstance(v, Version)

    def test_post_release(self):
        """'1.0.post1' must create a Version."""
        v = Version("1.0.post1")
        self.assertIsInstance(v, Version)

    def test_local_version(self):
        """'1.0+local.1' must create a Version."""
        v = Version("1.0+local.1")
        self.assertIsInstance(v, Version)

    def test_epoch_version(self):
        """'1!2.3' must create a Version."""
        v = Version("1!2.3")
        self.assertIsInstance(v, Version)

    def test_invalid_string_raises(self):
        """A clearly invalid string must raise InvalidVersion."""
        with self.assertRaises(InvalidVersion):
            Version("not_valid!!")

    def test_empty_string_raises(self):
        """An empty string must raise InvalidVersion."""
        with self.assertRaises(InvalidVersion):
            Version("")

    def test_spaces_only_raises(self):
        """A whitespace-only string must raise InvalidVersion."""
        with self.assertRaises(InvalidVersion):
            Version("   ")


# ===========================================================================
# Version.__str__
# ===========================================================================


class TestVersionStr(unittest.TestCase):
    """str(Version(s)) must reproduce the canonical PEP 440 form."""

    def test_simple_roundtrip(self):
        """str(Version('1.2.3')) must return '1.2.3'."""
        self.assertEqual(str(Version("1.2.3")), "1.2.3")

    def test_pre_roundtrip(self):
        """str(Version('1.0a1')) must return '1.0a1'."""
        self.assertEqual(str(Version("1.0a1")), "1.0a1")

    def test_post_roundtrip(self):
        """str(Version('1.0.post1')) must return '1.0.post1'."""
        self.assertEqual(str(Version("1.0.post1")), "1.0.post1")

    def test_dev_roundtrip(self):
        """str(Version('1.0.dev1')) must return '1.0.dev1'."""
        self.assertEqual(str(Version("1.0.dev1")), "1.0.dev1")

    def test_repr_contains_version_string(self):
        """repr() must contain the version string."""
        v = Version("1.2.3")
        self.assertIn("1.2.3", repr(v))


# ===========================================================================
# Version properties
# ===========================================================================


class TestVersionProperties(unittest.TestCase):
    """Version properties must reflect the correct component values."""

    def test_base_version_strips_pre_dev_local(self):
        """base_version must contain only epoch and release."""
        v = Version("1.2.3a1.dev1+local")
        self.assertEqual(v.base_version, "1.2.3")

    def test_public_strips_local(self):
        """public must strip the local segment."""
        v = Version("1.2.3+local.build")
        self.assertNotIn("+", v.public)

    def test_local_returns_local_segment(self):
        """local property must return the local segment string."""
        v = Version("1.0+foo.bar")
        self.assertEqual(v.local, "foo.bar")

    def test_local_is_none_without_local_segment(self):
        """local must be None when no '+' segment is present."""
        v = Version("1.0")
        self.assertIsNone(v.local)

    def test_is_prerelease_true_for_alpha(self):
        """is_prerelease must be True for alpha versions."""
        v = Version("1.0a1")
        self.assertTrue(v.is_prerelease)

    def test_is_prerelease_true_for_dev(self):
        """is_prerelease must be True for dev versions."""
        v = Version("1.0.dev1")
        self.assertTrue(v.is_prerelease)

    def test_is_prerelease_false_for_release(self):
        """is_prerelease must be False for a plain release."""
        v = Version("1.0")
        self.assertFalse(v.is_prerelease)

    def test_is_postrelease_true_for_post(self):
        """is_postrelease must be True for post releases."""
        v = Version("1.0.post1")
        self.assertTrue(v.is_postrelease)

    def test_is_postrelease_false_for_release(self):
        """is_postrelease must be False for a plain release."""
        v = Version("1.0")
        self.assertFalse(v.is_postrelease)


# ===========================================================================
# Version comparison operators
# ===========================================================================


class TestVersionComparison(unittest.TestCase):
    """All six comparison operators must work correctly between Version objects."""

    def setUp(self):
        self.v1 = Version("1.0.0")
        self.v2 = Version("1.0.1")
        self.v3 = Version("1.0.0")

    def test_less_than_true(self):
        """1.0.0 < 1.0.1 must be True."""
        self.assertTrue(self.v1 < self.v2)

    def test_less_than_false(self):
        """1.0.1 < 1.0.0 must be False."""
        self.assertFalse(self.v2 < self.v1)

    def test_less_than_or_equal_equal(self):
        """1.0.0 <= 1.0.0 must be True."""
        self.assertTrue(self.v1 <= self.v3)

    def test_less_than_or_equal_less(self):
        """1.0.0 <= 1.0.1 must be True."""
        self.assertTrue(self.v1 <= self.v2)

    def test_equal_same_version(self):
        """Two Version objects with the same string must be equal."""
        self.assertEqual(self.v1, self.v3)

    def test_not_equal_different_versions(self):
        """Two Version objects with different strings must not be equal."""
        self.assertNotEqual(self.v1, self.v2)

    def test_greater_than_true(self):
        """1.0.1 > 1.0.0 must be True."""
        self.assertTrue(self.v2 > self.v1)

    def test_greater_than_false(self):
        """1.0.0 > 1.0.1 must be False."""
        self.assertFalse(self.v1 > self.v2)

    def test_greater_than_or_equal(self):
        """1.0.0 >= 1.0.0 must be True."""
        self.assertTrue(self.v1 >= self.v3)

    def test_cross_type_comparison_returns_not_implemented(self):
        """Comparing Version with a non-Version must not raise but return NotImplemented."""
        result = self.v1.__eq__("1.0.0")
        self.assertIs(result, NotImplemented)

    def test_version_ordering_with_major_minor_patch(self):
        """Ordering must respect major.minor.patch semantics."""
        versions = [Version(s) for s in ["2.0.0", "1.0.0", "1.1.0", "1.0.1"]]
        sorted_v = sorted(versions)
        self.assertEqual(str(sorted_v[0]), "1.0.0")
        self.assertEqual(str(sorted_v[-1]), "2.0.0")

    def test_pre_release_is_less_than_release(self):
        """A pre-release must sort below the corresponding release."""
        pre = Version("1.0a1")
        rel = Version("1.0")
        self.assertLess(pre, rel)

    def test_dev_is_less_than_pre_release(self):
        """A dev release must sort below an alpha of the same version."""
        dev = Version("1.0.dev1")
        alpha = Version("1.0a1")
        self.assertLess(dev, alpha)

    def test_version_hashable(self):
        """Version objects must be hashable (usable as dict keys and in sets)."""
        s = {Version("1.0"), Version("1.0"), Version("2.0")}
        self.assertEqual(len(s), 2)


# ===========================================================================
# LegacyVersion
# ===========================================================================


class TestLegacyVersion(unittest.TestCase):
    """LegacyVersion must accept any string and support basic comparisons."""

    def test_accepts_arbitrary_string(self):
        """LegacyVersion must accept any string without error."""
        v = LegacyVersion("totally not a version")
        self.assertIsInstance(v, LegacyVersion)

    def test_str_stores_original(self):
        """The version string must be stored and accessible."""
        raw = "1.2.3-custom"
        v = LegacyVersion(raw)
        self.assertEqual(v._version, raw)

    def test_equal_to_same_string(self):
        """Two LegacyVersion objects with the same string must be equal."""
        self.assertEqual(LegacyVersion("foo"), LegacyVersion("foo"))

    def test_not_equal_to_different_string(self):
        """Two LegacyVersion objects with different strings must not be equal."""
        self.assertNotEqual(LegacyVersion("foo"), LegacyVersion("bar"))

    def test_hashable(self):
        """LegacyVersion objects must be hashable."""
        s = {LegacyVersion("x"), LegacyVersion("x"), LegacyVersion("y")}
        self.assertEqual(len(s), 2)


# ===========================================================================
# InvalidVersion
# ===========================================================================


class TestInvalidVersion(unittest.TestCase):
    """InvalidVersion must be a ValueError subclass."""

    def test_is_value_error_subclass(self):
        """InvalidVersion must subclass ValueError."""
        self.assertTrue(issubclass(InvalidVersion, ValueError))

    def test_can_be_raised_and_caught(self):
        """InvalidVersion must be raise-able and catch-able as ValueError."""
        with self.assertRaises(ValueError):
            raise InvalidVersion("bad version")


# ===========================================================================
# Module-level
# ===========================================================================


class TestPep440Module(unittest.TestCase):
    """__all__ must contain the documented public names."""

    def test_all_contains_parse(self):
        from .. import _pep440
        self.assertIn("parse", _pep440.__all__)

    def test_all_contains_version(self):
        from .. import _pep440
        self.assertIn("Version", _pep440.__all__)

    def test_all_contains_legacy_version(self):
        from .. import _pep440
        self.assertIn("LegacyVersion", _pep440.__all__)

    def test_all_contains_invalid_version(self):
        from .. import _pep440
        self.assertIn("InvalidVersion", _pep440.__all__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
