# config/tests/test__distributor_init.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._distributor_init`.

Coverage map
------------
_distributor_init          Import guard, graceful missing local module  → TestDistributorInit
_distributor_init_local    Optional override module (skipped if absent) → TestDistributorInitLocal

Notes
-----
Developer note
    ``_distributor_init.py`` is intentionally minimal: it tries to import an
    optional ``_distributor_init_local`` override (provided by downstream
    distributors) and silently ignores ``ImportError`` when the override is
    absent.  Tests must verify this contract without requiring the override.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
import unittest.mock as mock


class TestDistributorInit(unittest.TestCase):
    """_distributor_init must import cleanly and suppress missing-local errors."""

    def test_import_does_not_raise(self):
        """Importing _distributor_init must never raise, even without the local module."""
        try:
            from .. import _distributor_init  # noqa: F401
        except ImportError as exc:
            self.fail(
                f"_distributor_init raised ImportError even though "
                f"_distributor_init_local is optional: {exc}"
            )

    def test_module_is_importable_as_attribute(self):
        """_distributor_init must be accessible as a submodule attribute."""
        import importlib
        # Get the parent package name dynamically from the test's own package
        parent_pkg = __name__.rsplit(".", 2)[0]  # e.g. scikitplot.config
        try:
            mod = importlib.import_module(".._distributor_init", package=__name__.rsplit(".", 1)[0])
            self.assertIsNotNone(mod)
        except ImportError:
            self.fail("_distributor_init could not be imported via importlib")

    def test_local_import_error_is_suppressed(self):
        """
        If _distributor_init_local is absent, no ImportError must propagate.

        This is the core contract: distributors may or may not provide
        _distributor_init_local; the module must handle both cases.
        """
        parent_pkg = __name__.rsplit(".", 1)[0]  # e.g. scikitplot.config

        # Temporarily remove any cached local module so the try/except fires
        local_key = f"{parent_pkg}._distributor_init_local"
        original = sys.modules.pop(local_key, None)
        try:
            # Re-import to exercise the except ImportError branch
            dist_key = f"{parent_pkg}._distributor_init"
            sys.modules.pop(dist_key, None)
            try:
                mod = importlib.import_module("scikitplot.config._distributor_init", package=f"{parent_pkg}.tests")
                # Must succeed even when local module is absent
                self.assertIsNotNone(mod)
            except ImportError as exc:
                self.fail(
                    f"_distributor_init raised ImportError on missing local module: {exc}"
                )
        finally:
            if original is not None:
                sys.modules[local_key] = original

    def test_local_override_is_loaded_when_present(self):
        """When _distributor_init_local exists, it must be loaded without error."""
        parent_pkg = __name__.rsplit(".", 1)[0]

        # Inject a mock local module
        local_key = f"{parent_pkg}._distributor_init_local"
        sentinel = types.ModuleType(local_key)
        sentinel.LOADED = True

        with mock.patch.dict(sys.modules, {local_key: sentinel}):
            # Re-import _distributor_init to pick up the mock local
            dist_key = f"{parent_pkg}._distributor_init"
            sys.modules.pop(dist_key, None)
            try:
                mod = importlib.import_module("scikitplot.config._distributor_init", package=f"{parent_pkg}.tests")
                self.assertIsNotNone(mod)
            except ImportError as exc:
                self.fail(f"Import failed even with local module injected: {exc}")


class TestDistributorInitLocal(unittest.TestCase):
    """
    Optional: tests for _distributor_init_local when present.

    All tests in this class are skipped when the override module is absent,
    which is the normal case for a standard source distribution.
    """

    @classmethod
    def setUpClass(cls):
        parent_pkg = __name__.rsplit(".", 1)[0]
        local_key = f"{parent_pkg}._distributor_init_local"
        try:
            cls._local = importlib.import_module(
                "scikitplot.config._distributor_init_local",
                package=f"{parent_pkg}.tests",
            )
        except ImportError:
            cls._local = None

    def _require_local(self):
        if self._local is None:
            self.skipTest("_distributor_init_local not present in this distribution")

    def test_local_module_is_a_module(self):
        """If the local override exists, it must be a proper module object."""
        self._require_local()
        self.assertIsInstance(self._local, types.ModuleType)


if __name__ == "__main__":
    unittest.main(verbosity=2)
