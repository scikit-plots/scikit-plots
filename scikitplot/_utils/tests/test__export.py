# scikitplot/_utils/tests/test__export.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils._export`.

Coverage map
------------
export_all      normal path, missing __all__, MethodType skip,
                immutable type, non-str public_module               → TestExportAll
export_objects  normal path, missing name, immutable type,
                non-str public_module, empty names list             → TestExportObjects

Run standalone::

    python -m unittest scikitplot._utils.tests.test__export -v
"""

from __future__ import annotations

import types
import unittest

from .._export import export_all, export_objects


# ===========================================================================
# export_all
# ===========================================================================


class TestExportAll(unittest.TestCase):
    """export_all must set __module__ on every name declared in __all__."""

    def _make_ns(self, names=("Foo", "bar"), public_mod="mypkg"):
        """Build a minimal namespace dict with a class and a function."""

        class Foo:
            pass

        def bar():
            pass

        ns = {
            "__all__": list(names),
            "Foo": Foo,
            "bar": bar,
        }
        return ns

    # -- normal path --

    def test_function_module_updated(self):
        """export_all must set __module__ on a plain function."""
        ns = self._make_ns()
        export_all(ns, public_module="mypkg")
        self.assertEqual(ns["bar"].__module__, "mypkg")

    def test_class_module_updated(self):
        """export_all must set __module__ on a class."""
        ns = self._make_ns()
        export_all(ns, public_module="mypkg")
        self.assertEqual(ns["Foo"].__module__, "mypkg")

    def test_returns_none(self):
        """export_all must return None (no return value contract)."""
        ns = self._make_ns()
        result = export_all(ns, public_module="mypkg")
        self.assertIsNone(result)

    def test_only_all_names_are_exported(self):
        """Only names listed in __all__ are touched; extras are left alone."""
        class Extra:
            pass

        original_module = Extra.__module__

        def foo():
            pass

        ns = {
            "__all__": ["foo"],
            "foo": foo,
            "Extra": Extra,
        }
        export_all(ns, public_module="newpkg")
        self.assertEqual(foo.__module__, "newpkg")
        self.assertEqual(Extra.__module__, original_module)

    def test_multiple_names_all_updated(self):
        """All names in __all__ must be updated in a single call."""

        def f1():
            pass

        def f2():
            pass

        def f3():
            pass

        ns = {"__all__": ["f1", "f2", "f3"], "f1": f1, "f2": f2, "f3": f3}
        export_all(ns, public_module="pkg.sub")
        for fn in [f1, f2, f3]:
            with self.subTest(fn=fn):
                self.assertEqual(fn.__module__, "pkg.sub")

    # -- MethodType is silently skipped --

    def test_bound_method_skipped_without_error(self):
        """A bound method in __all__ must be silently skipped (no exception)."""

        class Owner:
            def meth(self):
                pass

        instance = Owner()
        bound = instance.meth

        ns = {"__all__": ["bound"], "bound": bound}
        try:
            export_all(ns, public_module="mypkg")
        except Exception as exc:
            self.fail(f"export_all raised on bound method: {exc}")

    # -- missing __all__ raises AttributeError --

    def test_missing_all_raises_attribute_error(self):
        """export_all must raise AttributeError when __all__ is absent."""
        ns = {"foo": lambda: None}
        with self.assertRaises(AttributeError):
            export_all(ns, public_module="mypkg")

    # -- non-str public_module raises TypeError --

    def test_non_str_public_module_raises_type_error(self):
        """export_all must raise TypeError when public_module is not a str."""
        ns = self._make_ns()
        with self.assertRaises(TypeError):
            export_all(ns, public_module=123)

    def test_none_public_module_raises_type_error(self):
        """None as public_module must raise TypeError."""
        ns = self._make_ns()
        with self.assertRaises(TypeError):
            export_all(ns, public_module=None)

    # -- immutable built-in types are skipped gracefully --

    def test_builtin_type_skipped_gracefully(self):
        """A C-extension type (immutable __module__) must not raise."""
        ns = {"__all__": ["int"], "int": int}
        try:
            export_all(ns, public_module="mypkg")
        except Exception as exc:
            self.fail(f"export_all raised on built-in type: {exc}")


# ===========================================================================
# export_objects
# ===========================================================================


class TestExportObjects(unittest.TestCase):
    """export_objects must set __module__ on a selected subset of names."""

    # -- normal path --

    def test_single_name_module_updated(self):
        """export_objects must update __module__ for a single named function."""

        def fn():
            pass

        ns = {"fn": fn, "other": lambda: None}
        export_objects(ns, public_module="mypkg", names=["fn"])
        self.assertEqual(fn.__module__, "mypkg")

    def test_other_names_not_touched(self):
        """Names not in 'names' must retain their original __module__."""

        def fn():
            pass

        def other():
            pass

        original = other.__module__
        ns = {"fn": fn, "other": other}
        export_objects(ns, public_module="mypkg", names=["fn"])
        self.assertEqual(other.__module__, original)

    def test_multiple_names_updated(self):
        """All names in the 'names' list must have __module__ updated."""

        def a():
            pass

        def b():
            pass

        ns = {"a": a, "b": b}
        export_objects(ns, public_module="pkg.api", names=["a", "b"])
        self.assertEqual(a.__module__, "pkg.api")
        self.assertEqual(b.__module__, "pkg.api")

    def test_empty_names_list_is_noop(self):
        """An empty names iterable must leave the namespace unchanged."""

        def fn():
            pass

        original = fn.__module__
        ns = {"fn": fn}
        export_objects(ns, public_module="mypkg", names=[])
        self.assertEqual(fn.__module__, original)

    def test_returns_none(self):
        """export_objects must return None."""

        def fn():
            pass

        ns = {"fn": fn}
        result = export_objects(ns, public_module="mypkg", names=["fn"])
        self.assertIsNone(result)

    # -- missing name raises KeyError --

    def test_missing_name_raises_key_error(self):
        """export_objects must raise KeyError when a name is absent."""
        ns = {"fn": lambda: None}
        with self.assertRaises(KeyError):
            export_objects(ns, public_module="mypkg", names=["missing"])

    # -- non-str public_module raises TypeError --

    def test_non_str_public_module_raises_type_error(self):
        """export_objects must raise TypeError for non-str public_module."""
        ns = {"fn": lambda: None}
        with self.assertRaises(TypeError):
            export_objects(ns, public_module=42, names=["fn"])

    # -- immutable object raises TypeError --

    def test_immutable_object_raises_type_error(self):
        """export_objects must raise TypeError for objects with immutable __module__."""
        ns = {"builtin": int}
        with self.assertRaises(TypeError):
            export_objects(ns, public_module="mypkg", names=["builtin"])


# ===========================================================================
# Module-level
# ===========================================================================


class TestExportModule(unittest.TestCase):
    """Both public symbols must be importable and callable."""

    def test_export_all_callable(self):
        """export_all must be callable."""
        self.assertTrue(callable(export_all))

    def test_export_objects_callable(self):
        """export_objects must be callable."""
        self.assertTrue(callable(export_objects))


if __name__ == "__main__":
    unittest.main(verbosity=2)
