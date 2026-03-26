# scikitplot/_utils/tests/test__inspect.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils._inspect`.

Coverage map
------------
ismethod          bound vs unbound vs function         → TestIsmethod
isfunction        functions, lambdas, non-functions    → TestIsfunction
iscode            code objects vs non-code             → TestIscode
getargs           various arg signatures               → TestGetargs
getargspec        normal, method, non-function error   → TestGetargspec
formatargspec     formatting with/without defaults     → TestFormatargspec
formatargvalues   formatting from frame-like data      → TestFormatargvalues
__all__           declared exports                     → TestInspectModule

Run standalone::

    python -m unittest scikitplot._utils.tests.test__inspect -v
"""

from __future__ import annotations

import types
import unittest

from .._inspect import (
    formatargspec,
    formatargvalues,
    getargspec,
    getargvalues,
    getargs,
    iscode,
    isfunction,
    ismethod,
)


# ===========================================================================
# ismethod
# ===========================================================================


class TestIsmethod(unittest.TestCase):
    """ismethod must return True only for bound method objects."""

    def test_bound_method_is_true(self):
        """A bound method must return True."""

        class C:
            def m(self):
                pass

        self.assertTrue(ismethod(C().m))

    def test_plain_function_is_false(self):
        """A plain function must return False."""

        def fn():
            pass

        self.assertFalse(ismethod(fn))

    def test_lambda_is_false(self):
        """A lambda must return False."""
        self.assertFalse(ismethod(lambda: None))

    def test_class_is_false(self):
        """A class object must return False."""
        self.assertFalse(ismethod(int))

    def test_builtin_function_is_false(self):
        """A built-in function must return False."""
        self.assertFalse(ismethod(len))

    def test_none_is_false(self):
        """None must return False."""
        self.assertFalse(ismethod(None))


# ===========================================================================
# isfunction
# ===========================================================================


class TestIsfunction(unittest.TestCase):
    """isfunction must return True only for user-defined functions."""

    def test_plain_function_is_true(self):
        """A plain def function must return True."""

        def fn():
            pass

        self.assertTrue(isfunction(fn))

    def test_lambda_is_true(self):
        """A lambda is a user-defined FunctionType — must return True."""
        self.assertTrue(isfunction(lambda: None))

    def test_bound_method_is_false(self):
        """A bound method is not a FunctionType — must return False."""

        class C:
            def m(self):
                pass

        self.assertFalse(isfunction(C().m))

    def test_builtin_is_false(self):
        """A built-in function must return False."""
        self.assertFalse(isfunction(len))

    def test_class_is_false(self):
        """A class must return False."""
        self.assertFalse(isfunction(int))

    def test_int_is_false(self):
        """An integer must return False."""
        self.assertFalse(isfunction(42))


# ===========================================================================
# iscode
# ===========================================================================


class TestIscode(unittest.TestCase):
    """iscode must return True only for code objects."""

    def test_function_code_is_true(self):
        """A function's __code__ attribute must return True."""

        def fn():
            pass

        self.assertTrue(iscode(fn.__code__))

    def test_plain_function_is_false(self):
        """A function itself (not its __code__) must return False."""

        def fn():
            pass

        self.assertFalse(iscode(fn))

    def test_string_is_false(self):
        """A string must return False."""
        self.assertFalse(iscode("not code"))

    def test_none_is_false(self):
        """None must return False."""
        self.assertFalse(iscode(None))


# ===========================================================================
# getargs
# ===========================================================================


class TestGetargs(unittest.TestCase):
    """getargs must extract (args, varargs, varkw) from a code object."""

    def test_simple_positional_args(self):
        """A function with simple positional args must return them in order."""

        def fn(a, b, c):
            pass

        args, varargs, varkw = getargs(fn.__code__)
        self.assertEqual(args, ["a", "b", "c"])
        self.assertIsNone(varargs)
        self.assertIsNone(varkw)

    def test_no_args(self):
        """A function with no parameters must return empty list."""

        def fn():
            pass

        args, varargs, varkw = getargs(fn.__code__)
        self.assertEqual(args, [])
        self.assertIsNone(varargs)
        self.assertIsNone(varkw)

    def test_varargs(self):
        """A *args parameter must be captured in varargs."""

        def fn(*args):
            pass

        _, varargs, _ = getargs(fn.__code__)
        self.assertEqual(varargs, "args")

    def test_varkw(self):
        """A **kwargs parameter must be captured in varkw."""

        def fn(**kwargs):
            pass

        _, _, varkw = getargs(fn.__code__)
        self.assertEqual(varkw, "kwargs")

    def test_all_combined(self):
        """A full signature (pos + *args + **kwargs) must be extracted correctly."""

        def fn(a, b, *args, **kwargs):
            pass

        args, varargs, varkw = getargs(fn.__code__)
        self.assertEqual(args, ["a", "b"])
        self.assertEqual(varargs, "args")
        self.assertEqual(varkw, "kwargs")

    def test_non_code_raises_type_error(self):
        """Passing a non-code object must raise TypeError."""
        with self.assertRaises(TypeError):
            getargs("not a code object")


# ===========================================================================
# getargspec
# ===========================================================================


class TestGetargspec(unittest.TestCase):
    """getargspec must return (args, varargs, varkw, defaults) for functions."""

    def test_simple_function(self):
        """Simple function without defaults must have None as defaults."""

        def fn(a, b):
            pass

        args, varargs, varkw, defaults = getargspec(fn)
        self.assertEqual(args, ["a", "b"])
        self.assertIsNone(varargs)
        self.assertIsNone(varkw)
        self.assertIsNone(defaults)

    def test_function_with_defaults(self):
        """Defaults must be captured as a tuple of their values."""

        def fn(a, b=2, c=3):
            pass

        args, varargs, varkw, defaults = getargspec(fn)
        self.assertEqual(args, ["a", "b", "c"])
        self.assertEqual(defaults, (2, 3))

    def test_bound_method_unwrapped(self):
        """getargspec on a bound method must unwrap __func__ first."""

        class C:
            def m(self, x):
                pass

        args, _, _, _ = getargspec(C.m)
        self.assertIn("self", args)
        self.assertIn("x", args)

    def test_non_function_raises_type_error(self):
        """Passing a non-function must raise TypeError."""
        with self.assertRaises(TypeError):
            getargspec("not a function")

    def test_non_function_int_raises_type_error(self):
        """Passing an integer must raise TypeError."""
        with self.assertRaises(TypeError):
            getargspec(42)

    def test_varargs_captured(self):
        """*args must appear in the varargs slot."""

        def fn(a, *args):
            pass

        _, varargs, _, _ = getargspec(fn)
        self.assertEqual(varargs, "args")

    def test_kwargs_captured(self):
        """**kwargs must appear in the varkw slot."""

        def fn(a, **kwargs):
            pass

        _, _, varkw, _ = getargspec(fn)
        self.assertEqual(varkw, "kwargs")


# ===========================================================================
# formatargspec
# ===========================================================================


class TestFormatargspec(unittest.TestCase):
    """formatargspec must produce a valid argument-list string."""

    def test_no_args_gives_empty_parens(self):
        """No arguments must give '()'."""
        result = formatargspec([])
        self.assertEqual(result, "()")

    def test_single_arg(self):
        """A single arg must appear between parens."""
        result = formatargspec(["a"])
        self.assertEqual(result, "(a)")

    def test_multiple_args(self):
        """Multiple args must be comma-separated."""
        result = formatargspec(["a", "b", "c"])
        self.assertEqual(result, "(a, b, c)")

    def test_varargs(self):
        """A *args must be formatted with a leading asterisk."""
        result = formatargspec(["a"], varargs="args")
        self.assertEqual(result, "(a, *args)")

    def test_varkw(self):
        """A **kwargs must be formatted with double asterisks."""
        result = formatargspec(["a"], varkw="kwargs")
        self.assertEqual(result, "(a, **kwargs)")

    def test_defaults_applied_to_last_args(self):
        """Default values must be attached to the last N arguments."""
        result = formatargspec(["a", "b", "c"], defaults=(10, 20))
        self.assertIn("b=10", result)
        self.assertIn("c=20", result)
        self.assertNotIn("a=", result)

    def test_full_signature(self):
        """A complete signature must render with all parts."""
        result = formatargspec(
            ["a", "b"],
            varargs="args",
            varkw="kwargs",
            defaults=(99,),
        )
        self.assertIn("a", result)
        self.assertIn("b=99", result)
        self.assertIn("*args", result)
        self.assertIn("**kwargs", result)


# ===========================================================================
# formatargvalues
# ===========================================================================


class TestFormatargvalues(unittest.TestCase):
    """formatargvalues must produce an argument-value string from frame data."""

    def test_simple_values(self):
        """Simple positional values must be rendered as name=value."""
        result = formatargvalues(["x", "y"], None, None, {"x": 1, "y": 2})
        self.assertIn("x=1", result)
        self.assertIn("y=2", result)

    def test_empty_args(self):
        """No args must produce empty parens."""
        result = formatargvalues([], None, None, {})
        self.assertEqual(result, "()")

    def test_with_varargs(self):
        """varargs entry must appear with leading asterisk."""
        result = formatargvalues(
            ["a"],
            "myargs",
            None,
            {"a": 1, "myargs": (2, 3)},
        )
        self.assertIn("*myargs", result)

    def test_with_varkw(self):
        """varkw entry must appear with double asterisks."""
        result = formatargvalues(
            ["a"],
            None,
            "mykw",
            {"a": 1, "mykw": {"k": "v"}},
        )
        self.assertIn("**mykw", result)


# ===========================================================================
# Module-level
# ===========================================================================


class TestInspectModule(unittest.TestCase):
    """__all__ must contain getargspec and formatargspec."""

    def test_all_contains_getargspec(self):
        """__all__ must include 'getargspec'."""
        from .. import _inspect
        self.assertIn("getargspec", _inspect.__all__)

    def test_all_contains_formatargspec(self):
        """__all__ must include 'formatargspec'."""
        from .. import _inspect
        self.assertIn("formatargspec", _inspect.__all__)

    def test_public_symbols_callable(self):
        """All __all__ symbols must be callable."""
        from .. import _inspect
        for name in _inspect.__all__:
            obj = getattr(_inspect, name)
            with self.subTest(name=name):
                self.assertTrue(callable(obj))


if __name__ == "__main__":
    unittest.main(verbosity=2)
