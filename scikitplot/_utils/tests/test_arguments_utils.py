# scikitplot/_utils/tests/test_arguments_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.arguments_utils`.

Coverage map
------------
_get_arg_names    plain function, no-args, *args, **kwargs,
                  defaults, positional-only, wrapped (functools.wraps),
                  lambda, classmethod, staticmethod, instance method,
                  callable object, built-in raises TypeError          -> TestGetArgNames

Run standalone::

    python -m unittest scikitplot._utils.tests.test_arguments_utils -v
"""

from __future__ import annotations

import functools
import unittest

from ..arguments_utils import _get_arg_names


# ===========================================================================
# _get_arg_names
# ===========================================================================


class TestGetArgNames(unittest.TestCase):
    """_get_arg_names must return the inspect.signature parameter names."""

    # -- return type --

    def test_returns_list(self):
        """Return value must be a list."""
        def fn(a, b):
            pass
        result = _get_arg_names(fn)
        self.assertIsInstance(result, list)

    # -- basic signatures --

    def test_no_args_returns_empty_list(self):
        """A function with no parameters must return an empty list."""
        def fn():
            pass
        self.assertEqual(_get_arg_names(fn), [])

    def test_single_positional_arg(self):
        """A single positional parameter must appear in the result."""
        def fn(x):
            pass
        self.assertEqual(_get_arg_names(fn), ["x"])

    def test_multiple_positional_args(self):
        """Multiple positional parameters must be listed in declaration order."""
        def fn(a, b, c):
            pass
        self.assertEqual(_get_arg_names(fn), ["a", "b", "c"])

    def test_args_with_defaults(self):
        """Parameters with defaults must be included in the result."""
        def fn(a, b=10, c=20):
            pass
        self.assertEqual(_get_arg_names(fn), ["a", "b", "c"])

    # -- *args and **kwargs --

    def test_var_positional_args(self):
        """*args must appear in the result."""
        def fn(*args):
            pass
        self.assertIn("args", _get_arg_names(fn))

    def test_var_keyword_args(self):
        """**kwargs must appear in the result."""
        def fn(**kwargs):
            pass
        self.assertIn("kwargs", _get_arg_names(fn))

    def test_mixed_all_param_kinds(self):
        """All parameter kinds must be included and ordered correctly."""
        def fn(a, b=2, *args, kw_only=3, **kwargs):
            pass
        names = _get_arg_names(fn)
        self.assertEqual(names, ["a", "b", "args", "kw_only", "kwargs"])

    # -- keyword-only (after bare *) --

    def test_keyword_only_param(self):
        """Keyword-only parameters (after bare *) must appear in the result."""
        def fn(a, *, b):
            pass
        self.assertEqual(_get_arg_names(fn), ["a", "b"])

    # -- wrapped function (functools.wraps compatibility) --

    def test_wrapped_function_reports_wrapper_signature(self):
        """A functools.wraps-decorated function must expose the wrapper's params."""
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper

        @decorator
        def original(x, y):
            pass

        # functools.wraps copies __wrapped__; inspect.signature follows __wrapped__
        names = _get_arg_names(original)
        self.assertIsInstance(names, list)
        # After wraps, signature resolves back to the original's (x, y)
        self.assertEqual(names, ["x", "y"])

    # -- lambda --

    def test_lambda_with_args(self):
        """Lambda parameters must be returned correctly."""
        fn = lambda x, y: x + y  # noqa: E731
        self.assertEqual(_get_arg_names(fn), ["x", "y"])

    def test_lambda_no_args(self):
        """Zero-argument lambda must return empty list."""
        fn = lambda: 42  # noqa: E731
        self.assertEqual(_get_arg_names(fn), [])

    # -- class-level callables --

    def test_instance_method(self):
        """Bound instance method must expose parameters (excluding self)."""
        class MyClass:
            def method(self, a, b):
                pass

        obj = MyClass()
        names = _get_arg_names(obj.method)
        # Bound method: self is already bound, so signature omits it
        self.assertEqual(names, ["a", "b"])

    def test_unbound_method(self):
        """Unbound method accessed from the class must include self."""
        class MyClass:
            def method(self, a, b):
                pass

        names = _get_arg_names(MyClass.method)
        self.assertEqual(names, ["self", "a", "b"])

    def test_staticmethod_via_class(self):
        """staticmethod must report its parameters without cls/self."""
        class MyClass:
            @staticmethod
            def fn(x, y):
                pass

        names = _get_arg_names(MyClass.fn)
        self.assertEqual(names, ["x", "y"])

    def test_classmethod_via_class(self):
        """classmethod accessed from the class must omit cls."""
        class MyClass:
            @classmethod
            def fn(cls, x):
                pass

        names = _get_arg_names(MyClass.fn)
        self.assertEqual(names, ["x"])

    # -- callable object --

    def test_callable_object_with_call_signature(self):
        """A callable object must expose its __call__ parameters."""
        class Callable:
            def __call__(self, a, b):
                pass

        obj = Callable()
        names = _get_arg_names(obj)
        # __call__ is a bound method; self is already bound
        self.assertEqual(names, ["a", "b"])

    # -- preserving order --

    def test_order_preserved_for_many_params(self):
        """Parameter order must exactly match the declaration order."""
        def fn(p1, p2, p3, p4, p5):
            pass
        self.assertEqual(_get_arg_names(fn), ["p1", "p2", "p3", "p4", "p5"])

    # -- built-in raises ValueError --

    def test_builtin_function_raises_value_error_or_type_error(self):
        """C built-in method descriptors without introspectable signatures must raise.

        Notes
        -----
        ``len`` acquired an Argument Clinic signature in CPython 3.11 and is
        therefore introspectable on modern Python.  ``str.count`` is a
        ``method_descriptor`` that has never had Argument Clinic coverage and
        reliably raises ``ValueError`` across all supported Python versions.
        """
        with self.assertRaises((ValueError, TypeError)):
            _get_arg_names(str.count)

    # -- idempotency --

    def test_called_twice_returns_same_result(self):
        """Calling _get_arg_names twice on the same function must give identical results."""
        def fn(a, b, c=1):
            pass
        self.assertEqual(_get_arg_names(fn), _get_arg_names(fn))

    # -- result is independent list (not a view) --

    def test_result_is_a_fresh_list(self):
        """Mutating the returned list must not affect subsequent calls."""
        def fn(a, b):
            pass
        result1 = _get_arg_names(fn)
        result1.append("injected")
        result2 = _get_arg_names(fn)
        self.assertNotIn("injected", result2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
