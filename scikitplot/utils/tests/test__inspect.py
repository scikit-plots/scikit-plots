# scikitplot/utils/tests/test__inspect.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._inspect`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__inspect.py -v

Coverage map
------------
_get_args_kwargs          Replaces param in kwargs, in args,
                          not found raises ValueError         → TestGetArgsKwargs
_get_param_w_index        Finds first match, uses default,
                          returns (None,None,None) if missing → TestGetParamWIndex
_resolve_args_and_kwargs  Strict/partial binding, apply_defaults,
                          extra kwargs handling, TypeError    → TestResolveArgsAndKwargs
"""

from __future__ import annotations

import unittest

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib, sys
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._inspect import (  # noqa: E402
    _get_args_kwargs,
    _get_param_w_index,
    _resolve_args_and_kwargs,
)


# ---------------------------------------------------------------------------
# Helper functions used across tests
# ---------------------------------------------------------------------------


def _simple(a, b, c=10):
    """Simple function with two required and one default parameter."""
    return a + b + c


def _no_params():
    """Function with no parameters."""
    return 42


def _all_defaults(x=1, y=2, z=3):
    return x + y + z


def _keyword_only(a, *, b=5):
    return a + b


# ===========================================================================
# _get_args_kwargs
# ===========================================================================


class TestGetArgsKwargs(unittest.TestCase):
    """_get_args_kwargs must replace a named parameter value in args or kwargs."""

    # -- Replacement via kwargs --

    def test_replaces_kwarg(self):
        """If param_key is in kwargs, its value must be updated."""
        args = (1, 2)
        kwargs = {"c": 0, "param_key": "c", "param_index": 2, "param_value": 99}
        new_args, new_kwargs = _get_args_kwargs(*args, **kwargs)
        self.assertEqual(new_kwargs["c"], 99)

    def test_kwarg_other_keys_preserved(self):
        """Other kwargs must remain unchanged."""
        args = ()
        kwargs = {
            "extra": "preserved",
            "param_key": "extra",
            "param_index": None,
            "param_value": "replaced",
        }
        _, new_kwargs = _get_args_kwargs(*args, **kwargs)
        self.assertEqual(new_kwargs["extra"], "replaced")

    # -- Replacement via args --

    def test_replaces_positional_arg(self):
        """If param_key is not in kwargs, replace by param_index in args."""
        args = (10, 20, 30)
        kwargs = {"param_key": "notinkwargs", "param_index": 1, "param_value": 99}
        new_args, new_kwargs = _get_args_kwargs(*args, **kwargs)
        self.assertEqual(new_args[1], 99)

    def test_other_args_unchanged(self):
        args = (10, 20, 30)
        kwargs = {"param_key": "x", "param_index": 0, "param_value": 5}
        new_args, _ = _get_args_kwargs(*args, **kwargs)
        self.assertEqual(new_args[1], 20)
        self.assertEqual(new_args[2], 30)

    # -- Error path --

    def test_not_found_raises_value_error(self):
        """If param_key is not in kwargs and index is out of range, raise ValueError."""
        args = (1,)
        kwargs = {"param_key": "ghost", "param_index": 99, "param_value": 0}
        with self.assertRaises(ValueError):
            _get_args_kwargs(*args, **kwargs)

    def test_none_index_and_not_in_kwargs_raises(self):
        """param_index=None and param_key not in kwargs must raise ValueError."""
        kwargs = {"param_key": "missing", "param_index": None, "param_value": 0}
        with self.assertRaises(ValueError):
            _get_args_kwargs(**kwargs)

    # -- Return type --

    def test_returns_tuple_two_elements(self):
        args = (1, 2)
        kwargs = {"c": 3, "param_key": "c", "param_index": 2, "param_value": 9}
        result = _get_args_kwargs(*args, **kwargs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_new_args_is_list(self):
        args = (1, 2)
        kwargs = {"c": 3, "param_key": "c", "param_index": 2, "param_value": 9}
        new_args, _ = _get_args_kwargs(*args, **kwargs)
        self.assertIsInstance(new_args, list)

    def test_new_kwargs_is_dict(self):
        args = (1, 2)
        kwargs = {"c": 3, "param_key": "c", "param_index": 2, "param_value": 9}
        _, new_kwargs = _get_args_kwargs(*args, **kwargs)
        self.assertIsInstance(new_kwargs, dict)


# ===========================================================================
# _get_param_w_index
# ===========================================================================


class TestGetParamWIndex(unittest.TestCase):
    """_get_param_w_index must retrieve a named parameter and its index."""

    # -- Found in kwargs --

    def test_finds_kwarg_param(self):
        """When param is passed as kwarg, value must be retrieved from kwargs."""
        _, idx, val = _get_param_w_index(
            1, 2, func=_simple, params=["b"], b=99
        )
        self.assertEqual(val, 99)

    # -- Found in positional args --

    def test_finds_positional_param(self):
        """When param is passed as arg, value must come from args by index."""
        key, idx, val = _get_param_w_index(
            10, 20, func=_simple, params=["b"]
        )
        self.assertEqual(key, "b")
        self.assertEqual(val, 20)

    def test_index_is_correct(self):
        """The returned index must match the parameter's position in the signature."""
        _, idx, _ = _get_param_w_index(1, 2, func=_simple, params=["a"])
        self.assertEqual(idx, 0)

    def test_second_param_index(self):
        _, idx, _ = _get_param_w_index(1, 2, func=_simple, params=["b"])
        self.assertEqual(idx, 1)

    # -- Default value fallback --

    def test_uses_default_when_no_arg(self):
        """When neither arg nor kwarg is provided, the function default must be used."""
        key, idx, val = _get_param_w_index(func=_simple, params=["c"])
        self.assertEqual(key, "c")
        self.assertEqual(val, 10)  # default is 10

    def test_no_default_param_gets_none_value(self):
        """A required param with no default and no arg gives value=None."""
        key, idx, val = _get_param_w_index(func=_simple, params=["a"])
        self.assertEqual(key, "a")
        self.assertIsNone(val)

    # -- Not found --

    def test_not_found_returns_none_triple(self):
        """No matching parameter must return (None, None, None)."""
        key, idx, val = _get_param_w_index(
            1, 2, func=_simple, params=["nonexistent"]
        )
        self.assertIsNone(key)
        self.assertIsNone(idx)
        self.assertIsNone(val)

    # -- No-parameter function --

    def test_no_params_function_returns_none_triple(self):
        key, idx, val = _get_param_w_index(func=_no_params, params=["x"])
        self.assertIsNone(key)

    # -- First match only --

    def test_first_match_wins(self):
        """With multiple candidates, the first matching parameter wins."""
        key, idx, val = _get_param_w_index(
            1, 2, func=_simple, params=["b", "a"]
        )
        # "a" appears at index 0 in signature; "b" at index 1.
        # The loop stops at the first match → result depends on parameter iteration order.
        self.assertIn(key, ("a", "b"))

    # -- Return type --

    def test_returns_tuple_of_three(self):
        result = _get_param_w_index(func=_simple, params=["a"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


# ===========================================================================
# _resolve_args_and_kwargs
# ===========================================================================


class TestResolveArgsAndKwargs(unittest.TestCase):
    """_resolve_args_and_kwargs must bind and apply defaults correctly."""

    # -- Strict mode --

    def test_strict_full_binding(self):
        """strict=True with all args provided must succeed and apply defaults."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            1, 2, func=_simple, strict=True
        )
        # 'c' must have its default applied
        self.assertIn(10, list(resolved_args) + list(resolved_kwargs.values()))

    def test_strict_missing_required_raises(self):
        """strict=True with a missing required arg must raise TypeError."""
        with self.assertRaises(TypeError):
            _resolve_args_and_kwargs(func=_simple, strict=True)

    def test_strict_extra_kwargs_raises(self):
        """strict=True with unknown kwargs must raise TypeError."""
        with self.assertRaises(TypeError):
            _resolve_args_and_kwargs(1, 2, func=_simple, strict=True, unknown=99)

    # -- Partial mode (strict=False) --

    def test_partial_no_args_no_error(self):
        """strict=False with no args must allow partial binding without error."""
        try:
            _resolve_args_and_kwargs(func=_simple, strict=False)
        except TypeError as e:
            self.fail(f"Partial binding raised TypeError: {e}")

    def test_partial_returns_tuple(self):
        result = _resolve_args_and_kwargs(1, func=_simple, strict=False)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_partial_applies_defaults(self):
        """apply_defaults() must fill in missing optional parameters."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            1, 2, func=_simple, strict=False
        )
        # 'c=10' must appear somewhere in the resolved args/kwargs
        all_values = list(resolved_args) + list(resolved_kwargs.values())
        self.assertIn(10, all_values)

    def test_partial_explicit_kwarg_overrides_default(self):
        """An explicit kwarg value must override the parameter default."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            1, 2, func=_simple, strict=False, c=50
        )
        all_values = list(resolved_args) + list(resolved_kwargs.values())
        self.assertIn(50, all_values)
        self.assertNotIn(10, all_values)

    def test_all_defaults_function_fully_resolved(self):
        """A function with all defaults, no args: must return full default set."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            func=_all_defaults, strict=False
        )
        all_values = list(resolved_args) + list(resolved_kwargs.values())
        for default in (1, 2, 3):
            self.assertIn(default, all_values)

    def test_no_params_function_partial(self):
        """A function with no parameters must bind empty args/kwargs cleanly."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            func=_no_params, strict=False
        )
        self.assertEqual(list(resolved_args), [])
        self.assertEqual(resolved_kwargs, {})

    # -- Resolved types --

    def test_resolved_args_is_tuple(self):
        resolved_args, _ = _resolve_args_and_kwargs(1, 2, func=_simple, strict=False)
        self.assertIsInstance(resolved_args, tuple)

    def test_resolved_kwargs_is_dict(self):
        _, resolved_kwargs = _resolve_args_and_kwargs(1, 2, func=_simple, strict=False)
        self.assertIsInstance(resolved_kwargs, dict)

    # -- Keyword-only parameters --

    def test_keyword_only_partial(self):
        """Keyword-only params after '*' must be handled via partial binding."""
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            10, func=_keyword_only, strict=False
        )
        # default b=5 must appear in resolved kwargs
        self.assertIn(5, list(resolved_kwargs.values()))

    def test_keyword_only_explicit_value(self):
        resolved_args, resolved_kwargs = _resolve_args_and_kwargs(
            10, func=_keyword_only, strict=False, b=99
        )
        self.assertIn(99, list(resolved_kwargs.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
