# scikitplot/_api/tests/test___init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
test___init__.py
================
Unified, comprehensive test suite for ``_api/__init__.py``.

Supersedes ``test___init__.py``, ``test___init__extended.py``, and
``test___init__stdlib.py``.  Every public symbol is covered at both the
happy-path level and at the branch-coverage level.

Covered symbols
---------------
* ``_Unset / UNSET``              — repr, identity, distinct-from-falsy, usage
* ``classproperty``               — class/instance access, subclass dispatch,
                                    fset/fdel guards, doc param, multiple on class
* ``check_isinstance``            — single/tuple types, None sentinel, None ordering,
                                    builtin vs qualified name, multiple kwargs,
                                    first-failure semantics
* ``check_in_list``               — valid/invalid, generator materialisation,
                                    _print_supported_values, int values,
                                    empty sequence, boundary conditions
* ``check_shape``                 — exact, free dims, 1-D trailing comma,
                                    >7 free dims (D-series labels), non-array
                                    TypeError, 0-D array, multiple arrays,
                                    mixed None/fixed, all-fixed, all-free
* ``check_getitem``               — key found, ≤5/=5/>5 keys, difflib path,
                                    non-string value, all-non-string keys,
                                    custom _error_cls, error-chain suppression,
                                    zero/multiple kwargs, falsy return values
* ``caching_module_getattr``      — property access, cache hit, missing attr,
                                    non-property skipped, independent caches
* ``define_aliases``              — getter+setter, getter-only, setter-only,
                                    conflict detection, direct call, _alias_map
                                    merge, multiple aliases per property,
                                    alias docstring format
* ``select_matching_signature``   — first-match, fallback, no-match, kwargs,
                                    single func, return-value propagation,
                                    None return, non-TypeError propagation
* ``nargs_error``                 — TypeError shape, string takes, zero counts
* ``kwarg_error``                 — string, iterable, dict, generator sources,
                                    empty-iterable StopIteration documented
* ``recursive_subclasses``        — base class, direct/indirect, depth-first,
                                    diamond topology, deep chain, generator type
* ``warn_external``               — UserWarning, message, instance, custom
                                    category, py≥3.12 path, py<3.12 frame-
                                    walker path, frame=None embedded context
* Package integration             — all re-exported symbols accessible, UNSET
                                    is module-level singleton

Bug fixes vs prior tests
------------------------
* ``import .. as api_module`` (syntax error in legacy file) replaced by
  ``importlib.import_module('..', package=__package__)``.
* ``from .. import __init__ as api_module`` (fragile) replaced by same.

Notes
-----
User note
    Run with either::

        python -m unittest _api.tests.test___init__
        pytest _api/tests/test___init__.py -v

Developer note
    All tests use ``unittest.TestCase`` as base class so the suite runs
    without pytest.  Monkeypatching uses explicit ``try / finally`` patterns
    instead of the pytest ``monkeypatch`` fixture.  NumPy is required only
    for ``check_shape`` tests; all others are NumPy-free.
"""

from __future__ import annotations

import importlib
import sys
import unittest
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Relative imports — preserve private-submodule relative-path structure.
# ---------------------------------------------------------------------------
from .. import (
    UNSET,
    _Unset,
    check_getitem,
    check_in_list,
    check_isinstance,
    check_shape,
    caching_module_getattr,
    classproperty,
    define_aliases,
    kwarg_error,
    nargs_error,
    recursive_subclasses,
    select_matching_signature,
    warn_external,
)

# Canonical way to obtain the parent package as a module object.
# Required for monkeypatching sys.version_info in warn_external tests.
# NOTE: ``import .. as api_module`` is a syntax error; ``from .. import
# __init__ as api_module`` is fragile.  importlib is the correct approach.
api_module = importlib.import_module('..', package=__package__)


# ===========================================================================
# Shared helper mixin
# ===========================================================================

class _WarnAssertMixin:
    """Helpers for warning assertions compatible with unittest.TestCase."""

    def _assert_warns(self, record, category=UserWarning, *, match=None, count=None):
        """Assert *record* contains at least one warning of *category*.

        Parameters
        ----------
        record : list
            Captured by ``warnings.catch_warnings(record=True)``.
        category : type, default UserWarning
            Expected warning class.
        match : str or None
            If given, assert at least one warning message matches this regex.
        count : int or None
            If given, assert exactly *count* matching warnings were emitted.
        """
        import re
        matching = [w for w in record if issubclass(w.category, category)]
        self.assertTrue(
            len(matching) >= 1,
            f"Expected ≥1 {category.__name__} but got: {record}",
        )
        if count is not None:
            self.assertEqual(
                len(matching), count,
                f"Expected exactly {count} {category.__name__} but got {len(matching)}",
            )
        if match is not None:
            msgs = [str(w.message) for w in matching]
            self.assertTrue(
                any(re.search(match, m) for m in msgs),
                f"No warning matched {match!r}. Messages: {msgs}",
            )

    def _assert_no_warn(self, record, category=UserWarning):
        """Assert *record* contains NO warning of *category*."""
        matching = [w for w in record if issubclass(w.category, category)]
        self.assertEqual(
            len(matching), 0,
            f"Expected no {category.__name__} but got: {matching}",
        )


# ===========================================================================
# _Unset / UNSET
# ===========================================================================

class TestUnset(unittest.TestCase):
    """Tests for the UNSET sentinel and _Unset class."""

    def test_repr(self):
        """UNSET must render as '<UNSET>' for readable error messages."""
        self.assertEqual(repr(UNSET), "<UNSET>")

    def test_is_instance_of_unset_class(self):
        """UNSET must be an instance of the private _Unset class."""
        self.assertIsInstance(UNSET, _Unset)

    def test_singleton_identity_on_reimport(self):
        """UNSET must be the same object every time it is imported."""
        from .. import UNSET as unset2
        self.assertIs(UNSET, unset2)

    def test_distinct_from_none(self):
        """UNSET must not compare equal to None."""
        self.assertIsNot(UNSET, None)
        # noqa: E711 — intentional identity check, not equality
        self.assertNotEqual(UNSET, None)

    def test_distinct_from_false(self):
        """UNSET must not be False."""
        self.assertIsNot(UNSET, False)

    def test_distinct_from_zero(self):
        """UNSET must not be 0."""
        self.assertIsNot(UNSET, 0)

    def test_distinct_from_empty_string(self):
        """UNSET must not be an empty string."""
        self.assertIsNot(UNSET, "")

    def test_as_default_not_given(self):
        """Typical usage: arg not given → arg is UNSET evaluates True."""
        def foo(arg=UNSET):
            return arg is UNSET
        self.assertTrue(foo())

    def test_as_default_none_passed(self):
        """Passing None explicitly must NOT look like 'not given'."""
        def foo(arg=UNSET):
            return arg is UNSET
        self.assertFalse(foo(None))

    def test_as_default_zero_passed(self):
        """Passing 0 explicitly must NOT look like 'not given'."""
        def foo(arg=UNSET):
            return arg is UNSET
        self.assertFalse(foo(0))

    def test_as_default_false_passed(self):
        """Passing False explicitly must NOT look like 'not given'."""
        def foo(arg=UNSET):
            return arg is UNSET
        self.assertFalse(foo(False))


# ===========================================================================
# classproperty
# ===========================================================================

class TestClassproperty(unittest.TestCase):
    """Tests for the classproperty descriptor."""

    def test_access_via_class(self):
        """Accessing a classproperty on the class must call the getter."""
        class C:
            @classproperty
            def name(cls):
                return cls.__name__
        self.assertEqual(C.name, "C")

    def test_access_via_instance(self):
        """Accessing a classproperty on an instance must also call the getter."""
        class C:
            @classproperty
            def name(cls):
                return cls.__name__
        self.assertEqual(C().name, "C")

    def test_cls_argument_is_class_not_instance(self):
        """The getter must receive the *class*, not the instance."""
        received = []
        class C:
            @classproperty
            def capture(cls):
                received.append(cls)
                return True
        _ = C.capture
        self.assertEqual(received, [C])

    def test_fget_property_returns_original_getter(self):
        """The .fget property must return the original getter function."""
        def getter(cls):
            return cls
        cp = classproperty(getter)
        self.assertIs(cp.fget, getter)

    def test_fset_raises_value_error(self):
        """Constructing with fset must raise ValueError immediately."""
        with self.assertRaisesRegex(ValueError, "only implements fget"):
            classproperty(lambda cls: None, fset=lambda cls, v: None)

    def test_fdel_raises_value_error(self):
        """Constructing with fdel must raise ValueError immediately."""
        with self.assertRaisesRegex(ValueError, "only implements fget"):
            classproperty(lambda cls: None, fdel=lambda cls: None)

    def test_fset_and_fdel_stored_as_none_when_omitted(self):
        """fset and fdel must be None when not supplied."""
        cp = classproperty(lambda cls: None)
        self.assertIsNone(cp.fset)
        self.assertIsNone(cp.fdel)

    def test_subclass_receives_own_class(self):
        """Each subclass must receive its own class, not the base class."""
        class Base:
            @classproperty
            def name(cls):
                return cls.__name__
        class Child(Base):
            pass
        self.assertEqual(Base.name, "Base")
        self.assertEqual(Child.name, "Child")

    def test_doc_parameter_stored_as_private_doc(self):
        """The ``doc`` keyword is stored as ``_doc`` on the descriptor."""
        cp = classproperty(lambda cls: None, doc="my documentation")
        self.assertEqual(cp._doc, "my documentation")

    def test_doc_parameter_does_not_break_get(self):
        """Providing ``doc`` must not interfere with attribute access."""
        class C:
            info = classproperty(lambda cls: cls.__name__, doc="The class name.")
        self.assertEqual(C.info, "C")

    def test_no_doc_stores_none(self):
        """Omitting ``doc`` leaves ``_doc`` as None."""
        cp = classproperty(lambda cls: None)
        self.assertIsNone(cp._doc)

    def test_subclass_access_with_doc(self):
        """Doc-bearing classproperty must still resolve correctly on a subclass."""
        class Base:
            name = classproperty(lambda cls: cls.__name__, doc="class name")
        class Child(Base):
            pass
        self.assertEqual(Child.name, "Child")

    def test_return_value_propagated(self):
        """The value returned by the getter must be returned to the caller."""
        sentinel = object()
        class C:
            @classproperty
            def val(cls):
                return sentinel
        self.assertIs(C.val, sentinel)

    def test_multiple_classpropertys_independent(self):
        """Multiple classproperty descriptors on one class must be independent."""
        class C:
            @classproperty
            def a(cls):
                return 1
            @classproperty
            def b(cls):
                return 2
        self.assertEqual(C.a, 1)
        self.assertEqual(C.b, 2)


# ===========================================================================
# check_isinstance
# ===========================================================================

class TestCheckIsinstance(unittest.TestCase):
    """Tests for check_isinstance."""

    def test_single_type_valid(self):
        check_isinstance(str, val="hello")  # must not raise

    def test_single_type_invalid(self):
        with self.assertRaisesRegex(TypeError, "'val' must be an instance of str"):
            check_isinstance(str, val=42)

    def test_bare_type_normalised(self):
        """A bare type (not wrapped in tuple) must work identically to (type,)."""
        check_isinstance(int, val=42)

    def test_tuple_of_types_valid(self):
        check_isinstance((str, int), val=42)
        check_isinstance((str, int), val="hello")

    def test_tuple_of_types_invalid(self):
        with self.assertRaisesRegex(TypeError, "str or int"):
            check_isinstance((str, int), val=3.14)

    def test_none_in_types_accepts_none(self):
        """None in types tuple must be treated as NoneType."""
        check_isinstance((str, None), val=None)

    def test_none_in_types_invalid(self):
        """None in types must appear last in the error message."""
        with self.assertRaisesRegex(TypeError, "or None"):
            check_isinstance((str, None), val=42)

    def test_none_moved_to_end_in_error_message(self):
        """Error message must move None to the end for natural English phrasing."""
        with self.assertRaises(TypeError) as ctx:
            check_isinstance((None, str), val=42)
        msg = str(ctx.exception)
        self.assertLess(msg.index("str"), msg.index("None"))

    def test_types_is_none_only_accepts_none(self):
        """Passing types=None (bare sentinel) means only NoneType is accepted."""
        check_isinstance(None, val=None)

    def test_types_is_none_only_invalid(self):
        with self.assertRaises(TypeError):
            check_isinstance(None, val=42)

    def test_multiple_kwargs_all_valid(self):
        """All kwargs must be checked; no early exit on first success."""
        check_isinstance((str, int), a="x", b=1)

    def test_multiple_kwargs_second_invalid(self):
        """Failure on the second kwarg must still be detected."""
        with self.assertRaisesRegex(TypeError, "'b'"):
            check_isinstance(str, a="x", b=42)

    def test_first_failing_kwarg_names_itself(self):
        """Error message must include the failing kwarg name."""
        with self.assertRaisesRegex(TypeError, "'bad'"):
            check_isinstance(str, good="hello", bad=123)

    def test_qualified_name_in_error(self):
        """Non-builtin types must show their fully-qualified name."""
        class MyType:
            pass
        with self.assertRaisesRegex(TypeError, "MyType"):
            check_isinstance(MyType, val="not my type")

    def test_builtins_show_short_name_without_module_prefix(self):
        """Builtin types must show just the unqualified name (no 'builtins.')."""
        with self.assertRaises(TypeError) as ctx:
            check_isinstance(str, val=42)
        self.assertNotIn("builtins", str(ctx.exception))


# ===========================================================================
# check_in_list
# ===========================================================================

class TestCheckInList(unittest.TestCase):
    """Tests for check_in_list."""

    def test_valid_single_kwarg(self):
        check_in_list(["foo", "bar"], arg="foo")

    def test_valid_multiple_kwargs(self):
        check_in_list(["foo", "bar"], a="foo", b="bar")

    def test_invalid_shows_supported_values(self):
        with self.assertRaisesRegex(ValueError, r"'foo'.*'bar'"):
            check_in_list(["foo", "bar"], arg="baz")

    def test_invalid_hides_supported_values(self):
        with self.assertRaises(ValueError) as ctx:
            check_in_list(["foo", "bar"], _print_supported_values=False, arg="baz")
        self.assertNotIn("foo", str(ctx.exception))
        self.assertNotIn("bar", str(ctx.exception))

    def test_no_kwargs_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "No argument to check"):
            check_in_list(["foo", "bar"])

    def test_first_invalid_kwarg_detected(self):
        """First bad kwarg must raise; later ones are not checked."""
        with self.assertRaisesRegex(ValueError, "for a"):
            check_in_list(["foo"], a="bad", b="foo")

    def test_generator_materialised_before_iteration(self):
        """Generator must be converted to tuple so all kwargs see the values."""
        gen = (x for x in ["foo", "bar", "baz"])
        with self.assertRaisesRegex(ValueError, r"'bad'.*for b.*'foo'.*'bar'.*'baz'"):
            check_in_list(gen, a="foo", b="bad")

    def test_generator_both_kwargs_valid(self):
        """Two valid kwargs against a generator must both succeed."""
        check_in_list((x for x in ["alpha", "beta", "gamma"]), a="alpha", b="beta")

    def test_tuple_values_accepted(self):
        check_in_list(("a", "b", "c"), val="b")

    def test_int_values_accepted(self):
        check_in_list([1, 2, 3], val=2)

    def test_none_in_values(self):
        check_in_list(["foo", None], val=None)

    def test_empty_values_rejects_everything(self):
        with self.assertRaises(ValueError):
            check_in_list([], val="anything")

    def test_single_valid_value_passes(self):
        check_in_list(["only"], val="only")

    def test_error_message_quotes_invalid_value(self):
        with self.assertRaisesRegex(ValueError, "'bad'"):
            check_in_list(["a", "b"], val="bad")

    def test_int_values_suppressed_when_print_false(self):
        with self.assertRaises(ValueError) as ctx:
            check_in_list([1, 2, 3], _print_supported_values=False, val=99)
        self.assertNotIn("1", str(ctx.exception))


# ===========================================================================
# check_shape
# ===========================================================================

class TestCheckShape(unittest.TestCase):
    """Tests for check_shape."""

    def test_exact_shape_valid(self):
        check_shape((3, 2), arr=np.zeros((3, 2)))

    def test_free_first_dimension(self):
        """None in shape means any size is accepted along that axis."""
        check_shape((None, 2), arr=np.zeros((100, 2)))

    def test_all_free_dimensions(self):
        check_shape((None, None), arr=np.zeros((7, 13)))

    def test_all_fixed_dimensions_valid(self):
        check_shape((2, 3, 4), arr=np.zeros((2, 3, 4)))

    def test_all_fixed_dimensions_mismatch(self):
        with self.assertRaises(ValueError):
            check_shape((2, 3, 4), arr=np.zeros((2, 3, 5)))

    def test_wrong_ndim(self):
        with self.assertRaisesRegex(ValueError, "arr"):
            check_shape((None, 2), arr=np.zeros((5,)))

    def test_wrong_fixed_size(self):
        with self.assertRaisesRegex(ValueError, "arr"):
            check_shape((3, 2), arr=np.zeros((4, 2)))

    def test_error_message_contains_key_name(self):
        with self.assertRaisesRegex(ValueError, "'my_array'"):
            check_shape((3,), my_array=np.zeros((5,)))

    def test_error_message_contains_actual_shape(self):
        with self.assertRaisesRegex(ValueError, r"\(5,\)"):
            check_shape((3,), arr=np.zeros((5,)))

    def test_1d_free_valid(self):
        check_shape((None,), arr=np.zeros((42,)))

    def test_1d_shape_error_message_has_trailing_comma(self):
        """1-D error must read '(N,)' not '(N)' to be unambiguous."""
        with self.assertRaises(ValueError) as ctx:
            check_shape((None,), arr=np.zeros((3, 2)))
        self.assertIn(",", str(ctx.exception))

    def test_1d_fixed_size_trailing_comma(self):
        with self.assertRaisesRegex(ValueError, r"\(3,\)"):
            check_shape((3,), arr=np.zeros((5,)))

    def test_dim_labels_exhaust_into_D_series(self):
        """More than 7 None dimensions must use the D{i} label series."""
        with self.assertRaises(ValueError) as ctx:
            check_shape((None,) * 8, arr=np.zeros((1,) * 9))
        self.assertIn("D0", str(ctx.exception))

    def test_no_kwargs_is_noop(self):
        """Calling with no array kwargs must silently succeed."""
        check_shape((3, 2))  # no arrays — nothing to validate

    def test_multiple_arrays_first_failure_raises(self):
        good = np.zeros((3, 2))
        bad = np.zeros((5, 2))
        with self.assertRaises(ValueError):
            check_shape((3, 2), a=good, b=bad)

    def test_non_array_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "'arr' must be a numpy array"):
            check_shape((3,), arr=[1, 2, 3])

    def test_non_array_error_mentions_type_name(self):
        with self.assertRaisesRegex(TypeError, "list"):
            check_shape((3,), arr=[1, 2, 3])

    def test_3d_shape_valid(self):
        check_shape((2, None, 4), vol=np.zeros((2, 7, 4)))

    def test_3d_shape_invalid(self):
        with self.assertRaises(ValueError):
            check_shape((2, None, 4), vol=np.zeros((2, 7, 5)))

    def test_zero_d_array_matches_empty_shape(self):
        """0-D array must match 0-D shape specification."""
        check_shape((), arr=np.array(42))

    def test_mixed_none_and_fixed_error_shows_fixed_size(self):
        """Error for (None, 3) shape must include the fixed size '3'."""
        with self.assertRaises(ValueError) as ctx:
            check_shape((None, 3), arr=np.zeros((5, 4)))
        self.assertIn("3", str(ctx.exception))


# ===========================================================================
# check_getitem
# ===========================================================================

class TestCheckGetitem(unittest.TestCase):
    """Tests for check_getitem."""

    def test_key_found_returns_value(self):
        self.assertEqual(check_getitem({"foo": 1, "bar": 2}, key="foo"), 1)

    def test_returns_mapped_value_identity(self):
        sentinel = object()
        self.assertIs(check_getitem({"a": sentinel}, key="a"), sentinel)

    def test_returns_zero(self):
        """Return value 0 (falsy) must propagate correctly."""
        self.assertEqual(check_getitem({"a": 0}, key="a"), 0)

    def test_returns_none_value(self):
        """Return value None must propagate correctly."""
        self.assertIsNone(check_getitem({"a": None}, key="a"))

    def test_returns_false_value(self):
        """Return value False must propagate correctly."""
        self.assertIs(check_getitem({"a": False}, key="a"), False)

    def test_key_not_found_few_keys_shows_supported(self):
        with self.assertRaisesRegex(ValueError, "Supported values"):
            check_getitem({"foo": 1, "bar": 2}, key="baz")

    def test_exactly_five_keys_shows_supported(self):
        """len(mapping)==5 is NOT >5, so 'Supported values' branch is taken."""
        mapping = {f"opt{i}": i for i in range(5)}
        with self.assertRaisesRegex(ValueError, "Supported values"):
            check_getitem(mapping, key="missing")

    def test_six_keys_triggers_difflib_path(self):
        """len(mapping)==6 IS >5, so the difflib branch runs."""
        mapping = {f"opt{i}": i for i in range(6)}
        with self.assertRaises(ValueError) as ctx:
            check_getitem(mapping, key="totally_different")
        self.assertNotIn("Supported values", str(ctx.exception))

    def test_many_keys_close_string_match_produces_suggestion(self):
        mapping = {f"option{i}": i for i in range(10)}
        with self.assertRaisesRegex(ValueError, "option0"):
            check_getitem(mapping, key="option0x")

    def test_many_keys_no_close_match_no_suggestion(self):
        mapping = {f"option{i}": i for i in range(10)}
        with self.assertRaises(ValueError) as ctx:
            check_getitem(mapping, key="zzzzz")
        self.assertNotIn("Did you mean", str(ctx.exception))

    def test_non_string_lookup_value_bypasses_difflib(self):
        """Integer value → difflib bypassed, no 'Did you mean'."""
        mapping = {f"key{i}": i for i in range(10)}
        with self.assertRaises(ValueError) as ctx:
            check_getitem(mapping, key=99999)
        self.assertNotIn("Did you mean", str(ctx.exception))

    def test_all_integer_keys_no_suggestion(self):
        """All-integer keys → str_keys empty → no difflib suggestion."""
        mapping = {i: i * 10 for i in range(10)}
        with self.assertRaises(ValueError) as ctx:
            check_getitem(mapping, key="not_there")
        self.assertNotIn("Did you mean", str(ctx.exception))

    def test_custom_error_class_small_mapping(self):
        with self.assertRaises(KeyError):
            check_getitem({"foo": 1}, _error_cls=KeyError, key="bar")

    def test_custom_error_class_large_mapping(self):
        """Custom _error_cls must also apply on the large-mapping path."""
        with self.assertRaises(RuntimeError):
            check_getitem(
                {f"k{i}": i for i in range(10)}, _error_cls=RuntimeError, key="missing"
            )

    def test_multiple_kwargs_raises(self):
        with self.assertRaisesRegex(ValueError, "single keyword argument"):
            check_getitem({"foo": 1}, a="x", b="y")

    def test_zero_kwargs_raises(self):
        with self.assertRaisesRegex(ValueError, "single keyword argument"):
            check_getitem({"foo": 1})

    def test_error_chain_suppressed_from_none(self):
        """KeyError context must not be chained (raises ... from None)."""
        with self.assertRaises(ValueError) as ctx:
            check_getitem({"foo": 1}, key="bar")
        self.assertIsNone(ctx.exception.__cause__)

    def test_integer_key_in_large_mapping_no_type_error(self):
        """Integer keys must not cause TypeError in the difflib suggestion path."""
        mapping = {i: i * 10 for i in range(10)}
        with self.assertRaises(ValueError):
            check_getitem(mapping, key=99)


# ===========================================================================
# caching_module_getattr
# ===========================================================================

class TestCachingModuleGetattr(unittest.TestCase):
    """Tests for caching_module_getattr."""

    def test_property_accessible(self):
        @caching_module_getattr
        class __getattr__:
            @property
            def answer(self):
                return 42
        self.assertEqual(__getattr__("answer"), 42)

    def test_missing_attribute_raises_attribute_error(self):
        @caching_module_getattr
        class __getattr__:
            @property
            def present(self):
                return True
        with self.assertRaisesRegex(AttributeError, "has no attribute 'missing'"):
            __getattr__("missing")

    def test_attribute_error_mentions_module(self):
        @caching_module_getattr
        class __getattr__:
            pass
        with self.assertRaisesRegex(AttributeError, "module"):
            __getattr__("anything")

    def test_result_is_cached(self):
        """The property must be evaluated only once across repeated accesses."""
        call_count = 0
        @caching_module_getattr
        class __getattr__:
            @property
            def counter(self):
                nonlocal call_count
                call_count += 1
                return call_count
        first = __getattr__("counter")
        second = __getattr__("counter")
        self.assertEqual(first, 1)
        self.assertEqual(second, 1)
        self.assertEqual(call_count, 1)

    def test_non_property_attributes_ignored(self):
        """Only @property descriptors must be considered; plain attrs are skipped."""
        @caching_module_getattr
        class __getattr__:
            plain = "not a property"
            @property
            def real(self):
                return "yes"
        self.assertEqual(__getattr__("real"), "yes")
        with self.assertRaises(AttributeError):
            __getattr__("plain")

    def test_two_invocations_return_different_functions(self):
        """Two calls to caching_module_getattr must produce distinct callables."""
        @caching_module_getattr
        class __getattr__:
            @property
            def value(self):
                return 10
        fn1 = __getattr__

        @caching_module_getattr
        class __getattr__:     # noqa: F811 — intentional redefinition for test
            @property
            def value(self):
                return 20
        fn2 = __getattr__

        self.assertEqual(fn1("value"), 10)
        self.assertEqual(fn2("value"), 20)

    def test_returns_exact_object_from_property(self):
        """Return value must be the exact object yielded by the property."""
        sentinel = object()
        @caching_module_getattr
        class __getattr__:
            @property
            def obj(self):
                return sentinel
        self.assertIs(__getattr__("obj"), sentinel)


# ===========================================================================
# define_aliases
# ===========================================================================

class TestDefineAliases(unittest.TestCase):
    """Tests for define_aliases."""

    def test_getter_alias_added(self):
        @define_aliases({"color": ["colour", "c"]})
        class C:
            def get_color(self):
                return "red"
        obj = C()
        self.assertEqual(obj.get_colour(), "red")
        self.assertEqual(obj.get_c(), "red")

    def test_setter_alias_added(self):
        @define_aliases({"color": ["colour"]})
        class C:
            def get_color(self):
                return self._color
            def set_color(self, v):
                self._color = v
        obj = C()
        obj.set_colour("blue")
        self.assertEqual(obj.get_colour(), "blue")

    def test_only_setter_creates_set_alias_no_get_alias(self):
        """When only set_<prop> exists, no getter alias must be created."""
        @define_aliases({"alpha": ["opacity"]})
        class C:
            def set_alpha(self, v):
                self._alpha = v
        obj = C()
        obj.set_opacity(0.5)
        self.assertEqual(obj._alpha, 0.5)
        self.assertFalse(hasattr(C, "get_opacity"))

    def test_alias_map_stored_on_class(self):
        @define_aliases({"size": ["sz"]})
        class C:
            def get_size(self):
                return 0
        self.assertIn("size", C._alias_map)

    def test_alias_map_contains_all_aliases(self):
        @define_aliases({"size": ["sz", "s"]})
        class C:
            def get_size(self):
                return 0
        self.assertIn("sz", C._alias_map["size"])
        self.assertIn("s", C._alias_map["size"])

    def test_missing_getter_and_setter_raises(self):
        with self.assertRaisesRegex(ValueError, "Neither getter nor setter"):
            @define_aliases({"nonexistent": ["alias"]})
            class C:
                pass

    def test_conflict_with_existing_alias_map_raises(self):
        """Re-applying the same alias key (via inheritance) must raise."""
        @define_aliases({"prop": ["alias1"]})
        class Base:
            def get_prop(self):
                return 1
        with self.assertRaisesRegex(NotImplementedError, "conflicting aliases"):
            @define_aliases({"prop": ["alias1"]})
            class Child(Base):
                def get_prop(self):
                    return 2

    def test_partial_application_returns_callable(self):
        """Calling define_aliases without cls must return a decorator callable."""
        decorator = define_aliases({"x": ["y"]})
        self.assertTrue(callable(decorator))

    def test_alias_docstring_references_original_method(self):
        """Alias method __doc__ must reference the original method name."""
        @define_aliases({"color": ["colour"]})
        class C:
            def get_color(self):
                return "red"
        self.assertIn("get_color", C.get_colour.__doc__)

    def test_direct_call_with_both_args_returns_class(self):
        """define_aliases can be called with (alias_d, cls) directly."""
        class C:
            def get_speed(self):
                return 100
        result = define_aliases({"speed": ["velocity"]}, C)
        self.assertIs(result, C)
        self.assertTrue(hasattr(C, "get_velocity"))
        self.assertEqual(C().get_velocity(), 100)

    def test_inheritance_merges_alias_maps(self):
        """Child class alias map must include parent's entries."""
        @define_aliases({"color": ["colour"]})
        class Base:
            def get_color(self):
                return "red"
        @define_aliases({"size": ["sz"]})
        class Child(Base):
            def get_size(self):
                return 0
        self.assertIn("color", Child._alias_map)
        self.assertIn("size", Child._alias_map)

    def test_multiple_aliases_per_property_all_created(self):
        """Multiple aliases for one property must all be registered."""
        @define_aliases({"color": ["colour", "c", "clr"]})
        class C:
            def get_color(self):
                return "green"
        obj = C()
        self.assertEqual(obj.get_colour(), "green")
        self.assertEqual(obj.get_c(), "green")
        self.assertEqual(obj.get_clr(), "green")


# ===========================================================================
# select_matching_signature
# ===========================================================================

class TestSelectMatchingSignature(unittest.TestCase):
    """Tests for select_matching_signature."""

    def test_first_match_wins(self):
        result = select_matching_signature(
            [lambda a, b: (a, b), lambda x: x], 10, 20)
        self.assertEqual(result, (10, 20))

    def test_fallback_to_second(self):
        result = select_matching_signature(
            [lambda a, b: (a, b), lambda x: (x,)], "only")
        self.assertEqual(result, ("only",))

    def test_no_match_reraises_last_type_error(self):
        with self.assertRaises(TypeError):
            select_matching_signature(
                [lambda a: a, lambda b, c: (b, c)], 1, 2, 3)

    def test_kwargs_forwarded(self):
        result = select_matching_signature([lambda *, name: name], name="hello")
        self.assertEqual(result, "hello")

    def test_single_func_list(self):
        result = select_matching_signature([lambda: 99])
        self.assertEqual(result, 99)

    def test_return_value_propagated(self):
        sentinel = object()
        result = select_matching_signature([lambda: sentinel])
        self.assertIs(result, sentinel)

    def test_return_none_propagated(self):
        """None return must not be confused with a failed match."""
        result = select_matching_signature([lambda: None])
        self.assertIsNone(result)

    def test_non_type_error_propagates_immediately(self):
        """A ValueError from the first-and-only func must propagate."""
        def raise_value_error():
            raise ValueError("not a type error")
        with self.assertRaises(ValueError):
            select_matching_signature([raise_value_error])

    def test_multiple_fallbacks_selects_correct_one(self):
        result = select_matching_signature(
            [lambda a, b, c: 3, lambda a, b: 2, lambda a: 1], "x")
        self.assertEqual(result, 1)


# ===========================================================================
# nargs_error
# ===========================================================================

class TestNargsError(unittest.TestCase):
    """Tests for nargs_error."""

    def test_returns_type_error_not_raises(self):
        err = nargs_error("foo", takes=2, given=3)
        self.assertIsInstance(err, TypeError)

    def test_function_name_in_message(self):
        err = nargs_error("myfunc", takes=1, given=2)
        self.assertIn("myfunc()", str(err))

    def test_expected_count_in_message(self):
        err = nargs_error("f", takes=2, given=5)
        self.assertIn("2", str(err))

    def test_given_count_in_message(self):
        err = nargs_error("f", takes=2, given=5)
        self.assertIn("5", str(err))

    def test_takes_can_be_descriptive_string(self):
        err = nargs_error("f", takes="1 or 2", given=3)
        self.assertIn("1 or 2", str(err))

    def test_zero_takes(self):
        err = nargs_error("f", takes=0, given=1)
        self.assertIn("0", str(err))

    def test_zero_given(self):
        err = nargs_error("f", takes=2, given=0)
        self.assertIn("0", str(err))

    def test_message_mentions_positional_arguments(self):
        err = nargs_error("myfunc", takes=3, given=5)
        self.assertIn("positional arguments", str(err))


# ===========================================================================
# kwarg_error
# ===========================================================================

class TestKwargError(unittest.TestCase):
    """Tests for kwarg_error."""

    def test_returns_type_error_not_raises(self):
        err = kwarg_error("foo", "bad_kw")
        self.assertIsInstance(err, TypeError)

    def test_function_name_in_message(self):
        err = kwarg_error("myfunc", "bad_kw")
        self.assertIn("myfunc()", str(err))

    def test_string_kwarg_appears_in_message(self):
        err = kwarg_error("f", "unexpected")
        self.assertIn("unexpected", str(err))

    def test_iterable_uses_first_key(self):
        err = kwarg_error("f", ["first_bad", "second_bad"])
        self.assertIn("first_bad", str(err))

    def test_dict_uses_first_insertion_order_key(self):
        bad_kwargs = {"alpha": 1, "beta": 2}
        err = kwarg_error("g", bad_kwargs)
        self.assertIn("alpha", str(err))

    def test_generator_consumes_only_first_element(self):
        consumed = []
        def gen():
            for k in ["first", "second", "third"]:
                consumed.append(k)
                yield k
        kwarg_error("f", gen())
        self.assertEqual(consumed, ["first"])

    def test_empty_iterable_raises_stop_iteration(self):
        """
        Passing an empty iterable causes ``next(iter([]))`` to raise
        StopIteration.  This is the current documented behaviour; any future
        graceful handling would change this test.
        """
        with self.assertRaises(StopIteration):
            kwarg_error("f", [])


# ===========================================================================
# recursive_subclasses
# ===========================================================================

class TestRecursiveSubclasses(unittest.TestCase):
    """Tests for recursive_subclasses."""

    def test_yields_base_class(self):
        class A:
            pass
        self.assertIn(A, list(recursive_subclasses(A)))

    def test_yields_direct_subclass(self):
        class A:
            pass
        class B(A):
            pass
        self.assertIn(B, list(recursive_subclasses(A)))

    def test_yields_indirect_subclass(self):
        class A:
            pass
        class B(A):
            pass
        class C(B):
            pass
        self.assertIn(C, list(recursive_subclasses(A)))

    def test_base_class_is_first(self):
        class A:
            pass
        class B(A):
            pass
        result = list(recursive_subclasses(A))
        self.assertIs(result[0], A)

    def test_isolated_class_yields_only_itself(self):
        class Isolated:
            pass
        self.assertEqual(list(recursive_subclasses(Isolated)), [Isolated])

    def test_multiple_direct_subclasses_all_present(self):
        class Root:
            pass
        class X(Root):
            pass
        class Y(Root):
            pass
        result = list(recursive_subclasses(Root))
        self.assertIn(X, result)
        self.assertIn(Y, result)

    def test_diamond_inheritance_all_classes_present(self):
        class A:
            pass
        class B(A):
            pass
        class C(A):
            pass
        class D(B, C):
            pass
        result = list(recursive_subclasses(A))
        for cls in (A, B, C, D):
            self.assertIn(cls, result)

    def test_deep_chain_all_levels_present(self):
        class L0:
            pass
        class L1(L0):
            pass
        class L2(L1):
            pass
        class L3(L2):
            pass
        class L4(L3):
            pass
        result = list(recursive_subclasses(L0))
        for cls in (L0, L1, L2, L3, L4):
            self.assertIn(cls, result)

    def test_order_is_depth_first(self):
        """LeftChild must appear before Right sibling (depth-first traversal)."""
        class Root:
            pass
        class Left(Root):
            pass
        class LeftChild(Left):
            pass
        class Right(Root):
            pass
        result = list(recursive_subclasses(Root))
        self.assertLess(result.index(Left), result.index(LeftChild))
        self.assertLess(result.index(LeftChild), result.index(Right))

    def test_is_a_generator(self):
        """recursive_subclasses must return a generator object."""
        class A:
            pass
        gen = recursive_subclasses(A)
        self.assertTrue(hasattr(gen, "__next__"),
                        "recursive_subclasses must return a generator")


# ===========================================================================
# warn_external
# ===========================================================================

class TestWarnExternal(_WarnAssertMixin, unittest.TestCase):
    """Tests for warn_external."""

    def test_emits_user_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_external("test message", category=UserWarning)
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, UserWarning))

    def test_message_preserved(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_external("hello world", category=UserWarning)
        self.assertIn("hello world", str(w[0].message))

    def test_warning_instance_accepted(self):
        """warn_external must accept a Warning instance, not just strings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_external(UserWarning("instance warning"), category=UserWarning)
        self.assertEqual(len(w), 1)

    def test_custom_category_propagated(self):
        class CustomWarning(UserWarning):
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_external("custom", category=CustomWarning)
        self.assertTrue(issubclass(w[0].category, CustomWarning))

    def test_current_interpreter_emits_without_error(self):
        """Whichever Python version branch is taken must succeed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            api_module.warn_external("version branch test", category=UserWarning)
        self.assertEqual(len(w), 1)
        self.assertIn("version branch test", str(w[0].message))

    def test_py312_branch_no_type_error(self):
        """On Python ≥ 3.12 the skip_file_prefixes kwarg must be accepted."""
        if sys.version_info < (3, 12):
            self.skipTest("Python ≥ 3.12 branch not active on this interpreter")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            api_module.warn_external("py312 test", category=UserWarning)
        self.assertEqual(len(w), 1)
        self.assertIn("py312 test", str(w[0].message))

    def test_pre312_frame_walker_branch(self):
        """Force the < 3.12 frame-walker branch via monkeypatching sys.version_info."""
        original_version_info = sys.version_info
        try:
            sys.version_info = (3, 11, 0, "final", 0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                api_module.warn_external("frame branch test", category=UserWarning)
        finally:
            sys.version_info = original_version_info
        self.assertEqual(len(w), 1)
        self.assertIn("frame branch test", str(w[0].message))

    def test_category_none_does_not_raise(self):
        """category=None must not raise TypeError."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                api_module.warn_external("no category", category=None)
            except TypeError:
                self.fail("warn_external raised TypeError with category=None")

    def test_frame_walker_frame_none_embedded_context(self):
        """
        The frame-walker loop guards for ``frame is None`` (embedded context
        where frames may be missing).  Simulate via monkeypatching _getframe
        to return a fake chain ending in None.  No exception must occur.
        """
        original_version_info = sys.version_info
        original_getframe = sys._getframe

        class _FakeFrame:
            """Fake frame that terminates the chain immediately."""
            @property
            def f_globals(self):
                return {"__name__": "some_external_module"}
            @property
            def f_back(self):
                return None

        try:
            sys.version_info = (3, 11, 0, "final", 0)
            sys._getframe = lambda depth=0: _FakeFrame()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                api_module.warn_external("embedded ctx test", category=UserWarning)
        finally:
            sys.version_info = original_version_info
            sys._getframe = original_getframe
        # Reaching here without exception is the assertion.


# ===========================================================================
# Package integration
# ===========================================================================

class TestPackageIntegration(unittest.TestCase):
    """Ensure the package exports all expected symbols and invariants hold."""

    def test_deprecation_symbols_re_exported(self):
        """All deprecation symbols listed in __init__.py must be importable."""
        from .. import (
            deprecated,
            warn_deprecated,
            rename_parameter,
            delete_parameter,
            make_keyword_only,
            deprecate_method_override,
            deprecate_privatize_attribute,
            suppress_matplotlib_deprecation_warning,
            MatplotlibDeprecationWarning,
        )
        self.assertTrue(callable(deprecated))
        self.assertTrue(callable(warn_deprecated))
        self.assertTrue(callable(rename_parameter))
        self.assertTrue(callable(delete_parameter))
        self.assertTrue(callable(make_keyword_only))
        self.assertTrue(callable(deprecate_method_override))
        self.assertTrue(callable(suppress_matplotlib_deprecation_warning))
        self.assertTrue(issubclass(MatplotlibDeprecationWarning, DeprecationWarning))

    def test_unset_is_module_level_singleton(self):
        """api_module.UNSET must be the same object as the imported UNSET."""
        self.assertIs(api_module.UNSET, UNSET)

    def test_all_public_symbols_callable_or_instantiable(self):
        """All primary API symbols must be callable."""
        for sym in (
            check_isinstance, check_in_list, check_shape, check_getitem,
            caching_module_getattr, classproperty, define_aliases,
            select_matching_signature, nargs_error, kwarg_error,
            recursive_subclasses, warn_external,
        ):
            self.assertTrue(callable(sym), f"{sym!r} must be callable")


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
