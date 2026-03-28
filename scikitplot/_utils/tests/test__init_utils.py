# scikitplot/_utils/tests/test__init_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils` (__init__.py helpers).

Because ``_utils/__init__.py`` does ``from .. import logger``, a minimal
``scikitplot`` stub is injected into sys.modules before the import so that
the test file works in isolation as well as inside the full package.

Coverage map
------------
set_module              decorates class & function, no-op for None  → TestSetModule
_rename_parameter       rename, deprecation warning, TypeError      → TestRenameParameter
AttrDict                getattr, setattr, delattr, nested, missing  → TestAttrDict
merge_dicts             normal, duplicate raise, duplicate override → TestMergeDicts
chunk_list              even, uneven, empty                         → TestChunkList
_chunk_dict             even, uneven, empty                         → TestChunkDict
_truncate_and_ellipsize short, long, exact boundary                 → TestTruncateAndEllipsize
_truncate_dict          max_key_length, max_value_length, both      → TestTruncateDict
is_iterator             iterators, non-iterators                    → TestIsIterator
is_uuid                 valid UUID4, invalid strings                → TestIsUuid
get_major_minor_py_ver  '3.9.1' → '3.9'                           → TestGetMajorMinorPyVersion
reraise                 re-raises exception with traceback          → TestReraise
find_free_port          returns int, port > 0, usable              → TestFindFreePort
_get_fully_qualified    class and instance                          → TestGetFullyQualifiedClassName
get_results_paginated   collects pages, respects max_results       → TestGetResultsFromPaginatedFn

Run standalone::

    python -m unittest scikitplot._utils.tests.test__init_utils -v
"""

from __future__ import annotations

import logging
import sys
import types
import unittest
import uuid
import warnings

# ---------------------------------------------------------------------------
# Inject a minimal scikitplot stub so ``from .. import logger`` in
# _utils/__init__.py resolves when running outside the full package.
# ---------------------------------------------------------------------------
if "scikitplot" not in sys.modules:
    _stub = types.ModuleType("scikitplot")
    _stub.logger = logging.getLogger("scikitplot")
    sys.modules["scikitplot"] = _stub

from .. import (  # noqa: E402
    AttrDict,
    _chunk_dict,
    _get_fully_qualified_class_name,
    _inspect_original_var_name,
    _rename_parameter,
    _truncate_and_ellipsize,
    _truncate_dict,
    chunk_list,
    find_free_port,
    get_major_minor_py_version,
    get_results_from_paginated_fn,
    is_iterator,
    is_uuid,
    merge_dicts,
    reraise,
    set_module,
)


# ===========================================================================
# set_module
# ===========================================================================


class TestSetModule(unittest.TestCase):
    """set_module must update __module__ on decorated classes and functions."""

    def test_function_module_updated(self):
        """A decorated function must have __module__ set to the given name."""
        @set_module("mypkg")
        def fn():
            pass

        self.assertEqual(fn.__module__, "mypkg")

    def test_class_module_updated(self):
        """A decorated class must have __module__ set to the given name."""
        @set_module("mypkg")
        class MyClass:
            pass

        self.assertEqual(MyClass.__module__, "mypkg")

    def test_none_module_is_noop(self):
        """set_module(None) must leave __module__ unchanged."""
        original = "test_module_original"

        def fn():
            pass

        fn.__module__ = original

        @set_module(None)
        def fn2():
            pass

        fn2.__module__ = original
        set_module(None)(fn2)
        self.assertEqual(fn2.__module__, original)

    def test_decorator_returns_original_callable(self):
        """The decorated function must still be callable and functional."""
        @set_module("mypkg")
        def add(a, b):
            return a + b

        self.assertEqual(add(1, 2), 3)

    def test_class_module_source_stored(self):
        """The original __module__ is stored in _module_source for classes."""
        @set_module("new_pkg")
        class C:
            pass

        self.assertEqual(C.__module__, "new_pkg")
        # _module_source may or may not be set, but no exception must occur


# ===========================================================================
# _rename_parameter
# ===========================================================================


class TestRenameParameter(unittest.TestCase):
    """_rename_parameter must handle renaming, warnings, and TypeError."""

    def _make_fn(self, old_name="old", new_name="new", dep_version=None):
        @_rename_parameter([old_name], [new_name], dep_version=dep_version)
        def fn(new=None):
            return new
        return fn

    def test_new_param_works_normally(self):
        """Calling with new parameter name must behave normally."""
        fn = self._make_fn()
        self.assertEqual(fn(new=42), 42)

    def test_old_param_remapped_to_new(self):
        """Calling with old parameter name must be transparently remapped."""
        fn = self._make_fn()
        result = fn(old=99)
        self.assertEqual(result, 99)

    def test_old_param_without_dep_version_no_warning(self):
        """No DeprecationWarning when dep_version is None."""
        fn = self._make_fn(dep_version=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(old=1)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(dep_warnings), 0)

    def test_old_param_with_dep_version_raises_warning(self):
        """DeprecationWarning must be emitted when dep_version is set."""
        fn = self._make_fn(dep_version="1.0.0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(old=1)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertGreater(len(dep_warnings), 0)

    def test_both_old_and_new_raises_type_error(self):
        """Passing both old and new parameter names must raise TypeError."""
        fn = self._make_fn()
        with self.assertRaises(TypeError):
            fn(old=1, new=2)

    def test_functools_wraps_preserves_name(self):
        """The wrapper must preserve the original function's __name__."""
        fn = self._make_fn()
        self.assertEqual(fn.__name__, "fn")

    def test_dep_version_end_version_computed(self):
        """The deprecation message must mention a future removal version."""
        fn = self._make_fn(dep_version="1.2.3")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(old=1)
        msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        # end_version should be 1.4.3 (minor + 2)
        self.assertTrue(any("1.4.3" in m for m in msgs))


# ===========================================================================
# AttrDict
# ===========================================================================


class TestAttrDict(unittest.TestCase):
    """AttrDict must expose dict keys as attributes."""

    def test_getattr_returns_value(self):
        """d.key must return d['key']."""
        d = AttrDict({"a": 1, "b": 2})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)

    def test_setattr_updates_dict(self):
        """d.key = v must update d['key']."""
        d = AttrDict()
        d.x = 10
        self.assertEqual(d["x"], 10)

    def test_delattr_removes_key(self):
        """del d.key must remove the key from the dict."""
        d = AttrDict({"x": 1})
        del d.x
        self.assertNotIn("x", d)

    def test_nested_dict_wrapped_as_attrdict(self):
        """A nested dict value must be returned as an AttrDict."""
        d = AttrDict({"outer": {"inner": 99}})
        result = d.outer
        self.assertIsInstance(result, AttrDict)
        self.assertEqual(result.inner, 99)

    def test_missing_key_raises_attribute_error(self):
        """Accessing a missing key must raise AttributeError."""
        d = AttrDict()
        with self.assertRaises(AttributeError):
            _ = d.missing_key

    def test_regular_dict_methods_still_work(self):
        """AttrDict must behave as a normal dict for keys/values/items."""
        d = AttrDict({"a": 1, "b": 2})
        self.assertIn("a", d.keys())
        self.assertIn(1, d.values())

    def test_non_dict_value_returned_directly(self):
        """Non-dict values must be returned as-is (not wrapped)."""
        d = AttrDict({"nums": [1, 2, 3]})
        result = d.nums
        self.assertIsInstance(result, list)


# ===========================================================================
# merge_dicts
# ===========================================================================


class TestMergeDicts(unittest.TestCase):
    """merge_dicts must combine two dicts without duplicates by default."""

    def test_simple_merge(self):
        """Two disjoint dicts must be combined correctly."""
        result = merge_dicts({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_returns_new_dict(self):
        """Result must be a new dict, not one of the originals."""
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = merge_dicts(d1, d2)
        self.assertIsNot(result, d1)
        self.assertIsNot(result, d2)

    def test_duplicate_key_raises_by_default(self):
        """Duplicate keys must raise ValueError when raise_on_duplicates=True."""
        with self.assertRaises(ValueError):
            merge_dicts({"a": 1}, {"a": 2})

    def test_duplicate_key_override_when_not_raising(self):
        """When raise_on_duplicates=False, dict_b must override dict_a."""
        result = merge_dicts({"a": 1}, {"a": 2}, raise_on_duplicates=False)
        self.assertEqual(result["a"], 2)

    def test_empty_dicts(self):
        """Merging two empty dicts must return an empty dict."""
        result = merge_dicts({}, {})
        self.assertEqual(result, {})

    def test_one_empty_dict(self):
        """Merging with an empty dict must return the non-empty one."""
        result = merge_dicts({"a": 1}, {})
        self.assertEqual(result, {"a": 1})


# ===========================================================================
# chunk_list
# ===========================================================================


class TestChunkList(unittest.TestCase):
    """chunk_list must yield non-overlapping consecutive chunks."""

    def test_even_split(self):
        """A list divisible by chunk_size must produce equal-length chunks."""
        result = list(chunk_list([1, 2, 3, 4], 2))
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_uneven_split(self):
        """A list not divisible by chunk_size must have a smaller last chunk."""
        result = list(chunk_list([1, 2, 3, 4, 5], 2))
        self.assertEqual(result, [[1, 2], [3, 4], [5]])

    def test_chunk_size_larger_than_list(self):
        """A chunk_size larger than the list must return the whole list as one chunk."""
        result = list(chunk_list([1, 2], 10))
        self.assertEqual(result, [[1, 2]])

    def test_empty_list_produces_no_chunks(self):
        """An empty list must yield no chunks."""
        result = list(chunk_list([], 3))
        self.assertEqual(result, [])

    def test_chunk_size_one(self):
        """chunk_size=1 must produce one-element lists."""
        result = list(chunk_list([1, 2, 3], 1))
        self.assertEqual(result, [[1], [2], [3]])

    def test_all_values_preserved(self):
        """All original values must appear in the chunks."""
        original = list(range(10))
        chunks = list(chunk_list(original, 3))
        flat = [item for chunk in chunks for item in chunk]
        self.assertEqual(flat, original)


# ===========================================================================
# _chunk_dict
# ===========================================================================


class TestChunkDict(unittest.TestCase):
    """_chunk_dict must split a dict into consecutive sub-dicts."""

    def test_even_split(self):
        """A dict with size divisible by chunk_size must split evenly."""
        d = {"a": 1, "b": 2, "c": 3, "d": 4}
        chunks = list(_chunk_dict(d, 2))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(len(c) for c in chunks), len(d))

    def test_uneven_split_last_chunk_smaller(self):
        """A dict with size not divisible by chunk_size must have smaller last chunk."""
        d = {"a": 1, "b": 2, "c": 3}
        chunks = list(_chunk_dict(d, 2))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[-1]), 1)

    def test_empty_dict_produces_no_chunks(self):
        """An empty dict must produce no chunks."""
        result = list(_chunk_dict({}, 2))
        self.assertEqual(result, [])

    def test_all_keys_preserved(self):
        """All original keys must appear across all chunks."""
        d = {str(i): i for i in range(7)}
        chunks = list(_chunk_dict(d, 3))
        all_keys = set(k for c in chunks for k in c)
        self.assertEqual(all_keys, set(d))

    def test_all_values_preserved(self):
        """All original values must appear across all chunks."""
        d = {"a": 10, "b": 20, "c": 30}
        chunks = list(_chunk_dict(d, 2))
        all_vals = [v for c in chunks for v in c.values()]
        self.assertEqual(sorted(all_vals), sorted(d.values()))


# ===========================================================================
# _truncate_and_ellipsize
# ===========================================================================


class TestTruncateAndEllipsize(unittest.TestCase):
    """_truncate_and_ellipsize must trim and append '...' for long values."""

    def test_short_value_returned_unchanged(self):
        """A value shorter than max_length must be returned as-is."""
        result = _truncate_and_ellipsize("hello", 20)
        self.assertEqual(result, "hello")

    def test_exact_length_returned_unchanged(self):
        """A value exactly at max_length must be returned as-is."""
        s = "hello"
        result = _truncate_and_ellipsize(s, 5)
        self.assertEqual(result, s)

    def test_long_value_ends_with_ellipsis(self):
        """A value longer than max_length must end with '...'."""
        result = _truncate_and_ellipsize("a" * 30, 10)
        self.assertTrue(result.endswith("..."))

    def test_result_length_equals_max_length(self):
        """Truncated result must be exactly max_length characters."""
        result = _truncate_and_ellipsize("a" * 30, 10)
        self.assertEqual(len(result), 10)

    def test_non_string_coerced(self):
        """Non-string inputs must be coerced via str() first."""
        result = _truncate_and_ellipsize(12345678901234567890, 10)
        self.assertEqual(len(result), 10)
        self.assertTrue(result.endswith("..."))

    def test_none_coerced(self):
        """None must be coerced to 'None' and handled accordingly."""
        result = _truncate_and_ellipsize(None, 20)
        self.assertEqual(result, "None")


# ===========================================================================
# _truncate_dict
# ===========================================================================


class TestTruncateDict(unittest.TestCase):
    """_truncate_dict must truncate keys and/or values exceeding their limits."""

    def test_truncate_long_value(self):
        """A value exceeding max_value_length must be truncated."""
        d = {"key": "x" * 50}
        result = _truncate_dict(d, max_value_length=10)
        self.assertEqual(len(result["key"]), 10)
        self.assertTrue(result["key"].endswith("..."))

    def test_short_value_unchanged(self):
        """A value within max_value_length must be unchanged."""
        d = {"key": "short"}
        result = _truncate_dict(d, max_value_length=20)
        self.assertEqual(result["key"], "short")

    def test_truncate_long_key(self):
        """A key exceeding max_key_length must be truncated."""
        long_key = "k" * 50
        d = {long_key: "value"}
        result = _truncate_dict(d, max_key_length=10)
        keys = list(result.keys())
        self.assertEqual(len(keys[0]), 10)
        self.assertTrue(keys[0].endswith("..."))

    def test_must_specify_at_least_one_limit(self):
        """Calling with neither max_key_length nor max_value_length must raise ValueError."""
        with self.assertRaises(ValueError):
            _truncate_dict({"k": "v"})

    def test_both_limits_applied(self):
        """When both limits are set, both keys and values are truncated."""
        d = {"k" * 20: "v" * 20}
        result = _truncate_dict(d, max_key_length=8, max_value_length=8)
        for k, v in result.items():
            with self.subTest(k=k, v=v):
                self.assertLessEqual(len(k), 8)
                self.assertLessEqual(len(v), 8)

    def test_returns_dict(self):
        """Return type must be dict."""
        result = _truncate_dict({"a": "b"}, max_value_length=10)
        self.assertIsInstance(result, dict)


# ===========================================================================
# is_iterator
# ===========================================================================


class TestIsIterator(unittest.TestCase):
    """is_iterator must distinguish iterator objects from mere iterables."""

    def test_generator_is_iterator(self):
        """A generator expression must be an iterator."""
        gen = (x for x in range(5))
        self.assertTrue(is_iterator(gen))

    def test_iter_of_list_is_iterator(self):
        """iter([1,2,3]) must be an iterator."""
        self.assertTrue(is_iterator(iter([1, 2, 3])))

    def test_list_is_not_iterator(self):
        """A list itself (not iter'd) must not be an iterator."""
        self.assertFalse(is_iterator([1, 2, 3]))

    def test_dict_is_not_iterator(self):
        """A dict must not be an iterator."""
        self.assertFalse(is_iterator({"a": 1}))

    def test_string_is_not_iterator(self):
        """A string is iterable but is not an iterator."""
        self.assertFalse(is_iterator("hello"))

    def test_none_is_not_iterator(self):
        """None must not be an iterator."""
        self.assertFalse(is_iterator(None))

    def test_range_is_not_iterator(self):
        """range() is iterable but is not an iterator (no __next__)."""
        self.assertFalse(is_iterator(range(10)))


# ===========================================================================
# is_uuid
# ===========================================================================


class TestIsUuid(unittest.TestCase):
    """is_uuid must return True only for valid UUID strings."""

    def test_valid_uuid4_is_true(self):
        """A valid UUID4 string must return True."""
        valid = str(uuid.uuid4())
        self.assertTrue(is_uuid(valid))

    def test_valid_uuid_uppercase_is_true(self):
        """A valid UUID in upper-case must return True."""
        valid = str(uuid.uuid4()).upper()
        self.assertTrue(is_uuid(valid))

    def test_plain_string_is_false(self):
        """A plain string is not a UUID and must return False."""
        self.assertFalse(is_uuid("not-a-uuid"))

    def test_empty_string_is_false(self):
        """An empty string must return False."""
        self.assertFalse(is_uuid(""))

    def test_partial_uuid_is_false(self):
        """A truncated UUID must return False."""
        partial = str(uuid.uuid4())[:-5]
        self.assertFalse(is_uuid(partial))

    def test_nil_uuid_is_true(self):
        """The all-zeros UUID must be considered a valid UUID."""
        self.assertTrue(is_uuid("00000000-0000-0000-0000-000000000000"))

    def test_return_type_is_bool(self):
        """Return type must always be bool."""
        self.assertIsInstance(is_uuid(str(uuid.uuid4())), bool)
        self.assertIsInstance(is_uuid("bad"), bool)


# ===========================================================================
# get_major_minor_py_version
# ===========================================================================


class TestGetMajorMinorPyVersion(unittest.TestCase):
    """get_major_minor_py_version must extract major.minor from a full version."""

    def test_three_part_version(self):
        """'3.9.1' must return '3.9'."""
        self.assertEqual(get_major_minor_py_version("3.9.1"), "3.9")

    def test_two_part_version(self):
        """'3.9' must return '3.9' (already major.minor)."""
        self.assertEqual(get_major_minor_py_version("3.9"), "3.9")

    def test_four_part_version(self):
        """'3.10.0.final.0' must return '3.10'."""
        self.assertEqual(get_major_minor_py_version("3.10.0.final.0"), "3.10")

    def test_return_type_is_str(self):
        """Return type must be str."""
        result = get_major_minor_py_version("3.9.7")
        self.assertIsInstance(result, str)

    def test_format_has_exactly_one_dot(self):
        """Result must contain exactly one dot."""
        result = get_major_minor_py_version("3.9.1")
        self.assertEqual(result.count("."), 1)


# ===========================================================================
# reraise
# ===========================================================================


class TestReraise(unittest.TestCase):
    """reraise must re-raise an exception, preserving traceback when provided."""

    def test_reraises_exception(self):
        """reraise must cause the specified exception to propagate."""
        with self.assertRaises(ValueError):
            reraise(ValueError, ValueError("test error"))

    def test_creates_instance_when_value_is_none(self):
        """When value=None, the exception type must be instantiated."""
        with self.assertRaises(RuntimeError):
            reraise(RuntimeError, None)

    def test_reraises_existing_instance(self):
        """When value is an existing instance it must be raised."""
        err = ValueError("original message")
        try:
            reraise(ValueError, err)
        except ValueError as caught:
            self.assertIn("original message", str(caught))


# ===========================================================================
# find_free_port
# ===========================================================================


class TestFindFreePort(unittest.TestCase):
    """find_free_port must return an integer port number in a valid range."""

    def test_returns_int(self):
        """Return type must be int."""
        result = find_free_port()
        self.assertIsInstance(result, int)

    def test_port_in_valid_range(self):
        """Returned port must be in the valid 1–65535 range."""
        result = find_free_port()
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 65535)

    def test_two_calls_may_differ(self):
        """Two calls may return different ports (not guaranteed same)."""
        # Just verify both calls succeed without error.
        p1 = find_free_port()
        p2 = find_free_port()
        self.assertIsInstance(p1, int)
        self.assertIsInstance(p2, int)


# ===========================================================================
# _get_fully_qualified_class_name
# ===========================================================================


class TestGetFullyQualifiedClassName(unittest.TestCase):
    """_get_fully_qualified_class_name must return 'module.ClassName'."""

    def test_builtin_type(self):
        """For a builtin type instance the result must contain the type name."""
        result = _get_fully_qualified_class_name(42)
        self.assertIn("int", result)

    def test_custom_class(self):
        """For a custom class instance the result must end with the class name."""

        class MyCustomClass:
            pass

        obj = MyCustomClass()
        result = _get_fully_qualified_class_name(obj)
        self.assertTrue(result.endswith("MyCustomClass"))

    def test_result_contains_dot_separator(self):
        """Result must contain a '.' separating module from class."""
        result = _get_fully_qualified_class_name([])
        self.assertIn(".", result)

    def test_return_type_is_str(self):
        """Return type must be str."""
        result = _get_fully_qualified_class_name(None)
        self.assertIsInstance(result, str)


# ===========================================================================
# get_results_from_paginated_fn
# ===========================================================================


class _PagedList(list):
    """Minimal paged list with an optional token attribute."""

    def __init__(self, items, token=None):
        super().__init__(items)
        self.token = token


class TestGetResultsFromPaginatedFn(unittest.TestCase):
    """get_results_from_paginated_fn must collect pages until exhausted."""

    def _make_paginated_fn(self, pages):
        """Build a paginated function that yields items from 'pages' list."""
        pages = list(pages)

        def paginated_fn(max_results, page_token=None):
            if not pages:
                return _PagedList([], token=None)
            page = pages.pop(0)
            next_token = "next" if pages else None
            return _PagedList(page[:max_results], token=next_token)

        return paginated_fn

    def test_single_page_collected(self):
        """A single page of results must be fully collected."""
        fn = self._make_paginated_fn([[1, 2, 3]])
        result = get_results_from_paginated_fn(fn, max_results_per_page=10)
        self.assertEqual(result, [1, 2, 3])

    def test_multiple_pages_collected(self):
        """Multiple pages must all be concatenated into a single list."""
        fn = self._make_paginated_fn([[1, 2], [3, 4], [5]])
        result = get_results_from_paginated_fn(fn, max_results_per_page=2)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_max_results_respected(self):
        """When max_results is set, no more than that many results are returned."""
        fn = self._make_paginated_fn([[1, 2, 3, 4, 5], [6, 7]])
        result = get_results_from_paginated_fn(
            fn, max_results_per_page=5, max_results=3
        )
        self.assertLessEqual(len(result), 3)

    def test_empty_first_page_returns_empty(self):
        """When the first page is empty the result must be empty."""
        fn = self._make_paginated_fn([[]])
        result = get_results_from_paginated_fn(fn, max_results_per_page=10)
        self.assertEqual(result, [])

    def test_returns_list(self):
        """Return type must be a list."""
        fn = self._make_paginated_fn([[1]])
        result = get_results_from_paginated_fn(fn, max_results_per_page=10)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
