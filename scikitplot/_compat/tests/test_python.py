# scikitplot/_compat/tests/test_python.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._compat.python`.

Runs standalone (``python -m unittest scikitplot._compat.tests.test_python``)
or under pytest.

Coverage map
------------
PYTHON_VERSION    Format and runtime agreement          → TestPythonVersion
lru_cache         Caching, signature/doc preservation,
                  maxsize, per-instance independence    → TestLruCache
cache             Caching, signature/doc preservation,
                  independence from lru_cache wrapper   → TestCache
"""

from __future__ import annotations

import sys
import functools
import unittest

from ..python import PYTHON_VERSION, cache, lru_cache


# ===========================================================================
# PYTHON_VERSION
# ===========================================================================


class TestPythonVersion(unittest.TestCase):
    """PYTHON_VERSION must match the running interpreter and be well-formed."""

    def test_is_string(self):
        """PYTHON_VERSION must be a str."""
        self.assertIsInstance(PYTHON_VERSION, str)

    def test_three_dot_separated_parts(self):
        """Must have exactly three dot-separated parts."""
        parts = PYTHON_VERSION.split(".")
        self.assertEqual(len(parts), 3, msg=f"Got {PYTHON_VERSION!r}")

    def test_all_parts_are_digits(self):
        """Each part must be a non-negative integer string."""
        for part in PYTHON_VERSION.split("."):
            self.assertTrue(
                part.isdigit(),
                msg=f"Part {part!r} in {PYTHON_VERSION!r} is not a digit string",
            )

    def test_major_matches_sys(self):
        """Major component must equal sys.version_info.major."""
        major = int(PYTHON_VERSION.split(".")[0])
        self.assertEqual(major, sys.version_info.major)

    def test_minor_matches_sys(self):
        """Minor component must equal sys.version_info.minor."""
        minor = int(PYTHON_VERSION.split(".")[1])
        self.assertEqual(minor, sys.version_info.minor)

    def test_micro_matches_sys(self):
        """Micro component must equal sys.version_info.micro."""
        micro = int(PYTHON_VERSION.split(".")[2])
        self.assertEqual(micro, sys.version_info.micro)


# ===========================================================================
# lru_cache
# ===========================================================================


class TestLruCache(unittest.TestCase):
    """lru_cache must cache, preserve metadata, and respect maxsize."""

    # ------------------------------------------------------------------
    # Metadata preservation (functools.wraps contract)
    # ------------------------------------------------------------------

    def test_preserves_function_name(self):
        """Wrapped function __name__ must equal the original function name."""

        def original_fn(x):
            """Original docstring."""
            return x

        wrapped = lru_cache(maxsize=16)(original_fn)
        self.assertEqual(wrapped.__name__, "original_fn")

    def test_preserves_docstring(self):
        """Wrapped function __doc__ must equal the original docstring."""

        def documented_fn(x):
            """Documented function."""
            return x

        wrapped = lru_cache(maxsize=16)(documented_fn)
        self.assertEqual(wrapped.__doc__, "Documented function.")

    def test_preserves_module(self):
        """Wrapped function __module__ must equal the original __module__."""

        def mod_fn(x):
            return x

        original_module = mod_fn.__module__
        wrapped = lru_cache(maxsize=16)(mod_fn)
        self.assertEqual(wrapped.__module__, original_module)

    # ------------------------------------------------------------------
    # Caching behaviour
    # ------------------------------------------------------------------

    def test_repeated_call_same_result(self):
        """Repeated calls with same args must return the same result."""

        @lru_cache(maxsize=32)
        def square(x):
            """Square function."""
            return x * x

        self.assertEqual(square(7), 49)
        self.assertEqual(square(7), 49)

    def test_caches_call_count(self):
        """Body of cached function must be executed only once per unique arg."""
        call_log = []

        @lru_cache(maxsize=32)
        def tracked(x):
            """Tracked function."""
            call_log.append(x)
            return x + 1

        tracked(10)
        tracked(10)
        tracked(10)
        self.assertEqual(call_log.count(10), 1)

    def test_different_args_both_cached(self):
        """Each unique argument must be cached independently."""
        call_log = []

        @lru_cache(maxsize=32)
        def recorded(x):
            """Recorded function."""
            call_log.append(x)
            return x

        recorded(1)
        recorded(2)
        recorded(1)
        recorded(2)
        self.assertEqual(call_log.count(1), 1)
        self.assertEqual(call_log.count(2), 1)

    def test_none_return_value_cached(self):
        """A function returning None must be cached correctly (not re-run)."""
        call_log = []

        @lru_cache(maxsize=8)
        def returns_none(x):
            """Returns None."""
            call_log.append(x)
            return None

        result1 = returns_none(5)
        result2 = returns_none(5)
        self.assertIsNone(result1)
        self.assertIsNone(result2)
        self.assertEqual(len(call_log), 1)

    def test_zero_return_value_cached(self):
        """A function returning 0 (falsy) must be cached correctly."""
        call_log = []

        @lru_cache(maxsize=8)
        def returns_zero(x):
            """Returns zero."""
            call_log.append(x)
            return 0

        _ = returns_zero(3)
        _ = returns_zero(3)
        self.assertEqual(len(call_log), 1)

    # ------------------------------------------------------------------
    # Independence between decorated functions
    # ------------------------------------------------------------------

    def test_two_decorations_are_independent(self):
        """Two separately decorated functions must not share a cache."""
        log_a, log_b = [], []

        @lru_cache(maxsize=16)
        def fn_a(x):
            """Function A."""
            log_a.append(x)
            return x

        @lru_cache(maxsize=16)
        def fn_b(x):
            """Function B."""
            log_b.append(x)
            return x

        fn_a(99)
        fn_b(99)
        fn_a(99)
        fn_b(99)
        self.assertEqual(len(log_a), 1)
        self.assertEqual(len(log_b), 1)

    # ------------------------------------------------------------------
    # Keyword arguments
    # ------------------------------------------------------------------

    def test_keyword_argument_cached(self):
        """Calls via keyword argument must be cached when args are equal."""
        call_log = []

        @lru_cache(maxsize=16)
        def kw_fn(x, y=0):
            """Keyword function."""
            call_log.append((x, y))
            return x + y

        kw_fn(1, y=2)
        kw_fn(1, y=2)
        self.assertEqual(len(call_log), 1)

    # ------------------------------------------------------------------
    # Decorator protocol
    # ------------------------------------------------------------------

    def test_returns_callable(self):
        """lru_cache(maxsize=N) must return a callable decorator."""
        decorator = lru_cache(maxsize=16)
        self.assertTrue(callable(decorator))

    def test_decorated_result_is_callable(self):
        """Applying the decorator must produce a callable."""

        @lru_cache(maxsize=8)
        def simple(x):
            """Simple."""
            return x

        self.assertTrue(callable(simple))


# ===========================================================================
# cache
# ===========================================================================


class TestCache(unittest.TestCase):
    """cache must cache results and preserve function metadata."""

    # ------------------------------------------------------------------
    # Metadata preservation
    # ------------------------------------------------------------------

    def test_preserves_function_name(self):
        """Wrapped function __name__ must equal the original."""

        def fn_cache_name(x):
            """Docstring for name test."""
            return x

        wrapped = cache(fn_cache_name)
        self.assertEqual(wrapped.__name__, "fn_cache_name")

    def test_preserves_docstring(self):
        """Wrapped function __doc__ must equal the original."""

        def fn_cache_doc(x):
            """Cache docstring."""
            return x

        wrapped = cache(fn_cache_doc)
        self.assertEqual(wrapped.__doc__, "Cache docstring.")

    def test_preserves_module(self):
        """Wrapped function __module__ must equal the original."""

        def fn_cache_mod(x):
            return x

        original_module = fn_cache_mod.__module__
        wrapped = cache(fn_cache_mod)
        self.assertEqual(wrapped.__module__, original_module)

    # ------------------------------------------------------------------
    # Caching behaviour
    # ------------------------------------------------------------------

    def test_repeated_call_same_result(self):
        """Repeated identical calls must return the same value."""

        @cache
        def cube(x):
            """Cube."""
            return x ** 3

        self.assertEqual(cube(3), 27)
        self.assertEqual(cube(3), 27)

    def test_body_called_once_per_unique_arg(self):
        """Function body must execute only once per unique argument."""
        call_log = []

        @cache
        def cache_tracked(x):
            """Cache tracked."""
            call_log.append(x)
            return x

        cache_tracked(42)
        cache_tracked(42)
        cache_tracked(42)
        self.assertEqual(call_log.count(42), 1)

    def test_different_args_independent(self):
        """Different arguments must produce independent cache entries."""
        call_log = []

        @cache
        def cache_multi(x):
            """Cache multi."""
            call_log.append(x)
            return x

        cache_multi("a")
        cache_multi("b")
        cache_multi("a")
        self.assertEqual(call_log.count("a"), 1)
        self.assertEqual(call_log.count("b"), 1)

    def test_none_return_value_cached(self):
        """Return value of None must be cached (not treated as cache-miss)."""
        call_log = []

        @cache
        def cache_none(x):
            """Cache none."""
            call_log.append(x)
            return None

        cache_none(1)
        cache_none(1)
        self.assertIsNone(cache_none(1))
        self.assertEqual(len(call_log), 1)

    # ------------------------------------------------------------------
    # Independence from lru_cache
    # ------------------------------------------------------------------

    def test_cache_and_lru_cache_are_independent(self):
        """cache-decorated and lru_cache-decorated functions must not share state."""
        log_cache, log_lru = [], []

        @cache
        def fn_via_cache(x):
            """Via cache."""
            log_cache.append(x)
            return x

        @lru_cache(maxsize=16)
        def fn_via_lru(x):
            """Via lru."""
            log_lru.append(x)
            return x

        fn_via_cache(7)
        fn_via_lru(7)
        fn_via_cache(7)
        fn_via_lru(7)
        self.assertEqual(len(log_cache), 1)
        self.assertEqual(len(log_lru), 1)

    def test_cache_decorator_applied_directly(self):
        """cache must be usable as a bare decorator without parentheses."""

        @cache
        def identity(x):
            """Identity."""
            return x

        self.assertEqual(identity("hello"), "hello")
        self.assertTrue(callable(identity))


if __name__ == "__main__":
    unittest.main(verbosity=2)
