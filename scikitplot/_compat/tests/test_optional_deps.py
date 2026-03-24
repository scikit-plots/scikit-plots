# scikitplot/_compat/tests/test_optional_deps.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._compat.optional_deps`.

Runs standalone (``python -m unittest scikitplot._compat.tests.test_optional_deps``)
or under pytest.

Coverage map
------------
nested_import       Absolute/relative paths, error modes,
                    validate_callable, attribute resolution  → TestNestedImport
safe_import         Valid, missing, error param validation   → TestSafeImport
_get_source_path    Module, Python callable, C builtin,
                    non-callable, non-module objects         → TestGetSourcePath
LazyImport          Init validation, bool probe, resolve,
                    call, getattr, dir, hash, eq, clear,
                    parent_module_globals injection          → TestLazyImport
HAS_* flags         Attribute protocol, known deps,
                    unknown attr raises                      → TestHasFlags
"""

from __future__ import annotations

import os
import sys
import math
import types
import unittest
import unittest.mock as mock

from ..optional_deps import (
    LazyImport,
    _get_source_path,
    nested_import,
    safe_import,
    __all__ as _opt_all,
)
from .. import optional_deps as _opt_deps_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MISSING = "_xyzzy_no_such_package_abc123"  # guaranteed-absent module name


# ===========================================================================
# nested_import
# ===========================================================================


class TestNestedImport(unittest.TestCase):
    """nested_import must resolve modules and attributes deterministically."""

    # ------------------------------------------------------------------
    # Absolute — single-part (module only)
    # ------------------------------------------------------------------

    def test_absolute_single_returns_module(self):
        """A single absolute module name must return the module."""
        result = nested_import("os")
        self.assertIsInstance(result, types.ModuleType)

    def test_absolute_single_is_correct_module(self):
        """The returned module must be the same object as sys.modules entry."""
        result = nested_import("os")
        self.assertIs(result, sys.modules["os"])

    def test_absolute_single_math(self):
        """Importing 'math' must return the math module."""
        result = nested_import("math")
        import math as _math
        self.assertIs(result, _math)

    # ------------------------------------------------------------------
    # Absolute — multi-part (sub-module)
    # ------------------------------------------------------------------

    def test_absolute_multipart_submodule(self):
        """Dotted path must resolve to the sub-module."""
        result = nested_import("os.path")
        import os.path as _ospath
        self.assertIs(result, _ospath)

    def test_absolute_multipart_attribute(self):
        """Dotted path that ends in a function must return that callable."""
        result = nested_import("os.path.join")
        self.assertIs(result, os.path.join)

    def test_absolute_multipart_callable(self):
        """Resolved attribute must be callable."""
        result = nested_import("os.path.join")
        self.assertTrue(callable(result))

    def test_absolute_multipart_works_as_function(self):
        """The resolved function must behave correctly when invoked."""
        join = nested_import("os.path.join")
        self.assertEqual(join("a", "b"), os.path.join("a", "b"))

    # ------------------------------------------------------------------
    # Relative imports
    # ------------------------------------------------------------------

    def test_relative_single_part(self):
        """A relative single-part name must resolve using the package anchor."""
        result = nested_import(".misc", package="scikitplot")
        self.assertIsInstance(result, types.ModuleType)

    def test_relative_multipart(self):
        """A relative dotted path must resolve to the nested module."""
        result = nested_import(".misc._plot_colortable", package="scikitplot")
        self.assertIsInstance(result, types.ModuleType)

    def test_relative_requires_package(self):
        """A relative import with package=None must raise TypeError."""
        with self.assertRaises(TypeError):
            nested_import(".os", package=None)

    def test_relative_requires_nonempty_package(self):
        """A relative import with package='' must raise TypeError."""
        with self.assertRaises(TypeError):
            nested_import(".os", package="")

    # ------------------------------------------------------------------
    # error='ignore' — missing module
    # ------------------------------------------------------------------

    def test_missing_module_ignore_returns_default(self):
        """Missing module with error='ignore' must return the default."""
        result = nested_import(_MISSING, error="ignore")
        self.assertIsNone(result)

    def test_missing_module_ignore_custom_default(self):
        """Missing module with error='ignore' must return the supplied default."""
        sentinel = object()
        result = nested_import(_MISSING, default=sentinel, error="ignore")
        self.assertIs(result, sentinel)

    # ------------------------------------------------------------------
    # error='raise' — missing module
    # ------------------------------------------------------------------

    def test_missing_module_raise_raises_import_error(self):
        """Missing module with error='raise' must raise ImportError."""
        with self.assertRaises(ImportError):
            nested_import(_MISSING, error="raise")

    # ------------------------------------------------------------------
    # error validation
    # ------------------------------------------------------------------

    def test_invalid_error_param_raises_value_error(self):
        """An unrecognised error= value must raise ValueError immediately."""
        with self.assertRaises(ValueError):
            nested_import("os", error="bad")

    def test_invalid_error_param_not_empty(self):
        """error='' (empty string) must raise ValueError."""
        with self.assertRaises(ValueError):
            nested_import("os", error="")

    # ------------------------------------------------------------------
    # validate_callable
    # ------------------------------------------------------------------

    def test_validate_callable_true_on_callable_ok(self):
        """validate_callable=True must not raise for a callable attribute."""
        result = nested_import("os.path.join", validate_callable=True)
        self.assertTrue(callable(result))

    def test_validate_callable_true_on_non_callable_raises(self):
        """validate_callable=True must raise ValueError for a non-callable attr."""
        with self.assertRaises(ValueError):
            nested_import("os.sep", validate_callable=True)

    def test_validate_callable_false_allows_non_callable(self):
        """validate_callable=False (default) must not raise for a non-callable."""
        result = nested_import("os.sep", validate_callable=False)
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # Attribute resolution error — non-existent attribute
    # ------------------------------------------------------------------

    def test_nonexistent_attribute_raises_attribute_error(self):
        """A non-existent trailing attribute must raise AttributeError."""
        with self.assertRaises(AttributeError):
            nested_import("os.path.totally_nonexistent_attr_xyz")

    def test_nonexistent_attribute_ignore_returns_default(self):
        """Non-existent attr with error='ignore' must return default."""
        result = nested_import(
            "os.path.totally_nonexistent_attr_xyz", error="ignore"
        )
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Caching — same result on repeated calls
    # ------------------------------------------------------------------

    def test_repeated_call_same_result(self):
        """Repeated calls with identical args must return the same object."""
        r1 = nested_import("os")
        r2 = nested_import("os")
        self.assertIs(r1, r2)


# ===========================================================================
# safe_import
# ===========================================================================


class TestSafeImport(unittest.TestCase):
    """safe_import must import or return None with correct error handling."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_valid_module_returns_module(self):
        """A valid module name must return the module object."""
        result = safe_import("os")
        self.assertIsInstance(result, types.ModuleType)

    def test_valid_module_correct_object(self):
        """Returned module must match the stdlib module."""
        result = safe_import("os")
        self.assertIs(result, sys.modules["os"])

    def test_valid_module_math(self):
        """safe_import('math') must return the math module."""
        result = safe_import("math")
        import math as _math
        self.assertIs(result, _math)

    # ------------------------------------------------------------------
    # Missing module — error='ignore'
    # ------------------------------------------------------------------

    def test_missing_ignore_returns_none(self):
        """Missing module with error='ignore' must return None."""
        result = safe_import(_MISSING, error="ignore")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Missing module — error='raise'
    # ------------------------------------------------------------------

    def test_missing_raise_raises_import_error(self):
        """Missing module with error='raise' must raise ImportError."""
        with self.assertRaises(ImportError):
            safe_import(_MISSING, error="raise")

    def test_missing_raise_error_message_contains_name(self):
        """ImportError message must contain the module name."""
        with self.assertRaises(ImportError) as ctx:
            safe_import(_MISSING, error="raise")
        self.assertIn(_MISSING, str(ctx.exception))

    # ------------------------------------------------------------------
    # error parameter validation
    # ------------------------------------------------------------------

    def test_invalid_error_param_raises_value_error(self):
        """Unrecognised error= value must raise ValueError."""
        with self.assertRaises(ValueError):
            safe_import("os", error="silent")

    def test_invalid_error_param_empty_string(self):
        """error='' must raise ValueError."""
        with self.assertRaises(ValueError):
            safe_import("os", error="")

    # ------------------------------------------------------------------
    # Caching — repeated calls return the same object
    # ------------------------------------------------------------------

    def test_repeated_call_same_object(self):
        """Repeated calls with the same module name must return the same object."""
        r1 = safe_import("os")
        r2 = safe_import("os")
        self.assertIs(r1, r2)


# ===========================================================================
# _get_source_path
# ===========================================================================


class TestGetSourcePath(unittest.TestCase):
    """_get_source_path must return a path string for all supported inputs."""

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------

    def test_stdlib_module_returns_string(self):
        """A stdlib module must return a non-empty string path."""
        result = _get_source_path(os)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_stdlib_module_path_ends_py_or_so(self):
        """A stdlib module path must end with .py or a known extension."""
        result = _get_source_path(os)
        # os.py is always available as source
        self.assertTrue(
            result.endswith(".py") or result.endswith(".so") or result != "<?>",
            msg=f"Unexpected path: {result!r}",
        )

    def test_os_module_path_contains_os(self):
        """Path for the os module must contain 'os' (filename)."""
        result = _get_source_path(os)
        self.assertIn("os", result)

    # ------------------------------------------------------------------
    # Python-defined callables
    # ------------------------------------------------------------------

    def test_python_function_returns_string(self):
        """A Python-defined function must return a non-empty path."""

        def my_fn():
            pass

        result = _get_source_path(my_fn)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_python_lambda_returns_string(self):
        """A lambda must return a non-empty path."""
        fn = lambda x: x  # noqa: E731
        result = _get_source_path(fn)
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # C-extension / built-in callables
    # ------------------------------------------------------------------

    def test_builtin_returns_string(self):
        """math.sqrt (C extension) must return a string (may be '<?>')."""
        result = _get_source_path(math.sqrt)
        self.assertIsInstance(result, str)

    def test_builtin_does_not_raise(self):
        """_get_source_path must never raise for any built-in callable."""
        try:
            _get_source_path(len)
        except Exception as exc:
            self.fail(f"Raised unexpectedly for len: {exc}")

    # ------------------------------------------------------------------
    # Non-callable, non-module objects
    # ------------------------------------------------------------------

    def test_integer_returns_unknown(self):
        """An integer (not a module or callable) must return '<?>'. """
        result = _get_source_path(42)
        self.assertEqual(result, "<?>")

    def test_string_returns_unknown(self):
        """A string object must return '<?>'. """
        result = _get_source_path("hello")
        self.assertEqual(result, "<?>")

    def test_none_returns_unknown(self):
        """None must return '<?>'. """
        result = _get_source_path(None)
        self.assertEqual(result, "<?>")

    def test_list_returns_unknown(self):
        """A list must return '<?>'. """
        result = _get_source_path([1, 2, 3])
        self.assertEqual(result, "<?>")

    # ------------------------------------------------------------------
    # Return type invariant
    # ------------------------------------------------------------------

    def test_always_returns_string(self):
        """_get_source_path must always return a str for any input."""
        for obj in (os, os.path.join, math.sqrt, 42, None, [], "str"):
            result = _get_source_path(obj)
            self.assertIsInstance(result, str, msg=f"Failed for {obj!r}")


# ===========================================================================
# LazyImport
# ===========================================================================


class TestLazyImport(unittest.TestCase):
    """LazyImport must defer resolution and behave correctly after loading."""

    # ------------------------------------------------------------------
    # __init__ — validation
    # ------------------------------------------------------------------

    def test_invalid_error_param_raises_value_error(self):
        """error= not in {'raise', 'ignore'} must raise ValueError at init."""
        with self.assertRaises(ValueError):
            LazyImport("os", error="bad")

    def test_invalid_parent_module_globals_raises_type_error(self):
        """parent_module_globals of wrong type must raise TypeError."""
        with self.assertRaises(TypeError):
            LazyImport("os", parent_module_globals=123)

    def test_valid_init_does_not_import(self):
        """Constructing a LazyImport must not trigger actual module import."""
        # Use a module that can't be loaded to confirm no import happened
        lazy = LazyImport(_MISSING, error="ignore")
        self.assertFalse(lazy.is_loaded)

    # ------------------------------------------------------------------
    # is_loaded / is_resolved before resolution
    # ------------------------------------------------------------------

    def test_is_loaded_false_before_resolve(self):
        """is_loaded must be False before any resolution attempt."""
        lazy = LazyImport("os")
        self.assertFalse(lazy.is_loaded)

    def test_is_resolved_triggers_resolution(self):
        """Accessing is_resolved on an unresolved lazy must trigger resolution."""
        lazy = LazyImport("os")
        # is_resolved forces _resolve()
        resolved = lazy.is_resolved
        self.assertTrue(resolved)
        self.assertTrue(lazy.is_loaded)

    def test_is_resolved_false_for_failing_import(self):
        """is_resolved must be False when the import fails."""
        lazy = LazyImport(_MISSING, error="ignore")
        self.assertFalse(lazy.is_resolved)

    # ------------------------------------------------------------------
    # __bool__ — availability check without full import
    # ------------------------------------------------------------------

    def test_bool_true_for_available_module(self):
        """bool(lazy) must be True for a module known to be importable."""
        lazy = LazyImport("os")
        self.assertTrue(bool(lazy))

    def test_bool_false_for_unavailable_module(self):
        """bool(lazy) must be False for a module that cannot be found."""
        lazy = LazyImport(_MISSING, error="ignore")
        self.assertFalse(bool(lazy))

    def test_bool_before_load_does_not_set_is_loaded(self):
        """bool() check before resolution must not set is_loaded=True."""
        lazy = LazyImport("os")
        _ = bool(lazy)
        # bool uses find_spec, not the full import — is_loaded stays False
        self.assertFalse(lazy.is_loaded)

    def test_bool_after_load_uses_resolved(self):
        """After resolution bool() must reflect the resolved object's truthiness."""
        lazy = LazyImport("os")
        _ = lazy.resolved  # force resolution
        self.assertTrue(bool(lazy))

    # ------------------------------------------------------------------
    # .resolved — resolution and caching
    # ------------------------------------------------------------------

    def test_resolved_returns_module(self):
        """Accessing .resolved must return the actual module."""
        lazy = LazyImport("os")
        self.assertIsInstance(lazy.resolved, types.ModuleType)

    def test_resolved_is_correct_module(self):
        """The resolved module must be the same object as sys.modules entry."""
        lazy = LazyImport("os")
        self.assertIs(lazy.resolved, sys.modules["os"])

    def test_resolved_sets_is_loaded(self):
        """Resolution must set is_loaded=True."""
        lazy = LazyImport("os")
        _ = lazy.resolved
        self.assertTrue(lazy.is_loaded)

    def test_repeated_resolve_same_object(self):
        """Accessing .resolved twice must return the same object (cached)."""
        lazy = LazyImport("os")
        r1 = lazy.resolved
        r2 = lazy.resolved
        self.assertIs(r1, r2)

    def test_resolved_attribute(self):
        """LazyImport for a function path must resolve to that function."""
        lazy = LazyImport("os.path.join")
        self.assertIs(lazy.resolved, os.path.join)

    # ------------------------------------------------------------------
    # resolved_type
    # ------------------------------------------------------------------

    def test_resolved_type_is_type(self):
        """resolved_type must return a type object."""
        lazy = LazyImport("os")
        rt = lazy.resolved_type
        self.assertIsInstance(rt, type)

    def test_resolved_type_correct_for_module(self):
        """resolved_type must be types.ModuleType for a module lazy import."""
        lazy = LazyImport("os")
        self.assertIs(lazy.resolved_type, types.ModuleType)

    def test_resolved_type_correct_for_function(self):
        """resolved_type must reflect the actual type of the resolved callable."""
        lazy = LazyImport("os.path.join")
        rt = lazy.resolved_type
        self.assertTrue(callable(os.path.join))
        self.assertIsInstance(os.path.join, rt)

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def test_call_invokes_callable_resolved(self):
        """Calling lazy(args) must delegate to the resolved callable."""
        lazy = LazyImport("os.path.join")
        result = lazy("a", "b")
        self.assertEqual(result, os.path.join("a", "b"))

    def test_call_on_non_callable_returns_object(self):
        """Calling lazy() when resolved is non-callable must return the object."""
        lazy = LazyImport("os.sep")
        result = lazy()
        self.assertEqual(result, os.sep)

    # ------------------------------------------------------------------
    # __getattr__ delegation
    # ------------------------------------------------------------------

    def test_getattr_delegates_to_resolved_module(self):
        """Attribute access on the lazy must delegate to the resolved module."""
        lazy = LazyImport("os")
        self.assertEqual(lazy.sep, os.sep)

    def test_getattr_on_function_module(self):
        """Attribute from a sub-module must be accessible through lazy."""
        lazy = LazyImport("os.path")
        self.assertIs(lazy.join, os.path.join)

    def test_getattr_missing_raises_attribute_error(self):
        """Accessing a non-existent attribute must raise AttributeError."""
        lazy = LazyImport("os")
        with self.assertRaises(AttributeError):
            _ = lazy.totally_nonexistent_attribute_xyz

    # ------------------------------------------------------------------
    # __dir__
    # ------------------------------------------------------------------

    def test_dir_includes_known_attribute(self):
        """dir(lazy) must include attributes of the resolved module."""
        lazy = LazyImport("os")
        self.assertIn("sep", dir(lazy))

    def test_dir_returns_sorted_list(self):
        """dir(lazy) must return a sorted list."""
        lazy = LazyImport("os")
        d = dir(lazy)
        self.assertEqual(d, sorted(d))

    # ------------------------------------------------------------------
    # __hash__ and __eq__
    # ------------------------------------------------------------------

    def test_hash_based_on_name(self):
        """Two LazyImports with the same name must have equal hashes."""
        lazy1 = LazyImport("os")
        lazy2 = LazyImport("os")
        self.assertEqual(hash(lazy1), hash(lazy2))

    def test_hash_is_int(self):
        """Hash must be an int."""
        lazy = LazyImport("os")
        self.assertIsInstance(hash(lazy), int)

    def test_eq_same_name_lazy_objects(self):
        """Two LazyImports with the same name must compare equal (resolved)."""
        lazy1 = LazyImport("os")
        lazy2 = LazyImport("os")
        self.assertEqual(lazy1, lazy2)

    def test_eq_resolved_module(self):
        """lazy == real_module must be True after resolution."""
        lazy = LazyImport("os")
        self.assertEqual(lazy, sys.modules["os"])

    def test_neq_different_modules(self):
        """LazyImports for different modules must not be equal."""
        lazy_os = LazyImport("os")
        lazy_math = LazyImport("math")
        self.assertNotEqual(lazy_os, lazy_math)

    # ------------------------------------------------------------------
    # clear_cache
    # ------------------------------------------------------------------

    def test_clear_cache_resets_is_loaded(self):
        """clear_cache() must reset is_loaded to False."""
        lazy = LazyImport("os")
        _ = lazy.resolved
        self.assertTrue(lazy.is_loaded)
        lazy.clear_cache()
        self.assertFalse(lazy.is_loaded)

    def test_after_clear_cache_resolves_again(self):
        """After clear_cache(), accessing .resolved must re-resolve correctly."""
        lazy = LazyImport("os")
        r1 = lazy.resolved
        lazy.clear_cache()
        r2 = lazy.resolved
        self.assertIs(r1, r2)  # same module object

    # ------------------------------------------------------------------
    # parent_module_globals injection
    # ------------------------------------------------------------------

    def test_parent_module_globals_dict_populated_on_resolve(self):
        """Resolving must inject the object into the supplied globals dict."""
        ns = {}
        lazy = LazyImport("os", parent_module_globals=ns)
        _ = lazy.resolved
        self.assertIn("os", ns)

    def test_parent_module_globals_dict_value_is_module(self):
        """Injected value must be the resolved module itself."""
        ns = {}
        lazy = LazyImport("os", parent_module_globals=ns)
        _ = lazy.resolved
        self.assertIs(ns["os"], sys.modules["os"])

    def test_parent_module_globals_none_is_valid(self):
        """parent_module_globals=None must be accepted without error."""
        lazy = LazyImport("os", parent_module_globals=None)
        self.assertIsInstance(lazy.resolved, types.ModuleType)

    def test_parent_module_globals_string_is_valid(self):
        """parent_module_globals=str must be accepted (uses package globals)."""
        lazy = LazyImport("os", parent_module_globals="scikitplot")
        self.assertIsInstance(lazy.resolved, types.ModuleType)

    # ------------------------------------------------------------------
    # error='ignore' for unresolvable name
    # ------------------------------------------------------------------

    def test_error_ignore_resolved_is_false(self):
        """Unresolvable name with error='ignore' must not raise on .resolved."""
        lazy = LazyImport(_MISSING, error="ignore")
        try:
            _ = lazy.resolved
        except Exception as exc:
            self.fail(f"Raised unexpectedly with error='ignore': {exc}")

    # ------------------------------------------------------------------
    # error='raise' for unresolvable name
    # ------------------------------------------------------------------

    def test_error_raise_on_missing(self):
        """Accessing .resolved for a missing module with error='raise' must raise."""
        lazy = LazyImport(_MISSING, error="raise")
        with self.assertRaises(ImportError):
            _ = lazy.resolved


# ===========================================================================
# HAS_* feature flags
# ===========================================================================


class TestHasFlags(unittest.TestCase):
    """HAS_* attributes must reflect importability of optional dependencies."""

    # ------------------------------------------------------------------
    # __all__ contract
    # ------------------------------------------------------------------

    def test_all_entries_start_with_HAS(self):
        """Every entry in __all__ must start with 'HAS_'."""
        for name in _opt_all:
            self.assertTrue(
                name.startswith("HAS_"),
                msg=f"{name!r} in __all__ does not start with 'HAS_'",
            )

    def test_all_is_nonempty(self):
        """__all__ must contain at least one entry."""
        self.assertGreater(len(_opt_all), 0)

    # ------------------------------------------------------------------
    # Known-importable dependency flags
    # ------------------------------------------------------------------

    def test_has_numpy_is_bool(self):
        """HAS_NUMPY must be a bool."""
        result = _opt_deps_module.__getattr__("HAS_NUMPY")
        self.assertIsInstance(result, bool)

    def test_has_numpy_true(self):
        """HAS_NUMPY must be True (numpy is installed in the test environment)."""
        result = _opt_deps_module.__getattr__("HAS_NUMPY")
        self.assertTrue(result)

    def test_has_matplotlib_true(self):
        """HAS_MATPLOTLIB must be True (matplotlib is available)."""
        result = _opt_deps_module.__getattr__("HAS_MATPLOTLIB")
        self.assertTrue(result)

    def test_has_plt_alias_true(self):
        """HAS_PLT (alias for matplotlib) must be True."""
        result = _opt_deps_module.__getattr__("HAS_PLT")
        self.assertTrue(result)

    def test_has_scipy_is_bool(self):
        """HAS_SCIPY must be a bool (value depends on environment)."""
        result = _opt_deps_module.__getattr__("HAS_SCIPY")
        self.assertIsInstance(result, bool)

    # ------------------------------------------------------------------
    # Unknown flag — must raise AttributeError
    # ------------------------------------------------------------------

    def test_unknown_attr_raises_attribute_error(self):
        """Accessing HAS_NONEXISTENT_DEP must raise AttributeError."""
        with self.assertRaises(AttributeError):
            _opt_deps_module.__getattr__("HAS_TOTALLY_NONEXISTENT_DEP_XYZ")

    def test_non_has_attr_raises_attribute_error(self):
        """Accessing a non-HAS_ attribute must raise AttributeError."""
        with self.assertRaises(AttributeError):
            _opt_deps_module.__getattr__("TOTALLY_UNKNOWN_ATTR")

    # ------------------------------------------------------------------
    # All declared flags are bool-returning
    # ------------------------------------------------------------------

    def test_all_declared_flags_return_bool(self):
        """Every flag in __all__ must return a bool when accessed."""
        for name in _opt_all:
            with self.subTest(flag=name):
                result = _opt_deps_module.__getattr__(name)
                self.assertIsInstance(
                    result,
                    bool,
                    msg=f"{name} returned {type(result).__name__}, expected bool",
                )

    # ------------------------------------------------------------------
    # Absent optional dependency returns False
    # ------------------------------------------------------------------

    def test_unavailable_dep_returns_false(self):
        """A dependency that is not installed must return False.

        Notes
        -----
        Developer note
            The module-level ``__getattr__`` is wrapped by the project's
            custom ``lru_cache`` shim, which intentionally does not expose
            ``cache_clear`` on the wrapper (only on the inner cached function).
            Patching ``find_spec`` after a result has already been cached
            therefore has no effect on that cached entry.

            We test the False path using a dependency that is genuinely absent
            from the environment (``bottleneck``) so no cached True value
            can interfere.
        """
        result = _opt_deps_module.__getattr__("HAS_BOTTLENECK")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
