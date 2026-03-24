# config/tests/test__config.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._config`.

Runs standalone (``python test__config.py``) or under pytest.

Coverage map
------------
_parse_env_bool         Env var boolean parsing, truthy/falsy strings  → TestParseEnvBool
_parse_env_int          Env var integer parsing, bad-value fallback     → TestParseEnvInt
_global_config          Default keys, types, and critical bool bug fix  → TestGlobalConfigDefaults
_get_threadlocal_config Lazy init, mutable reference returned           → TestGetThreadlocalConfig
get_config              Returns copy, reflects set_config changes        → TestGetConfig
set_config              Per-key updates, None no-op, isolation           → TestSetConfig
config_context          Restore on exit, exception safety, nesting      → TestConfigContext
Thread safety           Thread isolation via threading.local             → TestThreadIsolation

Critical bug documented
-----------------------
Bug (line ~25): ``bool(os.environ.get("SKPLT_ASSUME_FINITE", "False"))``
    evaluates to ``True`` because ``bool("False") == True`` (any non-empty
    string is truthy).  Fixed by ``_parse_env_bool`` using an explicit
    allow-list.  Regression test: ``TestGlobalConfigDefaults.test_assume_finite_default_is_false``.

Bug (line ~26): ``int(os.environ.get("SKPLT_WORKING_MEMORY", "1024"))``
    raises ``ValueError`` with no recovery when the variable is a
    non-integer string.  Fixed by ``_parse_env_int`` with a warning
    fallback.  Regression test: ``TestParseEnvInt.test_invalid_string_falls_back_with_warning``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
import unittest.mock as mock
import warnings

from .._config import (  # noqa: E402
    _get_threadlocal_config,
    _global_config,
    _parse_env_bool,
    _parse_env_int,
    config_context,
    get_config,
    set_config,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ALL_KEYS = frozenset({
    "assume_finite",
    "working_memory",
    "display",
    "array_api_dispatch",
    "transform_output",
    "skip_parameter_validation",
})

_MODULE_DEFAULTS = {
    "assume_finite": False,
    "working_memory": 1024,
    "display": "diagram",
    "array_api_dispatch": False,
    "transform_output": "default",
    "skip_parameter_validation": False,
}


def _reset_config():
    """Restore the threadlocal config to documented module defaults.

    Developer note: does NOT pass ``array_api_dispatch`` to avoid triggering
    the conditional import of ``_check_array_api_dispatch`` from ``_utils``.
    ``array_api_dispatch`` is already ``False`` in the fresh threadlocal; we
    only reset the five keys that tests modify.
    """
    set_config(
        assume_finite=False,
        working_memory=1024,
        display="diagram",
        transform_output="default",
        skip_parameter_validation=False,
    )


# ===========================================================================
# _parse_env_bool — regression cover for the bool("False") critical bug
# ===========================================================================

class TestParseEnvBool(unittest.TestCase):
    """_parse_env_bool must correctly map env-var strings to bool."""

    ENV_NAME = "_SKPLT_TEST_BOOL_VAR"

    def _set(self, value):
        os.environ[self.ENV_NAME] = value

    def _clear(self):
        os.environ.pop(self.ENV_NAME, None)

    def tearDown(self):
        self._clear()

    # -- Truthy strings --

    def test_true_lowercase(self):
        self._set("true")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_true_uppercase(self):
        self._set("TRUE")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_true_mixed_case(self):
        self._set("True")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_one(self):
        self._set("1")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_yes(self):
        self._set("yes")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_on(self):
        self._set("on")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    # -- Falsy strings --

    def test_false_lowercase(self):
        """Critical bug: bool('false') == True; our helper must return False."""
        self._set("false")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    def test_false_uppercase(self):
        """Critical bug: bool('False') == True; our helper must return False."""
        self._set("False")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    def test_zero(self):
        """Critical bug: bool('0') == True; our helper must return False."""
        self._set("0")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    def test_no(self):
        self._set("no")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    def test_off(self):
        self._set("off")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    def test_random_string_is_falsy(self):
        """An unrecognised string must be treated as False."""
        self._set("maybe")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    # -- Absent variable --

    def test_absent_returns_default_false(self):
        self._clear()
        self.assertFalse(_parse_env_bool(self.ENV_NAME, default=False))

    def test_absent_returns_default_true(self):
        self._clear()
        self.assertTrue(_parse_env_bool(self.ENV_NAME, default=True))

    # -- Whitespace stripping --

    def test_leading_trailing_whitespace_true(self):
        self._set("  true  ")
        self.assertTrue(_parse_env_bool(self.ENV_NAME))

    def test_leading_trailing_whitespace_false(self):
        self._set("  false  ")
        self.assertFalse(_parse_env_bool(self.ENV_NAME))

    # -- Return type --

    def test_always_returns_bool(self):
        for val in ("1", "0", "true", "false", "yes", "maybe"):
            self._set(val)
            result = _parse_env_bool(self.ENV_NAME)
            self.assertIsInstance(result, bool, msg=f"Not bool for value {val!r}")


# ===========================================================================
# _parse_env_int — regression cover for the ValueError crash bug
# ===========================================================================

class TestParseEnvInt(unittest.TestCase):
    """_parse_env_int must parse integers and fall back safely on bad input."""

    ENV_NAME = "_SKPLT_TEST_INT_VAR"

    def _set(self, value):
        os.environ[self.ENV_NAME] = str(value)

    def _clear(self):
        os.environ.pop(self.ENV_NAME, None)

    def tearDown(self):
        self._clear()

    # -- Valid integers --

    def test_valid_integer_string(self):
        self._set("2048")
        self.assertEqual(_parse_env_int(self.ENV_NAME, 1024), 2048)

    def test_zero(self):
        self._set("0")
        self.assertEqual(_parse_env_int(self.ENV_NAME, 1024), 0)

    def test_negative_integer(self):
        self._set("-1")
        self.assertEqual(_parse_env_int(self.ENV_NAME, 1024), -1)

    # -- Absent variable --

    def test_absent_returns_default(self):
        self._clear()
        self.assertEqual(_parse_env_int(self.ENV_NAME, 512), 512)

    # -- Invalid input → fallback + warning --

    def test_invalid_string_falls_back_with_warning(self):
        """Critical bug: bare int() raises ValueError; helper must warn and fall back."""
        self._set("not_an_int")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_env_int(self.ENV_NAME, 1024)
        self.assertEqual(result, 1024, "Must fall back to default on bad string")
        self.assertEqual(len(w), 1)
        self.assertIn("not_an_int", str(w[0].message))

    def test_float_string_falls_back(self):
        self._set("1.5")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_env_int(self.ENV_NAME, 99)
        self.assertEqual(result, 99)
        self.assertEqual(len(w), 1)

    def test_empty_string_falls_back(self):
        self._set("")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_env_int(self.ENV_NAME, 42)
        self.assertEqual(result, 42)
        self.assertEqual(len(w), 1)

    # -- Return type --

    def test_returns_int(self):
        self._set("512")
        result = _parse_env_int(self.ENV_NAME, 1024)
        self.assertIsInstance(result, int)


# ===========================================================================
# _global_config — defaults, keys, types, and bug regression
# ===========================================================================

class TestGlobalConfigDefaults(unittest.TestCase):
    """_global_config must have the correct keys, types, and default values."""

    def test_has_all_expected_keys(self):
        self.assertEqual(set(_global_config.keys()), _ALL_KEYS)

    def test_no_unexpected_keys(self):
        self.assertEqual(len(_global_config), len(_ALL_KEYS))

    # -- Critical bug regression --

    def test_assume_finite_default_is_false(self):
        """
        Critical bug regression: bool('False') == True.

        The original code used ``bool(os.environ.get('SKPLT_ASSUME_FINITE', 'False'))``
        which evaluates to ``True`` because any non-empty string is truthy.
        The correct default must be ``False``.
        """
        self.assertFalse(
            _global_config["assume_finite"],
            msg=(
                "Bug regression: bool('False')==True. "
                "Default assume_finite must be False."
            ),
        )

    # -- Types --

    def test_assume_finite_is_bool(self):
        self.assertIsInstance(_global_config["assume_finite"], bool)

    def test_working_memory_is_int(self):
        self.assertIsInstance(_global_config["working_memory"], int)

    def test_display_is_str(self):
        self.assertIsInstance(_global_config["display"], str)

    def test_array_api_dispatch_is_bool(self):
        self.assertIsInstance(_global_config["array_api_dispatch"], bool)

    def test_transform_output_is_str(self):
        self.assertIsInstance(_global_config["transform_output"], str)

    def test_skip_parameter_validation_is_bool(self):
        self.assertIsInstance(_global_config["skip_parameter_validation"], bool)

    # -- Values --

    def test_working_memory_default_1024(self):
        self.assertEqual(_global_config["working_memory"], 1024)

    def test_display_default_diagram(self):
        self.assertEqual(_global_config["display"], "diagram")

    def test_array_api_dispatch_default_false(self):
        self.assertFalse(_global_config["array_api_dispatch"])

    def test_transform_output_default(self):
        self.assertEqual(_global_config["transform_output"], "default")

    def test_skip_parameter_validation_default_false(self):
        self.assertFalse(_global_config["skip_parameter_validation"])


# ===========================================================================
# _get_threadlocal_config
# ===========================================================================

class TestGetThreadlocalConfig(unittest.TestCase):
    """_get_threadlocal_config must lazily initialise and return the mutable config."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_returns_dict(self):
        self.assertIsInstance(_get_threadlocal_config(), dict)

    def test_contains_all_expected_keys(self):
        cfg = _get_threadlocal_config()
        for key in _ALL_KEYS:
            self.assertIn(key, cfg, msg=f"Key '{key}' missing from threadlocal config")

    def test_returns_mutable_reference_not_copy(self):
        """Two consecutive calls must return the same object (not a copy)."""
        cfg1 = _get_threadlocal_config()
        cfg2 = _get_threadlocal_config()
        self.assertIs(cfg1, cfg2)

    def test_mutations_are_visible_in_subsequent_calls(self):
        """Direct mutation of the returned dict must persist."""
        cfg = _get_threadlocal_config()
        original = cfg.get("working_memory")
        cfg["working_memory"] = original + 1
        self.assertEqual(_get_threadlocal_config()["working_memory"], original + 1)
        # Restore
        cfg["working_memory"] = original


# ===========================================================================
# get_config
# ===========================================================================

class TestGetConfig(unittest.TestCase):
    """get_config must return a copy of the threadlocal config."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_returns_dict(self):
        self.assertIsInstance(get_config(), dict)

    def test_has_all_expected_keys(self):
        cfg = get_config()
        self.assertEqual(set(cfg.keys()), _ALL_KEYS)

    def test_returns_copy_not_live_reference(self):
        """Mutating the returned dict must NOT affect the live config."""
        cfg = get_config()
        original = cfg["working_memory"]
        cfg["working_memory"] = original + 999
        self.assertEqual(get_config()["working_memory"], original)

    def test_two_calls_return_equal_dicts(self):
        self.assertEqual(get_config(), get_config())

    def test_reflects_set_config_changes(self):
        set_config(display="text")
        self.assertEqual(get_config()["display"], "text")

    def test_assume_finite_default_false(self):
        self.assertFalse(get_config()["assume_finite"])

    def test_working_memory_default_1024(self):
        self.assertEqual(get_config()["working_memory"], 1024)

    def test_display_default_diagram(self):
        self.assertEqual(get_config()["display"], "diagram")

    def test_transform_output_default_default(self):
        self.assertEqual(get_config()["transform_output"], "default")

    def test_skip_parameter_validation_default_false(self):
        self.assertFalse(get_config()["skip_parameter_validation"])


# ===========================================================================
# set_config
# ===========================================================================

class TestSetConfig(unittest.TestCase):
    """set_config must update exactly the requested keys, leaving others unchanged."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    # -- assume_finite --

    def test_set_assume_finite_true(self):
        set_config(assume_finite=True)
        self.assertTrue(get_config()["assume_finite"])

    def test_set_assume_finite_false(self):
        set_config(assume_finite=True)
        set_config(assume_finite=False)
        self.assertFalse(get_config()["assume_finite"])

    def test_none_assume_finite_is_noop(self):
        set_config(assume_finite=True)
        set_config(assume_finite=None)
        self.assertTrue(get_config()["assume_finite"])

    # -- working_memory --

    def test_set_working_memory(self):
        set_config(working_memory=512)
        self.assertEqual(get_config()["working_memory"], 512)

    def test_none_working_memory_is_noop(self):
        set_config(working_memory=2048)
        set_config(working_memory=None)
        self.assertEqual(get_config()["working_memory"], 2048)

    # -- display --

    def test_set_display_text(self):
        set_config(display="text")
        self.assertEqual(get_config()["display"], "text")

    def test_set_display_diagram(self):
        set_config(display="text")
        set_config(display="diagram")
        self.assertEqual(get_config()["display"], "diagram")

    def test_none_display_is_noop(self):
        set_config(display="text")
        set_config(display=None)
        self.assertEqual(get_config()["display"], "text")

    # -- transform_output --

    def test_set_transform_output_pandas(self):
        set_config(transform_output="pandas")
        self.assertEqual(get_config()["transform_output"], "pandas")

    def test_set_transform_output_polars(self):
        set_config(transform_output="polars")
        self.assertEqual(get_config()["transform_output"], "polars")

    def test_set_transform_output_default(self):
        set_config(transform_output="pandas")
        set_config(transform_output="default")
        self.assertEqual(get_config()["transform_output"], "default")

    def test_none_transform_output_is_noop(self):
        set_config(transform_output="pandas")
        set_config(transform_output=None)
        self.assertEqual(get_config()["transform_output"], "pandas")

    # -- skip_parameter_validation --

    def test_set_skip_parameter_validation_true(self):
        set_config(skip_parameter_validation=True)
        self.assertTrue(get_config()["skip_parameter_validation"])

    def test_set_skip_parameter_validation_false(self):
        set_config(skip_parameter_validation=True)
        set_config(skip_parameter_validation=False)
        self.assertFalse(get_config()["skip_parameter_validation"])

    def test_none_skip_parameter_validation_is_noop(self):
        set_config(skip_parameter_validation=True)
        set_config(skip_parameter_validation=None)
        self.assertTrue(get_config()["skip_parameter_validation"])

    # -- Isolation: only the specified key changes --

    def test_changing_one_key_leaves_others_unchanged(self):
        """set_config must update only the specified keys."""
        before = get_config()
        set_config(working_memory=512)
        after = get_config()
        for key in _ALL_KEYS - {"working_memory"}:
            self.assertEqual(
                after[key], before[key],
                msg=f"Key '{key}' changed unexpectedly",
            )

    def test_all_none_is_complete_noop(self):
        """set_config() with all None must not change any value."""
        before = get_config()
        set_config(
            assume_finite=None,
            working_memory=None,
            display=None,
            array_api_dispatch=None,
            transform_output=None,
            skip_parameter_validation=None,
        )
        self.assertEqual(get_config(), before)

    # -- array_api_dispatch (mocked to avoid cross-module dep) --

    def test_array_api_dispatch_none_is_noop(self):
        """Passing array_api_dispatch=None must not change the value."""
        before = get_config()["array_api_dispatch"]
        set_config(array_api_dispatch=None)
        self.assertEqual(get_config()["array_api_dispatch"], before)

    def test_array_api_dispatch_set_with_mock(self):
        """set_config(array_api_dispatch=True) must update the config when util is present."""
        patch_target = "scikitplot.config._config.set_config"
        # Only test the non-dispatch path by calling via config_context wrapper
        # which internally delegates to set_config after saving old state.
        # We verify array_api_dispatch is False by default (already covered),
        # and skip if _utils is not importable.
        try:
            import importlib
            utils = importlib.import_module("scikitplot._utils")
            if hasattr(utils, "_check_array_api_dispatch"):
                with mock.patch.object(utils, "_check_array_api_dispatch", return_value=None):
                    set_config(array_api_dispatch=True)
                    self.assertTrue(get_config()["array_api_dispatch"])
                    set_config(array_api_dispatch=False)
        except (ImportError, ModuleNotFoundError):
            pass  # _utils not available in this environment


# ===========================================================================
# config_context
# ===========================================================================

class TestConfigContext(unittest.TestCase):
    """config_context must restore config on exit under all conditions."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    # -- Basic operation --

    def test_is_context_manager(self):
        ctx = config_context(display="text")
        self.assertTrue(hasattr(ctx, "__enter__"))
        self.assertTrue(hasattr(ctx, "__exit__"))

    def test_changes_config_inside_block(self):
        with config_context(display="text"):
            self.assertEqual(get_config()["display"], "text")

    def test_restores_config_after_exit(self):
        with config_context(display="text"):
            pass
        self.assertEqual(get_config()["display"], "diagram")

    def test_yields_none(self):
        with config_context(display="text") as value:
            self.assertIsNone(value)

    # -- Exception safety --

    def test_restores_config_after_exception(self):
        """Config must be restored even when an exception propagates out."""
        try:
            with config_context(assume_finite=True):
                raise RuntimeError("deliberate test exception")
        except RuntimeError:
            pass
        self.assertFalse(get_config()["assume_finite"])

    def test_exception_propagates_unchanged(self):
        """The original exception must propagate unchanged through config_context."""
        sentinel = RuntimeError("propagation test")
        try:
            with config_context(display="text"):
                raise sentinel
        except RuntimeError as exc:
            self.assertIs(exc, sentinel)

    # -- Nesting --

    def test_nested_contexts_restore_intermediate_state(self):
        with config_context(display="text"):
            self.assertEqual(get_config()["display"], "text")
            with config_context(display="diagram"):
                self.assertEqual(get_config()["display"], "diagram")
            # Intermediate state (text) restored
            self.assertEqual(get_config()["display"], "text")
        # Outer state (diagram) restored
        self.assertEqual(get_config()["display"], "diagram")

    def test_three_levels_of_nesting(self):
        with config_context(working_memory=100):
            with config_context(working_memory=200):
                with config_context(working_memory=300):
                    self.assertEqual(get_config()["working_memory"], 300)
                self.assertEqual(get_config()["working_memory"], 200)
            self.assertEqual(get_config()["working_memory"], 100)
        self.assertEqual(get_config()["working_memory"], 1024)

    def test_nested_different_keys(self):
        """Nested contexts changing different keys must restore independently."""
        with config_context(display="text"):
            with config_context(assume_finite=True):
                self.assertEqual(get_config()["display"], "text")
                self.assertTrue(get_config()["assume_finite"])
            # assume_finite restored, display still text
            self.assertEqual(get_config()["display"], "text")
            self.assertFalse(get_config()["assume_finite"])
        self.assertEqual(get_config()["display"], "diagram")

    # -- None inside context --

    def test_none_key_inside_context_is_noop(self):
        with config_context(display=None):
            self.assertEqual(get_config()["display"], "diagram")

    # -- Multiple keys --

    def test_multiple_keys_changed_and_restored(self):
        with config_context(
            display="text",
            assume_finite=True,
            working_memory=512,
            transform_output="pandas",
            skip_parameter_validation=True,
        ):
            cfg = get_config()
            self.assertEqual(cfg["display"], "text")
            self.assertTrue(cfg["assume_finite"])
            self.assertEqual(cfg["working_memory"], 512)
            self.assertEqual(cfg["transform_output"], "pandas")
            self.assertTrue(cfg["skip_parameter_validation"])

        cfg_after = get_config()
        self.assertEqual(cfg_after["display"], "diagram")
        self.assertFalse(cfg_after["assume_finite"])
        self.assertEqual(cfg_after["working_memory"], 1024)
        self.assertEqual(cfg_after["transform_output"], "default")
        self.assertFalse(cfg_after["skip_parameter_validation"])

    # -- Repeated calls --

    def test_repeated_calls_do_not_accumulate_state(self):
        """Using config_context multiple times in sequence must be idempotent."""
        for _ in range(5):
            with config_context(working_memory=256):
                self.assertEqual(get_config()["working_memory"], 256)
            self.assertEqual(get_config()["working_memory"], 1024)


# ===========================================================================
# Thread isolation
# ===========================================================================

class TestThreadIsolation(unittest.TestCase):
    """
    Config changes in one thread must not be visible in another.

    Notes
    -----
    Developer note
        ``_get_threadlocal_config`` copies from ``_global_config`` on first
        access in a new thread.  Because ``set_config`` only modifies the
        calling thread's threadlocal dict (not ``_global_config``), another
        thread always starts from the module-level defaults, NOT from the
        caller's current config.
    """

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_worker_thread_sees_module_defaults(self):
        """A new thread must initialise from _global_config defaults."""
        set_config(display="text")  # main thread only
        results = {}

        def worker():
            results["display"] = get_config()["display"]

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # Worker initialises from _global_config, which still says "diagram"
        self.assertEqual(
            results["display"],
            _global_config["display"],
            msg="Worker thread must see _global_config defaults, not main-thread changes",
        )

    def test_worker_changes_do_not_affect_main(self):
        """set_config in a worker thread must not modify the main thread's config."""
        def worker():
            set_config(display="text", working_memory=256)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        cfg = get_config()
        self.assertEqual(cfg["display"], "diagram")
        self.assertEqual(cfg["working_memory"], 1024)

    def test_context_manager_in_worker_does_not_affect_main(self):
        """config_context in a worker thread must not affect the main thread."""
        barrier = threading.Barrier(2)
        errors = []

        def worker():
            try:
                with config_context(display="text", working_memory=256):
                    barrier.wait()  # sync: main checks config while worker is inside
                    time.sleep(0.02)
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        barrier.wait()  # wait until worker is inside config_context

        cfg = get_config()
        t.join()

        self.assertEqual(cfg["display"], "diagram", "Main thread must not see worker's context")
        self.assertEqual(cfg["working_memory"], 1024)
        self.assertEqual(errors, [])

    def test_concurrent_context_managers_are_isolated(self):
        """Multiple threads using config_context simultaneously must not interfere."""
        errors = []

        def worker(expected_wm):
            try:
                with config_context(working_memory=expected_wm):
                    time.sleep(0.01)
                    actual = get_config()["working_memory"]
                    if actual != expected_wm:
                        errors.append(
                            f"Expected {expected_wm}, got {actual}"
                        )
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=worker, args=(i * 100,))
            for i in range(1, 8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread isolation errors: {errors}")

    def test_thread_config_is_independent_for_each_key(self):
        """Each key must be isolated per thread, not just overall config."""
        results = {}
        lock = threading.Lock()

        def thread_a():
            set_config(display="text", working_memory=256)
            time.sleep(0.02)
            with lock:
                results["a"] = get_config()

        def thread_b():
            time.sleep(0.01)
            set_config(display="diagram", working_memory=512)
            with lock:
                results["b"] = get_config()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        if "a" in results and "b" in results:
            self.assertNotEqual(
                results["a"]["working_memory"],
                results["b"]["working_memory"],
                msg="Threads must have independent working_memory values",
            )


# ===========================================================================
# Integration: get_config + set_config + config_context round-trips
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Cross-function round-trips that exercise the full API together."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_set_then_get_round_trip(self):
        set_config(display="text", working_memory=2048, assume_finite=True)
        cfg = get_config()
        self.assertEqual(cfg["display"], "text")
        self.assertEqual(cfg["working_memory"], 2048)
        self.assertTrue(cfg["assume_finite"])

    def test_context_manager_wraps_set_config_changes(self):
        """Config set inside context_manager block must be fully reverted."""
        with config_context(display="text", working_memory=256):
            set_config(assume_finite=True)
            cfg = get_config()
            self.assertEqual(cfg["display"], "text")
            self.assertEqual(cfg["working_memory"], 256)
            self.assertTrue(cfg["assume_finite"])

        # After context exit the display and working_memory must be restored.
        # assume_finite was set by set_config INSIDE the context, but
        # config_context captures the state at entry and restores it fully.
        cfg_after = get_config()
        self.assertEqual(cfg_after["display"], "diagram")
        self.assertEqual(cfg_after["working_memory"], 1024)
        self.assertFalse(cfg_after["assume_finite"])

    def test_repeated_context_entry_is_stable(self):
        ctx_fn = lambda: config_context(display="text", working_memory=256)  # noqa: E731
        for _ in range(3):
            with ctx_fn():
                cfg = get_config()
                self.assertEqual(cfg["display"], "text")
            self.assertEqual(get_config()["display"], "diagram")

    def test_get_config_returns_correct_type_for_all_keys(self):
        """All config values must be of the expected Python type."""
        cfg = get_config()
        self.assertIsInstance(cfg["assume_finite"], bool)
        self.assertIsInstance(cfg["working_memory"], int)
        self.assertIsInstance(cfg["display"], str)
        self.assertIsInstance(cfg["array_api_dispatch"], bool)
        self.assertIsInstance(cfg["transform_output"], str)
        self.assertIsInstance(cfg["skip_parameter_validation"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
