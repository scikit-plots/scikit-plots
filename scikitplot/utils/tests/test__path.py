# scikitplot/utils/tests/test__path.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit-tests for :mod:`scikitplot.utils._path`.

Coverage goals
--------------
normalize_directory_path
    expand_user / expand_vars / resolve flags, str vs Path input,
    already-absolute paths, non-existent targets.

sanitize_path_component
    space→underscore, invalid chars, collapse underscores, dot/dotdot,
    trailing dot/space, Windows reserved names, None / empty,
    max_len truncation, default fallback.

normalize_extension
    bare word, leading-dot, empty, None, whitespace.

PathNamer (dataclass)
    defaults, root= kwarg, directory= alias, directory wins over root,
    str root coerced to Path, by_day folder layout, subdir layout,
    add_secret / private tokens, mkdir=False, make_filename params,
    make_path params, frozen immutability guard,
    multiple calls produce unique names,
    day-boundary timestamp consistency.

make_path (convenience wrapper)
    zero-arg call, prefix/ext, by_day, private, subdir, root expansion.

make_temp_path
    creates file on disk, honours prefix / suffix / ext.

_ensure_aware_utc (private, tested via PathNamer.make_filename)
    naive datetime treated as UTC, aware datetime normalised.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .._path import (
    PathNamer,
    make_path,
    make_temp_path,
    normalize_directory_path,
    normalize_extension,
    sanitize_path_component,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_dir() -> str:
    """Return a fresh temporary directory path (caller cleans up)."""
    return tempfile.mkdtemp()


# ===========================================================================
# normalize_directory_path
# ===========================================================================

class TestNormalizeDirectoryPath(unittest.TestCase):

    def test_str_input_returns_path(self):
        result = normalize_directory_path("/tmp")
        self.assertIsInstance(result, Path)

    def test_path_input_returns_path(self):
        result = normalize_directory_path(Path("/tmp"))
        self.assertIsInstance(result, Path)

    def test_resolves_to_absolute(self):
        result = normalize_directory_path("relative/path", resolve=True)
        self.assertTrue(result.is_absolute())

    def test_no_resolve_keeps_relative_normalized(self):
        # With resolve=False the result should NOT call Path.resolve()
        result = normalize_directory_path("some/rel", resolve=False)
        self.assertIsInstance(result, Path)
        # Cannot assert is_absolute=False on all OSes, but parts should be consistent
        self.assertIn("rel", result.parts)

    def test_expand_user_replaces_tilde(self):
        result = normalize_directory_path("~/mydir", expand_user=True, resolve=False)
        self.assertNotIn("~", str(result))

    def test_no_expand_user_keeps_tilde_in_str_rep(self):
        # resolve=False so we don't hit filesystem, expand_user=False keeps literal ~
        result = normalize_directory_path("~/mydir", expand_user=False, resolve=False)
        self.assertIn("~", str(result))

    def test_expand_vars_replaces_env_var(self):
        os.environ["_SKPLT_TEST_VAR"] = "injected"
        try:
            result = normalize_directory_path(
                "$_SKPLT_TEST_VAR/sub", expand_user=False, resolve=False
            )
            self.assertIn("injected", str(result))
        finally:
            del os.environ["_SKPLT_TEST_VAR"]

    def test_non_existent_path_does_not_raise(self):
        # resolve(strict=False) must not raise even if the path does not exist.
        result = normalize_directory_path("/no/such/directory/at/all")
        self.assertIsInstance(result, Path)

    def test_dot_dot_collapsed(self):
        result = normalize_directory_path("/tmp/a/../b")
        self.assertNotIn("..", result.parts)


# ===========================================================================
# sanitize_path_component
# ===========================================================================

class TestSanitizePathComponent(unittest.TestCase):

    def test_spaces_replaced_with_underscore(self):
        self.assertEqual(sanitize_path_component("hello world"), "hello_world")

    def test_multiple_spaces_collapsed(self):
        self.assertEqual(sanitize_path_component("a   b"), "a_b")

    def test_invalid_chars_replaced(self):
        # Characters like : < > " are replaced.
        result = sanitize_path_component("report: v1")
        self.assertNotIn(":", result)

    def test_none_input_returns_default(self):
        self.assertEqual(sanitize_path_component(None, default="fallback"), "fallback")

    def test_empty_string_returns_default(self):
        self.assertEqual(sanitize_path_component("", default="fb"), "fb")

    def test_whitespace_only_returns_default(self):
        self.assertEqual(sanitize_path_component("   ", default="x"), "x")

    def test_dot_replaced_by_default(self):
        self.assertEqual(sanitize_path_component(".", default="safe"), "safe")

    def test_dotdot_replaced_by_default(self):
        self.assertEqual(sanitize_path_component("..", default="safe"), "safe")

    def test_trailing_dot_stripped(self):
        result = sanitize_path_component("file.")
        self.assertFalse(result.endswith("."))

    def test_trailing_space_stripped(self):
        result = sanitize_path_component("file ")
        self.assertFalse(result.endswith(" "))

    def test_windows_reserved_name_con(self):
        result = sanitize_path_component("CON")
        self.assertEqual(result, "CON_")

    def test_windows_reserved_name_nul(self):
        result = sanitize_path_component("NUL")
        self.assertEqual(result, "NUL_")

    def test_windows_reserved_name_com1(self):
        result = sanitize_path_component("COM1")
        self.assertEqual(result, "COM1_")

    def test_windows_reserved_name_case_insensitive(self):
        result = sanitize_path_component("con")
        self.assertEqual(result, "con_")

    def test_max_len_truncation(self):
        long_name = "a" * 200
        result = sanitize_path_component(long_name, max_len=80)
        self.assertEqual(len(result), 80)

    def test_max_len_short_name_not_padded(self):
        result = sanitize_path_component("short", max_len=80)
        self.assertLessEqual(len(result), 80)

    def test_normal_name_unchanged(self):
        self.assertEqual(sanitize_path_component("my_file"), "my_file")

    def test_none_default_empty_returns_empty(self):
        result = sanitize_path_component(None)
        self.assertEqual(result, "")

    def test_path_separator_replaced(self):
        # / and \ are invalid on various filesystems
        result = sanitize_path_component("a/b")
        self.assertNotIn("/", result)

    def test_collapse_multiple_underscores(self):
        result = sanitize_path_component("a___b")
        self.assertNotIn("__", result)


# ===========================================================================
# normalize_extension
# ===========================================================================

class TestNormalizeExtension(unittest.TestCase):

    def test_bare_word_gets_dot_prefix(self):
        self.assertEqual(normalize_extension("csv"), ".csv")

    def test_leading_dot_preserved(self):
        self.assertEqual(normalize_extension(".parquet"), ".parquet")

    def test_empty_string_returns_empty(self):
        self.assertEqual(normalize_extension(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(normalize_extension(None), "")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(normalize_extension("   "), "")

    def test_json_extension(self):
        self.assertEqual(normalize_extension("json"), ".json")

    def test_annoy_extension(self):
        self.assertEqual(normalize_extension(".annoy"), ".annoy")

    def test_double_dot_not_added(self):
        result = normalize_extension(".csv")
        self.assertFalse(result.startswith(".."))


# ===========================================================================
# PathNamer — construction and field defaults
# ===========================================================================

class TestPathNamerDefaults(unittest.TestCase):

    def test_default_root_is_path(self):
        namer = PathNamer()
        self.assertIsInstance(namer.root, Path)

    def test_default_root_value(self):
        namer = PathNamer()
        self.assertEqual(namer.root, Path("scikitplot-artifacts"))

    def test_str_root_coerced_to_path(self):
        namer = PathNamer(root="/tmp/myroot")
        self.assertIsInstance(namer.root, Path)

    def test_default_prefix_empty(self):
        self.assertEqual(PathNamer().prefix, "")

    def test_default_suffix_empty(self):
        self.assertEqual(PathNamer().suffix, "")

    def test_default_ext_empty(self):
        self.assertEqual(PathNamer().ext, "")

    def test_default_by_day_false(self):
        self.assertFalse(PathNamer().by_day)

    def test_default_add_secret_false(self):
        self.assertFalse(PathNamer().add_secret)

    def test_default_private_false(self):
        self.assertFalse(PathNamer().private)

    def test_default_mkdir_true(self):
        self.assertTrue(PathNamer().mkdir)

    def test_default_directory_none(self):
        self.assertIsNone(PathNamer().directory)


# ===========================================================================
# PathNamer — directory alias
# ===========================================================================

class TestPathNamerDirectoryAlias(unittest.TestCase):

    def test_directory_sets_root(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(directory=td)
            self.assertEqual(namer.root, Path(td).resolve())

    def test_directory_str_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(directory=td)
            self.assertIsInstance(namer.root, Path)

    def test_directory_path_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(directory=Path(td))
            self.assertIsInstance(namer.root, Path)

    def test_directory_wins_over_root(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root="/tmp/other", directory=td)
            self.assertEqual(namer.root, Path(td).resolve())

    def test_directory_with_prefix_and_ext(self):
        """Canonical test matching the failing tests in test__ann.py / test__privacy.py."""
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(prefix="test_", ext=".annoy", directory=td)
            p = namer.make_path()
            self.assertEqual(p.parent, Path(td).resolve())
            self.assertTrue(p.name.startswith("test_"))
            self.assertTrue(p.name.endswith(".annoy"))

    def test_directory_normalises_tilde(self):
        # Passing "~" must not raise; tilde is expanded.
        namer = PathNamer(directory="~", mkdir=False)
        self.assertNotIn("~", str(namer.root))

    def test_directory_none_leaves_root_unchanged(self):
        namer = PathNamer(root="/tmp", directory=None)
        self.assertEqual(namer.root, Path("/tmp"))


# ===========================================================================
# PathNamer — frozen immutability
# ===========================================================================

class TestPathNamerFrozen(unittest.TestCase):

    def test_cannot_reassign_root(self):
        namer = PathNamer()
        with self.assertRaises((TypeError, AttributeError)):
            namer.root = Path("/other")  # type: ignore[misc]

    def test_cannot_reassign_prefix(self):
        namer = PathNamer()
        with self.assertRaises((TypeError, AttributeError)):
            namer.prefix = "new"  # type: ignore[misc]


# ===========================================================================
# PathNamer.make_filename
# ===========================================================================

class TestPathNamerMakeFilename(unittest.TestCase):

    def _namer(self, **kw):
        return PathNamer(**kw)

    def test_returns_string(self):
        fname = self._namer().make_filename()
        self.assertIsInstance(fname, str)

    def test_prefix_in_filename(self):
        fname = self._namer(prefix="rep").make_filename()
        self.assertTrue(fname.startswith("rep-"))

    def test_extension_appended(self):
        fname = self._namer(ext="csv").make_filename()
        self.assertTrue(fname.endswith(".csv"))

    def test_extension_with_leading_dot(self):
        fname = self._namer(ext=".json").make_filename()
        self.assertTrue(fname.endswith(".json"))

    def test_no_prefix_no_leading_dash(self):
        fname = self._namer().make_filename()
        self.assertFalse(fname.startswith("-"))

    def test_suffix_included(self):
        fname = self._namer(suffix="run").make_filename()
        # suffix comes after the UUID segment
        self.assertIn("run", fname)

    def test_private_flag_adds_token(self):
        namer = self._namer(private=True)
        f1 = namer.make_filename()
        f2 = namer.make_filename()
        # Both names should contain a secret token (extra hex segment)
        # Private names are longer because of the extra token.
        self.assertGreater(len(f1), len(self._namer().make_filename()))

    def test_add_secret_flag_adds_token(self):
        namer = self._namer(add_secret=True)
        fname = namer.make_filename()
        self.assertGreater(len(fname), len(self._namer().make_filename()))

    def test_two_calls_produce_different_names(self):
        namer = self._namer()
        self.assertNotEqual(namer.make_filename(), namer.make_filename())

    def test_override_prefix_per_call(self):
        namer = self._namer(prefix="default")
        fname = namer.make_filename(prefix="override")
        self.assertTrue(fname.startswith("override-"))

    def test_override_ext_per_call(self):
        namer = self._namer(ext="csv")
        fname = namer.make_filename(ext="parquet")
        self.assertTrue(fname.endswith(".parquet"))

    def test_timestamp_in_filename(self):
        # Timestamp pattern: YYYYMMDDTHHMMSSsssZ
        fname = self._namer().make_filename()
        ts_pattern = re.compile(r"\d{8}T\d{6}\d{3}Z")
        self.assertRegex(fname, ts_pattern)

    def test_custom_datetime_used(self):
        fixed_dt = datetime(2026, 1, 15, 12, 0, 0, 500_000, tzinfo=timezone.utc)
        fname = self._namer().make_filename(now=fixed_dt)
        self.assertIn("20260115T120000500Z", fname)

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2026, 3, 1, 8, 30, 0)
        fname = self._namer().make_filename(now=naive)
        self.assertIn("20260301T083000000Z", fname)

    def test_aware_non_utc_normalised(self):
        # +02:00 → 06:00 UTC
        tz_plus2 = timezone(timedelta(hours=2))
        aware = datetime(2026, 6, 1, 8, 0, 0, tzinfo=tz_plus2)
        fname = self._namer().make_filename(now=aware)
        self.assertIn("20260601T060000000Z", fname)

    def test_concurrent_calls_all_unique(self):
        namer = self._namer()
        results = []
        lock = threading.Lock()

        def worker():
            fname = namer.make_filename()
            with lock:
                results.append(fname)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), len(set(results)), "Concurrent filenames not unique")


# ===========================================================================
# PathNamer.make_path
# ===========================================================================

class TestPathNamerMakePath(unittest.TestCase):

    def test_returns_path_object(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, mkdir=True)
            p = namer.make_path()
        self.assertIsInstance(p, Path)

    def test_directory_created_when_mkdir_true(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, mkdir=True)
            p = namer.make_path()
            self.assertTrue(p.parent.exists())

    def test_mkdir_false_does_not_create_missing_dir(self):
        non_existent = Path(tempfile.mkdtemp()) / "does_not_exist"
        namer = PathNamer(root=non_existent, mkdir=False)
        p = namer.make_path()
        # Parent should not be created
        self.assertFalse(non_existent.exists())

    def test_by_day_creates_date_subdirectory(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, by_day=True, mkdir=True)
            now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
            p = namer.make_path(now=now)
            # Path should contain 2026/07/04
            self.assertIn("2026", str(p))
            self.assertIn("07", str(p))
            self.assertIn("04", str(p))

    def test_subdir_used_when_by_day_false(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, by_day=False, mkdir=True)
            p = namer.make_path(subdir="models")
            self.assertIn("models", str(p))

    def test_subdir_ignored_when_by_day_true(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, by_day=True, mkdir=True)
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            p = namer.make_path(subdir="models", now=now)
            # by_day takes priority; subdir is not inserted
            self.assertNotIn("models", str(p))

    def test_two_paths_are_unique(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, mkdir=True)
            p1, p2 = namer.make_path(), namer.make_path()
            self.assertNotEqual(p1, p2)

    def test_prefix_reflected_in_filename(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, prefix="snap", mkdir=True)
            p = namer.make_path()
            self.assertTrue(p.name.startswith("snap-"))

    def test_ext_reflected_in_filename(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, ext="parquet", mkdir=True)
            p = namer.make_path()
            self.assertTrue(p.name.endswith(".parquet"))

    def test_directory_alias_used_as_root(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(directory=td, prefix="d", ext=".bin", mkdir=True)
            p = namer.make_path()
            self.assertEqual(p.parent, Path(td).resolve())

    def test_day_boundary_consistency(self):
        """Folder and filename share the same UTC timestamp (no day boundary mismatch)."""
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(root=td, by_day=True, mkdir=True)
            # Inject a time exactly at midnight UTC
            now = datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc)
            p = namer.make_path(now=now)
            # Date folder must be 2026/12/31
            self.assertIn("2026", str(p))
            self.assertIn("12", str(p))
            self.assertIn("31", str(p))
            # Filename timestamp starts with 20261231
            self.assertIn("20261231", p.name)


# ===========================================================================
# make_path convenience wrapper
# ===========================================================================

class TestMakePath(unittest.TestCase):

    def test_zero_args_returns_path(self):
        p = make_path()
        self.assertIsInstance(p, Path)
        # Clean up created directory
        try:
            import shutil
            shutil.rmtree(p.parent, ignore_errors=True)
        except Exception:
            pass

    def test_prefix_and_ext(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_path(root=td, prefix="report", ext="csv")
            self.assertTrue(p.name.startswith("report-"))
            self.assertTrue(p.name.endswith(".csv"))

    def test_private_adds_extra_token(self):
        with tempfile.TemporaryDirectory() as td:
            p_priv = make_path(root=td, private=True)
            p_plain = make_path(root=td)
            self.assertGreater(len(p_priv.name), len(p_plain.name))

    def test_subdir_in_path(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_path(root=td, subdir="runs")
            self.assertIn("runs", str(p))

    def test_by_day_nests_under_date(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_path(root=td, by_day=True, now=datetime(2026, 5, 3, tzinfo=timezone.utc))
            self.assertIn("2026", str(p))

    def test_root_tilde_expanded(self):
        home = Path("~").expanduser()
        p = make_path(root="~/skplt_test_tmp", prefix="x", mkdir=True)
        try:
            self.assertEqual(p.parent.parent, home)
        finally:
            import shutil
            shutil.rmtree(str(home / "skplt_test_tmp"), ignore_errors=True)

    def test_two_calls_unique(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertNotEqual(make_path(root=td), make_path(root=td))


# ===========================================================================
# make_temp_path
# ===========================================================================

class TestMakeTempPath(unittest.TestCase):

    def test_creates_file_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_temp_path(root=td)
            self.assertTrue(os.path.exists(p))
            os.remove(p)

    def test_returns_str(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_temp_path(root=td)
            self.assertIsInstance(p, str)
            try:
                os.remove(p)
            except OSError:
                pass

    def test_honours_ext(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_temp_path(root=td, ext=".annoy")
            self.assertTrue(p.endswith(".annoy"))
            os.remove(p)

    def test_honours_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            p = make_temp_path(root=td, prefix="idx")
            self.assertIn("idx", os.path.basename(p))
            os.remove(p)

    def test_two_calls_produce_distinct_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = make_temp_path(root=td)
            p2 = make_temp_path(root=td)
            self.assertNotEqual(p1, p2)
            for p in (p1, p2):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ===========================================================================
# PathNamer — edge cases and regressions
# ===========================================================================

class TestPathNamerEdgeCases(unittest.TestCase):

    def test_prefix_with_invalid_chars_sanitised(self):
        """Prefix containing filesystem-invalid chars must be sanitised, not crash."""
        with tempfile.TemporaryDirectory() as td:
            # Colon is invalid on Windows; sanitize_path_component should clean it.
            namer = PathNamer(root=td, prefix="my:run", mkdir=True)
            fname = namer.make_filename()
            self.assertNotIn(":", fname)

    def test_empty_prefix_no_leading_dash(self):
        namer = PathNamer(prefix="")
        fname = namer.make_filename()
        self.assertFalse(fname.startswith("-"))

    def test_no_extension_no_trailing_dot(self):
        namer = PathNamer(ext="")
        fname = namer.make_filename()
        self.assertFalse(fname.endswith("."))

    def test_100_unique_filenames(self):
        namer = PathNamer()
        names = {namer.make_filename() for _ in range(100)}
        self.assertEqual(len(names), 100)

    def test_pathlib_root_passed_to_directory(self):
        with tempfile.TemporaryDirectory() as td:
            namer = PathNamer(directory=Path(td))
            p = namer.make_path()
            self.assertTrue(str(p).startswith(str(Path(td).resolve())))


# ===========================================================================
# Integration: PathNamer → _store_index interaction pattern
# ===========================================================================

class TestPathNamerAsIndexStorePath(unittest.TestCase):
    """
    Mirror the exact usage pattern from test__privacy.py::test_pathnamer_input
    and test__ann.py::test_external_pathnamer to confirm the root fix
    integrates end-to-end.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.td = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_pathnamer_directory_kwarg_makes_valid_path(self):
        namer = PathNamer(prefix="tst_", ext=".annoy", directory=self.td)
        p = namer.make_path()
        # Parent directory must exist (mkdir=True by default)
        self.assertTrue(p.parent.exists())
        # File name starts with the expected prefix
        self.assertTrue(p.name.startswith("tst_"))
        self.assertTrue(p.name.endswith(".annoy"))
        # Root must be inside the supplied temp dir
        self.assertEqual(p.parent, Path(self.td).resolve())

    def test_pathnamer_directory_kwarg_caching_consistency(self):
        """Two make_path() calls on the same namer differ only in filename."""
        namer = PathNamer(prefix="p_", ext=".annoy", directory=self.td)
        p1 = namer.make_path()
        p2 = namer.make_path()
        self.assertNotEqual(p1, p2)
        self.assertEqual(p1.parent, p2.parent)  # same directory, different names


if __name__ == "__main__":
    unittest.main(verbosity=2)
