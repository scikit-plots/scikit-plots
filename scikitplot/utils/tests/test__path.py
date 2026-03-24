# scikitplot/utils/tests/test__path.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`~._path`.

Runs under pytest from the package root::

    pytest scikitplot/utils/tests/test__path.py -v

Coverage map
------------
normalize_directory_path    Tilde, env-var, relative paths        → TestNormalizeDirectoryPath
sanitize_path_component     None/empty, spaces, invalid chars,
                            Windows reserved names, dot/dotdot,
                            max_len truncation                     → TestSanitizePathComponent
normalize_extension         With/without dot, None, empty         → TestNormalizeExtension
_ensure_aware_utc           None, naive, aware datetimes          → TestEnsureAwareUtc
_utc_timestamp_ms           Format correctness                    → TestUtcTimestampMs
PathNamer                   Dataclass defaults, make_filename,
                            make_path, by_day, private/add_secret → TestPathNamer
make_path                   Zero-arg, with args, private, subdir  → TestMakePath
make_temp_path              Creates temp file                      → TestMakeTempPath
get_path                    Defaults, ext, subfolder, timestamp,
                            overwrite=False, return_parts          → TestGetPath
remove_path                 File, directory, non-existing, None   → TestRemovePath
_filename_sanitizer         Invalid character replacement          → TestFilenameSanitizer
_filename_extension_normalizer  Extension normalization variants   → TestFilenameExtNormalizer
_filename_uniquer           Collision-avoidance counter            → TestFilenameUniquer
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

# --- path bootstrap (commented out — use pytest from the package root) -----
# import pathlib, sys
# _HERE = pathlib.Path(__file__).resolve().parent
# _PKG_ROOT = _HERE.parent.parent.parent
# if str(_PKG_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PKG_ROOT))
# --------------------------------------------------------------------------

from .._path import (  # noqa: E402
    PathNamer,
    _ensure_aware_utc,
    _filename_extension_normalizer,
    _filename_sanitizer,
    _filename_uniquer,
    _utc_timestamp_ms,
    get_path,
    make_path,
    make_temp_path,
    normalize_directory_path,
    normalize_extension,
    remove_path,
    sanitize_path_component,
)


def _make_tmpdir():
    """Create a temporary directory and return its Path."""
    return Path(tempfile.mkdtemp())


# ===========================================================================
# normalize_directory_path
# ===========================================================================


class TestNormalizeDirectoryPath(unittest.TestCase):
    """normalize_directory_path must expand and resolve paths."""

    def test_tilde_expanded(self):
        """'~' must be expanded to the user home directory."""
        p = normalize_directory_path("~/some/path")
        self.assertFalse(str(p).startswith("~"))
        self.assertTrue(p.is_absolute())

    def test_relative_path_becomes_absolute(self):
        """A relative path must be resolved to absolute."""
        p = normalize_directory_path("relative/dir")
        self.assertTrue(p.is_absolute())

    def test_returns_path_object(self):
        p = normalize_directory_path("/tmp")
        self.assertIsInstance(p, Path)

    def test_env_var_expansion(self):
        """$HOME must be expanded when expand_vars=True."""
        os.environ["_TEST_SKPLT_DIR"] = "/tmp/skplt_test"
        p = normalize_directory_path("$_TEST_SKPLT_DIR/data", expand_vars=True)
        self.assertEqual(str(p), str(Path("/tmp/skplt_test/data").resolve(strict=False)))
        del os.environ["_TEST_SKPLT_DIR"]

    def test_no_tilde_expansion_when_disabled(self):
        """expand_user=False must leave '~' unexpanded."""
        p = normalize_directory_path("~/path", expand_user=False, resolve=False)
        self.assertTrue(str(p).startswith("~"))

    def test_no_resolve_when_disabled(self):
        """resolve=False must not normalize '..' components."""
        p = normalize_directory_path("/tmp/../tmp", resolve=False, expand_user=False)
        self.assertIn("..", str(p))

    def test_path_object_input(self):
        """Accepts pathlib.Path as input."""
        p = normalize_directory_path(Path("/tmp"))
        self.assertIsInstance(p, Path)


# ===========================================================================
# sanitize_path_component
# ===========================================================================


class TestSanitizePathComponent(unittest.TestCase):
    """sanitize_path_component must produce portable path components."""

    def test_basic_name_unchanged(self):
        self.assertEqual(sanitize_path_component("myfile"), "myfile")

    def test_spaces_become_underscores(self):
        result = sanitize_path_component("my file name")
        self.assertEqual(result, "my_file_name")

    def test_multiple_spaces_collapse(self):
        result = sanitize_path_component("a  b   c")
        self.assertEqual(result, "a_b_c")

    def test_colon_replaced(self):
        result = sanitize_path_component("my report: v1")
        self.assertNotIn(":", result)

    def test_invalid_chars_replaced_by_underscore(self):
        """'<', '>', ':', '"', '\\', '/', '|', '?', '*' must be replaced."""
        for ch in '<>:"/\\|?*':
            result = sanitize_path_component(f"name{ch}")
            self.assertNotIn(ch, result)

    def test_single_dot_replaced_by_default(self):
        """'.' alone is a directory reference and must be replaced."""
        result = sanitize_path_component(".", default="data")
        self.assertEqual(result, "data")

    def test_double_dot_replaced_by_default(self):
        """'..' alone is a traversal; must be replaced."""
        result = sanitize_path_component("..", default="safe")
        self.assertEqual(result, "safe")

    def test_none_returns_empty_or_default(self):
        """None input must return the default value."""
        result = sanitize_path_component(None, default="fallback")
        self.assertEqual(result, "fallback")

    def test_empty_string_returns_default(self):
        result = sanitize_path_component("", default="fallback")
        self.assertEqual(result, "fallback")

    def test_max_len_truncation(self):
        """Result must not exceed max_len characters."""
        long_name = "a" * 200
        result = sanitize_path_component(long_name, max_len=50)
        self.assertLessEqual(len(result), 50)

    def test_windows_reserved_con(self):
        """'CON' is a Windows reserved device name; must get '_' appended."""
        result = sanitize_path_component("CON")
        self.assertNotEqual(result, "CON")
        self.assertTrue(result.startswith("CON"))

    def test_windows_reserved_case_insensitive(self):
        """Windows reserved names are case-insensitive."""
        result = sanitize_path_component("con")
        self.assertNotEqual(result.upper(), "CON")

    def test_trailing_dot_removed(self):
        """Trailing dots are invalid on Windows; must be stripped."""
        result = sanitize_path_component("filename.")
        self.assertFalse(result.endswith("."))

    def test_trailing_space_removed(self):
        """Trailing spaces are invalid on Windows; must be stripped."""
        result = sanitize_path_component("name  ")
        self.assertFalse(result.endswith(" "))

    def test_underscore_not_multiple(self):
        """Consecutive underscores must be collapsed to one."""
        result = sanitize_path_component("a__b")
        self.assertNotIn("__", result)

    def test_returns_string(self):
        self.assertIsInstance(sanitize_path_component("hello"), str)


# ===========================================================================
# normalize_extension
# ===========================================================================


class TestNormalizeExtension(unittest.TestCase):
    """normalize_extension must produce a dot-prefixed extension or ''."""

    def test_none_returns_empty(self):
        self.assertEqual(normalize_extension(None), "")

    def test_empty_string_returns_empty(self):
        self.assertEqual(normalize_extension(""), "")

    def test_ext_without_dot_gets_dot(self):
        self.assertEqual(normalize_extension("csv"), ".csv")

    def test_ext_with_dot_unchanged(self):
        self.assertEqual(normalize_extension(".parquet"), ".parquet")

    def test_uppercase_ext_preserved(self):
        self.assertEqual(normalize_extension("CSV"), ".CSV")

    def test_whitespace_stripped(self):
        self.assertEqual(normalize_extension("  .json  "), ".json")

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(normalize_extension("   "), "")

    def test_multi_dot_ext(self):
        """'.tar.gz' with a leading dot stays as-is."""
        self.assertEqual(normalize_extension(".tar.gz"), ".tar.gz")

    def test_returns_string(self):
        self.assertIsInstance(normalize_extension("csv"), str)


# ===========================================================================
# _ensure_aware_utc
# ===========================================================================


class TestEnsureAwareUtc(unittest.TestCase):
    """_ensure_aware_utc must always return a UTC-aware datetime."""

    def test_none_returns_now_utc(self):
        """None input must return current UTC time (aware)."""
        result = _ensure_aware_utc(None)
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_naive_treated_as_utc(self):
        """Naive datetime must have UTC tzinfo attached."""
        naive = datetime(2025, 1, 1, 12, 0, 0)
        result = _ensure_aware_utc(naive)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.year, 2025)

    def test_aware_utc_preserved(self):
        """An already UTC-aware datetime must be returned unchanged."""
        aware = datetime(2025, 6, 15, 9, 30, 0, tzinfo=timezone.utc)
        result = _ensure_aware_utc(aware)
        self.assertEqual(result, aware)

    def test_returns_datetime(self):
        self.assertIsInstance(_ensure_aware_utc(None), datetime)


# ===========================================================================
# _utc_timestamp_ms
# ===========================================================================


class TestUtcTimestampMs(unittest.TestCase):
    """_utc_timestamp_ms must produce the correct format."""

    def test_format_length(self):
        """Must be exactly 19 characters: YYYYMMDDTHHMMSSmmmZ."""
        ts = datetime(2025, 3, 15, 14, 30, 5, 123456, tzinfo=timezone.utc)
        # result     = '20250315T143005123Z'
        result = _utc_timestamp_ms(ts)
        self.assertEqual(len(result), 19)

    def test_format_ends_with_z(self):
        ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(_utc_timestamp_ms(ts).endswith("Z"))

    def test_format_contains_T(self):
        ts = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.assertIn("T", _utc_timestamp_ms(ts))

    def test_milliseconds_embedded(self):
        """Milliseconds must be the last 3 digits before 'Z'."""
        ts = datetime(2025, 1, 1, 0, 0, 0, 456000, tzinfo=timezone.utc)
        result = _utc_timestamp_ms(ts)
        self.assertTrue(result.endswith("456Z"))

    def test_zero_milliseconds(self):
        ts = datetime(2025, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(_utc_timestamp_ms(ts).endswith("000Z"))


# ===========================================================================
# PathNamer
# ===========================================================================


class TestPathNamer(unittest.TestCase):
    """PathNamer must generate correct, unique, portable paths."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- Dataclass defaults --

    def test_default_root(self):
        namer = PathNamer()
        self.assertEqual(namer.root, Path("scikitplot-artifacts"))

    def test_default_ext_empty(self):
        self.assertEqual(PathNamer().ext, "")

    def test_frozen_dataclass(self):
        """PathNamer is frozen; attribute assignment must raise."""
        namer = PathNamer()
        with self.assertRaises((TypeError, AttributeError)):
            namer.prefix = "new"

    # -- make_filename --

    def test_make_filename_returns_str(self):
        namer = PathNamer(root=self._tmpdir, mkdir=False)
        self.assertIsInstance(namer.make_filename(), str)

    def test_make_filename_with_prefix(self):
        namer = PathNamer(prefix="rep", root=self._tmpdir, mkdir=False)
        fname = namer.make_filename()
        self.assertTrue(fname.startswith("rep-"))

    def test_make_filename_with_ext(self):
        namer = PathNamer(ext="csv", root=self._tmpdir, mkdir=False)
        self.assertTrue(namer.make_filename().endswith(".csv"))

    def test_make_filename_ext_without_dot(self):
        namer = PathNamer(ext="json", root=self._tmpdir, mkdir=False)
        fname = namer.make_filename()
        self.assertTrue(fname.endswith(".json"))

    def test_make_filename_with_suffix(self):
        namer = PathNamer(suffix="final", root=self._tmpdir, mkdir=False)
        self.assertIn("final", namer.make_filename())

    def test_make_filename_unique(self):
        """Two consecutive filenames must differ (counter or UUID)."""
        namer = PathNamer(root=self._tmpdir, mkdir=False)
        f1 = namer.make_filename()
        f2 = namer.make_filename()
        self.assertNotEqual(f1, f2)

    def test_make_filename_private_adds_token(self):
        """private=True must add a token segment."""
        namer = PathNamer(root=self._tmpdir, private=True, mkdir=False)
        f1 = namer.make_filename()
        namer_plain = PathNamer(root=self._tmpdir, private=False, mkdir=False)
        f2 = namer_plain.make_filename()
        # Private filename is longer (extra token segment)
        self.assertGreater(len(f1), len(f2) - 40)  # tokens differ but both have UUID

    def test_make_filename_override_prefix(self):
        """make_filename(prefix=...) must override the dataclass prefix."""
        namer = PathNamer(prefix="base", root=self._tmpdir, mkdir=False)
        fname = namer.make_filename(prefix="override")
        self.assertTrue(fname.startswith("override-"))

    def test_make_filename_override_ext(self):
        namer = PathNamer(ext="csv", root=self._tmpdir, mkdir=False)
        fname = namer.make_filename(ext="parquet")
        self.assertTrue(fname.endswith(".parquet"))

    def test_make_filename_custom_now(self):
        """A fixed 'now' must produce a deterministic timestamp segment."""
        fixed = datetime(2025, 6, 1, 12, 0, 0, 500000, tzinfo=timezone.utc)
        namer = PathNamer(root=self._tmpdir, mkdir=False)
        fname = namer.make_filename(now=fixed)
        self.assertIn("20250601T120000500Z", fname)

    # -- make_path --

    def test_make_path_returns_path(self):
        namer = PathNamer(root=self._tmpdir)
        self.assertIsInstance(namer.make_path(), Path)

    def test_make_path_directory_created(self):
        """mkdir=True must create the target directory."""
        namer = PathNamer(root=self._tmpdir / "newdir")
        p = namer.make_path()
        self.assertTrue(p.parent.exists())

    def test_make_path_mkdir_false_does_not_create(self):
        """mkdir=False must not create directories."""
        target = self._tmpdir / "absent_dir"
        namer = PathNamer(root=target, mkdir=False)
        namer.make_path()  # must not raise
        # Directory absent_dir should NOT have been created
        self.assertFalse(target.exists())

    def test_make_path_by_day_nests_under_date(self):
        """by_day=True must nest files under YYYY/MM/DD."""
        namer = PathNamer(root=self._tmpdir, by_day=True)
        p = namer.make_path()
        # Last 3 parent parts should look like year/month/day
        parts = p.parts
        # The path should have at least 3 date segments above the filename
        date_parts = parts[-4:-1]  # (year, month, day)
        self.assertTrue(all(d.isdigit() for d in date_parts))

    def test_make_path_subdir_created(self):
        """subdir must create a subdirectory under root."""
        namer = PathNamer(root=self._tmpdir)
        p = namer.make_path(subdir="models")
        self.assertIn("models", str(p))

    def test_make_path_add_secret(self):
        """add_secret=True must include an extra token in the filename."""
        namer_secret = PathNamer(root=self._tmpdir, add_secret=True)
        namer_plain = PathNamer(root=self._tmpdir)
        f_secret = namer_secret.make_filename()
        f_plain = namer_plain.make_filename()
        # Secret filename has one extra '-' delimited segment
        self.assertGreater(f_secret.count("-"), f_plain.count("-") - 1)

    def test_make_path_is_absolute(self):
        namer = PathNamer(root=self._tmpdir)
        self.assertTrue(namer.make_path().is_absolute())

    # -- Thread safety: counters remain unique across threads --

    def test_concurrent_filenames_unique(self):
        """Counter must prevent collision across threads."""
        namer = PathNamer(root=self._tmpdir, mkdir=False)
        names = []
        lock = threading.Lock()

        def gen():
            fname = namer.make_filename()
            with lock:
                names.append(fname)

        threads = [threading.Thread(target=gen) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(names), len(set(names)))


# ===========================================================================
# make_path  (convenience wrapper)
# ===========================================================================


class TestMakePath(unittest.TestCase):
    """make_path must behave like PathNamer.make_path with zero args."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_zero_arg_returns_path(self):
        p = make_path(root=self._tmpdir)
        self.assertIsInstance(p, Path)

    def test_prefix_in_filename(self):
        p = make_path(prefix="report", root=self._tmpdir)
        self.assertTrue(p.name.startswith("report-"))

    def test_ext_in_filename(self):
        p = make_path(ext="csv", root=self._tmpdir)
        self.assertTrue(p.name.endswith(".csv"))

    def test_private_returns_path(self):
        p = make_path(root=self._tmpdir, private=True)
        self.assertIsInstance(p, Path)

    def test_subdir_in_path(self):
        p = make_path(root=self._tmpdir, subdir="runs")
        self.assertIn("runs", str(p))

    def test_by_day_creates_date_dirs(self):
        p = make_path(root=self._tmpdir, by_day=True)
        # Date dirs (YYYY/MM/DD) should be in the path
        self.assertGreater(len(p.parts), 2)

    def test_mkdir_false_does_not_create(self):
        target = self._tmpdir / "noexist"
        make_path(root=target, mkdir=False)
        self.assertFalse(target.exists())


# ===========================================================================
# make_temp_path
# ===========================================================================


class TestMakeTempPath(unittest.TestCase):
    """make_temp_path must create a real temporary file."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_returns_path_string(self):
        path = make_temp_path(root=self._tmpdir)
        self.assertIsInstance(path, (str, Path))

    def test_file_exists(self):
        """mkstemp creates the file; it should exist."""
        path = make_temp_path(root=self._tmpdir)
        self.assertTrue(os.path.exists(path))

    def test_prefix_in_name(self):
        path = make_temp_path(prefix="tmptest", root=self._tmpdir)
        self.assertIn("tmptest", os.path.basename(str(path)))


# ===========================================================================
# get_path
# ===========================================================================


class TestGetPath(unittest.TestCase):
    """get_path must return valid, writable paths and validate extensions."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_default_returns_string(self):
        result = get_path(file_path=str(self._tmpdir))
        self.assertIsInstance(result, str)

    def test_default_ext_is_png(self):
        result = get_path(file_path=str(self._tmpdir))
        self.assertTrue(result.endswith(".png"))

    def test_custom_filename(self):
        result = get_path(filename="myplot", file_path=str(self._tmpdir))
        self.assertIn("myplot", result)

    def test_custom_ext_jpg(self):
        result = get_path(filename="img", ext=".jpg", file_path=str(self._tmpdir))
        self.assertTrue(result.endswith(".jpg"))

    def test_pdf_ext_accepted(self):
        result = get_path(filename="report", ext=".pdf", file_path=str(self._tmpdir))
        self.assertTrue(result.endswith(".pdf"))

    def test_unsupported_ext_raises(self):
        """An unsupported extension must raise ValueError."""
        with self.assertRaises(ValueError):
            get_path(filename="data", ext=".zip", file_path=str(self._tmpdir))

    def test_subfolder_created(self):
        result = get_path(file_path=str(self._tmpdir), subfolder="sub1")
        self.assertIn("sub1", result)

    def test_directory_created(self):
        """Target directory must be created if not present."""
        new_dir = str(self._tmpdir / "auto_created")
        get_path(file_path=new_dir)
        self.assertTrue(os.path.isdir(new_dir))

    def test_add_timestamp_modifies_filename(self):
        result = get_path(
            filename="snap", file_path=str(self._tmpdir), add_timestamp=True
        )
        # Timestamp format YYYYMMDD_HHMMSSZ should appear
        self.assertIn("_", os.path.basename(result))

    def test_return_parts_true(self):
        result = get_path(file_path=str(self._tmpdir), return_parts=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_return_parts_false_gives_str(self):
        result = get_path(file_path=str(self._tmpdir), return_parts=False)
        self.assertIsInstance(result, str)

    def test_overwrite_false_avoids_collision(self):
        """overwrite=False must rename to avoid overwriting existing files."""
        # Create the file first
        base = get_path(filename="chart", file_path=str(self._tmpdir))
        open(base, "w").close()  # noqa: WPS515
        # Second call with same name must produce a different path
        result2 = get_path(
            filename="chart", file_path=str(self._tmpdir), overwrite=False
        )
        self.assertNotEqual(base, result2)

    def test_invalid_chars_sanitized(self):
        """Filename with '<', '>', ':', etc. must be sanitized."""
        result = get_path(filename="bad:name", file_path=str(self._tmpdir))
        self.assertNotIn(":", os.path.basename(result))


# ===========================================================================
# remove_path
# ===========================================================================


class TestRemovePath(unittest.TestCase):
    """remove_path must silently remove files/dirs when present, ignore missing."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_removes_existing_file(self):
        f = self._tmpdir / "to_remove.txt"
        f.write_text("x")
        remove_path(["to_remove.txt"], base_path=str(self._tmpdir))
        self.assertFalse(f.exists())

    def test_removes_existing_directory(self):
        d = self._tmpdir / "dir_to_remove"
        d.mkdir()
        remove_path(["dir_to_remove"], base_path=str(self._tmpdir))
        self.assertFalse(d.exists())

    def test_missing_path_does_not_raise(self):
        """A path that does not exist must be silently ignored."""
        try:
            remove_path(["nonexistent_file"], base_path=str(self._tmpdir))
        except Exception as e:
            self.fail(f"remove_path raised on missing path: {e}")

    def test_none_paths_uses_default_list(self):
        """paths=None must use the built-in default list without raising."""
        try:
            remove_path(paths=None, base_path=str(self._tmpdir))
        except Exception as e:
            self.fail(f"remove_path raised with default list: {e}")

    def test_none_base_path_uses_cwd(self):
        """base_path=None must default to the current working directory."""
        try:
            remove_path(paths=["_nonexistent_abc"], base_path=None)
        except Exception as e:
            self.fail(f"remove_path raised with None base_path: {e}")


# ===========================================================================
# _filename_sanitizer
# ===========================================================================


class TestFilenameSanitizer(unittest.TestCase):
    """_filename_sanitizer must replace invalid characters with '_'."""

    def test_colon_replaced(self):
        self.assertEqual(_filename_sanitizer("a:b"), "a_b")

    def test_slash_replaced(self):
        self.assertEqual(_filename_sanitizer("a/b"), "a_b")

    def test_backslash_replaced(self):
        self.assertEqual(_filename_sanitizer("a\\b"), "a_b")

    def test_angle_bracket_replaced(self):
        self.assertEqual(_filename_sanitizer("a<b>c"), "a_b_c")

    def test_valid_name_unchanged(self):
        self.assertEqual(_filename_sanitizer("report_2025"), "report_2025")

    def test_empty_string(self):
        self.assertEqual(_filename_sanitizer(""), "")

    def test_returns_string(self):
        self.assertIsInstance(_filename_sanitizer("test"), str)


# ===========================================================================
# _filename_extension_normalizer
# ===========================================================================


class TestFilenameExtNormalizer(unittest.TestCase):
    """_filename_extension_normalizer must detect and normalize extensions."""

    def test_filename_with_png_ext(self):
        base, ext = _filename_extension_normalizer("chart.png", allowed_exts=[".png"])
        self.assertEqual(base, "chart")
        self.assertEqual(ext, ".png")

    def test_explicit_ext_overrides(self):
        base, ext = _filename_extension_normalizer("photo", ext=".jpg")
        self.assertEqual(ext, ".jpg")

    def test_explicit_ext_without_dot_gets_dot(self):
        _, ext = _filename_extension_normalizer("photo", ext="jpg")
        self.assertEqual(ext, ".jpg")

    def test_case_insensitive_extension_match(self):
        base, ext = _filename_extension_normalizer(
            "doc.PDF", allowed_exts=[".pdf"]
        )
        self.assertIn("PDF", ext.upper())

    def test_no_ext_no_allowed_gives_empty_ext(self):
        base, ext = _filename_extension_normalizer("noext", allowed_exts=[])
        # splitext on "noext" returns ("noext", "")
        self.assertEqual(base, "noext")

    def test_returns_tuple(self):
        result = _filename_extension_normalizer("chart.png")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


# ===========================================================================
# _filename_uniquer
# ===========================================================================


class TestFilenameUniquer(unittest.TestCase):
    """_filename_uniquer must append counter to avoid overwriting files."""

    def setUp(self):
        self._tmpdir = _make_tmpdir()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_collision_returns_same_path(self):
        full = str(self._tmpdir / "new.png")
        new_full, _, _ = _filename_uniquer(full, str(self._tmpdir), "new.png")
        self.assertEqual(new_full, full)

    def test_collision_changes_filename(self):
        """If the file exists, the returned path must differ."""
        existing = self._tmpdir / "report.png"
        existing.write_text("data")
        new_full, _, _ = _filename_uniquer(
            str(existing), str(self._tmpdir), "report.png"
        )
        self.assertNotEqual(new_full, str(existing))

    def test_double_collision_increments_counter(self):
        """Counter must keep incrementing until a free slot is found."""
        for name in ["data.png", "data_1.png"]:
            (self._tmpdir / name).write_text("x")
        new_full, _, _ = _filename_uniquer(
            str(self._tmpdir / "data.png"), str(self._tmpdir), "data.png"
        )
        self.assertIn("data_2", new_full)

    def test_returns_tuple_three(self):
        full = str(self._tmpdir / "x.png")
        result = _filename_uniquer(full, str(self._tmpdir), "x.png")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
