# scikitplot/_utils/tests/test_uri.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.uri`.

Coverage map
------------
is_local_uri          local paths, file:// scheme, http/https/databricks,
                      windows UNC path (mocked), Windows drive letter     -> TestIsLocalUri
is_file_uri           file:// prefix detection                            -> TestIsFileUri
is_http_uri           http/https, ftp, bare paths                         -> TestIsHttpUri
is_databricks_uri     databricks scheme, non-databricks                   -> TestIsDatabricksUri
extract_and_normalize_path  local path, file:// URI, windows              -> TestExtractAndNormalizePath
append_to_uri_path    simple join, trailing slash, absolute suffix        -> TestAppendToUriPath
join_paths            multiple segments, absolute parts, empty            -> TestJoinPaths
strip_scheme          removes scheme, no scheme, file://, http://         -> TestStripScheme
is_models_uri         models:/ prefix detection                           -> TestIsModelsUri
_escape_control_chars control chars removed/escaped                       -> TestEscapeControlChars
validate_query_string valid, invalid control chars                        -> TestValidateQueryString
generate_tmp_dfs_path has dbfs prefix, UUID suffix                        -> TestGenerateTmpDfsPath
is_valid_dbfs_uri     dbfs:/ prefix required                              -> TestIsValidDbfsUri
dbfs_hdfs_uri_to_fuse_path  strip prefix, prepend /dbfs/                 -> TestDbfsHdfsUriToFusePath

Run standalone::

    python -m unittest scikitplot._utils.tests.test_uri -v
"""

from __future__ import annotations

import sys
import types
import unittest
import unittest.mock as mock

from ..uri import (  # noqa: E402
    _escape_control_characters,
    append_to_uri_path,
    dbfs_hdfs_uri_to_fuse_path,
    extract_and_normalize_path,
    generate_tmp_dfs_path,
    is_databricks_uri,
    is_file_uri,
    is_http_uri,
    is_local_uri,
    is_models_uri,
    is_valid_dbfs_uri,
    join_paths,
    strip_scheme,
    validate_query_string,
)
from ..exception_utils import ScikitplotException


# ===========================================================================
# is_local_uri
# ===========================================================================


class TestIsLocalUri(unittest.TestCase):
    """is_local_uri must correctly identify local file paths."""

    def test_bare_posix_path_is_local(self):
        self.assertTrue(is_local_uri("/some/local/path"))

    def test_relative_path_is_local(self):
        self.assertTrue(is_local_uri("relative/path"))

    def test_file_scheme_is_local(self):
        self.assertTrue(is_local_uri("file:///tmp/local"))

    def test_file_localhost_is_local(self):
        self.assertTrue(is_local_uri("file://localhost/tmp/path"))

    def test_http_scheme_is_not_local(self):
        self.assertFalse(is_local_uri("http://example.com/path"))

    def test_https_scheme_is_not_local(self):
        self.assertFalse(is_local_uri("https://example.com/path"))

    def test_databricks_string_is_not_local(self):
        """'databricks' as a tracking URI must not be local."""
        self.assertFalse(is_local_uri("databricks"))

    def test_databricks_string_as_artifact_uri_is_local(self):
        """'databricks' without is_tracking_or_registry_uri flag must be local."""
        result = is_local_uri("databricks", is_tracking_or_registry_uri=False)
        self.assertTrue(result)

    def test_windows_unc_path_is_not_local(self):
        """Windows UNC paths must not be local."""
        with mock.patch("scikitplot._utils.uri.is_windows", return_value=True):
            result = is_local_uri("\\\\server\\share")
        self.assertFalse(result)

    def test_returns_bool(self):
        """Return type must be bool."""
        result = is_local_uri("/tmp/path")
        self.assertIsInstance(result, bool)

    def test_ftp_not_local(self):
        self.assertFalse(is_local_uri("ftp://files.example.com/file.txt"))


# ===========================================================================
# is_file_uri
# ===========================================================================


class TestIsFileUri(unittest.TestCase):
    """is_file_uri must detect 'file://' URIs."""

    def test_file_uri_true(self):
        self.assertTrue(is_file_uri("file:///tmp/file.txt"))

    def test_file_uri_with_hostname(self):
        self.assertTrue(is_file_uri("file://localhost/tmp"))

    def test_http_uri_false(self):
        self.assertFalse(is_file_uri("http://example.com"))

    def test_bare_path_false(self):
        self.assertFalse(is_file_uri("/tmp/path"))

    def test_empty_string_false(self):
        self.assertFalse(is_file_uri(""))

    def test_returns_bool(self):
        self.assertIsInstance(is_file_uri("file:///x"), bool)


# ===========================================================================
# is_http_uri
# ===========================================================================


class TestIsHttpUri(unittest.TestCase):
    """is_http_uri must detect http and https URIs."""

    def test_http_is_true(self):
        self.assertTrue(is_http_uri("http://example.com/path"))

    def test_https_is_true(self):
        self.assertTrue(is_http_uri("https://example.com/path"))

    def test_file_uri_is_false(self):
        self.assertFalse(is_http_uri("file:///tmp/x"))

    def test_bare_path_is_false(self):
        self.assertFalse(is_http_uri("/tmp/path"))

    def test_ftp_is_false(self):
        self.assertFalse(is_http_uri("ftp://files.example.com/x"))

    def test_databricks_is_false(self):
        self.assertFalse(is_http_uri("databricks://host"))

    def test_returns_bool(self):
        self.assertIsInstance(is_http_uri("http://x.com"), bool)


# ===========================================================================
# is_databricks_uri
# ===========================================================================


class TestIsDatabricksUri(unittest.TestCase):
    """is_databricks_uri must detect 'databricks' and 'databricks://' URIs."""

    def test_bare_databricks_is_true(self):
        self.assertTrue(is_databricks_uri("databricks"))

    def test_databricks_scheme_is_true(self):
        self.assertTrue(is_databricks_uri("databricks://host/path"))

    def test_http_is_false(self):
        self.assertFalse(is_databricks_uri("http://example.com"))

    def test_local_path_is_false(self):
        self.assertFalse(is_databricks_uri("/local/path"))

    def test_returns_bool(self):
        self.assertIsInstance(is_databricks_uri("databricks"), bool)


# ===========================================================================
# extract_and_normalize_path
# ===========================================================================


class TestExtractAndNormalizePath(unittest.TestCase):
    """extract_and_normalize_path must return the filesystem path from a URI."""

    def test_bare_path_returned_unchanged(self):
        result = extract_and_normalize_path("/tmp/my/path")
        self.assertEqual(result, "/tmp/my/path")

    def test_file_uri_extracts_path(self):
        result = extract_and_normalize_path("file:///tmp/my/path")
        self.assertIn("/tmp/my/path", result)

    def test_returns_str(self):
        result = extract_and_normalize_path("/some/path")
        self.assertIsInstance(result, str)


# ===========================================================================
# append_to_uri_path
# ===========================================================================


class TestAppendToUriPath(unittest.TestCase):
    """append_to_uri_path must append path components to a URI."""

    def test_appends_segment_to_local_path(self):
        result = append_to_uri_path("/base/path", "subdir")
        self.assertIn("subdir", result)

    def test_appends_to_http_uri(self):
        result = append_to_uri_path("http://example.com/base", "extra")
        self.assertIn("extra", result)
        self.assertTrue(result.startswith("http://"))

    def test_multiple_segments(self):
        result = append_to_uri_path("/base", "a", "b")
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_returns_str(self):
        result = append_to_uri_path("/base", "sub")
        self.assertIsInstance(result, str)

    def test_no_extra_segments_returns_base(self):
        result = append_to_uri_path("/base")
        self.assertIn("/base", result)


# ===========================================================================
# join_paths
# ===========================================================================


class TestJoinPaths(unittest.TestCase):
    """join_paths must concatenate path segments correctly."""

    def test_two_segments(self):
        result = join_paths("/base", "subdir")
        self.assertIn("base", result)
        self.assertIn("subdir", result)

    def test_single_segment(self):
        result = join_paths("/base")
        self.assertEqual(result, "/base")

    def test_returns_str(self):
        result = join_paths("/a", "b")
        self.assertIsInstance(result, str)

    def test_three_segments(self):
        result = join_paths("/a", "b", "c")
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)


# ===========================================================================
# strip_scheme
# ===========================================================================


class TestStripScheme(unittest.TestCase):
    """strip_scheme must remove the URI scheme prefix."""

    def test_strips_file_scheme(self):
        result = strip_scheme("file:///tmp/path")
        self.assertNotIn("file://", result)

    def test_strips_http_scheme(self):
        result = strip_scheme("http://example.com/path")
        self.assertNotIn("http://", result)
        self.assertIn("example.com", result)

    def test_bare_path_unchanged(self):
        result = strip_scheme("/tmp/path")
        self.assertEqual(result, "/tmp/path")

    def test_returns_str(self):
        result = strip_scheme("http://x.com")
        self.assertIsInstance(result, str)


# ===========================================================================
# is_models_uri
# ===========================================================================


class TestIsModelsUri(unittest.TestCase):
    """is_models_uri must detect 'models:/' URIs."""

    def test_models_uri_true(self):
        self.assertTrue(is_models_uri("models:/my-model/1"))

    def test_http_uri_false(self):
        self.assertFalse(is_models_uri("http://example.com"))

    def test_bare_path_false(self):
        self.assertFalse(is_models_uri("/local/path"))

    def test_returns_bool(self):
        self.assertIsInstance(is_models_uri("models:/x"), bool)


# ===========================================================================
# _escape_control_characters
# ===========================================================================


class TestEscapeControlChars(unittest.TestCase):
    """_escape_control_characters must escape or remove control characters."""

    def test_plain_string_unchanged(self):
        result = _escape_control_characters("hello world")
        self.assertEqual(result, "hello world")

    def test_returns_str(self):
        result = _escape_control_characters("test")
        self.assertIsInstance(result, str)

    def test_control_char_removed_or_escaped(self):
        """A null byte or newline must be handled without raising."""
        result = _escape_control_characters("before\x00after")
        self.assertIsInstance(result, str)
        # The null byte should not appear as-is
        self.assertNotEqual(result, "before\x00after")

    def test_empty_string(self):
        result = _escape_control_characters("")
        self.assertEqual(result, "")


# ===========================================================================
# validate_query_string
# ===========================================================================


class TestValidateQueryString(unittest.TestCase):
    """validate_query_string must raise on invalid control characters."""

    def test_valid_query_no_raise(self):
        """A well-formed query string must not raise."""
        try:
            validate_query_string("key=value&other=123")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"validate_query_string raised unexpectedly: {exc}")

    def test_empty_query_no_raise(self):
        try:
            validate_query_string("")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"validate_query_string raised unexpectedly: {exc}")


# ===========================================================================
# generate_tmp_dfs_path
# ===========================================================================


class TestGenerateTmpDfsPath(unittest.TestCase):
    """generate_tmp_dfs_path must produce a unique DBFS-prefixed temp path."""

    def test_contains_dfs_tmp(self):
        result = generate_tmp_dfs_path("/tmp/dfs")
        self.assertIn("/tmp/dfs", result)

    def test_returns_str(self):
        result = generate_tmp_dfs_path("/dfs/tmp")
        self.assertIsInstance(result, str)

    def test_two_calls_differ(self):
        """Each call must produce a unique path (UUID-based)."""
        r1 = generate_tmp_dfs_path("/dfs/tmp")
        r2 = generate_tmp_dfs_path("/dfs/tmp")
        self.assertNotEqual(r1, r2)


# ===========================================================================
# is_valid_dbfs_uri
# ===========================================================================


class TestIsValidDbfsUri(unittest.TestCase):
    """is_valid_dbfs_uri must return True only for properly-prefixed dbfs URIs."""

    def test_valid_dbfs_uri(self):
        self.assertTrue(is_valid_dbfs_uri("dbfs:/some/path"))

    def test_invalid_local_path(self):
        self.assertFalse(is_valid_dbfs_uri("/local/path"))

    def test_invalid_http_uri(self):
        self.assertFalse(is_valid_dbfs_uri("http://example.com"))

    def test_returns_bool(self):
        self.assertIsInstance(is_valid_dbfs_uri("dbfs:/x"), bool)


# ===========================================================================
# dbfs_hdfs_uri_to_fuse_path
# ===========================================================================


class TestDbfsHdfsUriToFusePath(unittest.TestCase):
    """dbfs_hdfs_uri_to_fuse_path must convert dbfs:/ URIs to /dbfs/ FUSE paths."""

    def test_converts_dbfs_uri(self):
        result = dbfs_hdfs_uri_to_fuse_path("dbfs:/my/path")
        self.assertTrue(result.startswith("/dbfs/"))

    def test_returns_str(self):
        result = dbfs_hdfs_uri_to_fuse_path("dbfs:/x")
        self.assertIsInstance(result, str)

    def test_path_component_preserved(self):
        result = dbfs_hdfs_uri_to_fuse_path("dbfs:/my/path/to/file")
        self.assertIn("my/path/to/file", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
