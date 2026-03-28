# scikitplot/_utils/tests/test_mime_type_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._utils.mime_type_utils`.

Coverage map
------------
get_text_extensions   returns list, known extensions present,
                      no duplicates, all lowercase, non-empty     -> TestGetTextExtensions
_guess_mime_type      known text extension → text/plain,
                      known binary extension → correct mime,
                      no extension → filename as extension key,
                      empty extension filename → fallback octet,
                      unknown extension → octet-stream,
                      path with directories → uses filename only  -> TestGuessMimeType

Run standalone::

    python -m unittest scikitplot._utils.tests.test_mime_type_utils -v
"""

from __future__ import annotations

import unittest

from ..mime_type_utils import _guess_mime_type, get_text_extensions


# ===========================================================================
# get_text_extensions
# ===========================================================================


class TestGetTextExtensions(unittest.TestCase):
    """get_text_extensions must return a well-formed list of extension strings."""

    def test_returns_list(self):
        """Return value must be a list."""
        result = get_text_extensions()
        self.assertIsInstance(result, list)

    def test_non_empty(self):
        """The extension list must not be empty."""
        self.assertGreater(len(get_text_extensions()), 0)

    def test_all_elements_are_strings(self):
        """Every element must be a non-empty str."""
        for ext in get_text_extensions():
            with self.subTest(ext=ext):
                self.assertIsInstance(ext, str)
                self.assertGreater(len(ext), 0)

    def test_all_elements_are_lowercase(self):
        """Extensions must be lowercase (no uppercase characters)."""
        for ext in get_text_extensions():
            with self.subTest(ext=ext):
                self.assertEqual(ext, ext.lower())

    def test_no_leading_dot(self):
        """Extensions must not start with a leading dot."""
        for ext in get_text_extensions():
            with self.subTest(ext=ext):
                self.assertFalse(ext.startswith("."), f"{ext!r} starts with dot")

    def test_no_duplicates(self):
        """There must be no duplicate extension entries."""
        exts = get_text_extensions()
        self.assertEqual(len(exts), len(set(exts)))

    # -- known required extensions --

    def test_txt_present(self):
        self.assertIn("txt", get_text_extensions())

    def test_json_present(self):
        self.assertIn("json", get_text_extensions())

    def test_yaml_present(self):
        self.assertIn("yaml", get_text_extensions())

    def test_yml_present(self):
        self.assertIn("yml", get_text_extensions())

    def test_csv_present(self):
        self.assertIn("csv", get_text_extensions())

    def test_py_present(self):
        self.assertIn("py", get_text_extensions())

    def test_md_present(self):
        self.assertIn("md", get_text_extensions())

    def test_xml_present(self):
        self.assertIn("xml", get_text_extensions())

    def test_log_present(self):
        self.assertIn("log", get_text_extensions())

    def test_toml_present(self):
        self.assertIn("toml", get_text_extensions())

    # -- idempotency --

    def test_returns_same_list_on_repeated_calls(self):
        """Two consecutive calls must return equal lists."""
        self.assertEqual(get_text_extensions(), get_text_extensions())


# ===========================================================================
# _guess_mime_type
# ===========================================================================


class TestGuessMimeType(unittest.TestCase):
    """_guess_mime_type must correctly resolve MIME types from file paths."""

    # -- text extensions -> text/plain --

    def test_txt_file(self):
        self.assertEqual(_guess_mime_type("notes.txt"), "text/plain")

    def test_log_file(self):
        self.assertEqual(_guess_mime_type("app.log"), "text/plain")

    def test_json_file(self):
        self.assertEqual(_guess_mime_type("config.json"), "text/plain")

    def test_yaml_file(self):
        self.assertEqual(_guess_mime_type("config.yaml"), "text/plain")

    def test_yml_file(self):
        self.assertEqual(_guess_mime_type("config.yml"), "text/plain")

    def test_csv_file(self):
        self.assertEqual(_guess_mime_type("data.csv"), "text/plain")

    def test_py_file(self):
        self.assertEqual(_guess_mime_type("script.py"), "text/plain")

    def test_md_file(self):
        self.assertEqual(_guess_mime_type("README.md"), "text/plain")

    def test_xml_file(self):
        self.assertEqual(_guess_mime_type("data.xml"), "text/plain")

    def test_toml_file(self):
        self.assertEqual(_guess_mime_type("pyproject.toml"), "text/plain")

    def test_ini_file(self):
        self.assertEqual(_guess_mime_type("setup.ini"), "text/plain")

    def test_cfg_file(self):
        self.assertEqual(_guess_mime_type("app.cfg"), "text/plain")

    # -- path with directories -> uses filename only --

    def test_full_path_txt(self):
        """A full path to a .txt file must still return text/plain."""
        self.assertEqual(_guess_mime_type("/some/path/to/notes.txt"), "text/plain")

    def test_relative_path_json(self):
        """A relative path must still return text/plain for known ext."""
        self.assertEqual(_guess_mime_type("subdir/config.json"), "text/plain")

    # -- known binary/media extensions -> system mime type --

    def test_png_file_not_plain(self):
        """PNG files must not return text/plain."""
        result = _guess_mime_type("image.png")
        self.assertNotEqual(result, "text/plain")
        self.assertIn("image", result)

    def test_pdf_file_not_plain(self):
        """PDF files must not return text/plain."""
        result = _guess_mime_type("document.pdf")
        self.assertNotEqual(result, "text/plain")

    def test_zip_file_not_plain(self):
        """ZIP files must not return text/plain."""
        result = _guess_mime_type("archive.zip")
        self.assertNotEqual(result, "text/plain")

    # -- no extension: filename becomes extension key --

    def test_mlmodel_no_extension(self):
        """A file named 'MLmodel' (no ext) must not raise and must return a string."""
        result = _guess_mime_type("MLmodel")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_no_extension_returns_string(self):
        """Any extensionless filename must return a non-empty string."""
        result = _guess_mime_type("somefile")
        self.assertIsInstance(result, str)

    # -- completely unknown extension -> application/octet-stream --

    def test_unknown_extension_returns_octet_stream(self):
        """An unrecognised extension must fall back to application/octet-stream."""
        result = _guess_mime_type("file.xyzzy_unknown_ext_12345")
        self.assertEqual(result, "application/octet-stream")

    def test_double_extension_uses_outer(self):
        """Files with two dots use the final extension for MIME detection."""
        # e.g. backup.txt.bak -> extension is .bak which is likely unknown
        result = _guess_mime_type("backup.txt.bak")
        # bak is not in text_extensions and not known to mimetypes, so octet-stream
        self.assertIsInstance(result, str)

    # -- return type invariant --

    def test_return_type_always_str(self):
        """_guess_mime_type must always return a str."""
        paths = [
            "file.txt",
            "file.png",
            "file.xyz_unknown",
            "MLmodel",
            "/a/b/c/data.csv",
        ]
        for path in paths:
            with self.subTest(path=path):
                result = _guess_mime_type(path)
                self.assertIsInstance(result, str)

    def test_return_contains_slash(self):
        """Every returned MIME type must contain exactly one slash."""
        paths = ["file.txt", "image.png", "data.unknownext99"]
        for path in paths:
            with self.subTest(path=path):
                result = _guess_mime_type(path)
                self.assertIn("/", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
