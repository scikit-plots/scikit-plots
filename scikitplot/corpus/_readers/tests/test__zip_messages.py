# corpus/_readers/tests/test__zip_messages.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
ZipReader member messages name the archive, not the temporary directory (CX-25).

Run with::

    pytest scikitplot/corpus/_readers/tests/test__zip_messages.py -v
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import pytest

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._readers._zip import ZipReader, _archive_relative


def _zip(tmp_path: Path) -> Path:
    path = tmp_path / "docs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a/bad.txt", "some text")
    return path


def _member_warnings(path: Path, caplog, captured_logger, monkeypatch) -> list:
    """Read *path* with a member reader whose error names the file it got.

    Pillow, pdfminer and other libraries put the path they were given in
    their messages; the stub does the same, so the test does not depend on
    which optional readers are installed.
    """

    def failing_reader(target, **kwargs):
        raise OSError(f"cannot identify file {str(target)!r}")

    monkeypatch.setattr(DocumentReader, "_create_one", staticmethod(failing_reader))
    caplog.clear()
    # captured_logger, not caplog.at_level: the "scikitplot" logger does not
    # propagate, and which pytest releases see its records differs (Rule 139).
    with captured_logger(logging.WARNING, "scikitplot.corpus._readers._zip"):
        list(ZipReader(input_path=path).get_raw_chunks())
    return [
        r.getMessage()
        for r in caplog.records
        if "failed to read member" in r.getMessage()
    ]


class TestMemberWarnings:
    def test_warning_names_the_archive_path(
        self, tmp_path, caplog, captured_logger, monkeypatch
    ):
        messages = _member_warnings(
            _zip(tmp_path), caplog, captured_logger, monkeypatch
        )
        assert len(messages) == 1
        assert "'docs.zip/a/bad.txt'" in messages[0]
        assert "skplt_zip_" not in messages[0]

    def test_warning_is_the_same_on_every_run(
        self, tmp_path, caplog, captured_logger, monkeypatch
    ):
        path = _zip(tmp_path)
        first = _member_warnings(path, caplog, captured_logger, monkeypatch)
        assert first == _member_warnings(path, caplog, captured_logger, monkeypatch)


class TestArchiveRelative:
    def test_replaces_the_directory_prefix(self, tmp_path):
        root = tmp_path / "skplt_zip_x"
        text = f"cannot open '{root / 'a' / 'b.png'}'"
        assert _archive_relative(text, root, "docs.zip") == (
            "cannot open 'docs.zip/a/b.png'"
        )

    def test_repr_of_the_path_is_replaced(self, tmp_path):
        root = tmp_path / "skplt_zip_x"
        text = f"cannot open {str(root / 'a' / 'b.png')!r}"
        assert _archive_relative(text, root, "docs.zip") == (
            "cannot open 'docs.zip/a/b.png'"
        )

    def test_forward_slash_prefix_is_replaced(self, tmp_path):
        root = tmp_path / "skplt_zip_x"
        text = f"{root}/a.txt"
        assert _archive_relative(text, root, "docs.zip") == "docs.zip/a.txt"

    def test_sibling_directory_with_same_prefix_is_kept(self, tmp_path):
        root = tmp_path / "skplt_zip_x"
        text = f"{root}2{os.sep}a.txt"
        assert _archive_relative(text, root, "docs.zip") == text

    def test_text_without_the_directory_is_unchanged(self, tmp_path):
        assert _archive_relative("plain", tmp_path, "docs.zip") == "plain"

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_resolved_form_is_replaced(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks not permitted")
        text = f"{real.resolve()}{os.sep}m.pdf and {link}{os.sep}m.pdf"
        assert _archive_relative(text, link, "z.zip") == "z.zip/m.pdf and z.zip/m.pdf"


class _WindowsDir:
    """A Windows extraction directory, so the test runs on any system."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __str__(self) -> str:
        return self._path

    def resolve(self) -> _WindowsDir:
        return self


class TestArchiveRelativeOnWindows:
    root = "C:\\Users\\runner\\AppData\\Local\\Temp\\skplt_zip_ab12"

    @pytest.fixture(autouse=True)
    def _windows_sep(self, monkeypatch):
        monkeypatch.setattr(os, "sep", "\\")

    def test_plain_path(self):
        text = f"cannot identify file '{self.root}\\a\\bad.txt'"
        got = _archive_relative(text, _WindowsDir(self.root), "docs.zip")
        assert got == "cannot identify file 'docs.zip/a/bad.txt'"

    def test_repr_with_doubled_backslashes(self):
        path = f"{self.root}\\a\\bad.txt"
        text = f"cannot identify file {path!r}"
        got = _archive_relative(text, _WindowsDir(self.root), "docs.zip")
        assert got == "cannot identify file 'docs.zip/a/bad.txt'"

    def test_forward_slashes(self):
        text = f"{self.root}/a/bad.txt"
        got = _archive_relative(text, _WindowsDir(self.root), "docs.zip")
        assert got == "docs.zip/a/bad.txt"
