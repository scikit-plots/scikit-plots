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

from scikitplot.corpus._readers._zip import ZipReader, _archive_relative


def _zip(tmp_path: Path) -> Path:
    path = tmp_path / "docs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a/bad.png", b"not an image")
    return path


def _member_warnings(path: Path, caplog: pytest.LogCaptureFixture) -> list:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._readers._zip"):
        list(ZipReader(input_path=path).get_raw_chunks())
    return [
        r.getMessage()
        for r in caplog.records
        if "failed to read member" in r.getMessage()
    ]


class TestMemberWarnings:
    def test_warning_names_the_archive_path(self, tmp_path, caplog):
        pytest.importorskip("PIL")
        messages = _member_warnings(_zip(tmp_path), caplog)
        assert len(messages) == 1
        assert "docs.zip/a/bad.png" in messages[0]
        assert "skplt_zip_" not in messages[0]

    def test_warning_is_the_same_on_every_run(self, tmp_path, caplog):
        pytest.importorskip("PIL")
        path = _zip(tmp_path)
        assert _member_warnings(path, caplog) == _member_warnings(path, caplog)


class TestArchiveRelative:
    def test_replaces_the_directory_prefix(self, tmp_path):
        root = tmp_path / "skplt_zip_x"
        text = f"cannot open '{root}{os.sep}a{os.sep}b.png'"
        assert _archive_relative(text, root, "docs.zip") == (
            f"cannot open 'docs.zip/a{os.sep}b.png'"
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
