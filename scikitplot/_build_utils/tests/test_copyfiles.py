"""
Tests for ``copyfiles.copy_file`` destination handling.
"""

from __future__ import annotations

from ..copyfiles import copy_file


def test_a_bare_destination_name_copies_into_the_working_directory(
    tmp_path, monkeypatch
):
    src = tmp_path / "in" / "a.txt"
    src.parent.mkdir()
    src.write_text("data\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert copy_file(str(src), "a.txt") is True
    assert (work / "a.txt").read_text(encoding="utf-8") == "data\n"


def test_missing_parent_directories_are_created(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("data\n", encoding="utf-8")
    dest = tmp_path / "x" / "y" / "a.txt"
    assert copy_file(str(src), str(dest)) is True
    assert dest.read_text(encoding="utf-8") == "data\n"


def test_an_existing_file_is_kept_without_overwrite(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("new\n", encoding="utf-8")
    dest = tmp_path / "b.txt"
    dest.write_text("old\n", encoding="utf-8")
    assert copy_file(str(src), str(dest), overwrite=False) is False
    assert dest.read_text(encoding="utf-8") == "old\n"
