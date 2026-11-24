"""A glob and the directory it names give the same corpus sources (CX-53)."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from unittest.mock import patch

import pytest

from .. import _corpus_builder as _builder_module
from .._corpus_builder import CorpusBuilder, _glob_base, _skip_reason

expand = CorpusBuilder._expand_sources


@pytest.fixture
def tree(tmp_path):
    """``<tmp>/.hidden_parent/docs`` with supported, unsupported and hidden files."""
    docs = tmp_path / ".hidden_parent" / "docs"
    for rel in ("a.md", "sub/b.txt", "c.unknownext", ".secret.md", ".notes/n.md", "__pycache__/x.txt"):
        (docs / rel).parent.mkdir(parents=True, exist_ok=True)
        (docs / rel).write_text("x", encoding="utf-8")
    return docs


def _names(paths, root):
    return sorted(Path(p).relative_to(root).as_posix() for p in paths)


def test_glob_under_a_hidden_parent_matches_the_directory(tree):
    want = ["a.md", "sub/b.txt"]
    assert _names(expand([tree]), tree) == want
    assert _names(expand([f"{tree}/**/*"]), tree) == want
    assert _names(expand([f"{tree}/*.md"]), tree) == ["a.md"]


def test_glob_filters_unsupported_types_like_the_directory(tree):
    assert "c.unknownext" not in _names(expand([f"{tree}/*"]), tree)


def test_a_hidden_folder_named_by_the_pattern_is_read(tree):
    assert _names(expand([f"{tree}/.notes/*.md"]), tree) == [".notes/n.md"]


def test_relative_patterns(tree, monkeypatch):
    monkeypatch.chdir(tree)
    assert sorted(Path(p).as_posix() for p in expand(["*.md"])) == ["a.md"]
    assert sorted(Path(p).as_posix() for p in expand(["**/*.txt"])) == ["sub/b.txt"]


# The module logger is patched, not captured: in a partial distribution the
# package logger does not propagate to the root logger caplog listens on.
def _warnings(call):
    with patch.object(_builder_module.logger, "warning") as warning:
        result = call()
    return result, [c.args[0] % c.args[1:] for c in warning.call_args_list]


def test_one_warning_when_every_match_is_skipped(tree):
    result, messages = _warnings(lambda: expand([f"{tree}/*.unknownext"]))
    assert result == []
    skipped = [m for m in messages if "skipped" in m]
    assert len(skipped) == 1 and "1 unsupported" in skipped[0]


def test_no_match_keeps_its_own_warning(tree):
    result, messages = _warnings(lambda: expand([f"{tree}/*.none"]))
    assert result == []
    assert any("matched no files" in m for m in messages)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_symlinks_are_still_skipped(tree):
    try:
        (tree / "link.md").symlink_to(tree / "a.md")
    except OSError:
        pytest.skip("symlinks not permitted")
    assert "link.md" not in _names(expand([f"{tree}/*.md"]), tree)
    assert "link.md" not in _names(expand([tree]), tree)


@pytest.mark.parametrize(
    ("cls", "pattern", "base"),
    [
        (PurePosixPath, "/x/.data/docs/*.md", "/x/.data/docs"),
        (PurePosixPath, "docs/**/*.md", "docs"),
        (PurePosixPath, "*.md", "."),
        (PurePosixPath, "a/b[0-9]/c", "a"),
        (PureWindowsPath, r"C:\data\.cache\*.pdf", r"C:\data\.cache"),
        (PureWindowsPath, r"C:\*.pdf", "C:\\"),
    ],
)
def test_glob_base(cls, pattern, base):
    assert _glob_base(pattern, cls) == cls(base)


def test_skip_reason_judges_only_parts_below_the_base(tmp_path):
    f = tmp_path / ".top" / "docs" / "a.md"
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    assert _skip_reason(f, tmp_path / ".top" / "docs", {".md"}) is None
    assert _skip_reason(f, tmp_path, {".md"}) == "hidden"
    assert _skip_reason(f, f.parent, {".txt"}) == "unsupported"
    assert _skip_reason(f, f.parent, set()) is None  # no registry: nothing refused


def test_unsupported_types_are_filtered_without_a_hidden_parent(tmp_path):
    for name in ("a.md", "b.unknownext"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert _names(expand([tmp_path]), tmp_path) == ["a.md"]
    assert _names(expand([f"{tmp_path}/*"]), tmp_path) == ["a.md"]
