"""
Tests for ``install_meson_features._needs_update``: content, not timestamps.
"""

from __future__ import annotations

import os

from ..install_meson_features import _needs_update


def _package(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


FILES = {"__init__.py": "", "module.py": "A = 1\n", "feature.py": "B = 2\n"}


def test_an_absent_destination_needs_an_update(tmp_path):
    assert _needs_update(str(_package(tmp_path / "src", FILES)), str(tmp_path / "dst"))


def test_an_identical_destination_is_up_to_date(tmp_path):
    src = _package(tmp_path / "src", FILES)
    dst = _package(tmp_path / "dst", FILES)
    assert not _needs_update(str(src), str(dst))


def test_a_newer_destination_with_other_content_needs_an_update(tmp_path):
    src = _package(tmp_path / "src", FILES)
    dst = _package(tmp_path / "dst", {**FILES, "module.py": "A = 'edited'\n"})
    old = os.stat(src / "module.py").st_mtime - 100
    os.utime(src / "module.py", (old, old))  # the source is older
    assert _needs_update(str(src), str(dst))


def test_a_destination_only_file_needs_an_update(tmp_path):
    src = _package(tmp_path / "src", FILES)
    dst = _package(tmp_path / "dst", {**FILES, "removed_from_source.py": "C = 3\n"})
    assert _needs_update(str(src), str(dst))


def test_a_missing_destination_file_needs_an_update(tmp_path):
    src = _package(tmp_path / "src", FILES)
    dst = _package(
        tmp_path / "dst", {k: v for k, v in FILES.items() if k != "feature.py"}
    )
    assert _needs_update(str(src), str(dst))


def test_bytecode_on_either_side_is_ignored(tmp_path):
    src = _package(
        tmp_path / "src", {**FILES, "__pycache__/module.cpython-313.pyc": "x"}
    )
    dst = _package(
        tmp_path / "dst",
        {**FILES, "__pycache__/feature.cpython-312.pyc": "y", "stray.pyc": "z"},
    )
    assert not _needs_update(str(src), str(dst))


def test_nested_files_are_compared(tmp_path):
    src = _package(tmp_path / "src", {**FILES, "sub/x.py": "1\n"})
    dst = _package(tmp_path / "dst", {**FILES, "sub/x.py": "2\n"})
    assert _needs_update(str(src), str(dst))
