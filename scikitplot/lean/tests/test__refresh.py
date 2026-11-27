# lean/tests/test__refresh.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A refreshed index is the index a new read of the same files gives
=================================================================

``refresh`` reads every file of a layer once for its digest and the
commands it defines, and parses again only new and changed files. The
index it leaves saves to the same bytes as a new read of the folder, in
one layer or under and over others, with commands that change or not.
"""

from __future__ import annotations

import contextlib
import io
import json
import textwrap

import pytest

from .._index import LeanIndex, main

FILES = {
    "A/Defs.lean": """
        def Good (n : Nat) : Prop := n = n
        theorem base_fact : 1 = 1 := rfl
    """,
    "A/Util.lean": """
        import A.Defs
        theorem good_all (n : Nat) : Good n := rfl
    """,
    "A/Gone.lean": """
        theorem gone : 2 = 2 := rfl
    """,
}


def _write(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


def _edit(root):
    """Change one file, add one and remove one."""
    _write(root, {"A/Util.lean": "import A.Defs\ntheorem good_all (n : Nat) : Good n := sorry\n"})
    _write(root, {"A/New.lean": "import A.Util\ntheorem uses : Good 1 := good_all 1\n"})
    (root / "A" / "Gone.lean").unlink()


@pytest.mark.parametrize("workers", [1, 2])
def test_a_refresh_saves_the_bytes_of_a_new_read(tmp_path, workers):
    root = _write(tmp_path / "proj", FILES)
    index = LeanIndex.from_folder(root, workers=workers)
    index.dossier("good_all")  # caches filled before the refresh
    _edit(root)
    report = index.refresh()
    assert report["report"] == "refresh"
    assert report["added"]["items"] == ["A/New.lean"]
    assert report["changed"]["items"] == ["A/Util.lean"]
    assert report["removed"]["items"] == ["A/Gone.lean"]
    assert (report["unchanged"], report["files_read"], report["commands_changed"]) == (1, 2, False)
    assert index.dossier("uses")["flags_found"] == ["sorry"]
    assert index.find("gone") == []
    fresh = LeanIndex.from_folder(root)
    assert index.save(tmp_path / "a.leanidx").read_bytes() == fresh.save(
        tmp_path / "b.leanidx"
    ).read_bytes()
    assert index.summary() == fresh.summary()


def test_nothing_changed_reads_nothing_again(tmp_path):
    root = _write(tmp_path / "proj", FILES)
    index = LeanIndex.from_folder(root)
    before = index.save(tmp_path / "a.leanidx").read_bytes()
    report = index.refresh()
    assert (report["files_read"], report["unchanged"], report["files"]) == (0, 3, 3)
    assert index.save(tmp_path / "b.leanidx").read_bytes() == before


def test_a_changed_command_reads_the_layer_again(tmp_path):
    files = dict(FILES)
    files["A/Cmd.lean"] = 'syntax "audit_now" : command\naudit_now\n'
    root = _write(tmp_path / "proj", files)
    index = LeanIndex.from_folder(root)
    _write(root, {"A/Cmd.lean": 'syntax "audit_later" : command\naudit_later\n'})
    report = index.refresh()
    assert report["commands_changed"] and report["layers_read_again"] == [""]
    assert report["files_read"] == 4
    assert sorted(index._learned) == ["audit_later"]
    fresh = LeanIndex.from_folder(root)
    assert index.save(tmp_path / "a").read_bytes() == fresh.save(tmp_path / "b").read_bytes()


def test_a_command_moving_within_the_layer_is_followed(tmp_path):
    files = dict(FILES)
    files["A/Cmd.lean"] = 'syntax "audit_now" : command\n'
    root = _write(tmp_path / "proj", files)
    index = LeanIndex.from_folder(root)
    _write(root, {"A/Cmd.lean": "-- moved\n", "A/Cmd0.lean": 'syntax "audit_now" : command\n'})
    report = index.refresh()
    assert not report["commands_changed"]
    assert index._learned["audit_now"][0] == "A/Cmd0.lean"
    fresh = LeanIndex.from_folder(root)
    assert index.save(tmp_path / "a").read_bytes() == fresh.save(tmp_path / "b").read_bytes()


def _layered(tmp_path, base_root, proj_root):
    saved = LeanIndex.from_folder(base_root).save(tmp_path / "base.leanidx")
    index = LeanIndex.load(saved, root=base_root)
    index.add_folder(proj_root, name="p")
    return index


PROJECT = {
    "P/Main.lean": """
        import A.Util
        theorem main : Good 2 := good_all 2
    """,
}


@pytest.mark.parametrize(
    "edit",
    ["project", "base", "base command"],
)
def test_a_layer_refreshes_under_and_over_others(tmp_path, edit):
    base = _write(tmp_path / "base", FILES)
    proj = _write(tmp_path / "proj", PROJECT)
    index = _layered(tmp_path, base, proj)
    if edit == "project":
        _write(proj, {"P/Main.lean": "import A.Util\ntheorem main : Good 2 := sorry\n"})
        report = index.refresh("p")
    elif edit == "base":
        _edit(base)
        report = index.refresh("")
    else:
        _write(base, {"A/Cmd.lean": 'syntax "audit_now" : command\n'})
        report = index.refresh("")
        assert report["layers_read_again"] == ["", "p"]
    fresh = _layered(tmp_path, base, proj)
    assert index.save(tmp_path / "a").read_bytes() == fresh.save(tmp_path / "b").read_bytes()
    assert report["layer"] == ("p" if edit == "project" else "")


def test_a_layer_without_a_folder_needs_one(tmp_path):
    root = _write(tmp_path / "proj", FILES)
    saved = LeanIndex.from_folder(root).save(tmp_path / "a.leanidx")
    index = LeanIndex.load(saved)
    with pytest.raises(ValueError, match="no known folder; give folder="):
        index.refresh()
    _edit(root)
    assert index.refresh(folder=root)["changed"]["count"] == 1
    with pytest.raises(ValueError, match="layer must be None or one of"):
        index.refresh("q")
    with pytest.raises(ValueError, match="is not a folder"):
        index.refresh(folder=tmp_path / "nowhere")


def test_layers_above_without_files_are_refused_before_any_change(tmp_path):
    base = _write(tmp_path / "base", FILES)
    proj = _write(tmp_path / "proj", PROJECT)
    saved = _layered(tmp_path, base, proj).save(tmp_path / "both.leanidx")
    index = LeanIndex.load(saved, root={"": base})  # the project's files unknown
    before = index.save(tmp_path / "before").read_bytes()
    _write(base, {"A/Cmd.lean": 'syntax "audit_now" : command\n'})
    with pytest.raises(ValueError, match="'p:P/Main.lean' has no known file"):
        index.refresh("")
    assert index.save(tmp_path / "after").read_bytes() == before


def test_a_skipped_file_is_read_again(tmp_path):
    root = _write(tmp_path / "proj", FILES)
    _write(root, {"A/Big.lean": "-- " + "x" * 200 + "\n"})
    index = LeanIndex.from_folder(root, max_file_bytes=100)
    assert index.summary()["skipped_files"][0]["file"] == "A/Big.lean"
    report = index.refresh()
    assert report["changed"]["items"] == ["A/Big.lean"]
    assert index.summary()["skipped_files"][0]["file"] == "A/Big.lean"


def test_a_new_first_layer_file_may_not_name_another_layer(tmp_path):
    base = _write(tmp_path / "base", FILES)
    index = _layered(tmp_path, base, _write(tmp_path / "proj", PROJECT))
    _write(base, {"p:/X.lean": "theorem x : 1 = 1 := rfl\n"})
    with pytest.raises(ValueError, match="would name another layer"):
        index.refresh("")


def test_the_command_line_refreshes_a_loaded_index(tmp_path):
    root = _write(tmp_path / "proj", FILES)
    saved = LeanIndex.from_folder(root).save(tmp_path / "a.leanidx")
    _edit(root)
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        status = main(
            [str(root), "--load", str(saved), "--refresh", "--save", str(saved), "--json"]
        )
    assert status == 0
    assert "1 added, 1 changed, 1 removed" in err.getvalue()
    assert json.loads(out.getvalue())["files"] == 3
    fresh = LeanIndex.from_folder(root).save(tmp_path / "b.leanidx")
    assert saved.read_bytes() == fresh.read_bytes()
