# lean/tests/test__layers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A project added as a layer over a saved dependency reads as one project
=======================================================================

``add_folder`` puts a project's files over what an index holds (a saved
mathlib, say): its names resolve into the layer below, its dossiers follow
proofs through both, and the reports are those of one index of both
folders, with the project's labels prefixed by its layer's name. A module
two layers provide is listed; layer names are checked; snapshots keep the
layers and quote each from its own folder.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import textwrap

import pytest

from .._index import LeanIndex, main

BASE = {
    "Base/Defs.lean": """
        def Good (n : Nat) : Prop := n = n
        theorem shaky : 1 = 1 := sorry
    """,
    "Base/Util.lean": """
        import Base.Defs
        theorem good_all (n : Nat) : Good n := rfl
        theorem uses_shaky : 1 = 1 := shaky
    """,
}
PROJECT = {
    "Proj/Main.lean": """
        import Base.Util
        theorem main : ∀ n, Good n := fun n => good_all n
        theorem main_shaky : 1 = 1 := uses_shaky
    """,
}


def _write(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


def _layered(tmp_path, **load):
    base = _write(tmp_path / "base", BASE)
    saved = LeanIndex.from_folder(base).save(tmp_path / "base.leanidx")
    index = LeanIndex.load(saved, **load)
    index.add_folder(_write(tmp_path / "proj", PROJECT), name="p")
    return index


def _essence(report):
    """A dossier without its labels' layer prefix and its quoted text."""
    text = json.dumps(report, sort_keys=True).replace('"p:', '"')
    found = json.loads(text)
    found["target"].pop("statement", None)
    for item in found["statement"]["local_definitions"]["items"]:
        item.pop("text", None)
        item.pop("text_cut", None)
    return found


def test_a_layer_reads_as_one_project_of_both_folders(tmp_path):
    both = _write(_write(tmp_path / "both", BASE), PROJECT)
    whole = LeanIndex.from_folder(both)
    layered = _layered(tmp_path, root=tmp_path / "base")
    for name in ("main", "main_shaky"):
        assert _essence(layered.dossier(name)) == _essence(whole.dossier(name))
    shaky = layered.dossier("main_shaky")
    assert shaky["flags_found"] == ["sorry"]
    assert [r["declaration"] for r in shaky["dependencies"]["red_flags"]["items"]] == ["shaky"]
    assert layered.find("main")[0]["file"] == "p:Proj/Main.lean"
    assert layered.summary()["imports"]["missing_modules"]["count"] == 0


def test_a_module_two_layers_provide_is_listed_and_the_lower_one_used(tmp_path):
    index = _layered(tmp_path)
    clone = _write(tmp_path / "clone", {"Base/Defs.lean": "def Good (n : Nat) : Prop := True\n"})
    index.add_folder(clone, name="clone")
    clashes = index.summary()["imports"]["module_clashes"]["items"]
    assert clashes == [
        {"module": "Base.Defs", "used": "Base/Defs.lean", "also_in": "clone:Base/Defs.lean"}
    ]
    assert "Lean refuses them together" in LeanIndex.render(index.summary())


def test_a_layer_root_is_found_from_its_own_imports(tmp_path):
    # The base imports Base.Defs often; a project's own Proj/Base/Defs.lean
    # must stay Proj.Base.Defs, not be read one folder down as Base.Defs.
    base = dict(BASE)
    for i in range(3):
        base[f"Base/More{i}.lean"] = "import Base.Defs\ntheorem more{0} : 1 = 1 := rfl\n".format(i)
    saved = LeanIndex.from_folder(_write(tmp_path / "base", base)).save(tmp_path / "b.leanidx")
    index = LeanIndex.load(saved)
    project = {
        "Proj/Base/Defs.lean": "theorem own : 2 = 2 := rfl\n",
        "Proj/Main.lean": "import Proj.Base.Defs\nimport Base.Defs\ntheorem t : 2 = 2 := own\n",
    }
    index.add_folder(_write(tmp_path / "proj", project), name="p")
    imports = index.summary()["imports"]
    assert imports["module_clashes"]["count"] == 0
    assert imports["missing_modules"]["count"] == 0
    assert index.dossier("t")["dependencies"]["files"] == 1  # p:Proj/Base/Defs.lean


@pytest.mark.parametrize("name", ["", "a:b", "a/b", "a b", 3, "p"])
def test_layer_names_are_checked(tmp_path, name):
    index = _layered(tmp_path)  # "p" is taken
    with pytest.raises(ValueError, match="add_folder"):
        index.add_folder(tmp_path / "proj", name=name)


def test_a_missing_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        LeanIndex().add_folder(tmp_path / "nowhere", name="x")


def test_a_layer_learns_its_own_commands(tmp_path):
    index = _layered(tmp_path)
    tool = _write(
        tmp_path / "tool",
        {"Tool/Cmd.lean": 'elab "audit_now" : command => pure ()\naudit_now\n'},
    )
    index.add_folder(tool, name="tool")
    assert "audit_now" in index._commands
    assert index._learned["audit_now"][0] == "tool:Tool/Cmd.lean"


def test_layers_are_saved_and_each_quoted_from_its_folder(tmp_path):
    index = _layered(tmp_path)
    saved = index.save(tmp_path / "both.leanidx")
    blind = LeanIndex.load(saved)
    assert blind._layers == [("", None), ("p", None)]
    assert blind.dossier("main")["target"]["statement"] is None
    seen = LeanIndex.load(saved, root={"": tmp_path / "base", "p": tmp_path / "proj"})
    assert seen.dossier("main")["target"]["statement"].startswith("theorem main")
    definitions = seen.dossier("main")["statement"]["local_definitions"]["items"]
    assert definitions[0]["text"].startswith("def Good")  # from the base's folder
    assert _essence(seen.dossier("main_shaky")) == _essence(index.dossier("main_shaky"))


def test_the_command_line_adds_a_folder_over_a_base(tmp_path):
    base = _write(tmp_path / "base", BASE)
    saved = LeanIndex.from_folder(base).save(tmp_path / "base.leanidx")
    project = _write(tmp_path / "proj", PROJECT)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main(
            [str(project), "--base", str(saved), "--base-root", str(base),
             "--target", "main_shaky", "--fail-on", "sorry", "--json"]
        )  # fmt: skip
    assert status == 1
    [report] = json.loads(out.getvalue())
    assert report["target"]["file"] == "proj:Proj/Main.lean"


def test_a_copy_of_the_base_is_not_needed(tmp_path):
    # The base's files may be gone: its saved index is enough to resolve.
    index = _layered(tmp_path)
    shutil.rmtree(tmp_path / "base")
    assert index.dossier("main_shaky")["flags_found"] == ["sorry"]
