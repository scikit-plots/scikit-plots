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


# -- reports on one layer -----------------------------------------------------


def test_a_layer_summary_counts_its_own_files_and_the_layers_add_up(tmp_path):
    index = _layered(tmp_path)
    whole = index.summary()
    base, project = index.summary(layer=""), index.summary(layer="p")
    assert (whole["layer"], base["layer"], project["layer"]) == (None, "", "p")
    assert (project["files"], project["declarations"]) == (1, 2)
    for key in ("files", "lines", "declarations"):
        assert base[key] + project[key] == whole[key]
    kinds = {k: base["kinds"].get(k, 0) + project["kinds"].get(k, 0) for k in whole["kinds"]}
    assert kinds == whole["kinds"]
    assert project["kinds"]["import"] == 1
    assert "sorry" in base["flags"] and project["flags"] == {}
    assert "in layer `p`" in LeanIndex.render(project)


def test_a_layer_discovery_lists_its_files_and_counts_every_use(tmp_path):
    index = _layered(tmp_path)
    project, base = index.discover(layer="p"), index.discover(layer="")
    assert project["layer"] == "p"
    assert project["claims"]["unproven"]["count"] == 0
    assert [i["declaration"] for i in base["claims"]["unproven"]["items"]] == ["shaky"]
    heads = sorted(i["declaration"] for i in project["headline_results"]["items"])
    assert heads == ["main", "main_shaky"]
    assert base["headline_results"]["count"] == 0  # the project names them all
    rules = {i["declaration"]: i for i in base["rules"]["load_bearing_theorems"]["items"]}
    assert rules["good_all"]["named_from_files"] == 1  # from the project's file
    assert project["rules"]["load_bearing_theorems"]["count"] == 0
    assert LeanIndex.render(project).startswith("Layer: `p`")


@pytest.mark.parametrize("layer", ["q", 0, "p:"])
def test_an_unknown_layer_is_refused(tmp_path, layer):
    index = _layered(tmp_path)
    for report in (index.summary, index.discover):
        with pytest.raises(ValueError, match="layer must be None or one of"):
            report(layer=layer)


def test_a_layer_export_writes_its_own_documents(tmp_path):
    index = _layered(tmp_path)
    index.export(tmp_path / "out", layer="p")
    summary = json.loads((tmp_path / "out" / "summary.json").read_text("utf-8"))
    assert (summary["layer"], summary["files"]) == ("p", 1)
    dossiers = sorted(p.name for p in (tmp_path / "out" / "dossiers").iterdir())
    assert len(dossiers) == 4  # main and main_shaky, .json and .md
    assert [d.split("-")[0] for d in dossiers[::2]] == ["main", "main_shaky"]


def test_a_snapshot_keeps_the_counts_of_each_layer(tmp_path):
    index = _layered(tmp_path)
    saved = index.save(tmp_path / "both.leanidx")
    loaded = LeanIndex.load(saved)
    for layer in ("", "p"):
        assert loaded.summary(layer=layer)["kinds"] == index.summary(layer=layer)["kinds"]


def _meta_changed(blob, change):
    from .test__snapshot import _resign, _section

    meta = json.loads(_section(blob, "meta"))
    change(meta)
    return _resign(blob, "meta", json.dumps(meta).encode())


def test_counts_by_file_and_commands_by_layer_are_checked(tmp_path):
    saved = _layered(tmp_path).save(tmp_path / "both.leanidx")
    blob = saved.read_bytes()
    bad = {
        "do not add up": lambda m: m["kinds_counted"].append(["axiom", 1]),
        "not sorted and unique": lambda m: m["file_kind_names"].reverse(),
        "commands by layer do not match": lambda m: m["layer_commands"].pop(),
        "not the given and learned": lambda m: m["layer_commands"][1].append("zz"),
        "without a digest are not sorted": lambda m: m["undigested"].extend([0, 0]),
        "names a layer it is not in": lambda m: m["labels"].__setitem__(0, "p:X.lean"),
    }
    for message, change in bad.items():
        saved.write_bytes(_meta_changed(blob, change))
        with pytest.raises(ValueError, match=message):
            LeanIndex.load(saved)


def test_a_snapshot_of_an_earlier_development_version_is_refused(tmp_path):
    from .test__snapshot import _resign

    saved = LeanIndex.from_folder(_write(tmp_path / "base", BASE)).save(tmp_path / "b.leanidx")
    blob = saved.read_bytes()
    saved.write_bytes(
        _resign(blob, header_change=lambda h: h["sections"].pop("digests"))
    )
    with pytest.raises(ValueError, match="sections missing: digests .*save it again"):
        LeanIndex.load(saved)


def test_a_layer_name_a_base_label_starts_with_is_refused(tmp_path):
    base = _write(tmp_path / "base", {"q:/A.lean": "theorem a : 1 = 1 := rfl\n"})
    index = LeanIndex.from_folder(base)
    with pytest.raises(ValueError, match="already starts with 'q:'"):
        index.add_folder(_write(tmp_path / "proj", PROJECT), name="q")


def test_the_command_line_reports_and_fails_on_one_layer(tmp_path):
    base = _write(tmp_path / "base", BASE)
    saved = LeanIndex.from_folder(base).save(tmp_path / "base.leanidx")
    project = _write(tmp_path / "proj", PROJECT)
    common = [str(project), "--base", str(saved), "--fail-on", "sorry", "--json"]
    for layer, status in (("proj", 0), ("", 1)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert main([*common, "--layer", layer]) == status
        assert json.loads(out.getvalue())["layer"] == layer
    with contextlib.redirect_stdout(io.StringIO()):
        assert main([*common, "--discover", "--layer", "proj"]) == 0


def test_each_layer_is_read_with_its_own_and_lower_commands(tmp_path):
    from .._index import defined_commands

    index = _layered(tmp_path)
    tool = _write(tmp_path / "tool", {"Tool/Cmd.lean": 'elab "audit_now" : command => pure ()\n'})
    index.add_folder(tool, name="tool")
    assert "audit_now" in index.commands("tool")
    assert "audit_now" not in index.commands("") | index.commands("p")
    with pytest.raises(ValueError, match="layer must be None or one of"):
        index.commands("nope")
    assert defined_commands('@[inline] macro "go" : command => `(#eval 1)') == {"go": 1}
    with pytest.raises(TypeError, match="text must be a str"):
        defined_commands(b"bytes")


@pytest.mark.parametrize(
    ("defined", "label", "error", "message"),
    [
        ({"go": 1}, 3, TypeError, "label must be a str"),
        ([("go", 1)], "A.lean", TypeError, "defined must be a dict"),
        ({"#go": 1}, "A.lean", ValueError, "is not a command keyword"),
        ({"go": 0}, "A.lean", ValueError, "must be an int >= 1"),
    ],
)
def test_learned_commands_are_checked(defined, label, error, message):
    with pytest.raises(error, match=message):
        LeanIndex().learn_commands(defined, label=label)


def test_the_first_definition_learned_is_kept(tmp_path):
    index = LeanIndex()
    index.learn_commands({"go": 3}, label="A.lean")
    index.learn_commands({"go": 1, "stop": 2}, label="B.lean")
    assert index._learned == {"go": ("A.lean", 3), "stop": ("B.lean", 2)}
    assert index.commands() == frozenset({"go", "stop"})
