# lean/tests/test__index.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
LeanIndex: evidence on a Lean project, every item with its file and lines
=========================================================================

Statements and the definitions they unfold to, what a proof names and the
path to each flagged dependency, what could be used without a name, the
name-resolution rules, project gaps, reports and the command line. Projects
are written in each test; no test reaches the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import textwrap
import time

import pytest

from .._index import LeanIndex, main


def _lean_reader():
    """Return corpus's LeanReader; skip where only scikit-plots-lean is installed."""
    return pytest.importorskip("scikitplot.corpus._readers._lean").LeanReader


def _project(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


PROJECT = {
    "Proj/Statements.lean": """
        /-- What the project claims. -/
        def MyClaim : Prop := ∀ n : ℕ, Helper n
        def Helper (n : ℕ) : Prop := n = n
    """,
    "Proj/Lemmas.lean": """
        import Proj.Statements
        namespace Proj
        theorem helper_ok (n : ℕ) : Helper n := rfl
        theorem bad : 1 = 2 := sorry
        end Proj
    """,
    "Proj/Main.lean": """
        import Proj.Lemmas
        import Proj.Inst
        import Mathlib.Data.Nat.Basic
        open Proj
        theorem main : MyClaim := fun n => helper_ok n
        theorem main2 (k : ℕ) : MyClaim ∧ k = k :=
          ⟨fun n => by have := bad; exact helper_ok n, rfl⟩
    """,
    "Proj/Inst.lean": """
        instance flaky : Inhabited Nat := sorry
    """,
}


@pytest.fixture
def index(tmp_path):
    return LeanIndex.from_folder(_project(tmp_path, PROJECT))


# ---------------------------------------------------------------------------
# Project summary
# ---------------------------------------------------------------------------


def test_summary_counts_the_project(index):
    summary = index.summary()
    assert (summary["files"], summary["declarations"]) == (4, 7)
    sorry = summary["flags"]["sorry"]
    assert sorry["count"] == 2 and not sorry["cut"]
    assert {i["declaration"] for i in sorry["items"]} == {"flaky", "Proj.bad"}
    assert summary["imports"]["external_roots"] == {"Mathlib": 1}
    assert summary["imports"]["missing_modules"]["count"] == 0


def test_imported_project_modules_that_no_file_provides_are_missing(tmp_path):
    files = dict(PROJECT, **{"Proj/Extra.lean": "import Proj.Missing\ndef e := 1\n"})
    summary = LeanIndex.from_folder(_project(tmp_path, files)).summary()
    assert summary["imports"]["missing_modules"]["items"] == [
        {"module": "Proj.Missing", "imported_by": ["Proj/Extra.lean"]}
    ]


# ---------------------------------------------------------------------------
# Dossiers
# ---------------------------------------------------------------------------


def test_a_clean_proof_still_reports_what_it_cannot_see(index):
    report = index.dossier("main")
    assert report["verdict"] == "no lexical red flags; kernel check pending"
    assert report["target"]["statement"] == "theorem main : MyClaim"
    assert report["dependencies"]["red_flags"]["count"] == 0
    # The flagged instance is never named, but could be used: listed.
    assert [i["declaration"] for i in report["implicit_risks"]["items"]] == ["flaky"]
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses == {
        "statement_definitions": "review",
        "named_dependencies": "pass",
        "implicit_dependencies": "review",
        "duplicate_names": "pass",
        "imported_modules_present": "pass",
        "kernel": "not_checked",
    }
    assert report["kernel_steps"][1].startswith("#print axioms main")


def test_a_flag_is_traced_through_open_and_namespace(index):
    report = index.dossier("main2")
    assert report["verdict"] == "lexical red flags: named_dependencies"
    (flag,) = report["dependencies"]["red_flags"]["items"]
    assert (flag["declaration"], flag["file"], flag["line_start"]) == (
        "Proj.bad", "Proj/Lemmas.lean", 4,
    )
    assert flag["path"] == [
        {
            "declaration": "Proj.bad",
            "file": "Proj/Lemmas.lean",
            "line_start": 4,
            "found_by": "exact",
            "from": "main2",
        }
    ]
    assert report["flags_found"] == ["sorry"]


def test_the_statement_unfolds_to_quoted_definitions(index):
    statement = index.dossier("main2")["statement"]
    definitions = statement["local_definitions"]["items"]
    # MyClaim's chunk starts at its doc comment, line 1.
    assert [(d["declaration"], d["line_start"]) for d in definitions] == [
        ("MyClaim", 1),
        ("Helper", 3),
    ]
    assert definitions[0]["text"].startswith("/-- What the project claims. -/")
    # `k` is bound by the statement; `ℕ` comes from a dependency.
    assert statement["external_names"]["items"] == ["ℕ"]


def test_targets_not_found_or_ambiguous(tmp_path):
    files = {"A.lean": "def dup := 1\n", "B.lean": "def dup := 2\n"}
    index = LeanIndex.from_folder(_project(tmp_path, files))
    assert index.dossier("nothing")["verdict"] == "target not found"
    report = index.dossier("dup")
    assert report["verdict"] == "target ambiguous"
    assert [m["file"] for m in report["matches"]] == ["A.lean", "B.lean"]
    assert index.summary()["duplicate_names"]["count"] == 1


def test_a_dependency_after_exit_is_a_red_flag(tmp_path):
    files = {"A.lean": "theorem t : P := ghost\n#exit\ntheorem ghost : P := trivial\n"}
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("t")
    (flag,) = report["dependencies"]["red_flags"]["items"]
    assert flag["declaration"] == "ghost" and flag["flags"] == ["after_exit"]


def test_syntax_extensions_are_implicit_risks(tmp_path):
    files = {
        "A.lean": """
            macro "trust_me" : tactic => `(tactic| sorry)
            theorem t : True := by trust_me
        """
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("t")
    assert report["dependencies"]["red_flags"]["count"] == 0
    (risk,) = report["implicit_risks"]["items"]
    assert (risk["kind"], risk["flags"]) == ("macro", ["sorry"])


def test_cycles_terminate(tmp_path):
    files = {
        "A.lean": """
            mutual
              theorem a : P := b
              theorem b : P := a
            end
        """
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("a")
    assert report["dependencies"]["declarations"] == 1


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def _depends(tmp_path, files, target, **options):
    index = LeanIndex.from_folder(_project(tmp_path, files), **options)
    report = index.dossier(target)
    reached = report["dependencies"]
    paths = [f["path"] for f in reached["red_flags"]["items"]]
    return reached, {hop["declaration"]: hop["found_by"] for p in paths for hop in p}


@pytest.mark.parametrize(
    ("files", "found"),
    [
        # the enclosing namespace, innermost first
        ({"A.lean": "namespace N\ntheorem x : P := sorry\ntheorem t : P := x\nend N\n"},
         {"N.x": "exact"}),
        # _root_ skips the namespaces
        ({"A.lean": "theorem x : P := sorry\nnamespace N\ntheorem t : P := _root_.x\nend N\n"},
         {"x": "exact"}),
        # open in the same scope
        ({"A.lean": "namespace M\ntheorem x : P := sorry\nend M\nopen M\ntheorem t : P := x\n"},
         {"M.x": "exact"}),
        # open ... in applies to its command
        ({"A.lean": "namespace M\ntheorem x : P := sorry\nend M\nopen M in\ntheorem t : P := x\n"},
         {"M.x": "exact"}),
        # a name Lean derives: Foo.mk is Foo's
        ({"A.lean": "structure Foo where\n  v : Nat := sorry\ntheorem t : P := Foo.mk 1\n"},
         {"Foo": "prefix"}),
        # dot notation on a hypothesis: the one declaration named myLemma
        ({"A.lean": "theorem MyProp.myLemma : P := sorry\ntheorem t (h : MyProp) : P := h.myLemma\n"},
         {"MyProp.myLemma": "suffix"}),
    ],
)
def test_resolution_rules(tmp_path, files, found):
    _reached, hops = _depends(tmp_path, files, "t")
    assert hops == found


def test_an_open_ends_with_its_scope(tmp_path):
    files = {
        "A.lean": """
            namespace M
            theorem x : P := sorry
            end M
            section
            open M
            end
            theorem t : P := x
        """
    }
    reached, _hops = _depends(tmp_path, files, "t")
    assert reached["red_flags"]["count"] == 0


def test_a_private_declaration_is_seen_from_its_own_file_only(tmp_path):
    files = {
        "A.lean": "private theorem x : P := sorry\ntheorem here : P := x\n",
        "B.lean": "theorem t : P := x\n",
    }
    index = LeanIndex.from_folder(_project(tmp_path, files))
    assert index.dossier("here")["dependencies"]["red_flags"]["count"] == 1
    assert index.dossier("t")["dependencies"]["red_flags"]["count"] == 0


def test_suffix_links_only_a_unique_short_name(tmp_path):
    files = {
        "A.lean": """
            theorem P.go : P := sorry
            theorem Q.go : Q := sorry
            theorem R.one : R := sorry
            theorem t (h : P) (k : R) : P := h.go.trans k.one
        """
    }
    _reached, hops = _depends(tmp_path, files, "t")
    assert hops == {"R.one": "suffix"}  # `go` names two declarations
    _reached, hops = _depends(tmp_path, files, "t", suffix="none")
    assert hops == {}


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------


def test_hidden_folders_and_unreadable_files(tmp_path):
    files = dict(PROJECT)
    files[".lake/packages/dep/Dep.lean"] = "theorem dep : P := sorry\n"
    files["Deep.lean"] = "namespace N\n" * 5 + "def d := 1\n"
    root = _project(tmp_path, files)
    summary = LeanIndex.from_folder(root, max_scope_depth=3).summary()
    assert summary["files"] == 5  # .lake left out
    assert summary["skipped_files"][0]["file"] == "Deep.lean"
    assert "max_scope_depth=3" in summary["skipped_files"][0]["reason"]
    with_hidden = LeanIndex.from_folder(root, include_hidden=True).summary()
    assert with_hidden["files"] == 6


def test_chunks_of_a_reader_can_be_indexed_without_files(tmp_path):
    root = _project(tmp_path, PROJECT)
    index = LeanIndex()
    for path in sorted(root.rglob("*.lean")):
        reader = _lean_reader()(input_path=path, references=True, lines_per_chunk=1)
        index.add_chunks(reader.get_raw_chunks(), label=path.relative_to(root).as_posix())
    report = index.dossier("main2")
    assert report["target"]["statement"] is None  # no file to quote
    assert report["target"]["line_end"] == 7  # the two parts joined
    assert [f["declaration"] for f in report["dependencies"]["red_flags"]["items"]] == [
        "Proj.bad"
    ]


def test_reports_are_json_and_deterministic(tmp_path):
    root = _project(tmp_path, PROJECT)
    first = LeanIndex.from_folder(root)
    second = LeanIndex.from_folder(root)
    for name in ("main", "main2"):
        a, b = first.dossier(name), second.dossier(name)
        assert json.loads(json.dumps(a, ensure_ascii=False)) == a == b
    assert first.summary() == second.summary()
    text = LeanIndex.render(first.dossier("main2"))
    assert "`main2` → `Proj.bad`" in text and "`Proj/Lemmas.lean:4`" in text
    assert "# Lean project summary" in LeanIndex.render(first.summary())


def test_lists_say_when_they_are_cut(tmp_path):
    files = {"A.lean": "".join(f"theorem s{i} : P := sorry\n" for i in range(5))}
    summary = LeanIndex.from_folder(_project(tmp_path, files)).summary(max_items=2)
    listing = summary["flags"]["sorry"]
    assert (listing["count"], len(listing["items"]), listing["cut"]) == (5, 2, True)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"dialect": "lean5"}, ValueError, "dialect must be"),
        ({"suffix": "any"}, ValueError, "suffix must be"),
        ({"max_scope_depth": 0}, ValueError, "max_scope_depth must be"),
        ({"max_file_bytes": True}, ValueError, "max_file_bytes must be"),
        ({"commands": "cmd"}, TypeError, "commands must be"),
    ],
)
def test_invalid_options_are_refused(options, error, message):
    with pytest.raises(error, match=message):
        LeanIndex(**options)


def test_a_missing_folder_and_bad_limits_are_refused(tmp_path, index):
    with pytest.raises(ValueError, match="is not a folder"):
        LeanIndex.from_folder(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="max_items must be"):
        index.summary(max_items=0)


def test_resolution_time_grows_linearly(tmp_path):
    # Each theorem names the one before: the dossier of the last reaches all.
    def build(n, folder):
        lines = ["theorem t0 : P := sorry"]
        lines += [f"theorem t{i} : P := t{i - 1}" for i in range(1, n)]
        _project(folder, {"A.lean": "\n".join(lines) + "\n"})
        start = time.perf_counter()
        report = LeanIndex.from_folder(folder).dossier(f"t{n - 1}")
        assert report["dependencies"]["declarations"] == n - 1
        return time.perf_counter() - start

    small = min(build(1500, tmp_path / f"s{i}") for i in range(2))
    large = min(build(6000, tmp_path / f"l{i}") for i in range(2))
    assert large / max(small, 1e-4) < 9, f"x{large / small:.1f} for 4x the declarations"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main(argv)
    return status, out.getvalue()


def test_the_command_line(tmp_path):
    root = str(_project(tmp_path, PROJECT))
    status, text = _run([root])
    assert status == 0 and text.startswith("# Lean project summary")
    status, text = _run([root, "--target", "main2", "--json"])
    assert status == 0 and json.loads(text)[0]["flags_found"] == ["sorry"]
    assert _run([root, "--target", "main2", "--fail-on", "sorry"])[0] == 1
    assert _run([root, "--target", "main", "--fail-on", "sorry"])[0] == 1  # implicit
    assert _run([root, "--target", "main", "--fail-on", "axiom"])[0] == 0
    assert _run([root, "--fail-on", "sorry"])[0] == 1


# ---------------------------------------------------------------------------
# Building: order, repetition, unreadable and linked files
# ---------------------------------------------------------------------------


def _local_definitions(report):
    return [d["declaration"] for d in report["statement"]["local_definitions"]["items"]]


def test_a_query_does_not_spend_the_names_a_later_query_needs(tmp_path):
    # Without files on disk the statement's names come from the index; a
    # first query (discover resolves every edge) must leave them in place.
    root = _project(tmp_path, PROJECT)
    index = LeanIndex()
    for path in sorted(root.rglob("*.lean")):
        reader = _lean_reader()(input_path=path, references=True)
        index.add_chunks(reader.get_raw_chunks(), label=path.relative_to(root).as_posix())
    before = _local_definitions(index.dossier("main"))
    index.discover()
    assert _local_definitions(index.dossier("main")) == before
    assert "MyClaim" in before


def test_files_added_after_a_query_are_linked(tmp_path):
    root = _project(tmp_path, PROJECT)
    names = ["Proj/Main.lean", "Proj/Inst.lean", "Proj/Lemmas.lean", "Proj/Statements.lean"]
    grown = LeanIndex()
    for name in names:
        grown.add_file(root / name, label=name)
        grown.dossier("main2")  # a query between additions
    whole = LeanIndex.from_folder(root)
    report = json.dumps(grown.dossier("main2"), sort_keys=True)
    assert report == json.dumps(whole.dossier("main2"), sort_keys=True)


def test_a_file_is_indexed_once(tmp_path):
    root = _project(tmp_path, {"A.lean": "def a := 1\n"})
    index = LeanIndex()
    index.add_file(root / "A.lean", label="A.lean")
    with pytest.raises(ValueError, match="already indexed"):
        index.add_file(root / "A.lean", label="A.lean")
    assert index.summary()["duplicate_names"]["count"] == 0


def test_a_skipped_file_adds_nothing_to_the_counts(tmp_path):
    files = {"A.lean": "def a := 1\n", "Deep.lean": "def early := 1\n" + "namespace N\n" * 5}
    summary = LeanIndex.from_folder(_project(tmp_path, files), max_scope_depth=3).summary()
    assert summary["kinds"] == {"def": 1}
    assert summary["declarations"] == 1


def test_a_missing_file_is_skipped_not_raised(tmp_path):
    index = LeanIndex()
    index.add_file(tmp_path / "Gone.lean", label="Gone.lean")
    skipped = index.summary()["skipped_files"]
    assert skipped[0]["file"] == "Gone.lean" and skipped[0]["reason"].startswith("unreadable")


def _symlink(link, target, *, folder=False):
    try:
        link.symlink_to(target, target_is_directory=folder)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")


def test_links_never_lead_outside_the_folder(tmp_path):
    root = _project(tmp_path / "proj", {"A.lean": "def a := 1\n"})
    outside = _project(tmp_path / "outside", {"Secret.lean": "def secret := 1\n", "D/In.lean": "def d := 1\n"})
    _symlink(root / "Out.lean", outside / "Secret.lean")
    _symlink(root / "Dir", outside / "D", folder=True)
    _symlink(root / "Again.lean", root / "A.lean")
    _symlink(root / "Broken.lean", root / "nothing.lean")
    index = LeanIndex.from_folder(root)
    assert sorted(index._labels) == ["A.lean", "Again.lean"]  # a link inside is kept
    reasons = dict((s["file"], s["reason"]) for s in index.summary()["skipped_files"])
    assert reasons == {
        "Broken.lean": "broken symbolic link",
        "Out.lean": "symbolic link to a file outside the folder",
    }
    assert index.find("secret") == [] and index.find("d") == []


def test_fail_on_refuses_an_unknown_flag(tmp_path):
    root = str(_project(tmp_path, PROJECT))
    with pytest.raises(SystemExit) as exit_info, contextlib.redirect_stderr(io.StringIO()):
        main([root, "--fail-on", "no_such_flag"])
    assert exit_info.value.code == 2


# ---------------------------------------------------------------------------
# Visibility: a name links only into files its own file can see
# ---------------------------------------------------------------------------


VISIBLE = {
    "V/Core.lean": "theorem helper : True := sorry\n",
    "V/Mid.lean": "import V.Core\ntheorem mid : True := helper\n",
    "V/Main.lean": "import V.Mid\ntheorem main : True := helper\n",
    "Scratch/Other.lean": "theorem helper' : True := sorry\ntheorem stray : True := sorry\n",
    "Scratch/Use.lean": "theorem uses_stray : True := stray\n",
}


def _named(report):
    """The flagged declarations a proof names, directly or through others."""
    return {d["declaration"] for d in report["dependencies"]["red_flags"]["items"]}


def test_a_name_in_a_file_not_imported_is_not_a_dependency(tmp_path):
    files = dict(VISIBLE)
    files["V/Main.lean"] = "import V.Mid\ntheorem main : True := stray\n"
    index = LeanIndex.from_folder(_project(tmp_path, files))
    report = index.dossier("main")
    assert report["visibility"] == "imports"
    assert _named(report) == set() and report["flags_found"] == []


def test_visibility_all_links_by_text_alone(tmp_path):
    files = dict(VISIBLE)
    files["V/Main.lean"] = "import V.Mid\ntheorem main : True := stray\n"
    report = LeanIndex.from_folder(_project(tmp_path, files), visibility="all").dossier("main")
    assert "stray" in _named(report) and report["flags_found"] == ["sorry"]


def test_imports_are_followed_through_other_files(tmp_path):
    report = LeanIndex.from_folder(_project(tmp_path, VISIBLE)).dossier("main")
    assert "helper" in _named(report)  # Main -> Mid -> Core


def test_an_index_of_a_subfolder_hides_nothing(tmp_path):
    # Indexed from V/, `import V.Mid` matches no label; the file's imports
    # are unknown, so it sees every declaration rather than none.
    files = dict(VISIBLE)
    files["V/Main.lean"] = "import V.Mid\ntheorem main : True := stray\n"
    root = _project(tmp_path, files)
    _project(root / "V", {"Stray.lean": "theorem stray : True := sorry\n"})
    report = LeanIndex.from_folder(root / "V").dossier("main")
    assert "stray" in _named(report) and report["flags_found"] == ["sorry"]


def test_a_missing_project_module_hides_nothing(tmp_path):
    files = {
        "W/Main.lean": "import W.Gone\ntheorem main : True := stray\n",
        "W/Other.lean": "theorem stray : True := sorry\n",
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("main")
    assert report["flags_found"] == ["sorry"]
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["imported_modules_present"] == "fail"


def test_an_import_cycle_terminates(tmp_path):
    files = {
        "C/A.lean": "import C.B\ntheorem a : True := b\n",
        "C/B.lean": "import C.A\ntheorem b : True := sorry\n",
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("a")
    assert report["flags_found"] == ["sorry"]


def test_visibility_is_validated_and_on_the_command_line(tmp_path):
    with pytest.raises(ValueError, match="visibility"):
        LeanIndex(visibility="some")
    files = dict(VISIBLE)
    files["V/Main.lean"] = "import V.Mid\ntheorem main : True := stray\n"
    root = str(_project(tmp_path, files))
    assert _run([root, "--target", "main", "--fail-on", "sorry"])[0] == 0
    assert _run([root, "--target", "main", "--fail-on", "sorry", "--visibility", "all"])[0] == 1


# ---------------------------------------------------------------------------
# Modules: named from the project's source root, never by a tail of a path
# ---------------------------------------------------------------------------


def test_a_core_import_is_not_a_project_file_of_the_same_tail(tmp_path):
    files = {
        "M/Lean/Elab/Term.lean": "theorem shadow : True := sorry\n",
        "M/Base.lean": "theorem base : True := trivial\n",
        "M/Use.lean": "import Lean.Elab.Term\nimport M.Base\ntheorem use : True := shadow\n",
        "M/Top.lean": "import M.Use\nimport M.Base\ntheorem top : True := use\n",
    }
    index = LeanIndex.from_folder(_project(tmp_path, files))
    imports = index.summary()["imports"]
    assert imports["external_roots"] == {"Lean": 1}
    assert imports["missing_modules"]["count"] == 0
    assert index.dossier("use")["flags_found"] == []  # Term.lean is not imported


def test_a_source_folder_is_found_as_the_root(tmp_path):
    files = {
        "src/A/B.lean": "theorem b : True := sorry\n",
        "src/A/C.lean": "import A.B\ntheorem c : True := b\n",
    }
    index = LeanIndex.from_folder(_project(tmp_path, files))
    assert index.summary()["imports"]["missing_modules"]["count"] == 0
    assert index.dossier("c")["flags_found"] == ["sorry"]


def test_the_root_resolving_the_most_imports_wins(tmp_path):
    # Depth 0 resolves `X.Y` once; depth 1 resolves `Y` and `Z` twice.
    files = {
        "X/Y.lean": "theorem y : True := sorry\n",
        "X/Z.lean": "import Y\ntheorem z : True := y\n",
        "X/W.lean": "import Z\nimport Y\ntheorem w : True := z\n",
        "Q.lean": "import X.Y\ntheorem q : True := trivial\n",
    }
    index = LeanIndex.from_folder(_project(tmp_path, files))
    assert set(index._module_files()) == {"Y", "Z", "W"}
    assert index.dossier("w")["flags_found"] == ["sorry"]


def test_an_ambiguous_root_can_be_set(tmp_path):
    # One import, which could name the dependency or the file: detection
    # takes the file; source_root="" starts module names at the index root,
    # and the import is the dependency's.
    files = {
        "M/Lean/Elab/Term.lean": "theorem shadow : True := sorry\n",
        "M/Use.lean": "import Lean.Elab.Term\ntheorem use : True := shadow\n",
    }
    root = _project(tmp_path, files)
    assert LeanIndex.from_folder(root).dossier("use")["flags_found"] == ["sorry"]
    index = LeanIndex.from_folder(root, source_root="")
    assert index.dossier("use")["flags_found"] == []
    assert index.summary()["imports"]["external_roots"] == {"Lean": 1}
    status, _ = _run([str(root), "--target", "use", "--source-root", "", "--fail-on", "sorry"])
    assert status == 0


@pytest.mark.parametrize("bad", ["/abs", "a/../b", "a\\b", 3])
def test_source_root_is_validated(bad):
    with pytest.raises(ValueError, match="source_root"):
        LeanIndex(source_root=bad)


def test_implicit_risks_are_those_in_the_targets_environment(tmp_path):
    files = {
        "R/Inst.lean": "instance seen : Inhabited Nat := sorry\n",
        "R/Far.lean": "instance unseen : Inhabited Nat := sorry\n",
        "R/Main.lean": (
            "import R.Inst\n"
            "theorem main : True := trivial\n"
            "instance later : Inhabited Nat := sorry\n"
        ),
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).dossier("main")
    # Far.lean is not imported; `later` comes after the theorem.
    assert [i["declaration"] for i in report["implicit_risks"]["items"]] == ["seen"]


def test_the_package_imports_no_corpus_code():
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = (
        "import sys, scikitplot.lean, scikitplot.lean._discovery\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('scikitplot.corpus'))\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_subfolder_index_says_its_imports_are_missing(tmp_path):
    # The import a file cannot resolve, and the check that reports it, agree.
    files = {
        "S/Core.lean": "theorem helper : True := trivial\n",
        "S/Old.lean": "theorem stale : True := sorry\n",
        "S/Main.lean": "import S.Core\ntheorem main : True := stale\n",
    }
    root = _project(tmp_path, files)
    whole = LeanIndex.from_folder(root).dossier("main")
    assert whole["flags_found"] == []  # Old.lean is not imported
    part = LeanIndex.from_folder(root / "S")
    report = part.dossier("main")
    assert report["flags_found"] == ["sorry"]  # imports unknown: hides nothing
    checks = {c["check"]: c["status"] for c in report["checks"]}
    assert checks["imported_modules_present"] == "fail"
    missing = part.summary()["imports"]["missing_modules"]["items"]
    assert [m["module"] for m in missing] == ["S.Core"]
