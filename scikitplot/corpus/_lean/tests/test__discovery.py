# corpus/_lean/tests/test__discovery.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
LeanIndex discovery and helper documents: rules, patterns, claims, anomalies
============================================================================

Each section of :meth:`LeanIndex.discover` is checked against a project
written in the test, then :meth:`LeanIndex.export` and the detection of
code that adds declarations at build time. No test reaches the network.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from .._discovery import _safe
from .._evidence import LeanIndex


def _project(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return root


PROJECT = {
    "P/Defs.lean": """
        def Good (n : ℕ) : Prop := n = n
        def Claim : Prop := ∀ n : ℕ, Good n
        def size (n : ℕ) : ℕ := n
        axiom choice_ax : Good 0
    """,
    "P/Lemmas.lean": """
        import P.Defs
        theorem good_one (a : ℕ) : a + 1 = a + 1 := rfl
        theorem good_two (b : ℕ) : b + 2 = b + 2 := rfl
        theorem good_three (c : ℕ) : c + 3 = c + 3 := rfl
        theorem step_1 : Good 1 := rfl
        theorem step_2 : Good 2 := rfl
        theorem step_3 : Good 3 := rfl
        theorem hole : Good 5 := sorry
        theorem unused_big (n : ℕ) : size n = n := by
          rfl
    """,
    "P/Uses.lean": """
        import P.Lemmas
        theorem use_a : Good 1 := step_1
        theorem use_b : Good 2 := step_2
    """,
    "P/Top.lean": """
        import P.Uses
        theorem final : Good 1 ∧ Good 2 := ⟨use_a, step_2⟩
    """,
    "Q/Dup.lean": """
        theorem twin (x : ℕ) : x = x := rfl
    """,
    "Q/Dup2.lean": """
        theorem twin (y : ℕ) : y = y := rfl
    """,
    "Q/Clash.lean": """
        def mean : ℕ := 1
    """,
    "Q/Clash2.lean": """
        def mean : Bool := true
    """,
}


@pytest.fixture
def report(tmp_path):
    return LeanIndex.from_folder(_project(tmp_path, PROJECT)).discover()


def _names(listing):
    return [item["declaration"] for item in listing["items"]]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_report_identity(report):
    assert report["report"] == "discovery"
    assert set(report) >= {"rules", "patterns", "claims", "anomalies", "headline_results"}


def test_rules_rank_by_files_then_declarations(report):
    theorems = report["rules"]["load_bearing_theorems"]
    names = _names(theorems)
    # step_2 is named from two files (Uses, Top); step_1 from one.
    assert names.index("step_2") < names.index("step_1")
    first = theorems["items"][0]
    assert first["named_from_files"] >= 1 and first["file"].endswith(".lean")
    assert "Good" in _names(report["rules"]["central_concepts"])


def test_statement_shapes_ignore_bound_names_and_numerals(report):
    shapes = report["patterns"]["statement_shapes"]
    families = [sorted(_names(f["members"])) for f in shapes["items"]]
    assert ["good_one", "good_three", "good_two"] in families
    family = next(f for f in shapes["items"] if f["instances"] == 3 and "good_one" in _names(f["members"]))
    assert family["files"] == 1 and "good_one" in family["exemplar"]


def test_min_instances_drops_small_families(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, PROJECT))
    shapes = index.discover(min_instances=4)["patterns"]["statement_shapes"]
    assert all(f["instances"] >= 4 for f in shapes["items"])


def test_numbered_series(report):
    series = report["patterns"]["numbered_series"]["items"]
    steps = next(s for s in series if s["series"] == "step")
    assert steps["numbers"] == [1, 2, 3]


def test_claims_and_assumptions(report):
    claims = report["claims"]
    assert "Claim" in _names(claims["claim_definitions"])
    assert "Good" in _names(claims["claim_definitions"])  # a Prop-valued definition
    assert "size" not in _names(claims["claim_definitions"])
    assert _names(claims["axioms"]) == ["choice_ax"]
    assert "hole" in _names(claims["unproven"])


def test_parallel_declarations(report):
    groups = {g["declaration"]: g for g in report["anomalies"]["parallel_declarations"]["items"]}
    assert groups["twin"]["same_statement"] is True
    assert groups["mean"]["same_statement"] is False
    assert groups["twin"]["one_imports_another"] is False
    assert groups["twin"]["at"]["count"] == 2


def test_headline_and_isolated_results(report):
    headline = _names(report["headline_results"])
    assert "final" in headline and "use_a" not in headline
    isolated = _names(report["anomalies"]["isolated_results"])
    assert "unused_big" in isolated and "final" not in isolated


def test_without_patterns_the_reread_is_skipped(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, PROJECT))
    report = index.discover(patterns=False)
    assert report["patterns"]["statement_shapes"]["count"] == 0
    assert report["claims"]["claim_definitions"]["count"] == 0
    groups = {g["declaration"]: g for g in report["anomalies"]["parallel_declarations"]["items"]}
    assert groups["twin"]["same_statement"] is False  # unknown without shapes
    assert report["patterns"]["numbered_series"]["count"] >= 1  # names only


def test_top_cuts_lists_and_keeps_counts(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, PROJECT))
    report = index.discover(top=1)
    isolated = report["anomalies"]["isolated_results"]
    assert len(isolated["items"]) <= 1
    assert isolated["count"] >= len(isolated["items"])
    assert isolated["cut"] == (isolated["count"] > 1)


@pytest.mark.parametrize("option", ["top", "min_instances"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_options_are_refused(tmp_path, option, value):
    index = LeanIndex.from_folder(_project(tmp_path, {"A.lean": "theorem t : 1 = 1 := rfl\n"}))
    with pytest.raises((TypeError, ValueError)):
        index.discover(**{option: value})


def test_an_empty_project(tmp_path):
    report = LeanIndex.from_folder(tmp_path).discover()
    assert report["headline_results"]["count"] == 0
    assert report["patterns"]["statement_shapes"]["count"] == 0


def test_discovery_is_deterministic(tmp_path):
    root = _project(tmp_path, PROJECT)
    first = json.dumps(LeanIndex.from_folder(root).discover(), sort_keys=True)
    second = json.dumps(LeanIndex.from_folder(root).discover(), sort_keys=True)
    assert first == second


def test_render(report):
    text = LeanIndex.render(report)
    for heading in ("# Rules", "# Patterns", "# Claims", "# Anomalies"):
        assert heading in text
    assert "`choice_ax`" in text


# ---------------------------------------------------------------------------
# Code that adds declarations at build time
# ---------------------------------------------------------------------------


GENERATORS = {
    "G/Meta.lean": """
        open Lean Elab Command
        def mkCopies : CommandElabM Unit := do
          liftCoreM (addDecl default)
        elab "copy_all" : command => mkCopies
    """,
    "G/Use.lean": """
        import G.Meta
        copy_all
        run_cmd mkCopies
        theorem plain : 1 = 1 := rfl
    """,
}


def test_generators_through_a_learned_command_and_run_cmd(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, GENERATORS))
    found = index.discover()["anomalies"]["generators"]["items"]
    kinds = sorted(item["invocation"]["kind"] for item in found if item["invocation"]["file"] == "G/Use.lean")
    assert kinds == ["copy_all", "run_cmd"]
    for item in found:
        assert item["adds_declarations_at"]["declaration"] == "mkCopies"


def test_no_generators_without_learning(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, GENERATORS), learn_commands=False)
    found = index.discover()["anomalies"]["generators"]["items"]
    assert [item["invocation"]["kind"] for item in found] == ["run_cmd"]


def test_no_generators_when_nothing_adds_declarations(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path, PROJECT))
    assert index.discover()["anomalies"]["generators"]["count"] == 0


# ---------------------------------------------------------------------------
# Helper documents
# ---------------------------------------------------------------------------


def test_export_writes_every_document(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path / "src", PROJECT))
    out = tmp_path / "helpers"
    written = index.export(out, targets=["use_a"])
    for name in ("README.md", "summary.json", "summary.md", "discovery.json", "rules.md",
                 "patterns.md", "claims.md", "anomalies.md", "records.jsonl"):
        assert name in written and (out / name).is_file()
    dossiers = [w for w in written if w.startswith("dossiers/")]
    assert any("final" in d for d in dossiers) and any("use_a" in d for d in dossiers)
    assert written[-1] == "records.jsonl"
    assert json.loads((out / "discovery.json").read_text(encoding="utf-8"))["report"] == "discovery"


def test_records_carry_their_place(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path / "src", PROJECT))
    out = tmp_path / "helpers"
    index.export(out, targets=["final"])
    lines = (out / "records.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    sections = {r["section"] for r in records}
    assert {"rule", "pattern", "series", "claim", "assumption", "open", "anomaly",
            "headline", "dossier"} <= sections
    for record in records:
        assert record["file"].endswith(".lean") and record["line_start"] >= 1
    dossier = next(r for r in records if r["section"] == "dossier")
    assert dossier["declaration"] == "final" and dossier["evidence"]["verdict"]


def test_export_twice_is_identical(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path / "src", PROJECT))
    a, b = tmp_path / "1" / "out", tmp_path / "2" / "out"  # README names the folder
    assert index.export(a) == index.export(b)
    for name in index.export(a):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_an_unknown_target_still_gets_a_dossier(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path / "src", PROJECT))
    written = index.export(tmp_path / "out", targets=["no_such_name"])
    assert any("no_such_name" in w for w in written)
    records = (tmp_path / "out" / "records.jsonl").read_text(encoding="utf-8")
    assert "no_such_name" not in records  # no target, no dossier record


def test_export_refuses_a_file_as_folder(tmp_path):
    index = LeanIndex.from_folder(_project(tmp_path / "src", PROJECT))
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a folder"):
        index.export(blocker)


@pytest.mark.parametrize(
    ("name", "kept"),
    [("Nat.add_comm", "Nat.add_comm"), ("../../etc", "_.._etc"), ("α β", "___"), ("", "item")],
)
def test_safe_names_stay_in_the_folder(name, kept):
    safe = _safe(name)
    assert "/" not in safe and "\\" not in safe and not safe.startswith(".")
    assert safe.rsplit("-", 1)[0] == kept


def test_safe_names_do_not_collide():
    assert _safe("a/b") != _safe("a_b")


def test_a_use_counts_only_into_files_the_user_imports(tmp_path):
    # `s` in each proof is a bound name; the top-level `s` in an unimported
    # test file must not become the project's most used declaration.
    files = {
        "T/Scratch.lean": "def s : Nat := 0\n",
        "A/Base.lean": "theorem keep (n : Nat) : n = n := rfl\n",
        "A/One.lean": "import A.Base\ntheorem one (s : Nat) : s = s := keep s\n",
        "A/Two.lean": "import A.Base\ntheorem two (s : Nat) : s = s := keep s\n",
        "A/Three.lean": "import A.One\ntheorem three (s : Nat) : s = s := one s\n",
    }
    report = LeanIndex.from_folder(_project(tmp_path, files)).discover()
    rules = report["rules"]
    assert "s" not in _names(rules["central_concepts"])
    keep = next(i for i in rules["load_bearing_theorems"]["items"] if i["declaration"] == "keep")
    assert keep["named_from_files"] == 2  # One and Two; Three only names `one`
