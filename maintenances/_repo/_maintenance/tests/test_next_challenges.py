"""Tests for ``next_challenges.py``: disjoint, deterministic slots around open work."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
nc = runpy.run_path(str(TOOLS / "next_challenges.py"))
plan, read_notes, main, overlaps = nc["plan"], nc["read_notes"], nc["main"], nc["overlaps"]


def _note(root, rel, *, status="open", kind="bug", area="scikitplot/a", paths=()):
    path = root / "upcoming_changes" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    section = "".join(f"- `{p}`\n" for p in paths)
    path.write_text(
        f'---\ntitle: "{rel}"\nstatus: {status}\nkind: "{kind}"\narea: "{area}"\n---\n\n'
        f"# t\n\n## Affected paths and ownership\n\n{section}\n## Constraints\n\nnone\n",
        encoding="utf-8",
    )


@pytest.fixture
def tree(tmp_path):
    _note(tmp_path, "a/one.md", paths=["scikitplot/a/x.py"])
    _note(tmp_path, "a/two.md", paths=["scikitplot/a/x.py", "scikitplot/a/y.py"])  # clashes with one
    _note(tmp_path, "a/three.md", kind="security", paths=["scikitplot/a/z.py"])
    _note(tmp_path, "b/four.md", area="scikitplot/b")  # no paths: whole area
    _note(tmp_path, "b/five.md", area="scikitplot/b", status="in-progress")
    _note(tmp_path, "c/six.md", kind="other", area="scikitplot/c",
          paths=["scikitplot/c/q.py", "docs/source/apis_reference.py", "`maintenances/c`"])
    return tmp_path


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("a/b", "a/b/c.py", True), ("a/b/", "a/b", True), ("a/b", "a/bc", False), ("a", "b", False)],
)
def test_overlap_is_by_path_segment(a, b, expected):
    assert overlaps(a, b) is expected and overlaps(b, a) is expected


def test_slots_are_disjoint_ranked_and_skip_unstartable(tree):
    report = plan(read_notes(tree), [], count=10)
    got = [s["path"] for s in report["slots"]]
    # security first; two waits on one (same file); in-progress never offered
    assert got == [
        "upcoming_changes/a/three.md",
        "upcoming_changes/a/one.md",
        "upcoming_changes/b/four.md",
        "upcoming_changes/c/six.md",
    ]
    assert report["waiting"] == {"upcoming_changes/a/two.md": "upcoming_changes/a/one.md"}
    assert [s["slot"] for s in report["slots"]] == [1, 2, 3, 4]
    six = report["slots"][3]
    assert six["footprint"] == ["scikitplot/c/q.py"]  # shared and owner paths left out
    assert six["branch"] == "chal/c/six"
    for i, s in enumerate(report["slots"]):
        for t in report["slots"][i + 1 :]:
            assert not any(overlaps(a, b) for a in s["footprint"] for b in t["footprint"])


def test_open_pull_requests_claim_and_busy(tree):
    prs = [
        {"number": 7, "title": "fix", "body": "Closes upcoming_changes/a/three.md", "files": []},
        {"number": 8, "title": "x", "body": "", "files": ["scikitplot/b/m.py"]},
        {"number": 9, "title": "y", "body": "", "files": ["docs/source/apis_reference.py"]},
    ]
    report = plan(read_notes(tree), prs, count=10)
    assert report["claimed"] == {"upcoming_changes/a/three.md": [7]}
    assert report["busy"] == {"upcoming_changes/b/four.md": [8]}  # shared file busies nothing
    assert [s["path"] for s in report["slots"]] == [
        "upcoming_changes/a/one.md",
        "upcoming_changes/c/six.md",
    ]


def test_a_shared_stem_claims_nothing(tmp_path):
    _note(tmp_path, "a/same.md", area="scikitplot/a")
    _note(tmp_path, "b/same.md", area="scikitplot/b")
    _note(tmp_path, "c/unique.md", area="scikitplot/c")
    prs = [{"number": 1, "title": "same and unique", "body": "", "files": []}]
    report = plan(read_notes(tmp_path), prs)
    assert report["claimed"] == {"upcoming_changes/c/unique.md": [1]}


def test_same_inputs_same_answer_and_count_bounds(tree):
    first = plan(read_notes(tree), [], count=2)
    assert first == plan(read_notes(tree), [], count=2)
    assert len(first["slots"]) == 2
    assert first["waiting"]["upcoming_changes/b/four.md"] == "(no slot left)"
    assert plan(read_notes(tree), [], count=3)["fingerprint"] != first["fingerprint"]
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="count"):
            plan([], [], count=bad)


def test_area_and_deferred_filter(tree):
    report = plan(read_notes(tree), [], area="scikitplot/a", deferred=["upcoming_changes/a/three.md"])
    assert [s["path"] for s in report["slots"]] == ["upcoming_changes/a/one.md"]


def test_cli_json_slot_and_errors(tree, capsys):
    prs = tree / "prs.json"
    prs.write_text(json.dumps([]), encoding="utf-8")
    assert main(["--root", str(tree), "--open-prs", str(prs), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["slots"][0]["path"] == "upcoming_changes/a/three.md"
    assert main(["--root", str(tree), "--slot", "2"]) == 0
    out = capsys.readouterr().out
    assert "## Slot 2:" in out and "## Slot 1:" not in out and "No open pull requests" in out
    assert main(["--root", str(tree), "--slot", "99"]) == 1
    capsys.readouterr()
    assert main(["--root", str(tree), "--count", "0"]) == 2
    assert main(["--root", str(tree), "--open-prs", str(tree / "missing.json")]) == 2
