"""Tests for the helper scripts of the developers' pull-request and CI skills.

Notes
-----
**Developer notes.** The scripts live in
``plugins/scikitplot-dev/skills/<skill>/scripts/`` and call ``gh`` only in
functions that fetch; every decision is a pure function, tested here without
a network. Each test of a rule also breaks the input the rule guards against,
so a weakened rule fails a test rather than passing silently.

Run with ``python -B maintenances/_repo/_maintenance/tests/test_workflow_skill_scripts.py``
or with pytest.
"""

from __future__ import annotations

import ast
import os
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SKILLS = ROOT / "plugins/scikitplot-dev/skills"
SCRIPTS = sorted(SKILLS.glob("*/scripts/*.py"))


def _load(skill: str, name: str) -> dict:
    return runpy.run_path(str(SKILLS / skill / "scripts" / name), run_name="workflow_script")


review = _load("scikitplot-pr-pending-review", "post_pending_review.py")
worklist = _load("scikitplot-pr-impact-brief", "worklist.py")
context = _load("scikitplot-pr-impact-brief", "pr_context.py")
ci = _load("scikitplot-ci-failure-forecast", "ci_runs.py")
paths = _load("scikitplot-pr-walkthrough", "pr_paths.py")

PATCH = "\n".join([
    "@@ -10,3 +10,4 @@ def f():",
    " a",
    "-b",
    "+b2",
    "+b3",
    " c",
    "@@ -40 +41 @@",
    "-x",
    "+y",
])


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_each_script_parses_as_python_3_8(script):
    ast.parse(script.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_every_skill_with_a_script_names_it():
    for script in SCRIPTS:
        text = (script.parent.parent / "SKILL.md").read_text(encoding="utf-8")
        assert f"scripts/{script.name}" in text, script


class TestPendingReview:
    def test_hunk_lines_maps_both_sides(self):
        index = review["hunk_lines"]("a.py", PATCH)
        assert {index[("a.py", "RIGHT", n)] for n in (10, 11, 12, 13)} == {0}
        assert {index[("a.py", "LEFT", n)] for n in (10, 11, 12)} == {0}
        assert ("a.py", "RIGHT", 14) not in index
        assert ("a.py", "LEFT", 13) not in index
        assert index[("a.py", "LEFT", 40)] == index[("a.py", "RIGHT", 41)] == 1

    def test_check_refuses_lines_outside_the_diff_and_ranges_across_hunks(self):
        index = review["hunk_lines"]("a.py", PATCH)
        norm = review["normalize"]
        assert review["check"](norm({"path": "a.py", "line": 12, "body": "x"}), index) is None
        assert review["check"](norm({"path": "a.py", "line": 20, "body": "x"}), index)
        assert review["check"](norm({"path": "b.py", "line": 12, "body": "x"}), index)
        assert review["check"](norm({"path": "a.py", "start_line": 10, "line": 13, "body": "x"}), index) is None
        assert "one diff hunk" in review["check"](
            norm({"path": "a.py", "start_line": 12, "line": 41, "body": "x"}), index
        )

    @pytest.mark.parametrize("raw", [
        {"path": "a.py", "line": 0, "body": "x"},
        {"path": "a.py", "line": True, "body": "x"},
        {"path": "a.py", "line": 3, "body": "  "},
        {"path": "", "line": 3, "body": "x"},
        {"path": "a.py", "line": 3, "body": "x", "side": "right"},
        {"path": "a.py", "line": 3, "body": "x", "start_line": 4},
    ])
    def test_normalize_refuses_bad_comments(self, raw):
        with pytest.raises(ValueError):
            review["normalize"](raw)

    def test_a_one_line_range_is_a_line(self):
        out = review["normalize"]({"path": "a.py", "start_line": 3, "line": 3, "body": "x"})
        assert "start_line" not in out and out["side"] == "RIGHT"

    def test_key_ignores_surrounding_space_only(self):
        a = review["key"]({"path": "a.py", "line": 3, "body": "text\n"})
        assert a == review["key"]({"path": "a.py", "line": 3, "side": "RIGHT", "body": " text"})
        assert a != review["key"]({"path": "a.py", "line": 3, "side": "LEFT", "body": "text"})
        assert a != review["key"]({"path": "a.py", "line": 3, "body": "Text"})


def _pr(number, **kw):
    pr = {"number": number, "draft": False, "files": ["scikitplot/x.py"], "labels": [],
          "maintainer_engaged": False}
    pr.update(kw)
    return pr


class TestWorklist:
    def test_each_stage_drops_its_own_pull_request(self):
        pulls = [
            _pr(1),
            _pr(2, draft=True),
            _pr(3, files=["docs/a.rst"]),
            _pr(4, labels=["Needs Triage"]),
            _pr(5, maintainer_engaged=True),
            _pr(6),
        ]
        stages = worklist["worklist"](pulls, ["scikitplot/"], {6}, False)
        assert [len(v) for v in stages.values()] == [6, 5, 4, 3, 2, 1]
        assert [p["number"] for p in stages["not briefed"]] == [1]
        again = worklist["worklist"](pulls, ["scikitplot/"], {6}, True)
        assert [p["number"] for p in again["not briefed"]] == [1, 6]

    def test_a_label_that_only_mentions_needs_is_kept(self):
        stages = worklist["worklist"]([_pr(1, labels=["Docs: Needs review after merge"])], ["scikitplot/"], set(), False)
        assert len(stages["not briefed"]) == 1


class TestPrContext:
    def test_linked_issues(self):
        text = "Fixes #12, closes #3 and resolved #12. See #99; issue #7 is unrelated."
        assert context["linked_issues"](text) == [12, 3, 99]
        assert context["linked_issues"]("https://github.com/o/r/issues/5") == [5]

    def test_public_api_follows_all_when_declared(self):
        source = (
            "__all__ = ['f', 'C']\n"
            "def f(a, b=1, *args, c, **kw): pass\n"
            "def g(): pass\n"
            "class C:\n"
            "    def __init__(self, x): pass\n"
            "    def m(self): pass\n"
            "    def _h(self): pass\n"
        )
        api = context["public_api"](source)
        assert set(api) == {"f", "C", "C.__init__", "C.m"}
        assert api["C"] == "class"
        without = context["public_api"](source.split("\n", 1)[1])
        assert "g" in without and "_h" not in without

    def test_api_delta(self):
        before = context["public_api"]("def f(a): pass\ndef g(): pass\n")
        after = context["public_api"]("def f(a, b=None): pass\ndef h(): pass\n")
        assert context["api_delta"](before, after) == {"added": ["h"], "removed": ["g"], "changed": ["f"]}
        assert context["api_delta"](before, before) == {"added": [], "removed": [], "changed": []}
        assert context["public_api"]("") == {}


class TestCiRuns:
    FORECAST = {"jobs": {"a": "carries over", "b": "breaks", "c": "passes", "d": "recovers", "e": "passes", "g": "breaks"}}

    def test_score(self):
        actual = {"a": "failure", "b": "success", "c": "cancelled", "d": "success", "f": "skipped"}
        result = ci["score"](self.FORECAST, actual)
        assert result["hits"] == ["a"]
        assert result["misses"] == ["c"]
        assert result["false_alarms"] == ["b"]
        assert result["unknown"] == ["e", "g"]
        assert (result["correct"], result["total"]) == (2, 4)

    def test_an_unknown_outcome_is_refused(self):
        with pytest.raises(ValueError):
            ci["score"]({"jobs": {"a": "fails"}}, {})


class TestPrPaths:
    def test_area_follows_the_maintenance_mirror(self):
        area = paths["area"]
        mirrored = sorted(p.name for p in (ROOT / "maintenances").iterdir() if p.is_dir() and not p.name.startswith("_"))
        assert mirrored, "maintenances/ has no package areas"
        name = mirrored[0]
        assert area(f"scikitplot/{name}/sub/deep.py", ROOT) == name
        assert area("scikitplot/no_such_package/x.py", ROOT) == "_repo"
        assert area("docs/source/index.rst", ROOT) == "_docs"
        assert area("plugins/scikitplot/README.md", ROOT) == "_plugins"
        assert area("maintenances/_plugins/todo/todo.md", ROOT) == "_plugins"
        assert area(f"maintenances/{name}/_maintenance/tools/x.py", ROOT) == name
        assert area("maintenances/README.md", ROOT) == "_repo"

    def test_kind(self):
        shared = paths["shared_skills"](ROOT)
        kind = paths["kind"]
        assert kind("docs/source/whats_new/upcoming_changes/x/1.fix.rst", shared)[0] == "release-note"
        assert kind("maintenances/_plugins/todo/todo.md", shared)[0] == "record"
        assert kind("maintenances/_repo/_maintenance/tests/test_records.py", shared)[0] == "test"
        assert kind("scikitplot/api/tests/test_x.py", shared)[0] == "test"
        assert kind("scikitplot/api/x.py", shared)[0] == "source"
        assert kind("plugins/INDEX.md", shared) == ("generated", kind("plugins/INDEX.md", shared)[1])
        assert kind("plugins/INDEX.md", shared)[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-c", os.devnull, "-q", "-p", "no:cacheprovider"]))
