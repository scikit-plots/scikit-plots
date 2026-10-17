"""Mutation tests for the work-records checker (`check_records.py`).

Notes
-----
**Developer notes.** A checker that passes the real tree proves little on its
own. Each test copies the records, breaks exactly one property, and asserts
that the check owning that property fails and names it.

The fixture copies every folder a test may change (``tasks/``,
``maintenances/``, ``skills/``, ``docs/tasks``, ``docs/maintenances``,
``docs/skills``) and symlinks the rest of the checkout, so links from the
records into runtime folders still resolve. The symlinks are writable: no
test may write below one.

Run with ``python -B maintenances/_repo/_maintenance/tests/test_records.py``
(it ignores the repository's pytest configuration, which loads plugins this
test does not need) or with pytest.
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
ROOT = Path(__file__).resolve().parents[4]
records = runpy.run_path(str(TOOLS / "check_records.py"))
run = records["run"]
main = records["main"]
parse_rules = records["parse_rules"]
MANIFEST = records["MANIFEST"]
INDEX = records["INDEX"]
HISTORY_INDEX = records["HISTORY_INDEX"]
ACTIVE_TODO_MAX_LINES = records["ACTIVE_TODO_MAX_LINES"]

#: Folders copied into each test tree; everything else is symlinked.
COPIED = (
    "tasks",
    "maintenances",
    "skills",
    "docs/tasks",
    "docs/maintenances",
    "docs/skills",
)


@pytest.fixture(scope="session")
def template(tmp_path_factory) -> Path:
    """One copy of the records, with the rest of the checkout symlinked."""
    base = tmp_path_factory.mktemp("records")
    for entry in ROOT.iterdir():
        if entry.name in (".git", "docs") or entry.name in COPIED:
            continue
        (base / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (base / "docs").mkdir()
    for entry in (ROOT / "docs").iterdir():
        if f"docs/{entry.name}" not in COPIED:
            (base / "docs" / entry.name).symlink_to(
                entry, target_is_directory=entry.is_dir()
            )
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in COPIED:
        if (ROOT / rel).is_dir():
            shutil.copytree(ROOT / rel, base / rel, symlinks=True, ignore=ignore)
    return base


@pytest.fixture
def tree(template: Path, tmp_path: Path) -> Path:
    """A fresh copy of *template* for one test."""
    root = tmp_path / "repo"
    shutil.copytree(template, root, symlinks=True)
    return root


def _failing(root: Path) -> dict[str, list[str]]:
    return {name: errors for name, errors in run(root).items() if errors}


def _rules(root: Path, name: str) -> Path:
    return root / "skills/_engineering/rules" / name


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def test_the_real_tree_passes():
    assert _failing(ROOT) == {}


def test_the_copy_passes(tree):
    assert _failing(tree) == {}


class TestRouter:
    def test_another_file_in_tasks_fails(self, tree):
        (tree / "tasks/todo.md").write_text("# Active Tasks\n", encoding="utf-8")
        failing = _failing(tree)
        assert list(failing) == ["router"]
        assert "tasks/todo.md" in failing["router"][0]

    def test_a_file_in_a_subfolder_of_tasks_fails(self, tree):
        (tree / "tasks/old").mkdir()
        (tree / "tasks/old/lessons.md").write_text("x\n", encoding="utf-8")
        assert any("tasks/old/lessons.md" in e for e in _failing(tree)["router"])

    def test_a_symlink_in_tasks_fails(self, tree):
        (tree / "tasks/todo.md").symlink_to(tree / "tasks/missing.md")
        assert any("tasks/todo.md" in e for e in _failing(tree)["router"])

    def test_a_missing_router_fails(self, tree):
        (tree / "tasks/README.md").unlink()
        assert any("missing" in e for e in _failing(tree)["router"])


class TestRules:
    def _move_rule(self, tree, number, source, target):
        """Move the block of Rule *number* from one rules file to another."""
        text = _rules(tree, source).read_text(encoding="utf-8")
        start = text.index(f"### Rule {number}:")
        end = text.find("\n### Rule ", start + 1)
        block = text[start : len(text) if end < 0 else end + 1]
        _rules(tree, source).write_text(text.replace(block, ""), encoding="utf-8")
        _append(_rules(tree, target), "\n" + block)

    def test_a_duplicate_number_fails(self, tree):
        text = _rules(tree, "text.md").read_text(encoding="utf-8")
        block = text[text.index("### Rule 58:") :].split("\n### Rule ")[0]
        _append(_rules(tree, "text.md"), "\n" + block + "\n")
        assert any("duplicate rule numbers: [58]" in e for e in _failing(tree)["rules"])

    def test_a_missing_number_fails(self, tree):
        path = _rules(tree, "text.md")
        text = path.read_text(encoding="utf-8")
        start = text.index("### Rule 117:")
        end = text.index("### Rule 122:")
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        errors = _failing(tree)["rules"]
        assert "missing rule numbers: [117]" in errors
        assert any("out of date" in e for e in errors)

    @pytest.mark.parametrize(
        "heading",
        [
            "### Rule 999: Records " + chr(0x2014) + " an em dash",
            "### Rule 999:Records - no space",
            "#### Rule 999: Records - a level-4 heading",
            "## rule 999 - lower case",
        ],
    )
    def test_a_malformed_rule_heading_fails(self, tree, heading):
        """Review 5: these were invisible, so a later rule took the number."""
        _append(_rules(tree, "delivery.md"), f"\n{heading}\n- **When:** x\n")
        assert any("malformed rule heading" in e for e in _failing(tree)["rules"])

    def test_a_rule_quoted_in_a_code_fence_is_not_a_rule(self, tree):
        _append(
            _rules(tree, "delivery.md"),
            "\n```markdown\n### Rule 999: Records - x\n```\n",
        )
        assert "rules" not in _failing(tree)

    def test_a_rules_file_in_a_subfolder_fails(self, tree):
        (tree / "skills/_engineering/rules/sub").mkdir()
        (tree / "skills/_engineering/rules/sub/more.md").write_text(
            "Categories: More\n\n### Rule 999: More - x\n", encoding="utf-8"
        )
        errors = _failing(tree)["rules"]
        assert any("rules files live directly in" in e for e in errors)

    def test_a_rule_in_the_wrong_file_fails(self, tree):
        self._move_rule(tree, 58, "text.md", "native.md")
        errors = _failing(tree)["rules"]
        assert any("Rule 58 has category 'Unicode'" in e for e in errors)

    def test_a_category_declared_twice_fails(self, tree):
        path = _rules(tree, "native.md")
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("Categories: ", "Categories: Unicode, ", 1), encoding="utf-8"
        )
        assert any("'Unicode' is also declared" in e for e in _failing(tree)["rules"])

    def test_a_file_without_categories_fails(self, tree):
        path = _rules(tree, "native.md")
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("Categories: ", "Kinds: ", 1), encoding="utf-8")
        assert any("no 'Categories:' line" in e for e in _failing(tree)["rules"])

    def test_a_rules_file_the_skill_does_not_link_fails(self, tree):
        _rules(tree, "extra.md").write_text(
            "# Extra\n\nCategories: Extra\n", encoding="utf-8"
        )
        assert any("does not link rules/extra.md" in e for e in _failing(tree)["rules"])

    def test_a_name_in_prose_is_not_a_link(self, tree):
        """Review 5: a substring match counted ``rules/text.md`` in prose."""
        path = tree / "skills/_engineering/SKILL.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace("[`rules/text.md`](rules/text.md)", "`rules/text.md`"),
            encoding="utf-8",
        )
        assert any("does not link rules/text.md" in e for e in _failing(tree)["rules"])

    def test_a_stale_index_fails_and_write_index_repairs_it(self, tree, capsys):
        (tree / INDEX).write_text("# Rule index\n", encoding="utf-8")
        assert any("out of date" in e for e in _failing(tree)["rules"])
        assert main(["--root", str(tree), "--write-index"]) == 0
        capsys.readouterr()
        assert _failing(tree) == {}
        assert b"\r\n" not in (tree / INDEX).read_bytes()

    def test_a_category_with_a_hyphen_is_parsed_whole(self):
        rules, errors = parse_rules(ROOT)
        assert errors == []
        by_number = {r.number: r for r in rules}
        assert by_number[105].category == "Caller-supplied resources"


class TestArchives:
    def _an_archive(self, tree):
        manifest = json.loads((tree / MANIFEST).read_text(encoding="utf-8"))
        part = next(
            p
            for r in manifest["sources"].values()
            for p in r["parts"]
            if "/history/" in p["dest"] and p["dest"].endswith(".md")
        )
        return tree / part["dest"], part

    def test_an_edited_archive_fails(self, tree):
        path, part = self._an_archive(tree)
        text = path.read_text(encoding="utf-8")
        i = text.index(part["heading"]) + len(part["heading"]) + 1
        path.write_text(text[:i] + "edited\n" + text[i:], encoding="utf-8")
        errors = _failing(tree)["archives"]
        assert any("was changed or removed" in e for e in errors)

    def test_a_missing_archive_fails(self, tree):
        path, _ = self._an_archive(tree)
        path.unlink()
        assert any("missing" in e for e in _failing(tree)["archives"])

    def test_text_added_around_the_parts_is_allowed(self, tree):
        path, _ = self._an_archive(tree)
        _append(path, "\nA note added after the archived parts.\n")
        assert "archives" not in _failing(tree)

    def test_a_stale_history_index_fails(self, tree):
        (tree / HISTORY_INDEX).write_text("# History index\n", encoding="utf-8")
        assert any(
            "HISTORY_INDEX.md is out of date" in e for e in _failing(tree)["archives"]
        )

    def test_the_history_index_names_the_areas(self):
        text = (ROOT / HISTORY_INDEX).read_text(encoding="utf-8")
        row = next(
            line for line in text.splitlines() if "r33_run1_correctness.md" in line
        )
        for area in ("`corpus`", "`levenshtein`", "`memmap`"):
            assert area in row


class TestLinks:
    @pytest.mark.parametrize(
        ("link", "message"),
        [
            ("[gone](history/gone.md)", "does not exist"),
            ('[titled](gone.md "a title")', "does not exist"),
            ("[angle](<gone.md>)", "does not exist"),
            ("[absolute](/etc/hosts)", "is absolute"),
            ("[outside](../../../../../tmp)", "points outside the repository"),
        ],
    )
    def test_a_bad_link_fails(self, tree, link, message):
        _append(tree / "docs/tasks/todo.md", f"\nSee {link}.\n")
        errors = _failing(tree)["links"]
        assert len(errors) == 1
        assert errors[0].startswith("docs/tasks/todo.md:")
        assert message in errors[0]

    def test_a_reference_link_is_checked(self, tree):
        _append(tree / "docs/tasks/todo.md", "\n[ref]: history/gone.md\n")
        assert any("history/gone.md" in e for e in _failing(tree)["links"])

    def test_valid_forms_pass(self, tree):
        (tree / "docs/tasks/a(1).md").write_text("x\n", encoding="utf-8")
        _append(
            tree / "docs/tasks/todo.md",
            "\n[a](https://example.org/x) [b](#top) [c](README.md#where)"
            ' [d](<README.md>) [e](a(1).md) [f](README.md "title")\n',
        )
        assert "links" not in _failing(tree)

    def test_a_link_in_a_code_fence_is_not_resolved(self, tree):
        _append(tree / "docs/tasks/todo.md", "\n```text\n[x](nowhere.md)\n```\n")
        assert "links" not in _failing(tree)

    def test_a_link_into_a_symlinked_folder_resolves(self, tree):
        _append(tree / "docs/tasks/todo.md", "\n[pkg](../../scikitplot/__init__.py)\n")
        assert "links" not in _failing(tree)


class TestBudgets:
    def test_an_active_todo_over_budget_fails(self, tree):
        _append(tree / "docs/tasks/todo.md", "- line\n" * ACTIVE_TODO_MAX_LINES)
        errors = _failing(tree)["budgets"]
        assert len(errors) == 1
        assert errors[0].startswith("docs/tasks/todo.md:")

    def test_an_archived_todo_has_no_budget(self, tree):
        """Review 11: a todo.md kept under history/ was budgeted."""
        archived = tree / "maintenances/corpus/_maintenance/history/old/todo/todo.md"
        archived.parent.mkdir(parents=True)
        archived.write_text("- line\n" * (ACTIVE_TODO_MAX_LINES + 1), encoding="utf-8")
        assert "budgets" not in _failing(tree)


def test_main_reports_json_and_exit_status(tree, capsys):
    assert main(["--root", str(tree), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    (tree / "tasks/extra.txt").write_text("x\n", encoding="utf-8")
    assert main(["--root", str(tree), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


if __name__ == "__main__":
    # ``-c os.devnull``: the repository's pytest configuration loads plugins
    # (sphinx.testing) that a clean environment with only pytest lacks.
    sys.exit(pytest.main([__file__, "-c", os.devnull, "-q", "-p", "no:cacheprovider"]))


class TestSecondReview:
    """One test per finding of the 4R re-review."""

    def test_a_checkout_under_a_history_folder_is_still_checked(
        self, template, tmp_path
    ):
        """Finding 1: '/history/' in the absolute path disabled two checks."""
        root = tmp_path / "history" / "repo"
        shutil.copytree(template, root, symlinks=True)
        _append(
            root / "docs/tasks/todo.md",
            "\n[x](gone.md)\n" + "- l\n" * ACTIVE_TODO_MAX_LINES,
        )
        failing = _failing(root)
        assert "links" in failing
        assert "budgets" in failing

    def test_a_live_folder_named_history_is_not_an_archive(self, tree):
        live = tree / "maintenances/corpus/my/history/todo/todo.md"
        live.parent.mkdir(parents=True)
        live.write_text("[x](gone.md)\n", encoding="utf-8")
        assert any("my/history/todo/todo.md" in e for e in _failing(tree)["links"])

    @pytest.mark.parametrize(
        "heading",
        [
            "   ### Rule 999: Records - indented",
            "### Rules 999: Records - x",
            "### Rule #999: Records - x",
        ],
    )
    def test_more_malformed_headings_fail(self, tree, heading):
        """Finding 3."""
        _append(_rules(tree, "delivery.md"), f"\n{heading}\n")
        assert any("malformed rule heading" in e for e in _failing(tree)["rules"])

    def test_an_unclosed_fence_fails_in_rules_and_links(self, tree):
        """Finding 3: an open fence hid every heading and link after it."""
        _append(
            _rules(tree, "delivery.md"), "\n```text\n### Rule 999: Records - hidden\n"
        )
        _append(tree / "docs/tasks/todo.md", "\n```\n[x](gone.md)\n")
        failing = _failing(tree)
        assert any("code fence never closed" in e for e in failing["rules"])
        assert any("code fence never closed" in e for e in failing["links"])

    def test_footnotes_code_spans_and_escapes(self, tree):
        """Finding 4: false positives in the link parser."""
        (tree / "docs/tasks/a b.md").write_text("x\n", encoding="utf-8")
        _append(
            tree / "docs/tasks/todo.md",
            "\n[^1]: some words\n\nSee `a[0](b)` and [c](a%20b.md).\n",
        )
        assert "links" not in _failing(tree)

    def test_a_nested_skill_is_link_checked(self, tree):
        """Finding 5: only skills/*/SKILL.md was checked."""
        nested = next(p for p in (tree / "skills").glob("*/*/*/SKILL.md"))
        _append(nested, "\n[x](gone.md)\n")
        assert any("gone.md" in e for e in _failing(tree)["links"])

    def test_the_index_resolves_nested_areas_and_leaves(self):
        """Finding 2: '_sphinx_feedback/...' named no area."""
        text = (ROOT / HISTORY_INDEX).read_text(encoding="utf-8")
        row = next(
            line for line in text.splitlines() if "r29b_feedback_storage.md" in line
        )
        assert "`_externals/_sphinx_ext/_sphinx_feedback`" in row

    @pytest.mark.parametrize(
        "text",
        [
            "scikitplot.random_projection and scikitplot/nc_extra.py",
            "a random/arbitrary choice, logging/tracing",
            "`random.Random(0)`",
        ],
    )
    def test_no_false_areas(self, text):
        """Finding 2: prose and longer names matched areas."""
        known = records["areas"](ROOT)
        assert records["areas_named"](text, ROOT, known) == []

    def test_an_empty_local_folder_does_not_change_the_index(self, tree):
        """Finding 2: two empty untracked folders made the index stale."""
        package = tree / "scikitplot"
        package.unlink()
        shutil.copytree(
            ROOT / "scikitplot",
            package,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "*.so", "*.pyc", "__pycache__", "node_modules"
            ),
        )
        (package / "tools").mkdir(exist_ok=True)
        (package / "build").mkdir(exist_ok=True)
        assert "archives" not in _failing(tree)


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git history")
def test_prove_rebuilds_the_move():
    """Finding 6: the proof of Rule 124 runs from the repository."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("needs git on PATH")
    log = subprocess.run(  # noqa: S603 (a fixed argument list, no shell)
        [git, "-C", str(ROOT), "log", "--format=%h %s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    move = next((line.split()[0] for line in log if " 4R-1:" in f" {line}"), None)
    if move is None:
        pytest.skip("the 4R-1 move is not in this history")
    assert records["prove"](ROOT, move) == []
    assert main(["--root", str(ROOT), "--prove", move]) == 0
