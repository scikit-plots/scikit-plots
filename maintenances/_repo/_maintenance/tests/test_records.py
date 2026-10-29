"""Mutation tests for the work-records checker (`check_records.py`).

Notes
-----
**Developer notes.** A checker that passes the real tree proves little on its
own. Each test copies the records, breaks exactly one property, and asserts
that the check owning that property fails and names it.

The fixture copies every folder a test may change (``maintenances/``,
``plugins/``, ``.claude-plugin/``) and symlinks the rest of the checkout, so
links from the
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
    "maintenances",
    "plugins",
    ".claude-plugin",
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
    return root / "plugins/_shared/skills/scikitplot-engineering-rules/rules" / name


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def test_the_real_tree_passes():
    assert _failing(ROOT) == {}


def test_the_copy_passes(tree):
    assert _failing(tree) == {}


class TestRouter:
    """Rule 167: the retired locations stay empty; the router exists."""

    @pytest.mark.parametrize("rel", [
        "tasks/todo.md",
        "tasks/old/lessons.md",
        "docs/tasks/todo.md",
        "docs/maintenances/MAINTAINING.md",
        "docs/skills/README.md",
        "skills/scikitplot-x-dev/SKILL.md",
        "agents/scikitplot-x-dev.md",
        "plugins/maintenances/README.md",
    ])
    def test_a_file_in_a_retired_location_fails(self, tree, rel):
        path = tree / rel
        top = tree / rel.split("/")[0]
        if top.is_symlink():  # docs/ is symlinked: give the test its own copy
            top.unlink()
            top.mkdir()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
        failing = _failing(tree)
        assert "router" in failing
        assert any(rel in e and "is retired" in e for e in failing["router"])

    def test_a_symlink_in_a_retired_location_fails(self, tree):
        (tree / "tasks").mkdir()
        (tree / "tasks/todo.md").symlink_to(tree / "tasks/missing.md")
        assert any("tasks/todo.md" in e for e in _failing(tree)["router"])

    def test_an_empty_retired_folder_passes(self, tree):
        (tree / "tasks").mkdir()
        assert "router" not in _failing(tree)

    def test_a_missing_router_fails(self, tree):
        (tree / "maintenances/README.md").unlink()
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
        (tree / "plugins/_shared/skills/scikitplot-engineering-rules/rules/sub").mkdir()
        (tree / "plugins/_shared/skills/scikitplot-engineering-rules/rules/sub/more.md").write_text(
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
        path = tree / "plugins/_shared/skills/scikitplot-engineering-rules/SKILL.md"
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

    def test_a_last_part_that_lost_its_trailing_blank_line_passes(self, tree):
        """pre-commit's end-of-file-fixer keeps one final newline (CI-6 root cause)."""
        manifest = json.loads((tree / MANIFEST).read_text(encoding="utf-8"))
        for part in (p for r in manifest["sources"].values() for p in r["parts"]):
            path = tree / part["dest"]
            if "/history/" not in part["dest"] or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            i = text.find(part["heading"])
            if i + part["chars"] == len(text) and text.endswith("\n\n"):
                path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
                assert "archives" not in _failing(tree)
                return
            if i + part["chars"] == len(text) + 1:  # already fixed in the tree
                assert "archives" not in _failing(tree)
                return
        pytest.skip("no archive ends with its last part")

    def test_a_last_part_that_lost_text_still_fails(self, tree):
        path, part = self._an_archive(tree)
        text = path.read_text(encoding="utf-8")
        i = text.index(part["heading"])
        path.write_text(text[: i + part["chars"] - 10] + "\n", encoding="utf-8")
        assert any("was changed or removed" in e for e in _failing(tree)["archives"])

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
        _append(tree / "maintenances/_docs/todo/todo.md", f"\nSee {link}.\n")
        errors = _failing(tree)["links"]
        assert len(errors) == 1
        assert errors[0].startswith("maintenances/_docs/todo/todo.md:")
        assert message in errors[0]

    def test_a_reference_link_is_checked(self, tree):
        _append(tree / "maintenances/_docs/todo/todo.md", "\n[ref]: history/gone.md\n")
        assert any("history/gone.md" in e for e in _failing(tree)["links"])

    def test_valid_forms_pass(self, tree):
        (tree / "maintenances/_docs/todo/a(1).md").write_text("x\n", encoding="utf-8")
        _append(
            tree / "maintenances/_docs/todo/todo.md",
            "\n[a](https://example.org/x) [b](#top) [c](todo.md#where)"
            ' [d](<todo.md>) [e](a(1).md) [f](todo.md "title")\n',
        )
        assert "links" not in _failing(tree)

    def test_a_link_in_a_code_fence_is_not_resolved(self, tree):
        _append(tree / "maintenances/_docs/todo/todo.md", "\n```text\n[x](nowhere.md)\n```\n")
        assert "links" not in _failing(tree)

    def test_a_link_into_a_symlinked_folder_resolves(self, tree):
        _append(tree / "maintenances/_docs/todo/todo.md", "\n[pkg](../../../scikitplot/__init__.py)\n")
        assert "links" not in _failing(tree)


class TestBudgets:
    def test_an_active_todo_over_budget_fails(self, tree):
        _append(tree / "maintenances/_docs/todo/todo.md", "- line\n" * ACTIVE_TODO_MAX_LINES)
        errors = _failing(tree)["budgets"]
        assert len(errors) == 1
        assert errors[0].startswith("maintenances/_docs/todo/todo.md:")

    def test_an_archived_todo_has_no_budget(self, tree):
        """Review 11: a todo.md kept under history/ was budgeted."""
        archived = tree / "maintenances/corpus/_maintenance/history/old/todo/todo.md"
        archived.parent.mkdir(parents=True)
        archived.write_text("- line\n" * (ACTIVE_TODO_MAX_LINES + 1), encoding="utf-8")
        assert "budgets" not in _failing(tree)


def test_main_reports_json_and_exit_status(tree, capsys):
    assert main(["--root", str(tree), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    (tree / "tasks").mkdir()
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
            root / "maintenances/_docs/todo/todo.md",
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
        _append(tree / "maintenances/_docs/todo/todo.md", "\n```\n[x](gone.md)\n")
        failing = _failing(tree)
        assert any("code fence never closed" in e for e in failing["rules"])
        assert any("code fence never closed" in e for e in failing["links"])

    def test_footnotes_code_spans_and_escapes(self, tree):
        """Finding 4: false positives in the link parser."""
        (tree / "maintenances/_docs/todo/a b.md").write_text("x\n", encoding="utf-8")
        _append(
            tree / "maintenances/_docs/todo/todo.md",
            "\n[^1]: some words\n\nSee `a[0](b)` and [c](a%20b.md).\n",
        )
        assert "links" not in _failing(tree)

    def test_a_nested_skill_is_link_checked(self, tree):
        """Finding 5: only plugins/scikitplot-dev/skills/*/SKILL.md was checked (skills are flat now)."""
        nested = tree / "plugins/scikitplot-dev/skills/scikitplot-annoy-dev/references/deep/SKILL.md"
        nested.parent.mkdir(parents=True)
        nested.write_text("---\nname: x\ndescription: y\n---\n[x](gone.md)\n", encoding="utf-8")
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


derived_skill_names = records["derived_skill_names"]
parse_skill_frontmatter = records["parse_skill_frontmatter"]
skill_mirrors = records["skill_mirrors"]


def _skill(root: Path, folder: str, base: str = "plugins/scikitplot-dev/skills") -> Path:
    return root / base / folder / "SKILL.md"


USAGE = "plugins/scikitplot/skills"


def _set_line(path: Path, prefix: str, line: str) -> None:
    text = path.read_text(encoding="utf-8").split("\n")
    (index,) = [i for i, old in enumerate(text) if old.startswith(prefix)]
    text[index] = line
    path.write_text("\n".join(text), encoding="utf-8")


def _skills_errors(tree: Path) -> list[str]:
    return _failing(tree).get("skills", [])


class TestSkills:
    """Rules 164 and 165: the specification, the names, the mirror, the agents."""

    def test_the_real_tree_passes(self):
        assert "skills" not in _failing(ROOT)

    def test_both_sets_are_found_through_their_metadata(self):
        assert skill_mirrors(ROOT)["_cli"] == "scikitplot-cli-dev"
        assert skill_mirrors(ROOT)["cexternals/_annoy"] == "scikitplot-cexternals-annoy-dev"
        usage = records["usage_mirrors"](ROOT)
        assert usage["annoy"] == "scikitplot-annoy"
        assert usage[""] == "scikitplot"

    def test_a_name_unlike_its_folder_fails(self, tree):
        _set_line(_skill(tree, "scikitplot-cli-dev"), "name:", "name: scikitplot-cli")
        assert any("must equal its folder" in e for e in _skills_errors(tree))

    @pytest.mark.parametrize("name", ["_cli", "Cli-dev", "cli--dev", '"-cli"', "a" * 65])
    def test_a_name_outside_the_specification_fails(self, tree, name):
        _set_line(_skill(tree, "scikitplot-cli-dev"), "name:", f"name: {name}")
        assert any("must be 1-64 characters" in e for e in _skills_errors(tree))

    def test_a_nested_skill_fails(self, tree):
        nested = tree / "plugins/scikitplot-dev/skills/scikitplot-annoy-dev/inner/SKILL.md"
        nested.parent.mkdir()
        nested.write_text("---\nname: inner\ndescription: d\n---\n", encoding="utf-8")
        assert any("a skill is plugins/scikitplot-dev/skills/<name>/SKILL.md" in e for e in _skills_errors(tree))

    def test_a_plain_value_yaml_reads_differently_fails(self, tree):
        """The mlflow skill's unquoted 'scikitplot.mlflow: ...' broke its YAML."""
        _set_line(
            _skill(tree, "scikitplot-mlflow-dev"),
            "description:",
            "description: Maintain scikitplot.mlflow: servers and sessions.",
        )
        assert any("put it in double quotes" in e for e in _skills_errors(tree))

    def test_a_dev_skill_without_a_mirror_fails(self, tree):
        path = _skill(tree, "scikitplot-cli-dev")
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('metadata:\n  scikitplot-mirror: "_cli"\n', ""), encoding="utf-8")
        assert any("names the path it maintains" in e for e in _skills_errors(tree))

    def test_a_mirror_that_is_no_path_fails(self, tree):
        _set_line(_skill(tree, "scikitplot-cli-dev"), "  scikitplot-mirror:", '  scikitplot-mirror: "_nope"')
        assert any("is neither a package" in e for e in _skills_errors(tree))

    def test_an_unprefixed_name_fails(self, tree):
        """'annoy-maintainer' named no project (the reason for the prefix)."""
        old, new = tree / "plugins/scikitplot-dev/skills/scikitplot-annoy-dev", tree / "plugins/scikitplot-dev/skills/annoy-maintainer"
        old.rename(new)
        _set_line(new / "SKILL.md", "name:", "name: annoy-maintainer")
        assert any(
            "is named 'scikitplot-annoy-dev' (Rule 164)" in e for e in _skills_errors(tree)
        )

    def test_a_usage_name_other_than_the_derived_one_fails(self, tree):
        old, new = tree / USAGE / "scikitplot-annoy", tree / USAGE / "annoy"
        old.rename(new)
        _set_line(new / "SKILL.md", "name:", "name: annoy")
        assert any(
            "usage skill for 'scikitplot.annoy' is named 'scikitplot-annoy'" in e
            for e in _skills_errors(tree)
        )

    def test_two_skills_for_one_path_fail(self, tree):
        _set_line(_skill(tree, "scikitplot-brand-dev"), "  scikitplot-mirror:", '  scikitplot-mirror: "_cli"')
        assert any("already has skill" in e for e in _skills_errors(tree))

    def test_a_folder_without_a_skill_fails(self, tree):
        (tree / "plugins/scikitplot-dev/skills/_externals").mkdir()
        assert any("no SKILL.md" in e for e in _skills_errors(tree))

    def test_a_hand_edited_agent_fails_and_write_index_repairs_it(self, tree):
        agent = tree / "plugins/scikitplot-dev/agents/scikitplot-annoy-dev.md"
        agent.write_text(agent.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
        assert any("plugins/scikitplot-dev/agents/scikitplot-annoy-dev.md is out of date" in e for e in _skills_errors(tree))
        records["write_indexes"](tree)
        assert "skills" not in _failing(tree)

    def test_an_agent_without_its_skill_fails(self, tree):
        (tree / "plugins/scikitplot/agents/scikitplot-gone.md").write_text("x\n", encoding="utf-8")
        assert any("no skill has this agent" in e for e in _skills_errors(tree))

    def test_every_skill_with_a_path_or_module_has_its_agent(self):
        for folder in skill_mirrors(ROOT).values():
            assert (ROOT / "plugins/scikitplot-dev/agents" / f"{folder}.md").is_file(), folder
        for folder in records["usage_mirrors"](ROOT).values():
            assert (ROOT / "plugins/scikitplot/agents" / f"{folder}.md").is_file(), folder

    def test_a_stale_skill_index_fails_and_write_index_repairs_it(self, tree):
        (tree / "plugins/INDEX.md").write_text("# stale\n", encoding="utf-8")
        assert any("plugins/INDEX.md is out of date" in e for e in _skills_errors(tree))
        records["write_indexes"](tree)
        assert "skills" not in _failing(tree)


class TestDerivedSkillNames:
    def test_a_public_and_a_private_twin_get_two_names(self):
        names = derived_skill_names({"externals", "_externals", "utils", "_utils"})
        assert names == {
            "externals": "externals",
            "_externals": "internal-externals",
            "utils": "utils",
            "_utils": "internal-utils",
        }

    def test_the_path_nearer_the_root_keeps_the_short_name(self):
        names = derived_skill_names(
            {"annoy", "annoy/_annoy", "cexternals", "cexternals/_annoy", "cexternals/_annoy/annoy"}
        )
        assert names["annoy"] == "annoy"
        assert names["cexternals/_annoy"] == "cexternals-annoy"
        assert names["annoy/_annoy"] == "annoy-annoy"
        assert names["cexternals/_annoy/annoy"] == "cexternals-annoy-annoy"
        assert len(set(names.values())) == len(names)

    def test_a_unique_leaf_is_enough(self):
        names = derived_skill_names({"_externals", "_externals/_sphinx_ext", "_externals/_sphinx_ext/_sphinx_feedback"})
        assert names["_externals/_sphinx_ext/_sphinx_feedback"] == "sphinx-feedback"

    def test_every_real_path_gets_a_distinct_name(self):
        names = derived_skill_names(records["mirror_keys"](ROOT))
        assert len(set(names.values())) == len(names)
        assert all(len("scikitplot-") + len(n) + len("-dev") <= 64 for n in names.values())


class TestSkillFrontmatter:
    @pytest.mark.parametrize(
        ("text", "fields"),
        [
            ("---\nname: a\ndescription: b c\n---\n", {"name": "a", "description": "b c"}),
            ('---\nname: a\ndescription: "x: y"\n---\n', {"name": "a", "description": "x: y"}),
            ("---\nname: a\ndescription: 'it''s'\n---\n", {"name": "a", "description": "it's"}),
            (
                '---\nname: a\ndescription: d\nmetadata:\n  scikitplot-mirror: "_cli"\n---\n',
                {"name": "a", "description": "d", "metadata": {"scikitplot-mirror": "_cli"}},
            ),
        ],
    )
    def test_the_subset_is_read(self, text, fields):
        assert parse_skill_frontmatter(text) == (fields, [])

    @pytest.mark.parametrize(
        ("text", "problem"),
        [
            ("name: a\n", "does not start"),
            ("---\nname: a\n", "no closing"),
            ("---\nname: a\nname: b\n---\n", "given twice"),
            ("---\n  x: y\n---\n", "indented outside"),
            ("---\ndescription: >\n---\n", "put it in double quotes"),
            ('---\ndescription: "open\n---\n', "unterminated"),
            ("---\ndescription: a #b\n---\n", "put it in double quotes"),
        ],
    )
    def test_anything_else_is_a_problem(self, text, problem):
        _, problems = parse_skill_frontmatter(text)
        assert any(problem in p for p in problems), problems


SHARED = "plugins/_shared/skills"


class TestSharedSkills:
    """Rule 167: one source in plugins/_shared/skills/, a byte copy in each plugin."""

    def _name(self, root: Path) -> str:
        names = records["shared_skill_names"](root)
        if not names:
            pytest.skip("no shared skill")
        return names[0]

    def test_every_plugin_carries_an_identical_copy(self):
        name = self._name(ROOT)
        for target in records["SHARED_TARGETS"]:
            for path in (ROOT / SHARED / name).rglob("*"):
                if path.is_file():
                    rel = path.relative_to(ROOT / SHARED).as_posix()
                    assert (ROOT / target / rel).read_bytes() == path.read_bytes(), rel

    def test_an_edited_copy_fails_and_write_index_repairs_it(self, tree):
        name = self._name(tree)
        copy = tree / USAGE / name / "SKILL.md"
        copy.write_text(copy.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")
        assert any("differs from its source" in e for e in _skills_errors(tree))
        records["write_indexes"](tree)
        assert "skills" not in _failing(tree)

    def test_an_extra_file_in_a_copy_fails_and_write_index_removes_it(self, tree):
        name = self._name(tree)
        extra = tree / "plugins/scikitplot-dev/skills" / name / "stray.md"
        extra.write_text("x\n", encoding="utf-8")
        assert any("stray.md: not in" in e for e in _skills_errors(tree))
        records["write_indexes"](tree)
        assert not extra.exists()
        assert "skills" not in _failing(tree)

    def test_a_source_edit_propagates_on_write_index(self, tree):
        name = self._name(tree)
        source = tree / SHARED / name / "SKILL.md"
        source.write_text(source.read_text(encoding="utf-8") + "\nNew line.\n", encoding="utf-8")
        assert any("differs from its source" in e for e in _skills_errors(tree))
        records["write_indexes"](tree)
        assert (tree / USAGE / name / "SKILL.md").read_bytes() == source.read_bytes()
        assert "skills" not in _failing(tree)

    def test_a_shared_skill_has_no_agent_and_no_mirror(self):
        name = self._name(ROOT)
        assert not (ROOT / "plugins/scikitplot/agents" / f"{name}.md").exists()
        assert not (ROOT / "plugins/scikitplot-dev/agents" / f"{name}.md").exists()


def _plugins_errors(tree: Path) -> list[str]:
    return _failing(tree).get("plugins", [])


def _market(tree: Path) -> tuple[Path, dict]:
    path = tree / ".claude-plugin/marketplace.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


class TestPlugins:
    """Rule 167: the marketplace lists ./plugins/<name>, names agree, no version."""

    def test_the_real_tree_passes(self):
        assert "plugins" not in _failing(ROOT)

    def test_every_plugin_is_under_plugins(self):
        _, market = _market(ROOT)
        assert {e["name"] for e in market["plugins"]} == {"scikitplot", "scikitplot-mcp", "scikitplot-dev"}
        assert all(e["source"] == f"./plugins/{e['name']}" for e in market["plugins"])

    def test_a_root_source_fails(self, tree):
        path, market = _market(tree)
        market["plugins"][0]["source"] = "./"
        path.write_text(json.dumps(market), encoding="utf-8")
        assert any("write './plugins/" in e for e in _plugins_errors(tree))

    def test_a_manifest_name_unlike_the_entry_fails(self, tree):
        manifest = tree / "plugins/scikitplot-mcp/.claude-plugin/plugin.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["name"] = "scikitplot-docs"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        assert any("must equal the marketplace entry" in e for e in _plugins_errors(tree))

    def test_a_version_fails(self, tree):
        manifest = tree / "plugins/scikitplot-dev/.claude-plugin/plugin.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["version"] = "0.1.0"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        assert any("sets no 'version'" in e for e in _plugins_errors(tree))

    def test_an_unlisted_plugin_fails(self, tree):
        manifest = tree / "plugins/scikitplot-extra/.claude-plugin/plugin.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"name": "scikitplot-extra"}', encoding="utf-8")
        assert any("does not list" in e for e in _plugins_errors(tree))

    def test_invalid_json_fails(self, tree):
        path, _ = _market(tree)
        path.write_text("{", encoding="utf-8")
        assert any("invalid JSON" in e for e in _plugins_errors(tree))

    def test_the_mcp_plugin_launches_the_server(self):
        servers = json.loads((ROOT / "plugins/scikitplot-mcp/.mcp.json").read_text(encoding="utf-8"))
        (entry,) = servers["mcpServers"].values()
        assert entry["command"] == "python" and entry["args"] == ["-m", "scikitplot.mcp"]

    def test_the_default_plugin_runs_nothing(self):
        for name in (".mcp.json", "hooks", "bin"):
            assert not (ROOT / "plugins/scikitplot" / name).exists(), name


class TestIndexTablesAndDisclosure:
    """CI-11: the index groups packages and areas; programs in a skill are disclosed."""

    def test_every_maintenance_area_is_listed(self):
        text = (ROOT / "plugins/INDEX.md").read_text(encoding="utf-8")
        for area in records["maintenance_areas"](ROOT):
            assert f"maintenances/{area}/" in text, area

    def test_packages_and_repository_areas_are_separate_tables(self):
        text = (ROOT / "plugins/INDEX.md").read_text(encoding="utf-8")
        packages = text.split("### Packages", 1)[1].split("### Repository-wide areas", 1)
        assert "`scikitplot/corpus/`" in packages[0]
        repo = packages[1].split("## Using scikit-plots", 1)[0]
        for area in ("_repo", "_docs", "_plugins"):
            assert f"maintenances/{area}/" in repo, area
        assert "scikitplot/" not in repo.replace("plugins/scikitplot", "")

    def test_a_container_folder_is_not_an_area(self):
        found = records["maintenance_areas"](ROOT)
        assert "_externals" not in found and "cexternals" not in found
        assert "cexternals/_annoy" in found

    def test_the_moved_skills_are_shared_and_copied(self):
        names = records["shared_skill_names"](ROOT)
        for name in ("scikitplot-engineering-rules", "skill-creator"):
            assert name in names
            for plugin in ("scikitplot", "scikitplot-dev"):
                assert (ROOT / "plugins" / plugin / "skills" / name / "SKILL.md").is_file()

    def test_programs_are_disclosed_and_markdown_only_is_not_claimed(self):
        programs = records["skills_with_programs"](ROOT, records["USAGE_SKILLS_DIR"])
        assert "skill-creator" in programs
        readme = (ROOT / "plugins/scikitplot/README.md").read_text(encoding="utf-8")
        assert "`skill-creator`" in readme and "ask" in readme
        assert "Markdown only" not in readme
        for program in programs["skill-creator"]:
            assert f"`{program}`" in readme, program

    def test_a_new_program_in_a_skill_makes_the_readme_stale(self, tree):
        script = tree / records["USAGE_SKILLS_DIR"] / "scikitplot-api" / "tool.py"
        script.write_text("print(1)\n", encoding="utf-8")
        assert any("README.md is out of date" in e for e in _failing(tree).get("skills", []))
        records["write_indexes"](tree)
        assert "`scikitplot-api`" in (tree / "plugins/scikitplot/README.md").read_text(encoding="utf-8")

    def test_skill_creator_asks_before_running_programs(self):
        text = (ROOT / "plugins/_shared/skills/skill-creator/SKILL.md").read_text(encoding="utf-8")
        head = text.split("## Creating a skill", 1)[0]
        assert "Modified by scikit-plots" in head
        assert "explicit yes" in head and "background" in head
        assert "3117" in head and "claude -p" in head
