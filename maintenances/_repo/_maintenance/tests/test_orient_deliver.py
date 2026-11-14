"""Tests for the start-of-chat and delivery tools (``orient.py``, ``deliver.py``)."""

from __future__ import annotations

import runpy
import shutil
import zipfile
from pathlib import Path

import pytest

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

TOOLS = Path(__file__).resolve().parent.parent / "tools"
orient = runpy.run_path(str(TOOLS / "orient.py"))
deliver = runpy.run_path(str(TOOLS / "deliver.py"))

_TEMPLATE = """# start

## FILL IN

- Pull request number for new fragments: `42`
- Task for this chat: `continue`
- Notes I am deferring (do not work on them unless I name them):
  - `upcoming_changes/a/later.md`
- Decisions I have already taken (do not reopen them):
  - none

## 1. Who I am
"""


def _note(path: Path, status: str, title: str = "T") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ntitle: "{title}"\nstatus: {status}\n---\n\nbody\n', encoding="utf-8"
    )


@pytest.fixture
def checkout(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/new-chat-template.md").write_text(_TEMPLATE, encoding="utf-8")
    for name in (
        "AGENTS.md",
        "maintenances/MAINTAINING.md",
        "maintenances/_repo/_maintenance/FRESH_CHAT_HANDOFF.md",
    ):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("# doc\n", encoding="utf-8")
    (tmp_path / ".claude/init").write_text(
        "- `.claude/new-chat-template.md` - start\n- `python -B x/orient.py` - run\n"
        "- `maintenances/<area>/MAINTAINING.md` - pattern\n",
        encoding="utf-8",
    )
    _note(tmp_path / "upcoming_changes/a/later.md", "open", "Later")
    _note(tmp_path / "upcoming_changes/a/now.md", "open", "Now")
    _note(tmp_path / "upcoming_changes/a/done.md", "promoted")
    _note(tmp_path / "upcoming_changes/a/wait.md", "blocked", "Wait")
    (tmp_path / "upcoming_changes/README.md").write_text(
        "status: open\n", encoding="utf-8"
    )
    frag = tmp_path / "docs/source/whats_new/upcoming_changes/sec"
    frag.mkdir(parents=True)
    (frag / "42.fix.rst").write_text("- x\n", encoding="utf-8")
    (frag / "41.fix.rst").write_text("- y\n", encoding="utf-8")
    area = tmp_path / "maintenances/demo"
    (area / "_maintenance").mkdir(parents=True)
    (area / "REVIEW.json").write_text(
        '{"findings": [{"id": "D-007"}, {"id": "D-012"}]}', encoding="utf-8"
    )
    (area / "_maintenance/RESUME.md").write_text(
        "# R\n\n## Next action\n\nDo the thing.\n\n## Other\n", encoding="utf-8"
    )
    return tmp_path


class TestOrient:
    def test_next_ids_keep_each_series_and_its_padding(self):
        assert orient["next_ids"](["CP-9", "CP-119", "LOG-FMT-001", "x"]) == {
            "CP-": "CP-120",
            "LOG-FMT-": "LOG-FMT-002",
        }

    def test_state_reads_only_the_checkout(self, checkout):
        report = orient["state"](checkout)
        assert report["pr"] == "42"
        assert report["open_notes"] == [
            {
                "path": "upcoming_changes/a/later.md",
                "title": "Later",
                "status": "open",
                "deferred": True,
            },
            {
                "path": "upcoming_changes/a/now.md",
                "title": "Now",
                "status": "open",
                "deferred": False,
            },
            {
                "path": "upcoming_changes/a/wait.md",
                "title": "Wait",
                "status": "blocked",
                "deferred": False,
            },
        ]
        assert report["fragments"] == ["sec/42.fix.rst"]
        assert report["areas"] == [
            {
                "area": "demo",
                "findings": 2,
                "next_ids": {"D-": "D-013"},
                "next_action": "Do the thing.",
            }
        ]
        assert report["problems"] == []

    def test_pr_override_and_deterministic_output(self, checkout):
        first = orient["render"](orient["state"](checkout, "41"))
        assert "sec/41.fix.rst" in first
        assert first == orient["render"](orient["state"](checkout, "41"))

    def test_fragments_under_another_number_are_reported(self, checkout):
        problems = orient["state"](checkout, "43")["problems"]
        assert any(
            "no fragments are filed under 43, but some are under 41 (1), 42 (1)" in p
            for p in problems
        )
        assert not any(
            "no fragments" in p for p in orient["state"](checkout)["problems"]
        )

    def test_problems_are_reported_not_raised(self, checkout):
        (checkout / "upcoming_changes/a/later.md").unlink()
        (checkout / "maintenances/demo/REVIEW.json").write_text("{", encoding="utf-8")
        problems = orient["state"](checkout)["problems"]
        assert any("deferred note not found" in p for p in problems)
        assert any("REVIEW.json" in p for p in problems)

    def test_a_moved_path_is_reported_once_per_document(self, checkout):
        init = checkout / ".claude/init"
        init.write_text(
            init.read_text(encoding="utf-8")
            + "- `docs/gone.md` and `python -B tools/gone.py --x`\n",
            encoding="utf-8",
        )
        dead = orient["dead_paths"](checkout)
        assert ".claude/init: docs/gone.md" in dead
        assert ".claude/init: tools/gone.py" in dead
        assert not any("<area>" in one for one in dead)
        assert any(
            "dead path in .claude/init: docs/gone.md" in p
            for p in orient["state"](checkout)["problems"]
        )

    def test_a_missing_read_order_document_is_reported(self, checkout):
        (checkout / "AGENTS.md").unlink()
        assert "AGENTS.md: (the document itself)" in orient["dead_paths"](checkout)

    def test_the_real_read_order_names_only_existing_paths(self):
        root = orient["repository_root"](Path(__file__).resolve())
        assert orient["dead_paths"](root) == []

    def test_without_a_number_the_next_one_is_derived(self, checkout):
        (checkout / ".claude/new-chat-template.md").unlink()
        report = orient["state"](checkout)
        # Fragments are filed under 41 and 42; no git history here.
        assert (report["pr"], report["pr_source"]) == ("43", "derived")
        assert not any("no fragments" in p for p in report["problems"])
        assert orient["render"](report).startswith("pull request: 43 (derived")

    def test_auto_in_the_template_derives_the_number(self, checkout):
        template = checkout / ".claude/new-chat-template.md"
        template.write_text(
            template.read_text(encoding="utf-8").replace("`42`", "`auto`"),
            encoding="utf-8",
        )
        report = orient["state"](checkout)
        assert (report["pr"], report["pr_source"]) == ("43", "derived")

    def test_an_explicit_number_wins(self, checkout):
        assert orient["state"](checkout)["pr_source"] == "template"
        report = orient["state"](checkout, "7")
        assert (report["pr"], report["pr_source"]) == ("7", "--pr")
        assert orient["state"](checkout, "auto")["pr"] == "43"

    def test_nothing_known_asks_for_a_number(self, checkout):
        (checkout / ".claude/new-chat-template.md").unlink()
        shutil.rmtree(checkout / "docs")
        report = orient["state"](checkout)
        assert report["pr"] is None
        assert any("pull request number" in p for p in report["problems"])

    @needs_git
    def test_merged_subjects_count(self, checkout):
        import subprocess

        def git(*args):
            subprocess.run(
                ["git", "-C", str(checkout), *args],
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        git(
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fix: something (#120)",
        )
        (checkout / ".claude/new-chat-template.md").unlink()
        assert orient["state"](checkout)["pr"] == "121"


def _trees(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        (root / "pkg").mkdir(parents=True)
    (old / "pkg/keep.txt").write_text("same\n", encoding="utf-8")
    (new / "pkg/keep.txt").write_text("same\n", encoding="utf-8")
    (old / "pkg/edit.txt").write_text("before\n", encoding="utf-8")
    (new / "pkg/edit.txt").write_text("after\n", encoding="utf-8")
    (old / "pkg/gone.txt").write_text("delete me\n", encoding="utf-8")
    body = "a long enough body to be detected as the same file\n" * 5
    (old / "pkg/moved_from.txt").write_text(body, encoding="utf-8")
    (new / "pkg/moved_to.txt").write_text(body, encoding="utf-8")
    (new / "pkg/new.bin").write_bytes(bytes(range(256)))
    (new / "pkg/__pycache__").mkdir()
    (new / "pkg/__pycache__/x.pyc").write_bytes(b"cache")
    return old, new


@needs_git
class TestDeliver:
    def test_delivers_adds_edits_deletes_renames_and_binaries(self, tmp_path):
        old, new = _trees(tmp_path)
        report = deliver["deliver"](old, new, "round", tmp_path / "out")
        assert report["removed"] == ["pkg/gone.txt", "pkg/moved_from.txt"]
        assert set(report["status"]) == {"A", "D", "M", "R"}
        names = zipfile.ZipFile(report["zip"]).namelist()
        assert names == [
            "REMOVED_FILES.txt",
            "pkg/edit.txt",
            "pkg/moved_to.txt",
            "pkg/new.bin",
        ]

    def test_outputs_are_byte_identical_on_a_second_run(self, tmp_path):
        old, new = _trees(tmp_path)
        one = deliver["deliver"](old, new, "r", tmp_path / "a")
        two = deliver["deliver"](old, new, "r", tmp_path / "b")
        for key in ("patch", "zip"):
            assert Path(one[key]).read_bytes() == Path(two[key]).read_bytes()

    def test_equal_trees_are_refused(self, tmp_path):
        old, _ = _trees(tmp_path)
        with pytest.raises(deliver["DeliveryError"], match="equal"):
            deliver["deliver"](old, old, "r", tmp_path / "out")

    def test_trailing_whitespace_is_refused(self, tmp_path):
        old, new = _trees(tmp_path)
        (new / "pkg/edit.txt").write_text("after   \n", encoding="utf-8")
        with pytest.raises(deliver["DeliveryError"], match="apply"):
            deliver["deliver"](old, new, "r", tmp_path / "out")

    def test_the_trees_are_left_untouched(self, tmp_path):
        old, new = _trees(tmp_path)
        before = deliver["tree_differences"](old, new)
        deliver["deliver"](old, new, "r", tmp_path / "out")
        assert deliver["tree_differences"](old, new) == before
        assert (new / "pkg/__pycache__/x.pyc").is_file()
