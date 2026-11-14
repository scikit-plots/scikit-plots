"""Tests for ``tools/maint_tools/security_review.py`` (no network, no git)."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "security_review.py"
_SPEC = importlib.util.spec_from_file_location("_security_review", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
sr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = sr
_SPEC.loader.exec_module(sr)

# Built at run time so this file holds no credential-shaped literal.
_GH_TOKEN = "ghp_" + "A1b2C3d4" * 5
_KEY_HEADER = "-----BEGIN RSA " + "PRIVATE KEY-----"
_PASSWORD_LINE = "pass" + "word = 'hunter2hunter2'"
_SESSION_LINK = "see https://claude.ai/code/" + "session_" + "0" * 24


def _diff(path: str, added=(), removed=(), status: str = "modified", start: int = 10) -> str:
    """A minimal ``git diff -U0`` for one file."""
    head = [f"diff --git a/{path} b/{path}"]
    if status == "added":
        head += ["new file mode 100644", "--- /dev/null", f"+++ b/{path}"]
    elif status == "deleted":
        head += ["deleted file mode 100644", f"--- a/{path}", "+++ /dev/null"]
    else:
        head += [f"--- a/{path}", f"+++ b/{path}"]
    body = [f"@@ -{start},{len(removed)} +{start},{len(added)} @@"]
    body += [f"-{line}" for line in removed] + [f"+{line}" for line in added]
    return "\n".join(head + body) + "\n"


def _hits(text: str, check: str | None = None) -> list:
    hits = sr.review(sr.parse_diff(text))
    return [h for h in hits if check is None or h.check == check]


class TestParse:
    def test_line_numbers_follow_the_hunk(self):
        (change,) = sr.parse_diff(_diff("a.py", ["x", "y"], ["old"], start=7))
        assert change.added == [(7, "x"), (8, "y")]
        assert change.removed == [(7, "old")]

    def test_added_deleted_renamed(self):
        text = (
            _diff("new.py", ["x"], status="added")
            + _diff("gone.py", removed=["x"], status="deleted")
            + "diff --git a/o.py b/n.py\nsimilarity index 90%\nrename from o.py\nrename to n.py\n"
        )
        assert [(c.path, c.status) for c in sr.parse_diff(text)] == [
            ("new.py", "added"),
            ("gone.py", "deleted"),
            ("n.py", "renamed"),
        ]

    def test_binary_and_empty(self):
        text = "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        assert [(c.path, c.added) for c in sr.parse_diff(text)] == [("i.png", [])]
        assert sr.parse_diff("") == []

    def test_a_removed_line_starting_with_dashes_is_not_a_header(self):
        (change,) = sr.parse_diff(_diff("a.md", removed=["-- note"]))
        assert change.removed == [(10, "-- note")]


class TestS1:
    @pytest.mark.parametrize(
        "line", [_KEY_HEADER, f"token = '{_GH_TOKEN}'", _PASSWORD_LINE]
    )
    def test_a_credential_fails_and_is_never_printed(self, line):
        hits = _hits(_diff("cfg.py", [line]), "S1")
        assert hits and all(h.fails for h in hits)
        out = sr.render(hits)
        assert _GH_TOKEN not in out and "hunter2" not in out and "PRIVATE KEY" not in out
        assert "`cfg.py:10`" in out

    def test_a_credential_in_documentation_still_fails(self):
        assert all(h.fails for h in _hits(_diff("README.md", [f"`{_GH_TOKEN}`"]), "S1"))

    @pytest.mark.parametrize(
        "line",
        [
            _SESSION_LINK,
            "Author: someone@users." + "noreply.github.com",
        ],
    )
    def test_identity_is_reported_not_failed(self, line):
        hits = _hits(_diff("notes.md", [line]), "S1")
        assert len(hits) == 1 and not hits[0].fails and hits[0].kind.endswith("doc")

    @pytest.mark.parametrize(
        "line",
        [
            "password = os.environ['PW']",
            "token = ''",
            "api_key: ${{ secrets.KEY }}",
            "`<name>@users.noreply.github.com` is a user's address",
            "pattern: session_[0-9a-z]+",
        ],
    )
    def test_not_a_secret(self, line):
        assert _hits(_diff("a.py", [line]), "S1") == []


class TestChecks:
    @pytest.mark.parametrize(
        ("check", "line"),
        [
            ("S3", "subprocess.run(cmd)"),
            ("S3", "value = eval(text)"),
            ("S3", "pickle.loads(blob)"),
            ("S3", "run(cmd, shell=True)"),
            ("S4", "urllib.request.urlopen(url)"),
            ("S4", "requests.get(url)"),
            ("S5", "archive.extractall(target)"),
            ("S5", "open(path, 'wb')"),
            ("S6", "tarfile.open(path)"),
            ("S6", "yaml.load(fh)"),
        ],
    )
    def test_code_hits(self, check, line):
        assert [h.check for h in _hits(_diff("m.py", [line]))] == [check]

    @pytest.mark.parametrize(
        "line", ["evaluate(x)", "my_subprocess_count = 1", "yaml.safe_load(fh)", "open(path)"]
    )
    def test_code_non_hits(self, line):
        assert _hits(_diff("m.py", [line])) == []

    def test_code_patterns_skip_documentation(self):
        assert _hits(_diff("guide.md", ["run subprocess.run(cmd) here"])) == []

    def test_dependencies(self):
        hits = _hits(_diff("requirements/core.txt", ["numpy==2.1.0", "scipy>=1.10"]), "S2")
        kinds = sorted(h.kind for h in hits)
        assert kinds == ["dependency file modified", "exact pin"]

    def test_workflow(self):
        lines = [
            "uses: actions/checkout@v4",
            "uses: actions/checkout@" + "a" * 40 + " # v4",
            "uses: ./.github/actions/local",
            "contents: write",
            "on: pull_request_target",
            "run: echo ${{ github.event.pull_request.title }}",
        ]
        kinds = {h.kind for h in _hits(_diff(".github/workflows/ci.yml", lines), "S7")}
        assert kinds == {
            "modified",
            "action not pinned to a commit",
            "write permission",
            "pull_request_target",
            "event data in an expression",
        }
        pinned = _hits(_diff(".github/workflows/ci.yml", [lines[1]]), "S7")
        assert [h.kind for h in pinned] == ["modified"]

    def test_build_and_tool_paths(self):
        for path in ("tools/x.py", "scikitplot/_build_utils/y.py", "meson.build", "sub/meson.build"):
            assert [h.kind for h in _hits(_diff(path, ["x = 1"]), "S7")] == ["modified"]
        assert _hits(_diff("scikitplot/corpus/a.py", ["x = 1"]), "S7") == []


class TestS8:
    def test_removed_test_function(self):
        hits = _hits(_diff("pkg/tests/test_a.py", removed=["def test_one():"]), "S8")
        assert [h.kind for h in hits] == ["test_one removed (old line)"]

    def test_a_moved_or_reformatted_test_is_not_removed(self):
        text = _diff("pkg/tests/test_a.py", ["def test_one(x):"], ["def test_one():"])
        assert _hits(text, "S8") == []

    def test_deleted_test_file_and_markers(self):
        assert [h.kind for h in _hits(_diff("pkg/tests/test_b.py", removed=["x"], status="deleted"), "S8")] == ["test file deleted"]
        marks = ["@pytest.mark.skip(reason='x')", "@pytest.mark.xfail(strict=True)", "pytest.skip('no')"]
        assert len(_hits(_diff("pkg/tests/test_c.py", marks), "S8")) == 3

    def test_skip_in_library_code_is_not_s8(self):
        assert _hits(_diff("pkg/a.py", ["pytest.skip('x')"]), "S8") == []


class TestRender:
    def test_every_check_has_a_row_and_none_when_clean(self):
        out = sr.render([])
        for check, _, _ in sr.CHECKS:
            assert f"| {check} |" in out
        assert out.count("| none |") == 8
        assert "approval" not in out

    def test_approval_named_for_code_not_for_documentation(self):
        code = sr.render(_hits(_diff("m.py", ["subprocess.run(c)"])))
        assert "approval:** S3" in code
        doc = sr.render([sr.Hit("S3", "a.md", 1, "doc", "subprocess")])
        assert "approval" not in doc

    def test_a_pipe_does_not_break_the_table(self):
        out = sr.render(_hits(_diff("m.py", ["eval(a | b)"])))
        row = next(line for line in out.splitlines() if line.startswith("| S3 |"))
        assert row.replace("\\|", "").count("|") == 4  # four cell borders
        assert "eval(a \\| b)" in row

    def test_long_lists_are_cut(self):
        text = _diff("m.py", [f"eval(x{i})" for i in range(30)])
        out = sr.render(_hits(text), max_per_check=5)
        assert "30: " in out and "... and 25 more" in out

    def test_deterministic(self):
        text = _diff("m.py", ["eval(a)", "requests.get(u)"]) + _diff("tools/t.py", ["x"])
        assert sr.render(_hits(text)) == sr.render(_hits(text))
        reordered = _diff("tools/t.py", ["x"]) + _diff("m.py", ["eval(a)", "requests.get(u)"])
        assert sr.render(_hits(text)) == sr.render(_hits(reordered))


class TestMain:
    def test_exit_status(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "stdin", io.StringIO(_diff("m.py", ["eval(x)"])))
        assert sr.main(["-"]) == 0
        monkeypatch.setattr(sys, "stdin", io.StringIO(_diff("m.py", [_KEY_HEADER])))
        assert sr.main(["-"]) == 1
        assert "rotate the secret" in capsys.readouterr().out

    def test_a_diff_file(self, tmp_path, capsys):
        path = tmp_path / "d.diff"
        path.write_text(_diff("m.py", ["requests.get(u)"]), encoding="utf-8")
        assert sr.main([str(path)]) == 0
        assert "| S4 | Network access | 1:" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [[], ["-", "--base", "main"]])
    def test_usage(self, argv):
        with pytest.raises(SystemExit) as caught:
            sr.main(argv)
        assert caught.value.code == 2

    def test_base_uses_the_merge_base(self, tmp_path):
        def git(*args):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.invalid")
        git("config", "user.name", "t")
        (tmp_path / "a.py").write_text("x = 1\n")
        git("add", "."), git("commit", "-qm", "base")
        git("checkout", "-qb", "topic")
        (tmp_path / "b.py").write_text("eval(x)\n")
        git("add", "."), git("commit", "-qm", "topic")
        git("checkout", "-q", "main")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_c.py").write_text("def test_c():\n    pass\n")
        git("add", "."), git("commit", "-qm", "main moves on")
        done = subprocess.run(
            [sys.executable, str(_PATH), "--base", "main", "--head", "topic"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "| S3 | Code that runs other code | 1:" in done.stdout
        assert "| S8 | Tests weakened | none |" in done.stdout  # main's new test is not "removed"
