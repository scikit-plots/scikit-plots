from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT, TESTS_ROOT

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SEC = ROOT / "_hf_spaces_proxy" / "security"


def _load_run149():
    path = TESTS_ROOT / "_hf_spaces_proxy" / "security" / "test_promote_release.py"
    spec = importlib.util.spec_from_file_location("run149_fixture_for_run170", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _hostile_git_environment(monkeypatch, tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad = tmp_path / "bad-diff"
    bad.write_text("#!/bin/sh\nexit 97\n")
    bad.chmod(0o755)
    hostile_home = tmp_path / "hostile-home"
    hostile_home.mkdir()
    (hostile_home / ".gitconfig").write_text(
        "[diff]\n\texternal = " + str(bad) + "\n"
        "[core]\n\tautocrlf = true\n\tfilemode = false\n"
    )
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(hostile_home / "xdg"))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(bad))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-tree"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "diff.external")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(bad))
    monkeypatch.setenv("LC_ALL", "tr_TR.UTF-8")
    monkeypatch.setenv("LANG", "tr_TR.UTF-8")


def test_run170_run149_patch_ignores_host_git_environment(tmp_path: Path, monkeypatch):
    run149 = _load_run149()
    _hostile_git_environment(monkeypatch, tmp_path)
    case = tmp_path / "case"
    case.mkdir()
    old, new = run149._source(case)
    patch = run149._patch(old, new, tmp_path / "case.patch").read_bytes()
    assert b"diff --git a/a.txt b/a.txt" in patch
    assert b"diff --git a/bin.sh b/bin.sh" in patch
    assert b"new file mode 100755" in patch
    assert b"-old" in patch and b"+new" in patch


def test_run170_patch_bytes_identical_across_hostile_environments(tmp_path: Path, monkeypatch):
    run149 = _load_run149()
    left_root = tmp_path / "left"
    left_root.mkdir()
    left_old, left_new = run149._source(left_root)
    left = run149._patch(left_old, left_new, tmp_path / "left.patch").read_bytes()

    _hostile_git_environment(monkeypatch, tmp_path / "hostile")
    right_root = tmp_path / "right"
    right_root.mkdir()
    right_old, right_new = run149._source(right_root)
    right = run149._patch(right_old, right_new, tmp_path / "right.patch").read_bytes()
    assert right == left


def test_run170_production_patch_replay_ignores_host_git_environment(tmp_path: Path, monkeypatch):
    run149 = _load_run149()
    case = tmp_path / "case"
    case.mkdir()
    old, new = run149._source(case)
    patch = run149._patch(old, new, tmp_path / "change.patch")
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "a.txt").write_bytes((old / "a.txt").read_bytes())

    _hostile_git_environment(monkeypatch, tmp_path / "hostile-prod")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-git")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-git")
    run149.promote._apply_patch(replay, patch)

    assert (replay / "a.txt").read_text() == "new\n"
    assert (replay / "bin.sh").read_text() == "#!/bin/sh\necho ok\n"
    assert (replay / "bin.sh").stat().st_mode & 0o777 == 0o755


def test_run170_production_git_resolution_ignores_ambient_path(tmp_path: Path, monkeypatch):
    run149 = _load_run149()
    case = tmp_path / "case"
    case.mkdir()
    old, new = run149._source(case)
    patch = run149._patch(old, new, tmp_path / "change.patch")
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "a.txt").write_bytes((old / "a.txt").read_bytes())

    fake_dir = tmp_path / "fake-bin"
    fake_dir.mkdir()
    fake_git = fake_dir / "git"
    marker_path = tmp_path / "fake-git-ran"
    fake_git.write_text(f"#!/bin/sh\ntouch {marker_path}\nexit 99\n")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_dir))
    monkeypatch.delenv("SCIKITPLOT_RELEASE_GIT_EXECUTABLE", raising=False)

    run149.promote._apply_patch(replay, patch)
    assert (replay / "a.txt").read_text() == "new\n"
    assert not marker_path.exists()


def test_run170_production_git_environment_excludes_ambient_secrets(tmp_path: Path, monkeypatch):
    run149 = _load_run149()
    for key in ("AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "GITHUB_TOKEN", "GIT_DIR", "GIT_EXTERNAL_DIFF"):
        monkeypatch.setenv(key, "secret-or-hostile")
    env = run149.promote._git_apply_environment(tmp_path)
    assert "PATH" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env
    assert "GIT_DIR" not in env
    assert "GIT_EXTERNAL_DIFF" not in env
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["LC_ALL"] == "C" and env["LANG"] == "C"


def test_run170_production_git_pin_must_be_absolute(monkeypatch):
    run149 = _load_run149()
    monkeypatch.setenv("SCIKITPLOT_RELEASE_GIT_EXECUTABLE", "relative/git")
    with pytest.raises(run149.promote.PromotionError, match="GIT_EXECUTABLE_PIN_INVALID"):
        run149.promote._trusted_git_executable()


def test_run170_git_fixture_uses_allowlisted_process_environment():
    text = (TESTS_ROOT / "_hf_spaces_proxy" / "security" / "test_promote_release.py").read_text()
    for marker in (
        "def _git_env(",
        '"GIT_CONFIG_NOSYSTEM": "1"',
        '"GIT_CONFIG_GLOBAL"',
        '"GIT_ATTR_NOSYSTEM": "1"',
        '"GIT_TERMINAL_PROMPT": "0"',
        '"LC_ALL": "C"',
        '"core.autocrlf=false"',
        '"core.filemode=true"',
        '"core.ignorecase=false"',
        "--template=",
    ):
        assert marker in text
    assert 'subprocess.run(["git"' not in text
    assert "env=_git_env(repo)" in text

    production = (SEC / "promote_release.py").read_text()
    for marker in (
        "def _trusted_git_executable(",
        "SCIKITPLOT_RELEASE_GIT_EXECUTABLE",
        "shutil.which(\"git\", path=os.defpath)",
        "def _git_apply_environment(",
        "env=env",
    ):
        assert marker in production



def test_run170_all_release_subprocesses_have_explicit_secret_free_environment():
    sites = []
    for path in sorted(SEC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess":
                continue
            if node.func.attr not in {"Popen", "run", "check_call", "check_output"}:
                continue
            env_kw = next((kw for kw in node.keywords if kw.arg == "env"), None)
            assert env_kw is not None, f"{path.name}:{node.lineno} inherits ambient environment"
            rendered = ast.unparse(env_kw.value)
            assert "os.environ" not in rendered, f"{path.name}:{node.lineno} inherits ambient environment"
            if path.name != "promote_release.py":
                assert "os.defpath" in rendered, f"{path.name}:{node.lineno} does not pin adapter PATH"
            sites.append((path.name, node.lineno))
    assert len(sites) == 14


def test_run170_command_adapter_cannot_read_parent_secret(tmp_path: Path, monkeypatch):
    path = SEC / "publish_release.py"
    spec = importlib.util.spec_from_file_location("run170_publish_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    script = tmp_path / "env-check.py"
    script.write_text(
        "import json,os,sys\n"
        "json.load(sys.stdin)\n"
        "if os.environ.get('RUN170_CANARY_SECRET'): sys.exit(91)\n"
        "print(json.dumps({'ok': True, 'path': os.environ.get('PATH'), 'locale': os.environ.get('LC_ALL')}))\n"
    )
    monkeypatch.setenv("RUN170_CANARY_SECRET", "must-not-cross-process-boundary")
    # Use the already-resolved interpreter as the executable.  The production
    # adapter intentionally pins PATH to os.defpath, so a virtualenv-only
    # ``#!/usr/bin/env python3`` shebang is not a hermetic execution contract.
    adapter = module.command_publisher([sys.executable, str(script)])
    result = adapter({"probe": True})
    assert result == {"ok": True, "path": os.defpath, "locale": "C"}

def test_run170_documentation_and_release_gates_are_wired():
    guide = SEC / "RELEASE_PROCESS_HERMETICITY_GUIDE.md"
    assert guide.exists()
    text = guide.read_text().lower()
    for phrase in ("run 170", "git", "environment", "locale", "production"):
        assert phrase in text
    gates = (SEC / "SECURITY_RELEASE_GATES.md").read_text().lower()
    evidence = (SEC / "RELEASE_EVIDENCE_GUIDE.md").read_text().lower()
    readme = (ROOT / "README.md").read_text().lower()
    assert "run 170" in gates and "run 170" in evidence and "run 170" in readme
