# scikitplot/_env/tests/test_env_executor.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The executor: approvals first, lock, steps, failures, receipt, events."""

from __future__ import annotations

import io
import json
import logging
import os
import socket
from pathlib import Path
from unittest import mock

import pytest

from .._core import _executor
from .._core._decisions import Decider
from .._core._errors import (
    ApprovalRequiredError,
    ConfigError,
    ExecutionError,
    LockBusyError,
    PlanningError,
)
from .._core._events import EventSink
from .._core._executor import execute
from .._core._lock import PrefixLock
from .._core._planner import plan_request
from .._core._receipt import RECEIPT_NAME, read_receipt
from .._core._request import EnvironmentRequest, Policy
from ._helpers import FakeRunner, fake_host, make_venv_like, write


def _plan(tmp_path, operation="ensure", **fields):
    host = fake_host(tmp_path, tools={"uv": "/opt/uv"})
    fields.setdefault("prefix", str(tmp_path / "env"))
    return plan_request(EnvironmentRequest(**fields), operation=operation, host=host)


def _fake_create(runner):
    """A runner that also makes the prefix look built after ``uv venv``."""

    def run(argv, *, cwd, env):
        if list(argv[1:2]) == ["venv"]:
            make_venv_like(Path(argv[2]))
        return runner(argv, cwd=cwd, env=env)

    return run


class TestRun:
    def test_success_writes_receipt_and_events(self, tmp_path):
        plan = _plan(tmp_path, packages=["six"], verify_imports=["six"])
        runner = FakeRunner()
        stream = io.StringIO()
        result = execute(plan, runner=_fake_create(runner), events=EventSink(stream),
                         environ={"PATH": "/bin", "PYTHONPATH": "/evil", "HOME": "/h"})
        assert result.ok
        assert result.changed
        assert [s.id for s in result.steps] == ["create", "install-pypi", "verify"]
        assert all("PYTHONPATH" not in c["env"] for c in runner.calls)
        assert runner.calls[0]["env"]["UV_NO_PROGRESS"] == "1"
        receipt = read_receipt(plan.prefix)
        assert receipt["state_digest"] == plan.state_digest
        assert receipt["backend"] == "uv"
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert [e["event"] for e in events] == [
            "plan.started", "step.started", "step.finished", "step.started",
            "step.finished", "step.started", "step.finished", "plan.finished"]
        assert [e["seq"] for e in events] == list(range(1, 9))
        assert all(e["schema"] == "scikitplot.env.event/v1" for e in events)
        assert not os.path.exists(plan.prefix + ".scikitplot-env.lock")

    def test_up_to_date_runs_nothing(self, tmp_path):
        plan = _plan(tmp_path)
        execute(plan, runner=_fake_create(FakeRunner()))
        again = _plan(tmp_path)
        assert again.up_to_date
        runner = FakeRunner()
        result = execute(again, runner=runner)
        assert result.ok
        assert not result.changed
        assert runner.calls == []

    def test_failure_carries_partial_result_and_no_receipt(self, tmp_path):
        plan = _plan(tmp_path, packages=["six"])
        runner = FakeRunner(fail_on="six", output="x\nERROR: https://u:pw@h/simple bad\n")
        with pytest.raises(ExecutionError) as info:
            execute(plan, runner=_fake_create(runner))
        exc = info.value
        assert exc.step == "install-pypi"
        assert exc.returncode == 1
        assert "pw" not in str(exc)
        assert "***@" in str(exc)
        assert [(s.id, s.ok) for s in exc.result.steps] == [
            ("create", True), ("install-pypi", False)]
        assert not (tmp_path / "env" / RECEIPT_NAME).exists()
        assert not os.path.exists(plan.prefix + ".scikitplot-env.lock")

    def test_missing_executable(self, tmp_path):
        plan = _plan(tmp_path)

        def broken(argv, *, cwd, env):
            raise FileNotFoundError(2, "No such file", argv[0])

        with pytest.raises(ExecutionError, match="cannot run") as info:
            execute(plan, runner=broken)
        assert "doctor" in info.value.hint

    def test_plan_only_target(self, tmp_path):
        plan = _plan(tmp_path, target="pyodide", packages=["six"])
        with pytest.raises(PlanningError, match="plan-only"):
            execute(plan)


class TestApprovals:
    def test_refused_approval_runs_nothing(self, tmp_path):
        make_venv_like(tmp_path / "env")
        plan = _plan(tmp_path, "recreate")
        runner = FakeRunner()
        actions = {"remove_tree": mock.Mock()}
        with pytest.raises(ApprovalRequiredError) as info:
            execute(plan, decider=Decider("never"), runner=runner, actions=actions)
        assert info.value.kind == "remove"
        assert runner.calls == []
        assert not actions["remove_tree"].called
        assert (tmp_path / "env" / "pyvenv.cfg").exists()

    def test_approved_by_kind_and_callback(self, tmp_path):
        make_venv_like(tmp_path / "env")
        plan = _plan(tmp_path, "remove")
        seen = []
        result = execute(plan, decider=Decider(
            "never", approve=lambda kind, msg: seen.append((kind, msg)) or True))
        assert result.ok
        assert not (tmp_path / "env").exists()
        assert seen[0][0] == "remove"
        assert str(tmp_path / "env") in seen[0][1]
        assert result.approvals[0]["how"] == "callback"

    def test_terminal_question(self, tmp_path):
        make_venv_like(tmp_path / "env")
        plan = _plan(tmp_path, "remove")
        tty = io.StringIO("y\n")
        tty.isatty = lambda: True
        decider = Decider("always", stdin=tty, stderr=io.StringIO(), environ={})
        assert execute(plan, decider=decider).ok

    def test_auto_does_not_ask_in_ci(self, tmp_path):
        make_venv_like(tmp_path / "env")
        tty = io.StringIO("y\n")
        tty.isatty = lambda: True
        decider = Decider("auto", stdin=tty, stderr=io.StringIO(), environ={"CI": "1"})
        with pytest.raises(ApprovalRequiredError):
            execute(_plan(tmp_path, "remove"), decider=decider)


class TestActions:
    def test_remove_tree_refuses_unknown_directory(self, tmp_path):
        write(tmp_path / "data" / "x.txt", "keep")
        with pytest.raises(ExecutionError, match="not an environment"):
            _executor._remove_tree(str(tmp_path / "data"))
        assert (tmp_path / "data" / "x.txt").exists()
        _executor._remove_tree(str(tmp_path / "missing"))  # nothing to do

    def test_unknown_action(self, tmp_path):
        plan = _plan(tmp_path, conda_packages=["x"], prefix=str(tmp_path / "e"),
                     policy=Policy(bootstrap="if-missing"))
        with pytest.raises(ExecutionError, match="unknown action"):
            execute(plan, actions={}, decider=Decider("never", approve=True),
                    runner=FakeRunner())

    def test_bootstrap_uv_finds_one_executable(self, tmp_path):
        def fake_pip(argv, **kwargs):
            assert "--require-hashes" in argv and "--only-binary=:all:" in argv
            pinned = Path(argv[argv.index("-r") + 1]).read_text()
            assert pinned.startswith("uv==0.11.32 --hash=sha256:")
            target = argv[argv.index("--target") + 1]
            write(Path(target) / "bin" / "uv", "#!bin")
            return mock.Mock(returncode=0, stdout="")

        _executor._bootstrap_uv(str(tmp_path / "tools"), "uv==0.11.32", "/py",
                                run=fake_pip)
        assert (tmp_path / "tools" / "bin" / "uv").read_text() == "#!bin"

        def no_binary(argv, **kwargs):
            return mock.Mock(returncode=0, stdout="")

        with pytest.raises(ExecutionError, match="expected one uv"):
            _executor._bootstrap_uv(str(tmp_path / "t2"), "uv==0.11.32", "/py",
                                    run=no_binary)
        with pytest.raises(ExecutionError, match="pinned requirement"):
            _executor._bootstrap_uv(str(tmp_path / "t3"), "uv", "/py", run=no_binary)


class TestLock:
    def test_busy_and_stale(self, tmp_path):
        prefix = tmp_path / "env"
        with PrefixLock(prefix), pytest.raises(LockBusyError):
            PrefixLock(prefix).acquire()
        lock = tmp_path / "env.scikitplot-env.lock"
        write(lock, json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345}))
        with PrefixLock(prefix):  # a dead owner on this host is replaced
            pass
        write(lock, json.dumps({"host": "other-host", "pid": 1}))
        with pytest.raises(LockBusyError, match="other-host"):
            PrefixLock(prefix).acquire()


def test_events_file_errors(tmp_path):
    write(tmp_path / "f", "")
    with pytest.raises(ConfigError, match="events file"):
        EventSink(tmp_path / "f" / "events.jsonl")


def test_events_are_redacted(tmp_path):
    stream = io.StringIO()
    EventSink(stream).emit("x", url="https://a:secret@h/p?token=abc")
    assert "secret" not in stream.getvalue()
    assert "abc" not in stream.getvalue()


def test_executor_logs_command_output(tmp_path, captured_logger, caplog):
    plan = _plan(tmp_path)
    with captured_logger(logging.DEBUG, _executor.logger):
        execute(plan, runner=_fake_create(FakeRunner(output="hello from tool\n")))
    assert any("hello from tool" in r.getMessage() for r in caplog.records)


def test_steps_that_build_nothing_fail_the_run(tmp_path):
    plan = _plan(tmp_path)
    with pytest.raises(ExecutionError, match="does not exist") as info:
        execute(plan, runner=FakeRunner())  # every command "succeeds", no prefix
    assert info.value.result.ok is False
    assert not (tmp_path / "env" / RECEIPT_NAME).exists()


def test_tampered_bootstrap_url_is_refused(tmp_path):
    with pytest.raises(ExecutionError, match="pinned archive"):
        _executor._bootstrap_micromamba(str(tmp_path), "linux-64", False,
                                        "https://evil.example/mm.tar.bz2", "0" * 64)
    assert not (tmp_path / "bin").exists()


class TestReviewFindings:
    def test_concurrent_run_finds_the_work_done(self, tmp_path):
        first = _plan(tmp_path)
        second = _plan(tmp_path)  # planned before the first one ran
        execute(first, runner=_fake_create(FakeRunner()))
        runner = FakeRunner()
        result = execute(second, runner=runner)
        assert result.ok
        assert not result.changed
        assert runner.calls == []

    def test_environment_vanishing_while_waiting_is_stale(self, tmp_path):
        execute(_plan(tmp_path), runner=_fake_create(FakeRunner()))
        plan = _plan(tmp_path, packages=["six"])  # an update plan
        _executor._remove_tree(plan.prefix)
        with pytest.raises(ExecutionError) as info:
            execute(plan, runner=FakeRunner())
        assert info.value.code == "plan_stale"

    def test_receipt_is_readable_by_others(self, tmp_path):
        plan = _plan(tmp_path)
        execute(plan, runner=_fake_create(FakeRunner()))
        assert ((tmp_path / "env" / RECEIPT_NAME).stat().st_mode & 0o777) == 0o644

    def test_stale_lock_breaking_never_removes_a_live_lock(self, tmp_path):
        prefix = tmp_path / "env"
        lock = PrefixLock(prefix)
        dead = {"host": socket.gethostname(), "pid": 2**22 + 7, "token": "dead"}
        write(lock.path, json.dumps({**dead, "token": "live"}))
        lock._break_stale(dead, "mine")  # the file changed owner since judged stale
        assert json.loads(lock.path.read_text())["token"] == "live"
        other = PrefixLock(prefix)
        other._held, other._token = True, "not-the-owner"
        other.release()
        assert lock.path.exists()  # release never removes someone else's lock


def test_always_needs_a_terminal(tmp_path):
    make_venv_like(tmp_path / "env")
    piped = io.StringIO("y\n")  # not a terminal: piped "y" is not a person
    decider = Decider("always", stdin=piped, stderr=io.StringIO(), environ={})
    with pytest.raises(ApprovalRequiredError):
        execute(_plan(tmp_path, "remove"), decider=decider)
