# scikitplot/_ipymagic/tests/test_ipymagic_bridge.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The engine: splitting, routing, in-process isolation, subprocess lifecycle."""

from __future__ import annotations

import _thread
import contextlib
import io
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from ..._cli import logging as cli_logging
from .. import _bridge
from .._bridge import route, run_command, split_line, wants_json
from .._errors import CommandFailedError, MagicUsageError
from .._settings import MagicSettings


def _quiet(**kw):
    return {"out": io.StringIO(), "err": io.StringIO(), **kw}


#: Process groups, ``/proc`` and ``os.kill(pid, 0)`` (which ends a process on
#: Windows) are POSIX.
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def _child(code: str):
    """Make the 'scikitplot' child run ``python -c code`` instead."""
    return mock.patch.object(
        _bridge, "_child_argv", autospec=True,
        side_effect=lambda argv: [sys.executable, "-c", code, *argv],
    )


class TestSplitAndRoute:
    def test_split(self):
        assert split_line("env plan -n 'a b' --x=1") == ["env", "plan", "-n", "a b", "--x=1"]
        assert split_line('"C:\\Users\\me"') == ["C:\\Users\\me"]
        with pytest.raises(MagicUsageError, match="split") as info:
            split_line("env 'open")
        assert info.value.hint == "close the quote"

    def test_route(self):
        assert route([]) == _bridge.Route(None, False)
        assert route(["-vv", "-q", "info"]) == _bridge.Route("info", False)
        assert route(["env", "plan"]) == _bridge.Route("env", True)
        assert route(["clean-prompt"]) == _bridge.Route("cleanprompt", True)  # alias
        assert route(["--bogus"]) == _bridge.Route(None, False)  # CLI reports
        assert route(["--bogus", "info"]) == _bridge.Route("info", False)

    @pytest.mark.parametrize("argv", [
        ["-vq", "env", "plan"], ["-Vv", "env"], ["--verb", "cleanprompt", "redact"],
        ["-vvq", "mcp"], ["-v", "--", "env"],
    ])
    def test_route_delegates_as_the_cli_does(self, argv):
        # Regression: combined and abbreviated root flags once routed a
        # delegated command in-process (no timeout, kernel stdin).
        from ..._cli._frontends._argparse import _split_delegated

        assert _split_delegated(argv) is not None
        assert route(argv).delegated

    def test_unknown_command_is_never_guessed(self):
        with pytest.raises(MagicUsageError) as info:
            route(["evn", "create", "--yes"])
        assert "'evn' is not a scikitplot command" in str(info.value)
        assert info.value.hint == "did you mean 'env'?"
        with pytest.raises(MagicUsageError) as info:
            route(["zzzzzz"])
        assert info.value.hint.startswith("commands: ")

    @pytest.mark.parametrize("argv, expected", [
        (["info", "--format", "json"], True),
        (["info", "--format=json"], True),
        (["info", "--format", "yaml"], False),
        (["info", "--format"], False),
        (["cleanprompt", "redact", "--format json"], False),
    ])
    def test_wants_json(self, argv, expected):
        assert wants_json(argv) is expected


class TestInProcess:
    def test_json_data_and_nothing_printed(self):
        streams = _quiet()
        result = run_command("info --format json", **streams)
        assert result.ok and result.mode == "in-process" and result.command == "info"
        assert result.data == json.loads(result.stdout)
        assert streams["out"].getvalue() == ""  # shown as data, not as text

    def test_text_is_shown(self):
        streams = _quiet()
        result = run_command(["info"], **streams)
        assert result.data is None
        assert streams["out"].getvalue() == result.stdout != ""

    def test_kernel_logging_state_is_restored(self):
        target = logging.getLogger("scikitplot")
        mine = logging.StreamHandler(io.StringIO())
        target.addHandler(mine)
        before = (target.level, list(target.handlers), target.propagate)
        saved = cli_logging._HANDLER
        try:
            run_command("-vv info", **_quiet())
            run_command("-q doctor --format json", **_quiet())
            assert (target.level, list(target.handlers), target.propagate) == before
            assert cli_logging._HANDLER is saved
        finally:
            target.removeHandler(mine)

    def test_existing_cli_handler_keeps_its_stream(self):
        cli_logging.configure(0)  # as if the kernel ran the CLI itself before
        handler = cli_logging._HANDLER
        stream = handler.stream
        try:
            run_command("info", **_quiet())
            assert cli_logging._HANDLER is handler and handler.stream is stream
        finally:
            logging.getLogger("scikitplot").removeHandler(handler)
            logging.getLogger("scikitplot").propagate = True
            cli_logging._HANDLER = None

    def test_usage_error_exit_status(self):
        streams = _quiet()
        result = run_command("info --format xml", **streams)
        assert result.exit_code == 2
        assert "invalid choice" in streams["err"].getvalue()

    def test_raise_on_error(self):
        settings = MagicSettings(raise_on_error=True)
        with pytest.raises(CommandFailedError) as info:
            run_command("info --format xml", settings=settings, **_quiet())
        assert info.value.result.exit_code == 2
        assert info.value._render_traceback_()[0].startswith("CommandFailedError [")

    def test_display_is_not_cut(self, monkeypatch):
        monkeypatch.setattr(_bridge, "MAX_CAPTURE", 20)
        streams = _quiet()
        result = run_command("info", **streams)
        assert result.truncated and len(result.stdout) == 20
        assert len(streams["out"].getvalue()) > 20

    def test_interrupt_is_re_raised(self, monkeypatch):
        # Regression: the CLI's main turns Ctrl-C into exit 130, which let
        # "Run all" continue after an interrupted in-process command.
        from ..._cli._frontends import _argparse

        monkeypatch.setenv("SCIKITPLOT_CLI_FRONTEND", "argparse")

        before = signal.getsignal(signal.SIGINT)
        timer = threading.Timer(0.5, _thread.interrupt_main)
        def slow(argv=None):  # short sleeps: interrupt_main wakes no sleep
            end = time.monotonic() + 10
            while time.monotonic() < end:
                time.sleep(0.01)
            return 0

        with mock.patch.object(_argparse, "run", autospec=True, side_effect=slow):
            timer.start()
            start = time.monotonic()
            with pytest.raises(KeyboardInterrupt):
                run_command("info", **_quiet())
        timer.cancel()
        assert time.monotonic() - start < 5
        assert signal.getsignal(signal.SIGINT) is before

    def test_bad_arguments(self):
        with pytest.raises(MagicUsageError, match="not str"):
            run_command(["info", 3])
        with pytest.raises(MagicUsageError, match="stdin"):
            run_command("info", stdin=b"x")


class TestSubprocess:
    def test_delegated_command_runs_in_a_child(self):
        streams = _quiet()
        result = run_command("env --format json doctor", **streams)
        assert result.ok and result.mode == "subprocess" and result.command == "env"
        assert "backends" in result.data
        assert streams["out"].getvalue() == ""

    def test_streaming_order_and_capture(self):
        code = ("import sys,time\n"
                "for i in range(3):\n"
                "    print('line', i, flush=True); time.sleep(0.05)\n"
                "print('warn', file=sys.stderr, flush=True)\n"
                "sys.exit(3)")
        streams = _quiet()
        with _child(code):
            result = run_command("env plan", **streams)
        assert result.exit_code == 3
        assert streams["out"].getvalue() == "line 0\nline 1\nline 2\n" == result.stdout
        assert streams["err"].getvalue() == "warn\n" == result.stderr

    def test_cell_text_is_stdin_and_forces_a_child(self):
        streams = _quiet()
        with _child("import sys; print(sys.stdin.read().upper())"):
            result = run_command("info", stdin="hello\nworld", **streams)
        assert result.mode == "subprocess"
        assert result.stdout == "HELLO\nWORLD\n"

    @posix_only
    def test_timeout_stops_the_child_and_its_children(self, tmp_path):
        pidfile = tmp_path / "pid"
        code = ("import subprocess,sys,time\n"
                "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                f"open({str(pidfile)!r},'w').write(str(c.pid))\n"
                "print('started', flush=True); time.sleep(60)")
        streams = _quiet()
        start = time.monotonic()
        with _child(code):
            result = run_command("env plan", settings=MagicSettings(timeout=2),
                                 **streams)
        assert time.monotonic() - start < 2 + _bridge.STOP_GRACE + 3
        assert result.timed_out and result.exit_code == _bridge.EXIT_TIMEOUT
        assert "stopped after the 2 s timeout" in streams["err"].getvalue()
        grandchild = int(pidfile.read_text())
        time.sleep(0.2)
        assert not _alive(grandchild)

    def test_interrupt_stops_the_child_and_is_re_raised(self):
        code = "import time; print('x', flush=True); time.sleep(60)"
        timer = threading.Timer(1.0, _thread.interrupt_main)
        with _child(code), pytest.raises(KeyboardInterrupt):
            timer.start()
            run_command("env plan", **_quiet())
        timer.cancel()

    @posix_only
    def test_lingering_grandchild_does_not_hang_the_cell(self, monkeypatch):
        monkeypatch.setattr(_bridge, "STOP_GRACE", 1.0)
        code = ("import subprocess,sys\n"
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                "print('done', flush=True)")
        start = time.monotonic()
        with _child(code):
            result = run_command("env plan", **_quiet())
        assert result.ok and result.stdout == "done\n"
        assert time.monotonic() - start < 10

    @posix_only
    def test_exited_command_is_not_timed_out_and_leftovers_stop(
        self, monkeypatch, tmp_path
    ):
        # Regression: a command that exited 0 while a process it started held
        # the pipes waited out that process (20 s) and was reported "timed out".
        monkeypatch.setattr(_bridge, "STOP_GRACE", 30.0)  # the deadline comes first
        pidfile = tmp_path / "pid"
        code = ("import subprocess,sys\n"
                "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                f"open({str(pidfile)!r},'w').write(str(c.pid))\n"
                "print('done', flush=True)")
        start = time.monotonic()
        with _child(code):
            result = run_command("env plan", settings=MagicSettings(timeout=1),
                                 **_quiet())
        assert time.monotonic() - start < 10
        assert result.ok and not result.timed_out and result.stdout == "done\n"
        time.sleep(0.2)
        assert not _alive(int(pidfile.read_text()))

    @posix_only
    def test_process_in_its_own_session_does_not_hang_the_cell(
        self, monkeypatch, tmp_path
    ):
        # Regression: closing a pipe that a reader thread was blocked on hung
        # the cell until a process that escaped the group (setsid) exited.
        monkeypatch.setattr(_bridge, "STOP_GRACE", 1.0)
        pidfile = tmp_path / "pid"
        code = ("import subprocess,sys\n"
                "c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
                " start_new_session=True)\n"
                f"open({str(pidfile)!r},'w').write(str(c.pid))\n"
                "print('done', flush=True)")
        start = time.monotonic()
        try:
            with _child(code):
                result = run_command("env plan", **_quiet())
            assert time.monotonic() - start < 10
            assert result.ok and result.stdout == "done\n"
        finally:
            if pidfile.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)

    def test_capture_is_bounded_but_display_is_not(self, monkeypatch):
        monkeypatch.setattr(_bridge, "MAX_CAPTURE", 10)
        streams = _quiet()
        with _child("print('x' * 50)"):
            result = run_command("env plan", **streams)
        assert result.truncated and len(result.stdout) == 10
        assert streams["out"].getvalue() == "x" * 50 + "\n"

    def test_undecodable_output_is_replaced(self):
        with _child("import sys; sys.stdout.buffer.write(b'ok \\xff\\n')"):
            result = run_command("env plan", **_quiet())
        assert result.stdout == "ok \ufffd\n"

    def test_json_requested_but_not_json(self):
        streams = _quiet()
        with _child("print('not json')"):
            result = run_command("env --format json doctor", **streams)
        assert result.data is None
        assert streams["out"].getvalue() == "not json\n"  # held back, then shown

    def test_child_that_cannot_start(self):
        with mock.patch.object(_bridge, "_child_argv", autospec=True,
                               return_value=["/no/such/python"]):
            streams = _quiet()
            result = run_command("env plan", **streams)
        assert result.exit_code == _bridge.EXIT_NOT_STARTED
        assert "cannot start /no/such/python" in result.stderr

    def test_isolate_runs_native_commands_in_a_child(self):
        result = run_command("info --format json", settings=MagicSettings(isolate=True),
                             **_quiet())
        assert result.mode == "subprocess" and result.ok
        assert result.data["python"]["executable"] == sys.executable

    def test_output_shown_after_the_run_says_it_was_cut(self, monkeypatch):
        monkeypatch.setattr(_bridge, "MAX_CAPTURE", 10)
        streams = _quiet()
        with _child("print('x' * 50)"):
            run_command("env plan", settings=MagicSettings(stream=False), **streams)
        assert streams["out"].getvalue() == "x" * 10
        assert "output cut to the first 10 characters" in streams["err"].getvalue()

    def test_no_stream_shows_output_at_the_end(self):
        streams = _quiet()
        with _child("print('a', flush=True)"):
            run_command("env plan", settings=MagicSettings(stream=False), **streams)
        assert streams["out"].getvalue() == "a\n"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover
        return True
    status = Path(f"/proc/{pid}/stat")
    return not (status.exists() and status.read_text().split()[2] == "Z")


def test_debug_record_for_each_run(captured_logger, caplog):
    with captured_logger(logging.DEBUG, _bridge.logger):
        run_command("info", **_quiet())
    assert any("scikitplot info: exit 0" in r.getMessage() for r in caplog.records)
