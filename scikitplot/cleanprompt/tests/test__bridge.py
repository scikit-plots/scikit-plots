"""
Tests for :mod:`scikitplot.cleanprompt._bridge`.

Notes
-----
**Developer notes.** The "model" in these tests is the running Python
interpreter with a one-line script, so the tests need no model, no network
and no shell, and run the same on every platform.
"""

from __future__ import annotations

import io
import sys

import pytest

from .. import FluentCleanPrompt, LeakError
from .._bridge import run_command, split_command
from .._exceptions import CleanPromptError

_ECHO = [
    sys.executable,
    "-c",
    "import sys; sys.stdout.write('Reply: ' + sys.stdin.read())",
]


def _run(argv, prompt, guard=None, timeout=None):
    guard = guard if guard is not None else FluentCleanPrompt().packs("patient").guard()
    out, err = io.StringIO(), io.StringIO()
    status = run_command(guard, argv, prompt, out, err, timeout=timeout)
    return status, out.getvalue(), err.getvalue()


def test_the_command_sees_placeholders_and_the_user_sees_values(tmp_path):
    record = tmp_path / "seen.txt"
    script = f"import sys; data = sys.stdin.read(); open({str(record)!r}, 'w').write(data); sys.stdout.write(data)"
    status, out, _ = _run(
        [sys.executable, "-c", script], "MRN: 00412345 for ann@example.com"
    )
    assert status == 0
    assert out == "MRN: 00412345 for ann@example.com"
    seen = record.read_text()
    assert "00412345" not in seen and "ann@example.com" not in seen


def test_a_large_reply_streams_without_deadlock():
    script = "import sys; sys.stdin.read(); sys.stderr.write('x' * 200000); sys.stdout.write('[EMAIL-1] ' * 20000)"
    guard = FluentCleanPrompt().guard()
    status, out, err = _run(
        [sys.executable, "-c", script], "mail ann@example.com", guard=guard
    )
    assert status == 0
    assert out.count("ann@example.com") == 20000
    assert len(err) == 200000


def test_the_exit_status_is_the_commands():
    status, _, err = _run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('bad [MRN-1]'); sys.exit(7)",
        ],
        "MRN: 00412345",
    )
    assert status == 7
    assert err == "bad 00412345"


def test_a_missing_command_is_reported():
    with pytest.raises(CleanPromptError, match="could not start"):
        _run(["cleanprompt-no-such-command"], "hi")


def test_a_timeout_stops_the_command():
    with pytest.raises(CleanPromptError, match="ran past"):
        _run([sys.executable, "-c", "import time; time.sleep(30)"], "hi", timeout=0.5)


def test_a_leak_is_refused_before_the_command_starts(tmp_path):
    marker = tmp_path / "started"
    guard = FluentCleanPrompt().packs("personal").remember(False).guard()
    guard.outgoing("name: Marion Holt")
    with pytest.raises(LeakError):
        _run(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
            "Marion Holt",
            guard=guard,
        )
    assert not marker.exists()


def test_a_command_that_ignores_its_input_is_noted():
    script = "import os, sys; os.close(0); sys.stdout.write('ok')"
    status, out, err = _run([sys.executable, "-c", script], "x " * 400_000)
    assert status == 0 and out == "ok"
    assert "closed its input before reading the whole prompt" in err


@pytest.mark.parametrize(
    ("line", "argv"),
    [
        ("ollama run llama3", ["ollama", "run", "llama3"]),
        ("llm -m 'gpt 4o'", ["llm", "-m", "gpt 4o"]),
    ],
)
def test_split_command(line, argv):
    assert split_command(line) == argv


@pytest.mark.parametrize("line", ["", "   ", "llm 'unclosed"])
def test_split_command_refuses(line):
    with pytest.raises(CleanPromptError):
        split_command(line)


def test_a_closed_output_pipe_stops_the_command_instead_of_hanging():
    class Closed(io.StringIO):
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

    script = "import sys\nwhile True:\n    sys.stdout.write('[EMAIL-1] ' * 1000); sys.stdout.flush()"
    guard = FluentCleanPrompt().guard()
    guard.outgoing("ann@example.com")
    with pytest.raises(BrokenPipeError):
        run_command(
            guard,
            [sys.executable, "-c", script],
            "hi",
            Closed(),
            io.StringIO(),
            timeout=20,
        )


class TestPipesAreClosed:
    """
    CP-087: every pipe is closed before ``run_command`` returns.

    Notes
    -----
    **Developer notes.** The standard-error pipe used to be left for the
    garbage collector. Nothing failed until the suite ran with warnings as
    errors, where the ``ResourceWarning`` became an error in whichever test
    was running at collection time. The pipes are inspected directly here, so
    the result does not depend on the warning filter or on when a collection
    happens.
    """

    @staticmethod
    def _started(monkeypatch):
        import subprocess

        from .. import _bridge

        started = []
        real = subprocess.Popen

        def recording(*args, **kwargs):
            process = real(*args, **kwargs)
            started.append(process)
            return process

        monkeypatch.setattr(_bridge.subprocess, "Popen", recording)
        return started

    @staticmethod
    def _assert_closed(started):
        assert len(started) == 1
        process = started[0]
        assert process.returncode is not None
        assert [
            name
            for name in ("stdin", "stdout", "stderr")
            if not getattr(process, name).closed
        ] == []

    def test_after_an_ordinary_run(self, monkeypatch):
        started = self._started(monkeypatch)
        assert _run(_ECHO, "mail ann@example.com")[0] == 0
        self._assert_closed(started)

    def test_after_a_failing_command(self, monkeypatch):
        started = self._started(monkeypatch)
        script = "import sys; sys.stdin.read(); sys.stderr.write('no'); sys.exit(3)"
        assert _run([sys.executable, "-c", script], "hi")[0] == 3
        self._assert_closed(started)

    def test_after_a_command_that_ignores_its_input(self, monkeypatch):
        started = self._started(monkeypatch)
        script = "import os, sys; os.close(0); sys.stdout.write('ok')"
        status, _, err = _run([sys.executable, "-c", script], "x " * 400_000)
        assert status == 0
        assert err.count("closed its input before reading the whole prompt") == 1
        self._assert_closed(started)

    def test_after_a_timeout(self, monkeypatch):
        started = self._started(monkeypatch)
        with pytest.raises(CleanPromptError, match="ran past"):
            _run([sys.executable, "-c", "import time; time.sleep(30)"], "hi", timeout=0.5)
        self._assert_closed(started)
