# scikitplot/_testing/tests/test__perf_telemetry.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for :mod:`scikitplot._testing._perf_telemetry`, the test-performance recorder.

Notes
-----
**Developer notes.** The recorder installs an audit hook, which no process
can remove, and a garbage-collection callback. So every test that runs it
does so in a child pytest session in a temporary directory, with
``-W error`` (the repository's ``filterwarnings = error``): the session of
these tests is left as it was. The child's environment is this one without
any ``SKPLT_TEST_PERF_*`` variable, so a recorded CI session does not leak
into its children.

A child session in a temporary project is hermetic: plugin autoloading is
off (``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``) and a test that needs a plugin
loads it with ``-p``. Otherwise every pytest plugin installed in the
environment joins the child, and under ``-W error`` a plugin's own
configure-time warning ends the child before any test runs. That is what
made these tests pass on a developer machine and fail in CI: pytest 9.2
applies ``-W`` filters to warnings raised in ``pytest_configure``, and
pytest-asyncio warns there when ``asyncio_default_fixture_loop_scope`` is
unset (the child's ``pytest.ini`` is empty; the repository's own file
ignores that warning).
"""

from __future__ import annotations

import json
import os
import pstats
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

from .. import _perf_telemetry as perf

REPO = Path(__file__).resolve().parents[3]

#: How a child session (a separate project in ``tmp_path``, outside this
#: package) imports the recorder: by the name this run imported it under,
#: so the tests do not hard-code the package's name (Rule 142).
CHILD_IMPORT = (
    f"import importlib\n_perf_telemetry = importlib.import_module({perf.__name__!r})\n"
)

#: A child session's conftest: the recorder, registered as the package does it.
CHILD_CONFTEST = CHILD_IMPORT + """

def pytest_configure(config):
    _perf_telemetry.register(config)
"""


def _child_env(*, hermetic=True, **extra):
    """Return the environment of a child session (hermetic: no plugin autoload)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SKPLT_TEST_PERF", "PYTEST_", "COV_CORE_"))
        and key != "COVERAGE_PROCESS_START"
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if hermetic:
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.update(extra)
    return env


def _project(tmp_path, files, conftest=CHILD_CONFTEST):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (project / "conftest.py").write_text(textwrap.dedent(conftest), encoding="utf-8")
    for name, text in files.items():
        (project / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return project


def _pytest_argv(project, *args):
    return [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(project / "pytest.ini"),
        "--rootdir",
        str(project),
        "-p",
        "no:cacheprovider",
        "-W",
        "error",
        "-q",
        *args,
    ]


def _run(project, out, *args, **env):
    proc = subprocess.run(
        _pytest_argv(project, *args),
        cwd=project,
        env=_child_env(SKPLT_TEST_PERF_DIR=str(out), **env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    return proc


def _records(out):
    files = sorted(out.glob("telemetry-*.jsonl"))
    lines = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            lines.append(json.loads(line))
    return files, lines


def _by_kind(lines, kind):
    return [line for line in lines if line["kind"] == kind]


def _tests(lines):
    return {line["nodeid"].split("::")[-1]: line for line in _by_kind(lines, "test")}


# -- options ------------------------------------------------------------


def test_the_recorder_is_off_without_a_directory():
    assert perf.settings_from_environ({}) is None
    assert perf.settings_from_environ({perf.ENV_DIR: "   "}) is None


def test_options_are_read_from_the_environment(tmp_path):
    settings = perf.settings_from_environ(
        {
            perf.ENV_DIR: str(tmp_path),
            perf.ENV_PROFILE: r"test_x\[",
            perf.ENV_TRACEMALLOC: "1",
            perf.ENV_LABEL: "shard-1-of-2",
        }
    )
    assert settings.directory == tmp_path
    assert settings.profile.search("t.py::test_x[1]")
    assert settings.tracemalloc is True
    assert settings.label == "shard-1-of-2"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (perf.ENV_PROFILE, "test_(unclosed"),
        (perf.ENV_TRACEMALLOC, "maybe"),
        (perf.ENV_LABEL, "has space"),
        (perf.ENV_LABEL, "x" * 65),
    ],
)
def test_an_invalid_option_is_refused_by_name(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        perf.settings_from_environ({perf.ENV_DIR: str(tmp_path), name: value})


def test_register_turns_an_invalid_option_into_a_usage_error(tmp_path):
    config = types.SimpleNamespace(
        pluginmanager=types.SimpleNamespace(has_plugin=lambda name: False),
        option=types.SimpleNamespace(dist="no"),
    )
    with pytest.raises(pytest.UsageError, match=perf.ENV_TRACEMALLOC):
        perf.register(config, {perf.ENV_DIR: str(tmp_path), perf.ENV_TRACEMALLOC: "2"})


def test_an_xdist_controller_records_nothing(tmp_path):
    config = types.SimpleNamespace(
        pluginmanager=types.SimpleNamespace(has_plugin=lambda name: False),
        option=types.SimpleNamespace(dist="load"),
    )
    assert perf.register(config, {perf.ENV_DIR: str(tmp_path / "out")}) is None
    assert not (tmp_path / "out").exists()


def test_this_session_registered_the_recorder_exactly_when_asked(pytestconfig):
    wanted = bool(os.environ.get(perf.ENV_DIR, "").strip())
    assert pytestconfig.pluginmanager.has_plugin(perf.PLUGIN_NAME) is wanted


# -- pure helpers -------------------------------------------------------


def test_a_launch_counts_its_program_once():
    # Regression: on Python 3.13+ ``subprocess.Popen`` starts a child with
    # ``os.posix_spawn``, so one launch raised two events and counted twice.
    state = perf._AuditState()
    state.active = True
    as_subprocess = {"__name__": "subprocess", "state": state}
    exec(  # noqa: S102 - a caller frame that belongs to "subprocess"
        "def launch(exe):\n"
        "    state.record('subprocess.Popen', (exe, [exe], None, None))\n"
        "    state.record('os.posix_spawn', (exe, [exe], {}))\n",
        as_subprocess,
    )
    as_subprocess["launch"]("/usr/bin/python3")
    state.record("os.posix_spawn", ("/bin/true", ["/bin/true"], {}))  # direct call
    assert state.programs == {"python3": 1, "true": 1}
    assert state.counts["subprocess"] == 1 and state.counts["spawn"] == 2


def test_paths_are_written_relocatably(tmp_path):
    root = str(tmp_path)
    inside = tmp_path / "pkg" / "mod.py"
    assert perf._display_path(str(inside), root) == "pkg/mod.py"
    site = os.sep.join(
        ["", "opt", "env", "lib", "python3.11", "site-packages", "x", "y.py"]
    )
    assert perf._display_path(site, root) == "site-packages/x/y.py"
    assert perf._display_path("<string>", root) == "<string>"
    other = os.sep.join(["", "elsewhere", "deep", "a", "b.py"])
    assert perf._display_path(other, root) == ".../a/b.py"
    assert perf._display_arg(f"--junitxml={inside}", root) == "--junitxml=pkg/mod.py"
    assert perf._display_arg("-k", root) == "-k"
    # Inside a value, and after a short option written without a space.
    secret = os.sep.join(["", "home", "someone", "private", "coverage.xml"])
    if os.sep == "/":
        assert perf._display_arg(f"--cov-report=xml:{secret}", root) == (
            "--cov-report=xml:.../private/coverage.xml"
        )
        assert perf._display_arg(f"-c{secret}", root) == "-c.../private/coverage.xml"
        assert perf._display_path("/x", root) == ".../x"


def test_a_forked_child_never_waits_for_the_parents_lock():
    import threading  # noqa: PLC0415

    state = perf._AuditState()
    held, release = threading.Event(), threading.Event()

    def hold():
        with state.lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait(10)
    previous = perf._HOOK.target
    perf._HOOK.target = state
    try:
        # What os.register_at_fork runs in the child: the lock it inherited
        # may be held by a thread that does not exist there.
        perf._new_lock_after_fork()
        assert state.lock.acquire(blocking=False)
        state.lock.release()
    finally:
        perf._HOOK.target = previous
        release.set()
        thread.join(10)


def _report(failed=False, skipped=False, wasxfail=None):
    report = types.SimpleNamespace(failed=failed, skipped=skipped)
    if wasxfail is not None:
        report.wasxfail = wasxfail
    return report


@pytest.mark.parametrize(
    ("reports", "expected"),
    [
        ({"setup": _report(failed=True)}, "error"),
        ({"setup": _report(skipped=True)}, "skipped"),
        ({"setup": _report(skipped=True, wasxfail="")}, "xfailed"),
        (
            {"setup": _report(), "call": _report(failed=True), "teardown": _report()},
            "failed",
        ),
        ({"setup": _report(), "call": _report(skipped=True, wasxfail="r")}, "xfailed"),
        (
            {"setup": _report(), "call": _report(wasxfail="r"), "teardown": _report()},
            "xpassed",
        ),
        (
            {"setup": _report(), "call": _report(), "teardown": _report(failed=True)},
            "error",
        ),
        ({"setup": _report(), "call": _report(), "teardown": _report()}, "passed"),
    ],
)
def test_every_outcome_is_named(reports, expected):
    assert perf._outcome(reports) == expected


def test_profile_names_are_safe_unique_and_bounded():
    names = {perf._profile_name(f"a/b.py::test_x[{c}]") for c in '/:*?"<>| '}
    assert len(names) == 9
    for name in names:
        assert len(name) <= 100
        assert name.endswith(".prof")
        assert not set(name) - set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )


# -- child sessions -----------------------------------------------------

SESSION_TESTS = {
    "helper_mod.py": "VALUE = 1\n",
    "test_session.py": (
        """
        import importlib.util
        import pathlib
        import subprocess
        import sys

        import pytest

        HERE = pathlib.Path(__file__).parent


        @pytest.fixture(scope="module")
        def slow_module():
            import time
            time.sleep(0.02)
            return 1


        @pytest.fixture
        def broken():
            raise RuntimeError("setup fails")


        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("teardown fails")


        def test_pass(slow_module):
            assert slow_module == 1


        def test_fail():
            assert False


        def test_skip():
            pytest.skip("not today")


        @pytest.mark.xfail(reason="known")
        def test_xfail():
            assert False


        @pytest.mark.xfail(reason="known")
        def test_xpass():
            pass


        def test_setup_error(broken):
            pass


        def test_teardown_error(broken_teardown):
            pass


        def test_io(tmp_path):
            (tmp_path / "d").mkdir()
            (tmp_path / "d" / "f.txt").write_text("x", encoding="utf-8")
            (tmp_path / "d" / "f.txt").read_text(encoding="utf-8")
            subprocess.run([sys.executable, "-c", "pass"], check=True)


        def test_reexec():
            for name in ("helper_copy_a", "helper_copy_b"):
                spec = importlib.util.spec_from_file_location(name, HERE / "helper_mod.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert module.VALUE == 1
    """
    ),
}


def test_a_session_records_every_test_with_its_outcome(tmp_path):
    project = _project(tmp_path, SESSION_TESTS)
    out = tmp_path / "out"
    proc = _run(project, out, "test_session.py", SKPLT_TEST_PERF_LABEL="unit")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    files, lines = _records(out)
    assert len(files) == 1
    header = lines[0]
    assert header["kind"] == "session"
    assert header["schema"] == perf.SCHEMA and header["version"] == perf.SCHEMA_VERSION
    assert header["worker"] == "main" and header["label"] == "unit"
    assert header["environment"]["SKPLT_TEST_PERF_LABEL"] == "unit"
    assert "SKPLT_TEST_PERF_DIR" not in header["environment"]
    assert not any(os.path.isabs(arg) for arg in header["args"])
    assert lines[-1]["kind"] == "session_end"
    assert lines[-1]["exitstatus"] == 1
    assert lines[-1]["tests"] == 9
    assert lines[-1]["recorder_errors"] == 0, lines[-1]["recorder_error_messages"]

    (collect,) = _by_kind(lines, "collect")
    assert collect["path"] == "test_session.py"
    assert collect["items"] == 9 and collect["module_exec"] >= 1
    assert _by_kind(lines, "collection")[0]["items"] == 9

    tests = _tests(lines)
    assert {name: t["outcome"] for name, t in tests.items()} == {
        "test_pass": "passed",
        "test_fail": "failed",
        "test_skip": "skipped",
        "test_xfail": "xfailed",
        "test_xpass": "xpassed",
        "test_setup_error": "error",
        "test_teardown_error": "error",
        "test_io": "passed",
        "test_reexec": "passed",
    }
    assert set(tests["test_pass"]["phases"]) == {"setup", "call", "teardown"}
    assert set(tests["test_setup_error"]["phases"]) == {"setup", "teardown"}
    fixtures = {f[0]: f for f in tests["test_pass"]["fixtures"]}
    name, scope, seconds, location = fixtures["slow_module"]
    assert (
        scope == "module"
        and seconds >= 0.02
        and location.startswith("test_session.py:")
    )

    io = tests["test_io"]
    assert io["events"]["mkdir"] >= 1
    assert io["events"]["open_write"] >= 1 and io["events"]["open"] >= 2
    assert io["events"]["subprocess"] == 1
    assert io["programs"] == {os.path.basename(sys.executable): 1}
    if "child_cpu" in io["phases"]["call"]:
        assert io["phases"]["call"]["child_cpu"] > 0

    # helper_mod.py runs twice under two names: the first run is new, the
    # second is the re-execution the profile analysis must know about.
    reexec = tests["test_reexec"]
    assert reexec["events"]["module_exec"] >= 2
    assert reexec["reexecuted"] == ["helper_mod.py"] and reexec["reexecuted_count"] == 1
    assert ["helper_mod.py", 2] in lines[-1]["modules_reexecuted"]


def test_lines_written_before_a_kill_survive(tmp_path):
    project = _project(
        tmp_path,
        {
            "test_kill.py": (
                """
                import time

                def test_a_first():
                    pass

                def test_b_hangs():
                    time.sleep(120)
            """
            )
        },
    )
    out = tmp_path / "out"
    proc = subprocess.Popen(
        _pytest_argv(project, "test_kill.py"),
        cwd=project,
        env=_child_env(SKPLT_TEST_PERF_DIR=str(out)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            files = list(out.glob("telemetry-*.jsonl"))
            if files and "test_a_first" in files[0].read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("the first test's line never appeared")
    finally:
        if os.name == "posix":
            proc.send_signal(signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=60)
    _, lines = _records(out)
    kinds = [line["kind"] for line in lines]
    assert kinds[0] == "session" and "session_end" not in kinds
    assert [t["nodeid"] for t in _by_kind(lines, "test")] == [
        "test_kill.py::test_a_first"
    ]


PROFILE_TESTS = {
    "test_profile.py": (
        """
        def work():
            return sum(range(10000))

        def test_profiled():
            assert work() > 0

        def test_other():
            assert work() > 0

        def test_allocates():
            block = bytearray(5_000_000)
            assert len(block) == 5_000_000
    """
    )
}


def test_selected_tests_are_profiled_and_peaks_recorded(tmp_path):
    project = _project(tmp_path, PROFILE_TESTS)
    out = tmp_path / "out"
    proc = _run(
        project,
        out,
        "test_profile.py",
        SKPLT_TEST_PERF_PROFILE="test_profiled$",
        SKPLT_TEST_PERF_TRACEMALLOC="1",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _, lines = _records(out)
    tests = _tests(lines)
    path = out / tests["test_profiled"]["profile"]
    stats = pstats.Stats(str(path)).stats
    assert any(key[2] == "test_profiled" for key in stats)
    assert "profile" not in tests["test_other"]
    start, end, peak = tests["test_allocates"]["tracemalloc"]
    assert peak - start >= 5_000_000
    assert lines[0]["capabilities"]["tracemalloc"] is True


def test_a_profiled_session_is_not_replaced(tmp_path):
    project = _project(tmp_path, PROFILE_TESTS)
    out = tmp_path / "out"
    outer = tmp_path / "outer.prof"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(outer),
            *_pytest_argv(project, "test_profile.py")[1:],
        ],
        cwd=project,
        env=_child_env(
            SKPLT_TEST_PERF_DIR=str(out), SKPLT_TEST_PERF_PROFILE="test_profiled$"
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _, lines = _records(out)
    skipped = _tests(lines)["test_profiled"]["profile"]
    assert "profiler" in skipped["skipped"] or "refused" in skipped["skipped"]
    # The outer profile kept recording through the selected test.
    outer_stats = pstats.Stats(str(outer)).stats
    assert any(key[2] == "test_other" for key in outer_stats)
    assert not list((out / "profiles").glob("*.prof"))


def test_an_error_inside_the_recorder_never_changes_an_outcome(tmp_path):
    conftest = CHILD_IMPORT + textwrap.dedent("""
        class Broken(_perf_telemetry.TelemetryRecorder):
            def _fixture_end(self, fixturedef, started):
                raise RuntimeError("recorder boom")


        def pytest_configure(config):
            settings = _perf_telemetry.settings_from_environ()
            config.pluginmanager.register(Broken(config, settings), _perf_telemetry.PLUGIN_NAME)
    """)
    project = _project(
        tmp_path,
        {"test_ok.py": "def test_ok(tmp_path):\n    assert tmp_path.is_dir()\n"},
        conftest=conftest,
    )
    out = tmp_path / "out"
    proc = _run(project, out, "test_ok.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _, lines = _records(out)
    end = lines[-1]
    assert end["recorder_errors"] >= 1
    assert any("recorder boom" in message for message in end["recorder_error_messages"])
    assert _tests(lines)["test_ok"]["outcome"] == "passed"


def test_an_unwritable_directory_is_a_usage_error(tmp_path):
    project = _project(tmp_path, {"test_ok.py": "def test_ok():\n    pass\n"})
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory", encoding="utf-8")
    proc = _run(project, blocker / "out", "test_ok.py")
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert perf.ENV_DIR in proc.stdout + proc.stderr


def test_the_cost_per_test_is_small(tmp_path):
    project = _project(
        tmp_path,
        {
            "test_many.py": (
                """
                import pytest

                @pytest.mark.parametrize("i", range(300))
                def test_n(i, tmp_path):
                    assert i >= 0
            """
            )
        },
    )
    out = tmp_path / "out"
    proc = _run(project, out, "test_many.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _, lines = _records(out)
    end = lines[-1]
    assert end["tests"] == 300
    # The recorder's own work, timed by itself in CPU time of its thread
    # (time spent waiting for a CPU is not its cost): well under a
    # millisecond a test (measured: about 0.2 ms); the bound leaves room.
    assert end["recorder_seconds"] / end["tests"] < 0.002
    # The once-per-session work (machine, packages, speed probe) is apart.
    assert end["recorder_startup_seconds"] >= 0


def test_xdist_workers_share_one_run(tmp_path):
    pytest.importorskip("xdist")
    project = _project(
        tmp_path,
        {
            "test_par.py": (
                "import pytest\n\n@pytest.mark.parametrize('i', range(8))\ndef test_p(i):\n    pass\n"
            )
        },
    )
    out = tmp_path / "out"
    proc = _run(project, out, "test_par.py", "-p", "xdist.plugin", "-n", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    files, lines = _records(out)
    sessions = _by_kind(lines, "session")
    assert sorted(s["worker"] for s in sessions) == ["gw0", "gw1"]
    assert len({s["run_id"] for s in sessions}) == 1
    nodeids = [t["nodeid"] for t in _by_kind(lines, "test")]
    assert sorted(nodeids) == sorted(f"test_par.py::test_p[{i}]" for i in range(8))


def test_the_package_conftest_registers_the_recorder(tmp_path):
    if not ((REPO / "pytest.ini").is_file() and (REPO / ".github" / "ci").is_dir()):
        pytest.skip(
            "needs the repository checkout (pytest.ini and .github/ci), not an installed copy"
        )
    out = tmp_path / "out"
    target = "scikitplot/_testing/tests/test__perf_telemetry.py::test_the_recorder_is_off_without_a_directory"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", target],
        cwd=REPO,
        # The repository's own configuration, with the plugins it expects.
        env=_child_env(hermetic=False, SKPLT_TEST_PERF_DIR=str(out)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    files, lines = _records(out)
    assert len(files) == 1
    assert lines[0]["planner"]["package_root"] == "scikitplot"
    assert [t["nodeid"] for t in _by_kind(lines, "test")] == [target]


def test_the_analyzer_reads_what_the_recorder_writes(tmp_path):
    analyzer = REPO / ".github" / "scripts" / "ci_test_perf.py"
    if not analyzer.is_file():
        pytest.skip("needs the repository checkout (.github/scripts/ci_test_perf.py)")
    project = _project(tmp_path, SESSION_TESTS)
    out = tmp_path / "out"
    session = _run(project, out, "test_session.py")
    # The session has failing tests (1); anything else means it never ran.
    assert session.returncode == 1, session.stdout + session.stderr
    report = tmp_path / "report"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(analyzer),
            "report",
            str(out),
            "--out",
            str(report),
            "--bundle",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["tests"] == 9
    assert summary["totals"]["outcomes"] == {
        "error": 2,
        "failed": 1,
        "passed": 3,
        "skipped": 1,
        "xfailed": 1,
        "xpassed": 1,
    }
    assert any(
        f["name"] == "slow_module" and f["scope"] == "module"
        for f in summary["fixtures"]
    )
    assert any(
        f["rule"] == "module-re-executed" and f["subject"] == "helper_mod.py"
        for f in summary["findings"]
    )
    assert (report / "ANALYZE_ME.md").is_file() and (
        report / "test-perf-bundle.zip"
    ).is_file()
