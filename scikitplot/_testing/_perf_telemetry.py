# scikitplot/_testing/_perf_telemetry.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Record where a pytest session spends its time and memory, one line per test.

The recorder is a pytest plugin. ``scikitplot/conftest.py`` registers it only
when :data:`ENV_DIR` names a directory; otherwise nothing in this module runs.
Each process of a run writes one file of JSON Lines there, and every line is
complete when it is written, so a session that is killed or times out leaves
everything it measured up to that point.

Notes
-----
**User notes.** Record a run, then analyse the directory::

    SKPLT_TEST_PERF_DIR=build/test-perf python -m pytest scikitplot/_externals
    python .github/scripts/ci_test_perf.py report build/test-perf --out build/test-perf/report

Options, all environment variables:

``SKPLT_TEST_PERF_DIR``
    Where to write. Unset or empty: the recorder is off.
``SKPLT_TEST_PERF_PROFILE``
    A regular expression; every test whose node id it matches (``re.search``)
    is profiled with :mod:`cProfile` from the start of its setup to the end of
    its teardown, into ``profiles/`` under the directory. This replaces timing
    a fixture by editing it: the analyzer lists what each fixture called.
``SKPLT_TEST_PERF_TRACEMALLOC``
    ``1`` records each test's peak of Python allocations (:mod:`tracemalloc`).
    Allocation-heavy tests run several times slower while it is on.
``SKPLT_TEST_PERF_LABEL``
    A name for the run (a CI shard name), stored in every file.

**What each test line holds.** Wall and CPU time of setup, call and teardown;
CPU time of child processes that ended in each phase; resident memory at the
start and end and the growth of its peak; garbage collections per generation
and their time; every fixture setup of a millisecond or more, with its scope
and where it is defined; and counts from Python's audit events: files opened
(and opened for writing), directories made and removed, subprocesses started
and the programs they ran, network connections, module bodies executed and
which of them had already been executed in this process. Collection time is
recorded per test file. The schema is versioned (:data:`SCHEMA`,
:data:`SCHEMA_VERSION`); the analyzer refuses a version it does not know.

**Developer notes.**

* *Never change a test's outcome.* Every hook body runs inside
  :meth:`TelemetryRecorder._safely`, which counts an exception instead of
  raising it; the garbage-collection callback and the audit hook do the same.
  The recorder emits no warning (the suite runs with ``filterwarnings =
  error``) and never collects garbage (``scikitplot/conftest.py``, rule 3).
* *Cost.* What runs per test is a few clock and ``getrusage`` reads, two
  ``pread`` calls on ``/proc`` file descriptors opened once, copies of small
  counter dictionaries, and one JSON line. The recorder times its own work
  and reports it at the end of the session (``recorder_seconds``).
* *The audit hook cannot be removed* (:func:`sys.addaudithook`). One hook is
  installed per process, the first time a recorder is created, and forwards
  to the recorder of the current session only; between sessions it returns
  at once. It does no I/O, so it cannot trigger itself.
* *Profilers.* On Python 3.11 enabling a second :class:`cProfile.Profile`
  silently replaces the profiler already installed (for example
  ``python -m cProfile -m pytest``), which then stops recording; on 3.12 and
  later it raises ``ValueError``. A test is therefore profiled only when no
  profiler is active, and its line says why when it was not (checked on
  CPython 3.11.17, 3.12.3 and 3.13.17).
* *Audit events by version.* ``time.sleep`` raises its event on 3.13 and
  later only (not on 3.11.17 or 3.12.3); on older interpreters the sleep
  total is absent, not zero.
* *Paths.* Node ids are relative to the root directory already. Other paths
  (where a fixture is defined, which module body ran) are written relative to
  the root directory, or as ``site-packages/...`` or ``stdlib/...``, never as
  absolute paths: a record says what ran, not where the machine keeps it.
"""

from __future__ import annotations

import contextlib
import cProfile
import gc
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import NamedTuple

import pytest

__all__ = [
    "ENV_DIR",
    "ENV_LABEL",
    "ENV_PROFILE",
    "ENV_TRACEMALLOC",
    "PLUGIN_NAME",
    "SCHEMA",
    "SCHEMA_VERSION",
    "Settings",
    "TelemetryRecorder",
    "register",
    "settings_from_environ",
]

#: Name under which the recorder is registered with pytest's plugin manager.
PLUGIN_NAME = "skplt-test-perf"

#: Name of the record format; the first line of every file carries it.
SCHEMA = "skplt-test-perf"

#: Version of the record format. Raise it when a field changes meaning.
SCHEMA_VERSION = 1

#: Directory to write to; the recorder is off while this is unset or empty.
ENV_DIR = "SKPLT_TEST_PERF_DIR"

#: Regular expression of node ids to profile one by one.
ENV_PROFILE = "SKPLT_TEST_PERF_PROFILE"

#: ``1`` to record peaks of Python allocations per test.
ENV_TRACEMALLOC = "SKPLT_TEST_PERF_TRACEMALLOC"

#: Name of the run, such as a CI shard.
ENV_LABEL = "SKPLT_TEST_PERF_LABEL"

#: Environment variables copied into the session line. Each one describes
#: how the run was made; none can hold a credential, and the environment is
#: never copied as a whole.
ENV_RECORDED = (
    "SKPLT_TEST_GC",
    "SKPLT_TEST_FLOOR_BUDGET",
    ENV_PROFILE,
    ENV_TRACEMALLOC,
    ENV_LABEL,
    "PYTEST_XDIST_WORKER_COUNT",
    "COVERAGE_CORE",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_SHA",
    "GITHUB_REF_NAME",
    "GITHUB_EVENT_NAME",
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "RUNNER_OS",
    "RUNNER_ARCH",
)

#: Distributions whose versions are recorded, when installed.
PACKAGES_RECORDED = (
    "pytest",
    "pluggy",
    "pytest-cov",
    "coverage",
    "pytest-xdist",
    "hypothesis",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "sphinx",
    "cryptography",
    "scikit-plots",
)

#: A fixture setup shorter than this is counted, not listed.
FIXTURE_MIN_SECONDS = 0.001

#: At most this many re-executed module names are listed per line.
REEXECUTED_LIST_MAX = 20

#: At most this many recorder errors are kept with their message.
ERROR_MESSAGES_MAX = 20

#: Labels are file-name safe.
_LABEL = re.compile(r"[A-Za-z0-9._-]{1,64}")

#: Characters kept when a node id becomes a file name.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Audit event -> counter name. Events not listed are ignored at once.
AUDIT_COUNTERS = {
    "open": "open",
    "os.mkdir": "mkdir",
    "os.rmdir": "rmdir",
    "os.remove": "remove",
    "os.rename": "rename",
    "os.scandir": "scandir",
    "os.listdir": "listdir",
    "os.chmod": "chmod",
    "os.link": "link",
    "os.symlink": "symlink",
    "os.truncate": "truncate",
    "os.utime": "utime",
    "shutil.copyfile": "copyfile",
    "shutil.copytree": "copytree",
    "shutil.rmtree": "rmtree",
    "shutil.move": "move",
    "tempfile.mkdtemp": "mkdtemp",
    "tempfile.mkstemp": "mkstemp",
    "subprocess.Popen": "subprocess",
    "os.posix_spawn": "spawn",
    "os.system": "system",
    "os.exec": "exec_program",
    "os.fork": "fork",
    "socket.connect": "socket_connect",
    "socket.getaddrinfo": "dns_lookup",
    "urllib.Request": "http_request",
    "compile": "compile",
    "import": "import",
    "exec": "exec",
    "time.sleep": "sleep",
}

_WRITE_FLAGS = (
    getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2) | getattr(os, "O_APPEND", 0)
)
_WRITE_MODES = frozenset("wax+")

#: Positions of the mode and flags in the arguments of the ``open`` audit event.
_OPEN_MODE = 1
_OPEN_FLAGS = 2


class Settings(NamedTuple):
    """
    The recorder's options, read from the environment.

    Attributes
    ----------
    directory : pathlib.Path
        Where the files are written.
    profile : re.Pattern or None
        Node ids to profile, or None.
    tracemalloc : bool
        Whether to record Python allocation peaks.
    label : str or None
        Name of the run.
    """

    directory: Path
    profile: re.Pattern | None
    tracemalloc: bool
    label: str | None


def settings_from_environ(environ=None):
    """
    Read the recorder's options from the environment.

    Parameters
    ----------
    environ : mapping of str to str, optional
        The environment; :data:`os.environ` by default.

    Returns
    -------
    Settings or None
        None when :data:`ENV_DIR` is unset or empty: the recorder is off.

    Raises
    ------
    ValueError
        If a value is set but invalid. A typo must not silently record
        something else, or nothing.

    Examples
    --------
    >>> settings_from_environ({}) is None
    True
    >>> settings_from_environ({"SKPLT_TEST_PERF_DIR": "out"}).tracemalloc
    False
    """
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_DIR) or "").strip()
    if not raw:
        return None
    pattern_text = (environ.get(ENV_PROFILE) or "").strip()
    profile = None
    if pattern_text:
        try:
            profile = re.compile(pattern_text)
        except re.error as exc:
            raise ValueError(
                f"{ENV_PROFILE}={pattern_text!r} is not a regular expression: {exc}"
            ) from None
    flag = (environ.get(ENV_TRACEMALLOC) or "").strip().lower()
    if flag in ("", "0", "false", "no", "off"):
        trace = False
    elif flag in ("1", "true", "yes", "on"):
        trace = True
    else:
        raise ValueError(f"{ENV_TRACEMALLOC}={flag!r} is not a switch; use 1 or 0")
    label = (environ.get(ENV_LABEL) or "").strip() or None
    if label is not None and not _LABEL.fullmatch(label):
        raise ValueError(
            f"{ENV_LABEL}={label!r} must be 1-64 characters of letters, digits, '.', '_' or '-'"
        )
    return Settings(Path(raw), profile, trace, label)


def register(config, environ=None):
    """
    Register the recorder for this session when the environment asks for it.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration, during ``pytest_configure``.
    environ : mapping of str to str, optional
        The environment; :data:`os.environ` by default.

    Returns
    -------
    TelemetryRecorder or None
        The registered recorder; None when the recorder is off, is already
        registered, or this process is a pytest-xdist controller (which runs
        no test: each worker records its own).

    Raises
    ------
    pytest.UsageError
        If an option is invalid or the output cannot be written.
    """
    try:
        settings = settings_from_environ(environ)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    if settings is None or config.pluginmanager.has_plugin(PLUGIN_NAME):
        return None
    if (
        not hasattr(config, "workerinput")
        and getattr(config.option, "dist", "no") != "no"
    ):
        return None
    recorder = TelemetryRecorder(config, settings)
    config.pluginmanager.register(recorder, PLUGIN_NAME)
    return recorder


def _display_path(path, root):
    """
    Return ``path`` as a relocatable name.

    Parameters
    ----------
    path : str
        A file name, as Python reports it (``co_filename``).
    root : str
        The session's root directory, absolute.

    Returns
    -------
    str
        The POSIX path relative to ``root`` when the file is inside it; else
        ``site-packages/<rest>`` or ``stdlib/<rest>``; else ``<name>`` (an
        interpreter pseudo-file such as ``<string>``) or ``.../<last two
        parts>``. Never an absolute path.
    """
    text = str(path)
    if text.startswith("<"):
        return text
    norm = os.path.normcase(os.path.abspath(text))
    base = os.path.normcase(root)
    if norm == base or norm.startswith(base + os.sep):
        return Path(os.path.relpath(norm, base)).as_posix()
    parts = Path(text).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            return "site-packages/" + "/".join(parts[parts.index(marker) + 1 :])
    stdlib = os.path.normcase(os.path.dirname(os.__file__))
    if norm.startswith(stdlib + os.sep):
        return "stdlib/" + Path(os.path.relpath(norm, stdlib)).as_posix()
    # The last two parts, never the root itself ("/x" is ".../x").
    tail = [part for part in parts if part != Path(text).anchor][-2:]
    return ".../" + "/".join(tail)


#: An absolute path inside a command-line argument: at its start, after a
#: short option (``-c/x``), or after ``=``, ``:`` or ``,`` (``--cov-report=xml:/x``).
_ABSOLUTE_IN_ARG = re.compile(
    r"(?:^-[A-Za-z]|^|(?<=[=:,]))(/[^\s:,;]*|[A-Za-z]:[\\/][^\s,;]*)"
)


def _display_arg(arg, root):
    """Return a command-line argument with every absolute path in it made relocatable."""
    text = str(arg)
    if os.path.isabs(text):
        return _display_path(text, root)

    def relocatable(match):
        prefix = match.group(0)[: match.start(1) - match.start(0)]
        return prefix + _display_path(match.group(1), root)

    return _ABSOLUTE_IN_ARG.sub(relocatable, text)


def _profile_name(nodeid):
    """Return a short, unique, file-name-safe name for a node id."""
    digest = hashlib.sha256(nodeid.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    stem = _UNSAFE.sub("_", nodeid).strip("._")[-80:]
    return f"{stem}-{digest}.prof"


def _active_profiler():
    """
    Say why a :class:`cProfile.Profile` must not be enabled now, if it must not.

    Returns
    -------
    str or None
        A reason, or None when no profiler is active.
    """
    if sys.getprofile() is not None:
        return "another profiler is installed (sys.getprofile)"
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        try:
            tool = monitoring.get_tool(monitoring.PROFILER_ID)
        except Exception:  # noqa: BLE001 - an unknown interpreter: fall through
            tool = None
        if tool is not None:
            return f"the sys.monitoring profiler slot is used by {tool!r}"
    return None


class _AuditState:
    """
    Counters fed by the process's audit hook.

    Notes
    -----
    **Developer notes.** The hook (:func:`_audit_hook`) is installed once per
    process and forwards to the instance in ``_HOOK.target``; setting
    ``active`` to False pauses it (the recorder does so while it does its
    own work). Increments are made under a re-entrant lock so that threads
    do not lose counts, and so that a signal handler that triggers an event
    inside the hook cannot deadlock it.
    """

    def __init__(self):
        self.active = False
        self.counts = {}
        self.programs = {}
        self.module_runs = {}
        self.reexecuted = []
        self.sleep_seconds = 0.0
        self.sleep_seen = False
        self.errors = 0
        self.lock = threading.RLock()

    def record(self, event, args):
        name = AUDIT_COUNTERS.get(event)
        if name is None or not self.active:
            return
        with self.lock:
            counts = self.counts
            counts[name] = counts.get(name, 0) + 1
            if name == "open":
                mode = args[_OPEN_MODE] if len(args) > _OPEN_MODE else None
                flags = args[_OPEN_FLAGS] if len(args) > _OPEN_FLAGS else 0
                if (isinstance(mode, str) and not _WRITE_MODES.isdisjoint(mode)) or (
                    mode is None and isinstance(flags, int) and flags & _WRITE_FLAGS
                ):
                    counts["open_write"] = counts.get("open_write", 0) + 1
            elif name == "exec":
                code = args[0] if args else None
                filename = getattr(code, "co_filename", "")
                if getattr(
                    code, "co_name", ""
                ) == "<module>" and not filename.startswith("<"):
                    runs = self.module_runs.get(filename, 0) + 1
                    self.module_runs[filename] = runs
                    counts["module_exec"] = counts.get("module_exec", 0) + 1
                    if runs > 1:
                        self.reexecuted.append(filename)
            elif name in ("subprocess", "spawn", "exec_program"):
                program = args[0] if args else None
                if program is None and name == "subprocess" and len(args) > 1:
                    argv = args[1]
                    program = (
                        argv
                        if isinstance(argv, (str, bytes))
                        else (argv[0] if argv else None)
                    )
                if isinstance(program, bytes):
                    program = os.fsdecode(program)
                if program is not None:
                    key = (
                        os.path.basename(str(program).split()[0])
                        if str(program)
                        else ""
                    )
                    self.programs[key] = self.programs.get(key, 0) + 1
            elif name == "sleep":
                self.sleep_seen = True
                seconds = args[0] if args else 0
                if isinstance(seconds, (int, float)) and seconds > 0:
                    self.sleep_seconds += float(seconds)


class _HookSlot:
    """The process's one audit hook, and the session state it feeds.

    Python cannot remove an audit hook, so one is installed per process and
    ``target`` points at the current session's :class:`_AuditState`, or is
    None between sessions.
    """

    installed = False
    target = None


_HOOK = _HookSlot()


def _audit_hook(event, args):
    """Forward an audit event to the current session's counters; never raise."""
    target = _HOOK.target
    if target is None:
        return
    # An audit hook that raises makes the audited call fail: count instead.
    try:
        target.record(event, args)
    except Exception:  # ruff: ignore[blind-except]
        target.errors += 1


def _install_audit_hook():
    """Install :func:`_audit_hook` once per process; return False if unavailable."""
    if _HOOK.installed:
        return True
    adder = getattr(sys, "addaudithook", None)
    if adder is None:
        return False
    adder(_audit_hook)
    _HOOK.installed = True
    if hasattr(os, "register_at_fork"):
        # A child forked while another thread held the lock would wait for
        # it forever at its first counted event (opening a file).
        os.register_at_fork(after_in_child=_new_lock_after_fork)
    return True


def _new_lock_after_fork():
    target = _HOOK.target
    if target is not None:
        target.lock = threading.RLock()


class _GCMonitor:
    """
    Count garbage collections per generation, and their time.

    Notes
    -----
    **Developer notes.** A callback in :data:`gc.callbacks` that raises is
    reported as unraisable, which pytest turns into a warning and this suite
    into an error, so the body cannot raise.
    """

    def __init__(self, clock):
        self.clock = clock
        self.counts = [0, 0, 0]
        self.seconds = [0.0, 0.0, 0.0]
        self.errors = 0
        self._started = None

    def __call__(self, phase, info):
        try:
            if phase == "start":
                self._started = self.clock()
                return
            if self._started is None:
                return
            generation = min(max(int(info.get("generation", 2)), 0), 2)
            self.counts[generation] += 1
            self.seconds[generation] += self.clock() - self._started
            self._started = None
        except Exception:  # noqa: BLE001 - see the class notes
            self.errors += 1

    def snapshot(self):
        return (tuple(self.counts), tuple(self.seconds))


class _Probe:
    """
    Read process resources cheaply: ``getrusage`` and ``/proc`` descriptors.

    Notes
    -----
    **Developer notes.** ``/proc/self/statm`` and ``/proc/self/io`` are opened
    once and read with ``os.pread`` at offset 0, which regenerates them; a
    file that cannot be opened or read (another system, a restricted
    container) makes its fields ``None`` for the whole session. ``ru_maxrss``
    is kibibytes on Linux and bytes on macOS.
    """

    def __init__(self):
        try:
            import resource  # noqa: PLC0415 - not available on Windows
        except ImportError:
            resource = None
        self.resource = resource
        self.maxrss_unit = 1 if sys.platform == "darwin" else 1024
        self.page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
        self.statm = self._open("/proc/self/statm")
        self.io = self._open("/proc/self/io")

    @staticmethod
    def _open(path):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return None
        try:
            os.pread(fd, 1024, 0)
        except (OSError, AttributeError):
            os.close(fd)
            return None
        return fd

    def capabilities(self):
        return {
            "rusage": self.resource is not None,
            "proc_statm": self.statm is not None and self.page is not None,
            "proc_io": self.io is not None,
        }

    def rss(self):
        if self.statm is None or self.page is None:
            return None
        try:
            return int(os.pread(self.statm, 256, 0).split()[1]) * self.page
        except (OSError, ValueError, IndexError):
            return None

    def io_counters(self):
        if self.io is None:
            return None
        try:
            fields = os.pread(self.io, 1024, 0).decode("ascii").split()
            return {
                fields[i].rstrip(":"): int(fields[i + 1])
                for i in range(0, len(fields) - 1, 2)
            }
        except (OSError, ValueError, UnicodeDecodeError):
            return None

    def own(self):
        if self.resource is None:
            return None
        usage = self.resource.getrusage(self.resource.RUSAGE_SELF)
        return (usage.ru_maxrss * self.maxrss_unit, usage.ru_nvcsw, usage.ru_nivcsw)

    def children(self):
        if self.resource is None:
            return None
        usage = self.resource.getrusage(self.resource.RUSAGE_CHILDREN)
        return (usage.ru_utime + usage.ru_stime, usage.ru_maxrss * self.maxrss_unit)

    def close(self):
        for fd in (self.statm, self.io):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self.statm = self.io = None


def _machine():
    """Describe the machine once per session; cheap and side-effect free."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            info["cpu_affinity"] = len(os.sched_getaffinity(0))
    for path, key, field in (
        ("/proc/cpuinfo", "cpu_model", "model name"),
        ("/proc/meminfo", "memory_bytes", "MemTotal"),
    ):
        try:
            with open(path, encoding="ascii", errors="replace") as handle:
                for line in handle:
                    if line.startswith(field):
                        value = line.split(":", 1)[1].strip()
                        info[key] = (
                            int(value.split()[0]) * 1024
                            if key == "memory_bytes"
                            else value
                        )
                        break
        except (OSError, ValueError):
            pass
    return info


def _version(name):
    """Return an installed distribution's version, or None if absent or unreadable."""
    from importlib import metadata  # noqa: PLC0415

    # Absent, or metadata a broken install left behind: either way not recorded.
    try:
        return metadata.version(name)
    except Exception:  # ruff: ignore[blind-except]
        return None


def _packages():
    versions = {name: _version(name) for name in PACKAGES_RECORDED}
    return {name: version for name, version in versions.items() if version is not None}


def _planner(root):
    """Return the planner's package root and containers, when the checkout has them."""
    path = Path(root) / ".github" / "ci" / "test_plan.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    package, containers = config.get("package_root"), config.get("containers")
    if not isinstance(package, str) or not isinstance(containers, list):
        return None
    return {"package_root": package, "containers": [str(c) for c in containers]}


def _speed_probe(clock, directory, root):
    """
    Measure the machine once, so runs on different machines can be compared.

    Returns
    -------
    dict
        Median microseconds of: a fixed pure-Python loop (``python_loop_us``);
        ``stat`` of a file in the root directory; ``open`` + ``read`` +
        ``close`` of a small file, ``mkdir`` + ``rmdir`` and write + ``fsync``
        in the output directory. A step that fails is left out.
    """
    out = {}

    def median_us(func, repeats):
        samples = []
        for _ in range(repeats):
            start = clock()
            func()
            samples.append(clock() - start)
        return round(statistics.median(samples) * 1e6, 3)

    out["python_loop_us"] = median_us(lambda: sum(range(100_000)), 5)
    probe = Path(directory) / f".probe-{uuid.uuid4().hex}"
    try:
        target = Path(root) / "pytest.ini"
        if not target.is_file():
            target = probe
            target.write_bytes(b"x")
        out["stat_us"] = median_us(lambda: os.stat(target), 200)
        probe.write_bytes(b"x" * 4096)
        out["open_read_us"] = median_us(probe.read_bytes, 100)
        folder = probe.with_suffix(".d")
        out["mkdir_rmdir_us"] = median_us(lambda: (folder.mkdir(), folder.rmdir()), 50)

        def write_fsync():
            with open(probe, "wb") as handle:
                handle.write(b"y" * 4096)
                handle.flush()
                os.fsync(handle.fileno())

        out["write_fsync_us"] = median_us(write_fsync, 3)
    except OSError:
        # A step the file system refuses is left out, with the steps after it.
        out["incomplete"] = True
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    return out


def _outcome(reports):  # ruff: ignore[too-many-return-statements]
    """
    Name a test's outcome from its phase reports.

    Parameters
    ----------
    reports : dict of str to pytest.TestReport
        By phase (``setup``, ``call``, ``teardown``); missing phases did not run.

    Returns
    -------
    str
        ``error`` (setup or teardown failed), ``failed``, ``xfailed``,
        ``xpassed``, ``skipped`` or ``passed``.
    """
    setup, call, teardown = (reports.get(w) for w in ("setup", "call", "teardown"))
    if setup is not None and setup.failed:
        return "error"
    if setup is not None and setup.skipped:
        return "xfailed" if getattr(setup, "wasxfail", None) is not None else "skipped"
    if call is not None:
        xfail = getattr(call, "wasxfail", None) is not None
        if call.failed:
            return "failed"
        if call.skipped:
            return "xfailed" if xfail else "skipped"
        if xfail:
            return "xpassed"
    if teardown is not None and teardown.failed:
        return "error"
    return "passed"


class _Current:
    """What is being measured for the test in progress."""

    __slots__ = (
        "children",
        "counts",
        "fixtures",
        "gc",
        "io",
        "nodeid",
        "other_fixtures",
        "own",
        "phases",
        "profile",
        "profiler",
        "programs",
        "reexecuted_from",
        "reports",
        "rss",
        "sleep",
        "start",
        "trace",
    )


class TelemetryRecorder:
    """
    The pytest plugin that writes the records.

    Parameters
    ----------
    config : pytest.Config
        The session's configuration.
    settings : Settings
        From :func:`settings_from_environ`.
    clock, cpu_clock : callable
        The wall and process-CPU clocks of the measurements.
    cost_clock : callable
        The clock of the recorder's own cost (``recorder_seconds``): the CPU
        time of the calling thread, so time the process spends waiting for
        a CPU on a busy machine is not counted as the recorder's.

    Raises
    ------
    pytest.UsageError
        If the output directory cannot be created or its file cannot be
        created there (the recorder was asked for, so a silent run without
        it would be wrong).

    Notes
    -----
    **Developer notes.** ``pytest_runtest_protocol`` brackets the whole
    test, the three phase hooks bracket each phase, and
    ``pytest_fixture_setup`` brackets each fixture. Fixture times are
    exclusive: pytest sets a fixture's dependencies up before the hook runs,
    except for ``request.getfixturevalue`` called inside a fixture body,
    whose time is then inside the caller's too.
    """

    def __init__(
        self,
        config,
        settings,
        *,
        clock=time.perf_counter,
        cpu_clock=time.process_time,
        cost_clock=time.thread_time,
    ):
        self.config = config
        self.settings = settings
        self.clock = clock
        self.cpu_clock = cpu_clock
        self.cost_clock = cost_clock
        workerinput = getattr(config, "workerinput", None) or {}
        self.worker = str(workerinput.get("workerid", "main"))
        self.run_id = str(workerinput.get("testrunuid") or uuid.uuid4().hex)
        self.session_id = uuid.uuid4().hex
        self.root = (
            str(config.rootpath) if hasattr(config, "rootpath") else str(config.rootdir)
        )
        self.directory = settings.directory
        if not self.directory.is_absolute():
            self.directory = Path(config.invocation_params.dir) / self.directory
        self.path = self.directory / f"telemetry-{self.run_id}-{self.worker}.jsonl"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if settings.profile is not None:
                (self.directory / "profiles").mkdir(exist_ok=True)
            self._handle = (
                open(  # noqa: SIM115 - kept open for the session, closed in unconfigure
                    self.path, "x", encoding="utf-8", newline="\n", buffering=1
                )
            )
        except OSError as exc:
            raise pytest.UsageError(
                f"{ENV_DIR}={settings.directory}: cannot write the telemetry file {self.path}: {exc}"
            ) from None
        self.errors = 0
        self.error_messages = []
        self.recorder_seconds = 0.0
        self.startup_seconds = 0.0
        self.tests = 0
        self.probe = _Probe()
        self.gc = _GCMonitor(clock)
        self.audit = _AuditState()
        self.audit_available = False
        self.current = None
        self.started = None
        self.collection_files = 0
        self.tracemalloc_started = False
        self.fixture_locations = {}
        # An inline session inside this one (``pytester``) installs its own
        # target; each session puts back the one it found.
        self.previous_target = None

    # -- plumbing --------------------------------------------------------

    def _safely(self, where, func, *args):
        """Run ``func``; on an exception count it and return None."""
        started = self.cost_clock()
        paused = self.audit.active
        self.audit.active = False
        try:
            return func(*args)
        except Exception as exc:  # noqa: BLE001 - the recorder never fails a test
            self.errors += 1
            if len(self.error_messages) < ERROR_MESSAGES_MAX:
                self.error_messages.append(f"{where}: {type(exc).__name__}: {exc}")
            return None
        finally:
            self.audit.active = paused
            # Once per session (machine, packages, the speed probe): kept
            # apart, so recorder_seconds / tests is the cost of a test.
            spent = self.cost_clock() - started
            if where == "sessionstart":
                self.startup_seconds += spent
            else:
                self.recorder_seconds += spent

    def _write(self, record):
        record.setdefault("schema", SCHEMA)
        record.setdefault("version", SCHEMA_VERSION)
        text = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._handle.write(text + "\n")

    def _since_start(self):
        return (
            round(self.clock() - self.started, 6) if self.started is not None else None
        )

    def _location(self, fixturedef):
        known = self.fixture_locations.get(fixturedef)
        if known is not None:
            return known
        code = getattr(getattr(fixturedef, "func", None), "__code__", None)
        where = (
            f"{_display_path(code.co_filename, self.root)}:{code.co_firstlineno}"
            if code
            else "?"
        )
        self.fixture_locations[fixturedef] = where
        return where

    # -- session ---------------------------------------------------------

    def _session_start(self, session):
        self.started = self.clock()
        self.previous_target = _HOOK.target
        if self.settings.tracemalloc:
            import tracemalloc  # noqa: PLC0415

            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self.tracemalloc_started = True
        gc.callbacks.append(self.gc)
        capabilities = dict(self.probe.capabilities())
        capabilities["audit"] = _install_audit_hook()
        capabilities["sleep_events"] = sys.version_info >= (3, 13)
        capabilities["tracemalloc"] = bool(self.settings.tracemalloc)
        self.audit_available = capabilities["audit"]
        header = {
            "kind": "session",
            "run_id": self.run_id,
            "session_id": self.session_id,
            "worker": self.worker,
            "label": self.settings.label,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "rootdir_name": Path(self.root).name,
            "args": [
                _display_arg(a, self.root) for a in self.config.invocation_params.args
            ],
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "machine": _machine(),
            "packages": _packages(),
            "environment": {k: os.environ[k] for k in ENV_RECORDED if k in os.environ},
            "planner": _planner(self.root),
            "capabilities": capabilities,
            "fixture_min_seconds": FIXTURE_MIN_SECONDS,
        }
        if self.worker in ("main", "gw0"):
            header["speed_probe"] = _speed_probe(self.clock, self.directory, self.root)
        self._write(header)
        if self.audit_available:
            _HOOK.target = self.audit

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionstart(self, session):
        self._safely("sessionstart", self._session_start, session)
        self.audit.active = True

    def _collection_finish(self, session):
        self._write(
            {
                "kind": "collection",
                "seconds": self._since_start(),
                "items": len(getattr(session, "items", ()) or ()),
                "files": self.collection_files,
                "module_exec": self.audit.counts.get("module_exec", 0),
            }
        )

    def pytest_collection_finish(self, session):
        self._safely("collection_finish", self._collection_finish, session)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector):
        state = self._safely("collect_start", self._collect_start, collector)
        outcome = yield
        if state is not None:
            self._safely("collect_end", self._collect_end, collector, state, outcome)

    def _collect_start(self, collector):
        if not isinstance(collector, pytest.File):
            return None
        return (
            self.clock(),
            self.cpu_clock(),
            self.audit.counts.get("module_exec", 0),
            len(self.audit.reexecuted),
        )

    def _collect_end(self, collector, state, outcome):
        wall0, cpu0, execs0, reexec0 = state
        report = (
            outcome.get_result() if getattr(outcome, "excinfo", None) is None else None
        )
        self.collection_files += 1
        reexecuted = self.audit.reexecuted[reexec0:]
        self._write(
            {
                "kind": "collect",
                "path": collector.nodeid,
                "seconds": round(self.clock() - wall0, 6),
                "cpu": round(self.cpu_clock() - cpu0, 6),
                "outcome": getattr(report, "outcome", "error"),
                "items": len(getattr(report, "result", ()) or ()),
                "module_exec": self.audit.counts.get("module_exec", 0) - execs0,
                "reexecuted": sorted({_display_path(p, self.root) for p in reexecuted})[
                    :REEXECUTED_LIST_MAX
                ],
            }
        )

    def _session_finish(self, session, exitstatus):
        own = self.probe.own()
        children = self.probe.children()
        record = {
            "kind": "session_end",
            # After every other sessionfinish hook (trylast): the cost monitor
            # of scikitplot/conftest.py may have failed an otherwise green run.
            "exitstatus": int(getattr(session, "exitstatus", exitstatus)),
            "seconds": self._since_start(),
            "cpu": round(self.cpu_clock(), 6),
            "tests": self.tests,
            "peak_rss": own[0] if own else None,
            "child_cpu": round(children[0], 6) if children else None,
            "child_peak_rss": children[1] if children else None,
            "gc": {
                "collections": list(self.gc.counts),
                "seconds": [round(s, 6) for s in self.gc.seconds],
            },
            "events": dict(sorted(self.audit.counts.items())),
            "modules_executed": len(self.audit.module_runs),
            "modules_reexecuted": sorted(
                [_display_path(p, self.root), n]
                for p, n in self.audit.module_runs.items()
                if n > 1
            ),
            "recorder_seconds": round(self.recorder_seconds, 6),
            "recorder_startup_seconds": round(self.startup_seconds, 6),
            "recorder_errors": self.errors + self.gc.errors + self.audit.errors,
            "recorder_error_messages": list(self.error_messages),
        }
        self._write(record)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.audit.active = False
        self._safely("sessionfinish", self._session_finish, session, exitstatus)
        self._close()

    def pytest_unconfigure(self, config):
        # A session that never started or finished (a usage error) still
        # releases what the recorder holds.
        self._close()

    def _close(self):
        if self.gc in gc.callbacks:
            gc.callbacks.remove(self.gc)
        if _HOOK.target is self.audit:
            _HOOK.target = self.previous_target
        if self.tracemalloc_started:
            import tracemalloc  # noqa: PLC0415

            tracemalloc.stop()
            self.tracemalloc_started = False
        self.probe.close()
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()

    # -- tests -----------------------------------------------------------

    def _test_start(self, item):
        cur = _Current()
        cur.nodeid = item.nodeid
        cur.start = self._since_start()
        cur.phases = {}
        cur.reports = {}
        cur.fixtures = []
        cur.other_fixtures = [0, 0.0]
        cur.rss = self.probe.rss()
        cur.own = self.probe.own()
        cur.children = self.probe.children()
        cur.io = self.probe.io_counters()
        cur.gc = self.gc.snapshot()
        cur.counts = dict(self.audit.counts)
        cur.programs = dict(self.audit.programs)
        cur.reexecuted_from = len(self.audit.reexecuted)
        cur.sleep = self.audit.sleep_seconds
        cur.trace = None
        cur.profiler = None
        cur.profile = None
        if self.settings.tracemalloc:
            import tracemalloc  # noqa: PLC0415

            if tracemalloc.is_tracing():
                if hasattr(tracemalloc, "reset_peak"):
                    tracemalloc.reset_peak()
                cur.trace = tracemalloc.get_traced_memory()[0]
        pattern = self.settings.profile
        if pattern is not None and pattern.search(item.nodeid):
            reason = _active_profiler()
            if reason is None:
                profiler = cProfile.Profile()
                try:
                    profiler.enable()
                    cur.profiler = profiler
                except ValueError as exc:
                    reason = f"cProfile refused: {exc}"
            if reason is not None:
                cur.profile = {"skipped": reason}
        self.current = cur

    def _test_end(self, item):
        cur, self.current = self.current, None
        if cur is None:
            return
        if cur.profiler is not None:
            cur.profiler.disable()
            name = _profile_name(cur.nodeid)
            cur.profiler.dump_stats(str(self.directory / "profiles" / name))
            cur.profile = f"profiles/{name}"
        counts = self.audit.counts
        events = {
            k: counts[k] - cur.counts.get(k, 0)
            for k in sorted(counts)
            if counts[k] != cur.counts.get(k, 0)
        }
        programs = {
            k: self.audit.programs[k] - cur.programs.get(k, 0)
            for k in sorted(self.audit.programs)
            if self.audit.programs[k] != cur.programs.get(k, 0)
        }
        gc_counts, gc_seconds = self.gc.snapshot()
        rss = self.probe.rss()
        own = self.probe.own()
        children = self.probe.children()
        io = self.probe.io_counters()
        record = {
            "kind": "test",
            "nodeid": cur.nodeid,
            "path": cur.nodeid.split("::", 1)[0],
            "outcome": _outcome(cur.reports),
            "start": cur.start,
            "phases": cur.phases,
            "fixtures": cur.fixtures,
            "fixtures_other": [cur.other_fixtures[0], round(cur.other_fixtures[1], 6)],
            "gc": {
                "collections": [a - b for a, b in zip(gc_counts, cur.gc[0])],
                "seconds": [round(a - b, 6) for a, b in zip(gc_seconds, cur.gc[1])],
            },
            "events": events,
        }
        if programs:
            record["programs"] = programs
        reexecuted = self.audit.reexecuted[cur.reexecuted_from :]
        if reexecuted:
            names = sorted({_display_path(p, self.root) for p in reexecuted})
            record["reexecuted"] = names[:REEXECUTED_LIST_MAX]
            record["reexecuted_count"] = len(reexecuted)
        if self.audit.sleep_seen:
            record["sleep_requested"] = round(self.audit.sleep_seconds - cur.sleep, 6)
        record.update(self._resource_fields(cur, rss, own, children, io))
        if cur.profile is not None:
            record["profile"] = cur.profile
        self.tests += 1
        self._write(record)

    @staticmethod
    def _resource_fields(cur, rss, own, children, io):
        """Return the memory, context-switch and I/O fields measured for a test.

        A field is left out when the platform does not give it (no
        ``/proc``, no ``resource`` module), never written as zero.
        """
        fields = {}
        if cur.rss is not None and rss is not None:
            fields["rss"] = [cur.rss, rss]
        if cur.own is not None and own is not None:
            fields["peak_rss_growth"] = own[0] - cur.own[0]
            fields["ctx_switches"] = [own[1] - cur.own[1], own[2] - cur.own[2]]
        if (
            cur.children is not None
            and children is not None
            and children[1] > cur.children[1]
        ):
            fields["child_peak_rss"] = children[1]
        if cur.io is not None and io is not None:
            fields["io"] = {
                k: io[k] - cur.io.get(k, 0)
                for k in sorted(io)
                if io[k] != cur.io.get(k, 0)
            }
        if cur.trace is not None:
            import tracemalloc  # noqa: PLC0415

            # A test that stopped tracemalloc leaves nothing to read: None.
            fields["tracemalloc"] = (
                [cur.trace, *tracemalloc.get_traced_memory()]
                if tracemalloc.is_tracing()
                else None
            )
        return fields

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self._safely("test_start", self._test_start, item)
        self.audit.active = True
        yield
        self.audit.active = False
        self._safely("test_end", self._test_end, item)
        self.audit.active = True

    def _phase_start(self):
        children = self.probe.children()
        return (self.clock(), self.cpu_clock(), children[0] if children else None)

    def _phase_end(self, when, state):
        cur = self.current
        if cur is None or state is None:
            return
        wall0, cpu0, child0 = state
        children = self.probe.children()
        phase = {
            "wall": round(self.clock() - wall0, 6),
            "cpu": round(self.cpu_clock() - cpu0, 6),
        }
        if child0 is not None and children is not None:
            phase["child_cpu"] = round(children[0] - child0, 6)
        cur.phases[when] = phase

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        state = self._safely("setup_start", self._phase_start)
        yield
        self._safely("setup_end", self._phase_end, "setup", state)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        state = self._safely("call_start", self._phase_start)
        yield
        self._safely("call_end", self._phase_end, "call", state)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        state = self._safely("teardown_start", self._phase_start)
        yield
        self._safely("teardown_end", self._phase_end, "teardown", state)

    def _logreport(self, report):
        cur = self.current
        if (
            cur is not None
            and report.nodeid == cur.nodeid
            and report.when in ("setup", "call", "teardown")
        ):
            cur.reports[report.when] = report

    def pytest_runtest_logreport(self, report):
        self._safely("logreport", self._logreport, report)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(self, fixturedef, request):
        started = self.clock()
        yield
        self._safely("fixture", self._fixture_end, fixturedef, started)

    def _fixture_end(self, fixturedef, started):
        cur = self.current
        if cur is None:
            return
        seconds = self.clock() - started
        if seconds >= FIXTURE_MIN_SECONDS:
            cur.fixtures.append(
                [
                    fixturedef.argname,
                    fixturedef.scope,
                    round(seconds, 6),
                    self._location(fixturedef),
                ]
            )
        else:
            cur.other_fixtures[0] += 1
            cur.other_fixtures[1] += seconds
