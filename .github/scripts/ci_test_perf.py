# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Analyse where a test run spends its time and memory, from any evidence of it.

Notes
-----
**User notes.** Three commands::

    # A report from anything: directories, files, GitHub artifact zips
    # (nested zips included), in any mix.
    python .github/scripts/ci_test_perf.py report build/test-perf --out build/test-perf/report --bundle

    # Before and after, like ``asv compare``: tests and units whose time
    # changed by more than --factor and by more than --min-seconds.
    python .github/scripts/ci_test_perf.py compare --base before/ --head after/ --out cmp

    # The callers and callees of chosen functions in cProfile files.
    python .github/scripts/ci_test_perf.py profile run.prof --callees "(base)$"

What it reads, recognised by content:

* the recorder's JSON Lines (``SKPLT_TEST_PERF_DIR``, see
  ``scikitplot/_testing/_perf_telemetry.py``): the full picture;
* JUnit XML (``pytest --junitxml``): time and outcome per test;
* cProfile files (``.prof``): function-level detail;
* Cobertura XML (``coverage xml``): coverage per unit, next to its time;
* ``test_durations.json``: the planner's estimates, next to the measurement;
* a ``summary.json`` of an earlier report: for ``compare``.

``report`` writes ``report.md`` (for people), ``summary.json`` (versioned,
for programs, and complete enough for a later ``compare``) and
``ANALYZE_ME.md`` (a self-contained brief: what was measured, how to read
it, the findings with their evidence, and what to run next; give it to an
AI assistant in a new chat as it is). ``--bundle`` adds a zip of those, of
the inputs and of this script, built byte for byte the same from the same
inputs, so it can be analysed again anywhere.

**Developer notes.**

* *Standard library only*, one file: the report job runs it without
  installing anything, and the bundle carries it.
* *Every number is read from one measurement*, or is a sum of them. A
  finding is a fixed rule with a fixed threshold (:data:`THRESHOLDS`),
  printed beside its evidence; nothing is apportioned by assumption.
* *Units are the planner's.* A test belongs to the unit
  ``.github/scripts/ci_test_plan.py`` would run it in; the containers come
  from the recorder's session line (it reads ``.github/ci/test_plan.json``),
  so an analysis made outside the checkout groups the same way.
* *Inputs are untrusted.* Archives are read in memory, never extracted; a
  member whose name is absolute or climbs out (``..``), a link, an
  oversized member, more than :data:`MAX_ZIP_DEPTH` levels of nesting, or an
  XML document that declares an entity, is refused or skipped and reported.
  A cProfile file is checked field by field after it is unmarshalled.
* *cProfile keeps one record per (file, line, function)*. When a module is
  executed twice in one process, the record covers one copy while its
  edges add up both (``cProfile.Profile.snapshot_stats``): such functions
  are listed, and their cumulative times must not be used. The recorder
  counts module executions, so the report names the re-executed modules.
* *Deterministic*: sorted keys and rows, fixed rounding, zip entries with a
  fixed date. Outputs depend on the inputs only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import marshal
import math
import re
import stat
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

__all__ = [
    "THRESHOLDS",
    "PerfError",
    "analyze",
    "build_dataset",
    "compare",
    "iter_inputs",
    "main",
    "render_brief",
    "render_compare",
    "render_report",
    "unit_of",
]

#: Version of this analyzer; written into every output.
TOOL_VERSION = "1"

#: The recorder's format and the versions this analyzer reads.
TELEMETRY_SCHEMA = "skplt-test-perf"
TELEMETRY_VERSIONS = (1,)

#: The format of ``summary.json``.
SUMMARY_SCHEMA = "skplt-test-perf-summary"
SUMMARY_VERSION = 1

#: Limits on what is read. A member above a limit is skipped and reported.
MAX_FILE_BYTES = 256 << 20
MAX_TOTAL_BYTES = 1 << 30
MAX_MEMBERS = 100_000
MAX_ZIP_DEPTH = 3
#: A zip member larger than ``MIN_RATIO_CHECK_BYTES`` may expand at most
#: ``MAX_RATIO`` times (recorder lines compress about 10x).
MAX_RATIO = 200
MIN_RATIO_CHECK_BYTES = 16 << 20
#: Reasons for inputs not read that are listed; the rest are counted.
MAX_SKIPPED_LISTED = 200

#: Default number of rows in each table.
DEFAULT_TOP = 25

#: Package directory when no recorder line names it.
DEFAULT_PACKAGE = "scikitplot"

#: Fixed thresholds of the findings. Each finding prints the values it used.
THRESHOLDS = {
    # A test whose setup takes at least this long and at least twice its call.
    "setup_dominated_seconds": 1.0,
    "setup_dominated_ratio": 2.0,
    # A function-scoped fixture set up this often, this slowly on average.
    "repeated_fixture_setups": 10,
    "repeated_fixture_mean_seconds": 0.1,
    # A test of at least this wall time whose child processes used at least
    # this share of it.
    "subprocess_min_seconds": 1.0,
    "subprocess_share": 0.5,
    # A test of at least this wall time whose CPU (own and children) is
    # below this share of it: it waited.
    "waiting_min_seconds": 1.0,
    "waiting_cpu_share": 0.5,
    # Requested sleep per test.
    "sleep_seconds": 1.0,
    # Resident memory: growth kept after a test, growth of the peak.
    "rss_retained_bytes": 100 * 2**20,
    "rss_peak_growth_bytes": 200 * 2**20,
    # Share of a session spent in full garbage collections.
    "gc_full_share": 0.1,
    # Collection (import) of one test file.
    "collect_file_seconds": 2.0,
    # A profiled function whose callee edges exceed its recorded cumulative
    # time by this much was executed as more than one copy.
    "profile_excess_seconds": 1.0,
    # A profiled fixture whose cumulative time exceeds its own time and its
    # callees' edges by at least this many seconds and this share of it:
    # callees lost their edges (a module executed more than once).
    "unattributed_seconds": 1.0,
    "unattributed_share": 0.05,
    # Cost every test pays: this quantile of whole-test times, over at least
    # this many tests (the conftest monitor's rule). The finding fires at
    # half the budget the coverage workflow enforces (0.1 s), before a run
    # fails on it.
    "floor_quantile": 0.05,
    "floor_min_tests": 200,
    "floor_seconds": 0.05,
}

#: Every rule a finding can come from, in the order findings are listed.
_RULES = (
    "incomplete-session",
    "invalid-lines",
    "recorder-errors",
    "cost-per-test",
    "gc-full-collections",
    "module-re-executed",
    "profile-records-incomplete",
    "setup-dominated",
    "repeated-fixture",
    "subprocess-dominated",
    "waiting",
    "slow-collection",
    "memory",
    "sleeps",
    "network",
)
_RULE_ORDER = {rule: index for index, rule in enumerate(_RULES)}

#: Recorder events shown beside a test that waited, and those that mean
#: the network was used.
_WAITING_EVENTS = ("dns_lookup", "open", "sleep", "socket_connect", "subprocess")
_NETWORK_EVENTS = ("socket_connect", "dns_lookup", "http_request")

#: Shapes of the records read. A cProfile key is ``(file, line, function)``,
#: an entry ``(primitive calls, calls, own time, cumulative time, callers)``
#: and an edge from a caller ``(primitive calls, calls, own time, cumulative
#: time)`` (``cProfile.Profile.snapshot_stats``). In a recorder test line,
#: ``rss`` is ``[start, end]``, ``tracemalloc`` ``[start, end, peak]`` and a
#: fixture ``[name, scope, seconds, location]``.
_PROFILE_KEY_FIELDS = 3
_PROFILE_ENTRY_FIELDS = 5
_PROFILE_EDGE_FIELDS = 4
_RSS_FIELDS = 2
_TRACEMALLOC_FIELDS = 3
_FIXTURE_FIELDS = 4
#: A session end's ``modules_reexecuted`` entry is ``[module, executions]``.
_REEXECUTION_FIELDS = 2
#: Garbage-collector generations; the last one is a full collection.
_GC_GENERATIONS = 3

#: The ``schema`` of ``test_durations.json`` this analyzer reads.
_DURATIONS_SCHEMA = 1

#: Root elements of the XML reports this analyzer reads.
_XML_ROOTS = {"testsuites": "junit", "testsuite": "junit", "coverage": "cobertura"}

#: Test files named in a table cell before the rest are counted.
_NAMES_SHOWN = 2
#: Profiled fixtures, and callees of each, listed in report.md and in the brief.
_FIXTURES_SHOWN = 5
_CALLEES_SHOWN = 8
_BRIEF_FIXTURES_SHOWN = 3
_BRIEF_CALLEES_SHOWN = 6

_PHASES = ("setup", "call", "teardown")
_OUTCOMES = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class PerfError(Exception):
    """An input or an option that cannot be used; the message says why."""


# ---------------------------------------------------------------------------
# Reading inputs
# ---------------------------------------------------------------------------


class Input:
    """
    One file of evidence, read into memory.

    Attributes
    ----------
    name : str
        Where it came from: a path, and ``!member`` for each archive level.
    data : bytes
        Its content.
    """

    __slots__ = ("data", "name")

    def __init__(self, name, data):
        self.name = name
        self.data = data


class _Budget:
    """What may still be read, and what was not read and why."""

    def __init__(self):
        self.total = 0
        self.members = 0
        self.skipped = []
        self.more = 0

    def skip(self, reason):
        if len(self.skipped) < MAX_SKIPPED_LISTED:
            self.skipped.append(reason)
        else:
            self.more += 1

    def listing(self):
        """Return the reasons, the last one counting those not listed."""
        if not self.more:
            return list(self.skipped)
        return [*self.skipped, f"... and {self.more} more inputs not read"]

    def take(self, name, size, packed=None):
        if size > MAX_FILE_BYTES:
            self.skip(f"{name}: {size} bytes is over the {MAX_FILE_BYTES}-byte limit")
            return False
        if (
            packed is not None
            and size > MIN_RATIO_CHECK_BYTES
            and size > MAX_RATIO * max(packed, 1)
        ):
            self.skip(
                f"{name}: {size} bytes from {packed} compressed is over the {MAX_RATIO}x limit"
            )
            return False
        if self.total + size > MAX_TOTAL_BYTES or self.members >= MAX_MEMBERS:
            self.skip(f"{name}: the limit on the total input was reached")
            return False
        self.total += size
        self.members += 1
        return True


def _member_problem(info):
    """Return why an archive member is refused, or None."""
    name = info.filename
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        return "absolute path"
    if "\\" in name:
        return "backslash in the name"
    if ".." in PurePosixPath(name).parts:
        return "the name climbs out of the archive"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "a symbolic link"
    return None


#: What reading a damaged or unsupported zip member can raise.
_MEMBER_ERRORS = (
    zipfile.BadZipFile,
    zlib.error,
    NotImplementedError,
    RuntimeError,
    EOFError,
    OSError,
    ValueError,
)


def _read_member(archive, info, member, budget):
    try:
        with archive.open(info) as handle:
            content = handle.read(MAX_FILE_BYTES + 1)
    except _MEMBER_ERRORS as exc:
        budget.skip(f"{member}: not readable ({type(exc).__name__}: {exc})")
        return None
    if len(content) != info.file_size:
        budget.skip(f"{member}: its size differs from what its header says")
        return None
    return content


def _iter_zip(name, data, depth, budget):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, *_MEMBER_ERRORS) as exc:
        budget.skip(f"{name}: not a readable zip ({exc})")
        return
    with archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            member = f"{name}!{info.filename}"
            problem = _member_problem(info)
            if problem:
                budget.skip(f"{member}: refused ({problem})")
                continue
            if not budget.take(member, info.file_size, info.compress_size):
                continue
            content = _read_member(archive, info, member, budget)
            if content is None:
                continue
            if content.startswith(_ZIP_MAGIC):
                if depth >= MAX_ZIP_DEPTH:
                    budget.skip(
                        f"{member}: archives nested deeper than {MAX_ZIP_DEPTH}"
                    )
                    continue
                yield from _iter_zip(member, content, depth + 1, budget)
            else:
                yield Input(member, content)


def _labels(paths):
    """
    Return the name each input path is reported under.

    A relative path without ``..`` is kept as given; any other is reduced to
    its last component, so that no output holds an absolute path of the
    machine it ran on. A name given twice gets ``#<position>``.
    """
    labels, used = [], set()
    for index, raw in enumerate(paths):
        path = PurePosixPath(Path(raw).as_posix())
        keep = not path.is_absolute() and ".." not in path.parts and str(path) != "."
        label = str(path) if keep else (Path(raw).resolve().name or "input")
        if label in used:
            label = f"{label}#{index + 1}"
        used.add(label)
        labels.append(label)
    return labels


def _excluded(file, exclude):
    resolved = file.resolve()
    return any(resolved == e or e in resolved.parents for e in exclude)


def iter_inputs(paths, budget=None, exclude=()):
    """
    Read every file under ``paths``, opening archives in memory.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Files, directories (walked in sorted order; links are not followed,
        and are listed as not read) or zip archives.
    budget : _Budget, optional
        Collects the members that were skipped and why.
    exclude : iterable of str or pathlib.Path
        Files and directories never read: the outputs of this run, which
        may lie inside the evidence directory.

    Yields
    ------
    Input
        In a stable order: the order of ``paths``, then sorted names. A
        name is the input's label (see :func:`_labels`) followed by the
        path below it, never an absolute path.

    Raises
    ------
    PerfError
        If a path does not exist.
    """
    budget = budget if budget is not None else _Budget()
    paths = list(paths)
    exclude = [Path(e).resolve() for e in exclude]
    for raw, label in zip(paths, _labels(paths)):
        path = Path(raw)
        if path.is_symlink() or not path.exists():
            raise PerfError(
                f"{raw}: no such file or directory (links are not followed)"
            )
        if path.is_file():
            files = [(path, label)]
        else:
            found = sorted(path.rglob("*"), key=lambda p: p.as_posix())
            files = []
            for p in found:
                name = f"{label}/{p.relative_to(path).as_posix()}"
                if p.is_symlink():
                    budget.skip(f"{name}: a symbolic link (not followed)")
                elif p.is_file() and not _excluded(p, exclude):
                    files.append((p, name))
        for file, name in files:
            if not budget.take(name, file.stat().st_size):
                continue
            data = file.read_bytes()
            if data.startswith(_ZIP_MAGIC):
                yield from _iter_zip(name, data, 1, budget)
            else:
                yield Input(name, data)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _first_line(data):
    end = data.find(b"\n")
    return data[: end if end >= 0 else len(data)].strip()


def _xml_problem(data):
    """
    Return why an XML input is not read, or None when it may be parsed.

    Notes
    -----
    **Developer notes.** Only UTF-8 without NUL characters and without
    entity declarations is parsed. Searching the bytes for ``<!ENTITY`` is
    not enough by itself: expat recognises UTF-16 from the first bytes
    (``3C 00``) whatever encoding it is told to use, and in UTF-16 the
    declaration has other bytes. JUnit and Cobertura reports are UTF-8, so
    nothing they contain is refused.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "XML that is not UTF-8 is not read"
    if "\x00" in text:
        return "XML with NUL characters (UTF-16 or UTF-32) is not read"
    if "<!ENTITY" in text:
        return "XML entity declarations are not read"
    return None


def _xml_root(data):
    """Parse XML that :func:`_xml_problem` accepted; return its root element."""
    # A str is parsed as UTF-8 whatever the document declares (pyexpat).
    return ET.fromstring(data.decode("utf-8"))  # noqa: S314 - see _xml_problem


def _classify_json(data):
    try:
        first = json.loads(_first_line(data))
    except _JSON_ERRORS:
        first = None
    if isinstance(first, dict) and first.get("schema") == TELEMETRY_SCHEMA:
        return "telemetry"
    try:
        whole = json.loads(data)
    except _JSON_ERRORS:
        return None
    if not isinstance(whole, dict):
        return None
    if whole.get("schema") == SUMMARY_SCHEMA:
        return "summary"
    if whole.get("schema") == _DURATIONS_SCHEMA and isinstance(
        whole.get("units"), dict
    ):
        return "durations"
    return None


def _classify_xml(data):
    if _xml_problem(data) is not None:
        return "xml-refused"
    try:
        root = _xml_root(data)
    except ET.ParseError:
        return None
    return _XML_ROOTS.get(root.tag)


def classify(item):
    """
    Say what kind of evidence an input is.

    Returns
    -------
    str or None
        ``telemetry``, ``summary``, ``durations``, ``junit``, ``cobertura``,
        ``profile``; ``xml-refused`` for XML that is not read (see
        :func:`_xml_problem`); None for anything else (ignored, and listed).
    """
    # By name first: a marshalled dict (a cProfile file) may itself start
    # with the byte "{" (marshal's TYPE_DICT) and must not be read as JSON.
    if item.name.endswith(".prof"):
        return "profile"
    head = item.data.lstrip()[:1]
    if head == b"{":
        return _classify_json(item.data)
    if head == b"<":
        return _classify_xml(item.data)
    return None


_NOT_JSON = object()


#: What ``json.loads`` raises on hostile text: bad syntax, and nesting deeper
#: than the interpreter's recursion limit.
_JSON_ERRORS = (ValueError, RecursionError)


def _json_or_marker(raw):
    try:
        return json.loads(raw)
    except _JSON_ERRORS:
        return _NOT_JSON


def _check_version(item, record):
    if record.get("version") not in TELEMETRY_VERSIONS:
        raise PerfError(
            f"{item.name}: recorder format version {record.get('version')!r}; this analyzer "
            f"reads {', '.join(map(str, TELEMETRY_VERSIONS))} (tool version {TOOL_VERSION})"
        )


# Checks of the fields this analyzer reads, so that no record can make the
# analysis fail half way: a record whose fields do not pass is skipped and
# counted (``invalid_lines``, rule ``invalid-lines``).


#: The largest magnitude of an integer read: any real count, byte size or
#: time fits, and arithmetic on it never overflows a float.
_MAX_INTEGER = 2**63


def _integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -_MAX_INTEGER <= value <= _MAX_INTEGER
    )


def _number(value):
    if isinstance(value, float):
        return math.isfinite(value)
    return _integer(value)


def _count(value):
    return _integer(value) and value >= 0


def _text(value):
    return isinstance(value, str)


def _mapping(value):
    return isinstance(value, dict)


def _optional(check):
    return lambda value: value is None or check(value)


def _list_of(check, length=None):
    return lambda value: (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(check(v) for v in value)
    )


def _map_of(check):
    return lambda value: (
        isinstance(value, dict)
        and all(isinstance(k, str) and check(v) for k, v in value.items())
    )


def _phases_ok(value):
    return isinstance(value, dict) and all(
        isinstance(times, dict)
        and all(_optional(_number)(times.get(f)) for f in ("wall", "cpu", "child_cpu"))
        for times in value.values()
    )


def _gc_ok(value):
    return (
        isinstance(value, dict)
        and _optional(_list_of(_number, _GC_GENERATIONS))(value.get("seconds"))
        and _optional(_list_of(_count, _GC_GENERATIONS))(value.get("collections"))
    )


def _fixture_ok(value):
    return (
        isinstance(value, list)
        and len(value) == _FIXTURE_FIELDS
        and _text(value[0])
        and _text(value[1])
        and _number(value[2])
        and _text(value[3])
    )


def _planner_ok(value):
    return (
        isinstance(value, dict)
        and _text(value.get("package_root"))
        and _list_of(_text)(value.get("containers"))
    )


def _reexecution_ok(value):
    return (
        isinstance(value, list)
        and len(value) == _REEXECUTION_FIELDS
        and _text(value[0])
        and _count(value[1])
    )


def _probe_value_ok(value):
    # A number of microseconds, or the flag ``incomplete``.
    return value is None or isinstance(value, bool) or _number(value)


#: The fields of each record kind this analyzer reads, with their check.
#: ``None`` passes every check but those marked required.
_RECORD_FIELDS = {
    "session": {
        "run_id": _text,
        "session_id": _optional(_text),
        "worker": _optional(_text),
        "label": _optional(_text),
        "started_at": _optional(_text),
        "python": _optional(_mapping),
        "machine": _optional(_mapping),
        "speed_probe": _optional(_map_of(_probe_value_ok)),
        "packages": _optional(_mapping),
        "environment": _optional(_mapping),
        "capabilities": _optional(_mapping),
        "planner": _optional(_planner_ok),
    },
    "collection": {"seconds": _optional(_number)},
    "collect": {
        "path": _optional(_text),
        "seconds": _optional(_number),
        "module_exec": _optional(_count),
        "items": _optional(_count),
        "outcome": _optional(_text),
        "reexecuted": _optional(_list_of(_text)),
    },
    "test": {
        "nodeid": _text,
        "path": _optional(_text),
        "outcome": _optional(_text),
        "phases": _optional(_phases_ok),
        "gc": _optional(_gc_ok),
        "events": _optional(_map_of(_count)),
        "programs": _optional(_map_of(_count)),
        "fixtures": _optional(_list_of(_fixture_ok)),
        "rss": _optional(_list_of(_count, _RSS_FIELDS)),
        "peak_rss_growth": _optional(_integer),
        "tracemalloc": _optional(_list_of(_count, _TRACEMALLOC_FIELDS)),
        "sleep_requested": _optional(_number),
        "reexecuted": _optional(_list_of(_text)),
        "profile": _optional(lambda v: _text(v) or _map_of(_text)(v)),
    },
    "session_end": {
        "exitstatus": _optional(_integer),
        "seconds": _optional(_number),
        "gc": _optional(_gc_ok),
        "peak_rss": _optional(_count),
        "recorder_seconds": _optional(_number),
        "recorder_errors": _optional(_count),
        "recorder_error_messages": _optional(_list_of(_text)),
        "modules_reexecuted": _optional(_list_of(_reexecution_ok)),
    },
}

#: Fields a record of the kind cannot do without.
_REQUIRED_FIELDS = {
    "session": ("run_id",),
    "collect": ("path",),
    "test": ("nodeid", "outcome"),
}


def _record_ok(record):
    """Return True when every field this analyzer reads from ``record`` passes its check."""
    kind = record.get("kind")
    if not isinstance(kind, str):
        return False
    checks = _RECORD_FIELDS.get(kind)
    if checks is None:
        # A kind this analyzer does not read; it is kept and never used.
        return True
    if any(record.get(name) is None for name in _REQUIRED_FIELDS.get(kind, ())):
        return False
    return all(check(record.get(name)) for name, check in checks.items())


def parse_telemetry(item):
    """
    Read one recorder file.

    Returns
    -------
    dict
        ``{"session", "lines", "end", "invalid", "truncated"}``: the session
        line, the other lines in order, the ``session_end`` line or None (the
        process was killed or timed out), and counts of bad lines.

    Raises
    ------
    PerfError
        If the format version is not one this analyzer reads.
    """
    text = item.data.decode("utf-8", "replace")
    raw_lines = text.split("\n")
    # A file that does not end with a newline ends with a line whose write
    # was cut short (the process was killed while writing it).
    cut_short = len(raw_lines) - 1 if not text.endswith("\n") else None
    session, lines, end = None, [], None
    invalid = truncated = 0
    for index, raw in enumerate(raw_lines):
        if not raw.strip():
            continue
        record = _json_or_marker(raw)
        if record is _NOT_JSON and index == cut_short:
            truncated += 1
            continue
        if not isinstance(record, dict) or record.get("schema") != TELEMETRY_SCHEMA:
            invalid += 1
            continue
        if record.get("version") not in TELEMETRY_VERSIONS:
            # A session of another format cannot be read; a stray line can.
            if record.get("kind") == "session" and session is None:
                _check_version(item, record)
            invalid += 1
            continue
        if not _record_ok(record):
            invalid += 1
            continue
        kind = record.get("kind")
        if kind == "session" and session is None:
            session = record
        elif kind == "session":
            invalid += 1
        elif kind == "session_end":
            end = record
        else:
            lines.append(record)
    if session is None:
        raise PerfError(f"{item.name}: a recorder file without its session line")
    return {
        "session": session,
        "lines": lines,
        "end": end,
        "invalid": invalid,
        "truncated": truncated,
    }


def parse_junit(item):
    """
    Read the test cases of a JUnit XML report.

    Returns
    -------
    list of dict
        ``{"classname", "name", "seconds", "outcome"}``; ``seconds`` is the
        test's whole time as pytest writes it (setup, call and teardown).
    """
    root = _xml_root(item.data)
    cases = []
    for case in root.iter("testcase"):
        try:
            seconds = float(case.get("time") or 0.0)
        except ValueError:
            seconds = math.nan
        if not math.isfinite(seconds):
            raise PerfError(
                f"{item.name}: testcase {case.get('name')!r} has time="
                f"{case.get('time')!r}, not a number of seconds"
            )
        seconds = max(seconds, 0.0)
        tags = {child.tag for child in case}
        if "error" in tags:
            outcome = "error"
        elif "failure" in tags:
            outcome = "failed"
        elif "skipped" in tags:
            outcome = "skipped"
        else:
            outcome = "passed"
        cases.append(
            {
                "classname": case.get("classname") or "",
                "name": case.get("name") or "",
                "seconds": seconds,
                "outcome": outcome,
            }
        )
    return cases


_CONDITION = re.compile(r"\(([0-9]{1,18})/([0-9]{1,18})\)")
_COUNT = re.compile(r"[0-9]{1,18}")


def parse_cobertura(item, package):
    """
    Read per-file line and branch counts from a Cobertura report.

    Returns
    -------
    list of tuple
        ``(path, lines, covered, branches, branches_covered)``; ``path`` is
        prefixed with ``package/`` when the report's source is that
        directory.

    Raises
    ------
    PerfError
        If a line has no ``hits`` or its ``hits`` is not a count. (A branch
        line without ``condition-coverage``, which Cobertura allows, adds no
        branches.)
    """
    root = _xml_root(item.data)
    sources = [(s.text or "").strip().replace("\\", "/") for s in root.iter("source")]
    prefix = (
        f"{package}/"
        if any(s.rstrip("/").endswith("/" + package) or s == package for s in sources)
        else ""
    )
    rows = {}
    for cls in root.iter("class"):
        filename = (cls.get("filename") or "").replace("\\", "/")
        if prefix and not filename.startswith(prefix):
            filename = prefix + filename
        lines = covered = branches = branches_covered = 0
        for line in cls.iter("line"):
            lines += 1
            hits = line.get("hits")
            if hits is None or not _COUNT.fullmatch(hits.strip()):
                raise PerfError(
                    f"{item.name}: line {line.get('number')} of {filename} has "
                    f"hits={hits!r}, not a count"
                )
            covered += int(hits) > 0
            match = _CONDITION.search(line.get("condition-coverage") or "")
            if line.get("branch") == "true" and match:
                branches_covered += int(match.group(1))
                branches += int(match.group(2))
        old = rows.get(filename, (0, 0, 0, 0))
        rows[filename] = (
            old[0] + lines,
            old[1] + covered,
            old[2] + branches,
            old[3] + branches_covered,
        )
    return sorted((name, *values) for name, values in rows.items())


def _is_key(value):
    return (
        isinstance(value, tuple)
        and len(value) == _PROFILE_KEY_FIELDS
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and isinstance(value[2], str)
    )


def _is_numbers(value, length):
    return (
        isinstance(value, tuple)
        and len(value) == length
        and all(_number(v) for v in value)
    )


def _is_entry(value):
    return (
        isinstance(value, tuple)
        and len(value) == _PROFILE_ENTRY_FIELDS
        and _is_numbers(value[:-1], _PROFILE_ENTRY_FIELDS - 1)
        and isinstance(value[-1], dict)
        and all(
            _is_key(caller) and _is_numbers(edge, _PROFILE_EDGE_FIELDS)
            for caller, edge in value[-1].items()
        )
    )


#: Marshal type codes a cProfile table is made of (``marshal.c``), by how
#: many bytes follow the code: a fixed number, or a length whose size is given.
_MARSHAL_FIXED = {"0": 0, "N": 0, "F": 0, "T": 0, "{": 0, "i": 4, "g": 8, "r": 4}
_MARSHAL_SIZED = {
    "z": 1,  # short ASCII: 1-byte length, then that many bytes
    "Z": 1,  # short ASCII, interned
    "a": 4,  # ASCII: 4-byte length
    "A": 4,  # ASCII, interned
    "u": 4,  # UTF-8 text
    "t": 4,  # interned UTF-8 text
    "s": 4,  # bytes
}
_MARSHAL_COUNTED = {")": 1, "(": 4, "[": 4}  # containers: a count of items
_MARSHAL_FLAG_REF = 0x80
#: Bytes of marshal's 32-bit lengths, counts and digit counts (signed).
_MARSHAL_WORD = 4


def _marshal_problem(data):
    """
    Return why ``data`` is not a marshal stream of a cProfile table, or None.

    Notes
    -----
    **Developer notes.** ``marshal.loads`` trusts the lengths and counts in
    its input: five bytes claiming a list of two thousand million items make
    it allocate gigabytes before it finds the end of the data. This scan
    reads the stream once, without building anything, and accepts only the
    types a cProfile table uses, each length within the bytes that remain
    and each count at most one item per remaining byte, so what
    ``marshal.loads`` then allocates is bounded by the size of the file.
    Containers need no nesting here: marshal writes their items inline.
    """
    pos, end = 0, len(data)
    while pos < end:
        code = chr(data[pos] & ~_MARSHAL_FLAG_REF & 0xFF)
        pos += 1
        if code in _MARSHAL_FIXED:
            pos += _MARSHAL_FIXED[code]
        elif code in _MARSHAL_SIZED or code in _MARSHAL_COUNTED or code == "l":
            width = (
                _MARSHAL_SIZED.get(code) or _MARSHAL_COUNTED.get(code) or _MARSHAL_WORD
            )
            if pos + width > end:
                return "truncated"
            size = int.from_bytes(
                data[pos : pos + width], "little", signed=width == _MARSHAL_WORD
            )
            pos += width
            if code == "l":
                size = abs(size) * 2  # digits of 15 bits, two bytes each
            if size < 0 or size > end - pos:
                return f"a length of {size} where {end - pos} bytes remain"
            if code in _MARSHAL_SIZED or code == "l":
                pos += size
        else:
            return f"marshal type {code!r} is not part of a cProfile table"
        if pos > end:
            return "truncated"
    return None


def load_profile(item):
    """
    Unmarshal a cProfile file and check its structure.

    Returns
    -------
    dict
        The ``pstats`` table: ``(file, line, name) -> (primitive calls, calls,
        own time, cumulative time, callers)``.

    Raises
    ------
    PerfError
        If the content is not a cProfile table.
    """
    problem = _marshal_problem(item.data)
    if problem:
        raise PerfError(f"{item.name}: not a cProfile file ({problem})")
    try:
        table = marshal.loads(item.data)  # noqa: S302 - scanned first, checked below
    except (EOFError, ValueError, TypeError, MemoryError, OverflowError) as exc:
        raise PerfError(f"{item.name}: not a cProfile file ({exc})") from None
    if not isinstance(table, dict):
        raise PerfError(f"{item.name}: not a cProfile table")
    for key, value in table.items():
        if not (_is_key(key) and _is_entry(value)):
            raise PerfError(
                f"{item.name}: an entry of the cProfile table has an unexpected shape"
            )
    return table


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def unit_of(path, package, containers):
    """
    Return the planner unit that owns a test file.

    Parameters
    ----------
    path : str
        POSIX path of the test file, relative to the repository.
    package : str
        The package directory (``scikitplot``).
    containers : collection of str or None
        The planner's containers; None selects the rule below.

    Returns
    -------
    str
        With containers: the planner's rule (``Planner.owner``): the first
        directory below the nested containers; a file directly in a
        container belongs to ``<container> (loose files)``. Without: the
        directory that holds the first ``tests`` directory, else the file's
        directory. ``(outside <package>)`` for anything else.

    Examples
    --------
    >>> unit_of("scikitplot/corpus/tests/test_a.py", "scikitplot", {"scikitplot"})
    'scikitplot/corpus'
    >>> unit_of("scikitplot/tests/test_a.py", "scikitplot", {"scikitplot"})
    'scikitplot/tests'
    >>> unit_of("scikitplot/corpus/tests/test_a.py", "scikitplot", None)
    'scikitplot/corpus'
    """
    parts = PurePosixPath(path).parts
    if not parts or parts[0] != package:
        return f"(outside {package})"
    if containers is None:
        directories = parts[:-1]
        if "tests" in directories[1:]:
            index = directories.index("tests", 1)
            return "/".join(directories[:index]) or package
        return "/".join(directories) or package
    current = package
    for index in range(1, len(parts)):
        if index == len(parts) - 1:
            return f"{current} (loose files)"
        candidate = f"{current}/{parts[index]}"
        if candidate in containers:
            current = candidate
            continue
        return candidate
    return current


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _repo_planner(repo_root):
    if repo_root is None:
        return None
    path = Path(repo_root) / ".github" / "ci" / "test_plan.json"
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PerfError(f"{path}: not JSON ({exc})") from None
    planner = (
        {
            "package_root": config.get("package_root"),
            "containers": config.get("containers", []),
        }
        if isinstance(config, dict)
        else None
    )
    if not _planner_ok(planner):
        raise PerfError(
            f"{path}: package_root must be a string and containers a list of strings"
        )
    return planner


def build_dataset(paths, *, runs=None, repo_root=None, exclude=(), check_runs=True):
    """
    Read and group every input.

    Parameters
    ----------
    paths : list of str
        As for :func:`iter_inputs`.
    runs : list of str, optional
        Recorder run ids to keep (a prefix is enough); ``["latest"]`` keeps
        the run that started last. All runs by default.
    repo_root : str, optional
        A checkout whose ``.github/ci/test_plan.json`` gives the units when
        no recorder line does.
    exclude : iterable of str
        Paths never read (see :func:`iter_inputs`).
    check_runs : bool
        Refuse two runs of the same tests; ``False`` where no test time is
        added up (``profile``).

    Returns
    -------
    dict
        The parsed evidence, a manifest of every input, and the inputs
        themselves (``"inputs"``, for a bundle of exactly what was read).

    Raises
    ------
    PerfError
        If a path is missing, a recorder file has an unknown version, a
        cProfile file is malformed, or two runs hold the same test (they are
        two measurements of it: choose one with ``runs``, or ``compare``
        them).

    Notes
    -----
    **Developer notes.** Inputs with the same content are read once (a
    bundle's ``raw/`` copy beside its original, a report directory inside
    the evidence directory), and of two recorder files of one session the
    one with more lines is kept (a copy taken while the run was going). The
    manifest lists what was left out and why.
    """
    budget = _Budget()
    manifest, ignored, kept, seen = [], [], [], {}
    found = {kind: [] for kind in _PARSERS}
    for item in iter_inputs(paths, budget, exclude=exclude):
        kind = classify(item)
        digest = _sha256(item.data)
        if kind == "xml-refused":
            budget.skip(f"{item.name}: refused ({_xml_problem(item.data)})")
            continue
        entry = {
            "name": item.name,
            "kind": kind or "ignored",
            "bytes": len(item.data),
            "sha256": digest,
        }
        manifest.append(entry)
        if kind is None:
            # Not remembered by its digest: the same bytes under a name that
            # makes them evidence (a ``.prof`` suffix) are still read.
            ignored.append(item.name)
        elif digest in seen:
            entry["kind"] = f"duplicate of {seen[digest]}"
        else:
            seen[digest] = item.name
            kept.append(item)
            found[kind].append(_PARSERS[kind](item))
    telemetry = _one_file_per_session(found["telemetry"], budget)
    if runs:
        telemetry = _select_runs(telemetry, runs)
    if check_runs:
        _refuse_overlapping_runs(telemetry)
    planner, unit_rule = _planner_of(telemetry, repo_root)
    package = planner.get("package_root") or DEFAULT_PACKAGE
    containers = (
        set(planner["containers"]) if planner.get("containers") is not None else None
    )
    return {
        "inputs": kept,
        "manifest": manifest,
        "ignored": ignored,
        "skipped": budget.listing(),
        "telemetry": telemetry,
        "junit": found["junit"],
        "cobertura": [
            (item.name, parse_cobertura(item, package)) for item in found["cobertura"]
        ],
        "durations": found["durations"],
        "summaries": found["summary"],
        "profiles": dict(found["profile"]),
        "package": package,
        "containers": containers,
        "unit_rule": unit_rule,
    }


def _parse_durations(item):
    doc = json.loads(item.data)
    for unit, value in doc["units"].items():
        if not (
            isinstance(value, dict)
            and _optional(_count)(value.get("tests"))
            and _optional(_number)(value.get("work_seconds"))
        ):
            raise PerfError(
                f"{item.name}: unit {unit!r} is not {{'tests': <count>, 'work_seconds': <seconds>}}"
            )
    return (item.name, doc)


#: How each kind of input is read. Cobertura reports are parsed once the
#: package directory is known, so they are kept as they are; a summary is
#: checked where it is used (``compare``), so that an old report among the
#: evidence does not stop a new one.
_PARSERS = {
    "telemetry": lambda item: {**parse_telemetry(item), "name": item.name},
    "junit": lambda item: (item.name, parse_junit(item)),
    "cobertura": lambda item: item,
    "durations": _parse_durations,
    "summary": lambda item: (item.name, json.loads(item.data)),
    "profile": lambda item: (item.name, load_profile(item)),
}


def _planner_of(telemetry, repo_root):
    """
    Return the planner settings that group tests into units, and the rule's name.

    Raises
    ------
    PerfError
        If two recorded sessions name different planner settings: they
        come from different checkouts, and one grouping would be wrong for
        the other's tests.
    """
    planners = {}
    for parsed in telemetry:
        found = parsed["session"].get("planner")
        if found:
            key = (
                found.get("package_root"),
                tuple(sorted(set(found.get("containers") or ()))),
            )
            planners.setdefault(key, (parsed["name"], found))
    if len(planners) > 1:
        names = ", ".join(sorted(name for name, _ in planners.values()))
        raise PerfError(
            f"the recorded sessions come from checkouts with different unit settings "
            f"(.github/ci/test_plan.json): {names}. Report them separately, or compare them."
        )
    if planners:
        return next(iter(planners.values()))[1], "planner"
    found = _repo_planner(repo_root)
    if found is not None:
        return found, "planner"
    return {"package_root": DEFAULT_PACKAGE, "containers": None}, "tests-directory"


def _one_file_per_session(telemetry, budget):
    best = {}
    for parsed in telemetry:
        sid = parsed["session"].get("session_id") or parsed["name"]
        size = (len(parsed["lines"]), parsed["end"] is not None, parsed["name"])
        current = best.get(sid)
        if current is None or size > current[0]:
            if current is not None:
                budget.skip(
                    f"{current[1]['name']}: an incomplete copy of session {sid[:12]}; {parsed['name']} is used"
                )
            best[sid] = (size, parsed)
        else:
            budget.skip(
                f"{parsed['name']}: an incomplete copy of session {sid[:12]}; {current[1]['name']} is used"
            )
    return [best[sid][1] for sid in sorted(best)]


def _select_runs(telemetry, runs):
    if list(runs) == ["latest"]:
        if not telemetry:
            return telemetry
        latest = max(
            telemetry,
            key=lambda p: (
                p["session"].get("started_at") or "",
                p["session"]["run_id"],
            ),
        )
        wanted = {latest["session"]["run_id"]}
    else:
        wanted = {
            p["session"]["run_id"]
            for p in telemetry
            for prefix in runs
            if p["session"]["run_id"].startswith(prefix)
        }
        missing = [r for r in runs if not any(w.startswith(r) for w in wanted)]
        if missing:
            raise PerfError(f"no recorded run matches {', '.join(missing)}")
    return [p for p in telemetry if p["session"]["run_id"] in wanted]


def _refuse_overlapping_runs(telemetry):
    owner = {}
    clash = {}
    for parsed in telemetry:
        run = parsed["session"]["run_id"]
        for line in parsed["lines"]:
            if line.get("kind") != "test":
                continue
            first = owner.setdefault(line["nodeid"], run)
            if first != run:
                clash.setdefault((first, run), line["nodeid"])
    if clash:
        described = {}
        for parsed in telemetry:
            s = parsed["session"]
            described.setdefault(
                s["run_id"],
                f"{s['run_id'][:12]} (label {s.get('label')}, started {s.get('started_at')})",
            )
        pairs = "; ".join(
            f"{described[a]} and {described[b]} both ran {nodeid}"
            for (a, b), nodeid in sorted(clash.items())
        )
        raise PerfError(
            "two recorded runs hold the same tests, so their times cannot be added: "
            f"{pairs}. For report, keep one with --run <id prefix> or --run latest; to compare "
            "them, give each side the files of one run: compare --base <run A> --head <run B>"
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _r(value, digits=6):
    return None if value is None else round(float(value), digits)


def _quantile(values, q):
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))] if ordered else None


def _outcome_of(line):
    """Return the test's outcome: one of :data:`_OUTCOMES`, or ``other``."""
    return line["outcome"] if line["outcome"] in _OUTCOMES else "other"


def _new_test_row(line, session, package, containers):
    path = line.get("path") or line["nodeid"].split("::", 1)[0]
    return {
        "nodeid": line["nodeid"],
        "path": path,
        "unit": unit_of(path, package, containers),
        "run_id": session["run_id"],
        "worker": session.get("worker"),
        "label": session.get("label"),
        "runs": 0,
        "outcome": _outcome_of(line),
        "setup": 0.0,
        "call": 0.0,
        "teardown": 0.0,
        "cpu": 0.0,
        "child_cpu": 0.0,
        "gc_seconds": [0.0] * _GC_GENERATIONS,
        "gc_collections": [0] * _GC_GENERATIONS,
        "events": {},
        "programs": {},
        "fixtures": [],
        "rss_retained": None,
        "peak_rss_growth": None,
        "tracemalloc_peak": None,
        "sleep_requested": None,
        "reexecuted": [],
        "profile": None,
    }


def _add_counts(into, counts):
    for name, count in (counts or {}).items():
        into[name] = into.get(name, 0) + count


def _larger(current, value):
    return value if current is None else max(current, value)


def _add_test_line(row, line):
    """Add one recorded run of a test to its row (all fields were checked)."""
    row["runs"] += 1
    row["outcome"] = _outcome_of(line)
    phases = line.get("phases") or {}
    for phase in _PHASES:
        values = phases.get(phase) or {}
        row[phase] += values.get("wall") or 0.0
        row["cpu"] += values.get("cpu") or 0.0
        row["child_cpu"] += values.get("child_cpu") or 0.0
    gcinfo = line.get("gc") or {}
    seconds = gcinfo.get("seconds") or [0.0] * _GC_GENERATIONS
    collections = gcinfo.get("collections") or [0] * _GC_GENERATIONS
    for generation in range(_GC_GENERATIONS):
        row["gc_seconds"][generation] += seconds[generation]
        row["gc_collections"][generation] += collections[generation]
    _add_counts(row["events"], line.get("events"))
    _add_counts(row["programs"], line.get("programs"))
    row["fixtures"].extend(line.get("fixtures") or [])
    rss, trace = line.get("rss"), line.get("tracemalloc")
    if rss is not None:
        row["rss_retained"] = _larger(row["rss_retained"], rss[1] - rss[0])
    if line.get("peak_rss_growth") is not None:
        row["peak_rss_growth"] = _larger(
            row["peak_rss_growth"], line["peak_rss_growth"]
        )
    if trace is not None:
        row["tracemalloc_peak"] = _larger(row["tracemalloc_peak"], trace[2] - trace[0])
    if line.get("sleep_requested") is not None:
        row["sleep_requested"] = (row["sleep_requested"] or 0.0) + line[
            "sleep_requested"
        ]
    row["reexecuted"].extend(line.get("reexecuted") or [])
    if line.get("profile") is not None:
        row["profile"] = line["profile"]


def _test_rows(dataset):
    """One row per node id; the runs of a test in one run are added up."""
    rows = {}
    package, containers = dataset["package"], dataset["containers"]
    for parsed in dataset["telemetry"]:
        session = parsed["session"]
        for line in parsed["lines"]:
            if line.get("kind") != "test":
                continue
            row = rows.get(line["nodeid"])
            if row is None:
                row = rows[line["nodeid"]] = _new_test_row(
                    line, session, package, containers
                )
            _add_test_line(row, line)
    for row in rows.values():
        row["total"] = row["setup"] + row["call"] + row["teardown"]
    return sorted(rows.values(), key=lambda r: r["nodeid"])


def _aggregate(rows, key):
    groups = {}
    for row in rows:
        name = row[key]
        g = groups.get(name)
        if g is None:
            g = groups[name] = {
                key: name,
                "tests": 0,
                "total": 0.0,
                "setup": 0.0,
                "call": 0.0,
                "teardown": 0.0,
                "cpu": 0.0,
                "child_cpu": 0.0,
                "gc_seconds": 0.0,
                "outcomes": {},
                "events": {},
                "max_peak_rss_growth": None,
            }
        g["tests"] += 1
        for field in ("total", "setup", "call", "teardown", "cpu", "child_cpu"):
            g[field] += row[field]
        g["gc_seconds"] += sum(row["gc_seconds"])
        g["outcomes"][row["outcome"]] = g["outcomes"].get(row["outcome"], 0) + 1
        for name_, count in row["events"].items():
            g["events"][name_] = g["events"].get(name_, 0) + count
        if row["peak_rss_growth"] is not None:
            g["max_peak_rss_growth"] = max(
                row["peak_rss_growth"], g["max_peak_rss_growth"] or 0
            )
    out = []
    for g in groups.values():
        for field in (
            "total",
            "setup",
            "call",
            "teardown",
            "cpu",
            "child_cpu",
            "gc_seconds",
        ):
            g[field] = _r(g[field])
        g["outcomes"] = dict(sorted(g["outcomes"].items()))
        g["events"] = dict(sorted(g["events"].items()))
        out.append(g)
    return sorted(out, key=lambda g: (-g["total"], g[key]))


def _fixtures(rows):
    table = {}
    for row in rows:
        for name, scope, seconds, location in row["fixtures"]:
            t = table.get((name, location))
            if t is None:
                t = table[(name, location)] = {
                    "name": name,
                    "location": location,
                    "scopes": set(),
                    "setups": 0,
                    "seconds": 0.0,
                    "max": 0.0,
                }
            t["scopes"].add(scope)
            t["setups"] += 1
            t["seconds"] += seconds
            t["max"] = max(t["max"], seconds)
    out = [
        {
            "name": t["name"],
            "location": t["location"],
            "scope": "/".join(sorted(t["scopes"])),
            "setups": t["setups"],
            "seconds": _r(t["seconds"]),
            "mean": _r(t["seconds"] / t["setups"]),
            "max": _r(t["max"]),
        }
        for t in table.values()
    ]
    return sorted(out, key=lambda t: (-t["seconds"], t["name"], t["location"]))


def _full_collections(end, field):
    """Return the session's full collections (``collections`` or ``seconds``), or None."""
    values = ((end or {}).get("gc") or {}).get(field)
    return None if values is None else values[_GC_GENERATIONS - 1]


def _session_rows(dataset):
    rows = []
    for parsed in dataset["telemetry"]:
        s, end = parsed["session"], parsed["end"]
        tests = sum(1 for line in parsed["lines"] if line.get("kind") == "test")
        collection = next(
            (line for line in parsed["lines"] if line.get("kind") == "collection"), None
        )
        last_test = next(
            (
                line["nodeid"]
                for line in reversed(parsed["lines"])
                if line.get("kind") == "test"
            ),
            None,
        )
        machine = s.get("machine") or {}
        rows.append(
            {
                "name": parsed["name"],
                "run_id": s["run_id"],
                "session_id": s.get("session_id"),
                "worker": s.get("worker"),
                "label": s.get("label"),
                "started_at": s.get("started_at"),
                "python": (s.get("python") or {}).get("version"),
                "machine": {
                    k: machine.get(k)
                    for k in (
                        "system",
                        "machine",
                        "cpu_count",
                        "cpu_model",
                        "memory_bytes",
                    )
                },
                "speed_probe": s.get("speed_probe"),
                "packages": s.get("packages") or {},
                "environment": s.get("environment") or {},
                "capabilities": s.get("capabilities") or {},
                "tests": tests,
                "complete": end is not None,
                "last_test": None if end is not None else last_test,
                "exitstatus": end.get("exitstatus") if end else None,
                "seconds": end.get("seconds") if end else None,
                "collection_seconds": collection.get("seconds") if collection else None,
                "gc_full_seconds": _full_collections(end, "seconds"),
                "gc_full_collections": _full_collections(end, "collections"),
                "recorder_seconds": end.get("recorder_seconds") if end else None,
                "recorder_errors": end.get("recorder_errors") if end else None,
                "recorder_error_messages": (
                    (end or {}).get("recorder_error_messages") or []
                ),
                "peak_rss": end.get("peak_rss") if end else None,
                "invalid_lines": parsed["invalid"],
                "truncated_lines": parsed["truncated"],
                "modules_reexecuted": (end or {}).get("modules_reexecuted") or [],
            }
        )
    return sorted(rows, key=lambda r: (r["run_id"], r["worker"] or "", r["name"]))


def _collections(dataset):
    rows = [
        {
            "path": line["path"],
            "seconds": _r(line.get("seconds")),
            "module_exec": line.get("module_exec"),
            "items": line.get("items"),
            "outcome": line.get("outcome"),
            "reexecuted": line.get("reexecuted") or [],
        }
        for parsed in dataset["telemetry"]
        for line in parsed["lines"]
        if line.get("kind") == "collect"
    ]
    return sorted(rows, key=lambda r: (-(r["seconds"] or 0.0), r["path"]))


def _reexecuted(sessions, rows, collections):
    """Modules run more than once in a process, with where it happened.

    ``collected_by`` lists the test files whose import (collection) ran the
    module again, ``tests`` the tests during which it ran again.
    """
    found = {}

    def entry_for(module):
        return found.setdefault(
            module,
            {
                "module": module,
                "executions": 0,
                "sessions": 0,
                "collected_by": [],
                "tests": [],
            },
        )

    for session in sessions:
        for module, count in session["modules_reexecuted"]:
            entry = entry_for(module)
            entry["executions"] = max(entry["executions"], int(count))
            entry["sessions"] += 1
    for collect in collections:
        for module in collect["reexecuted"]:
            entry = entry_for(module)
            if collect["path"] not in entry["collected_by"]:
                entry["collected_by"].append(collect["path"])
    for row in rows:
        for module in row["reexecuted"]:
            entry = entry_for(module)
            if row["nodeid"] not in entry["tests"]:
                entry["tests"].append(row["nodeid"])
    for entry in found.values():
        entry["collected_by_count"] = len(entry["collected_by"])
        entry["collected_by"] = sorted(entry["collected_by"])[:_EVIDENCE_ITEMS]
        entry["tests_count"] = len(entry["tests"])
        entry["tests"] = sorted(entry["tests"])[:_EVIDENCE_ITEMS]
    return sorted(found.values(), key=lambda e: (-e["executions"], e["module"]))


def _junit_rows(dataset):
    rows = []
    package, containers = dataset["package"], dataset["containers"]
    for _, cases in dataset["junit"]:
        for case in cases:
            path = case["classname"].replace(".", "/") + "/x"
            rows.append(
                {
                    "nodeid": f"{case['classname']}::{case['name']}",
                    "unit": unit_of(path, package, containers),
                    "seconds": case["seconds"],
                    "outcome": case["outcome"],
                }
            )
    return sorted(rows, key=lambda r: r["nodeid"])


def _coverage_by_unit(dataset):
    package, containers = dataset["package"], dataset["containers"]
    units = {}
    for _, files in dataset["cobertura"]:
        for path, lines, covered, branches, bcovered in files:
            if "/tests/" in f"/{path}" or PurePosixPath(path).name.startswith("test_"):
                continue
            unit = unit_of(path, package, containers)
            if unit.endswith("(loose files)"):
                unit = unit[: -len(" (loose files)")]
            u = units.setdefault(unit, [0, 0, 0, 0])
            u[0] += lines
            u[1] += covered
            u[2] += branches
            u[3] += bcovered
    return {
        unit: {
            "lines": v[0],
            "lines_covered": v[1],
            "line_rate": _r(v[1] / v[0], 4) if v[0] else None,
            "branches": v[2],
            "branches_covered": v[3],
            "branch_rate": _r(v[3] / v[2], 4) if v[2] else None,
        }
        for unit, v in sorted(units.items())
    }


def _finding(rule, subject, evidence, next_step, weight=0.0):
    """One finding: the rule that fired, on what, its evidence, what to run next.

    ``weight`` orders the findings of one rule (seconds, or a count), largest first.
    """
    return {
        "rule": rule,
        "subject": subject,
        "evidence": evidence,
        "next_step": next_step,
        "weight": _r(weight),
    }


#: Items of a list shown in a finding's evidence; the rest are in summary.json.
_EVIDENCE_ITEMS = 5
#: Slowest fixtures named beside a setup-dominated test.
_FIXTURES_NAMED = 3


def _session_subject(s):
    return f"{s['label'] or s['worker'] or 'session'} ({s['run_id'][:12]})"


def _session_findings(s):
    th = THRESHOLDS
    if not s["complete"]:
        yield _finding(
            "incomplete-session",
            _session_subject(s),
            {"tests_recorded": s["tests"], "last_test": s["last_test"]},
            "The process ended without finishing (killed, crashed or timed out). The last test "
            "recorded is the last one that finished; the next one in collection order was running.",
            weight=s["tests"],
        )
    if s["invalid_lines"]:
        yield _finding(
            "invalid-lines",
            s["name"],
            {"invalid_lines": s["invalid_lines"]},
            "Lines that are not recorder records of a version this analyzer reads, or whose fields "
            "have unexpected types, were skipped and are in no total. The file was written by "
            "something else, or edited.",
            weight=s["invalid_lines"],
        )
    if s["recorder_errors"]:
        yield _finding(
            "recorder-errors",
            s["name"],
            {
                "errors": s["recorder_errors"],
                "messages": s["recorder_error_messages"][:_EVIDENCE_ITEMS],
            },
            "The recorder skipped what it could not measure; report the messages as a bug of "
            "scikitplot/_testing/_perf_telemetry.py.",
            weight=s["recorder_errors"],
        )
    if s["seconds"] and s["gc_full_seconds"] is not None:
        share = s["gc_full_seconds"] / s["seconds"]
        if share >= th["gc_full_share"]:
            yield _finding(
                "gc-full-collections",
                _session_subject(s),
                {
                    "full_collections": s["gc_full_collections"],
                    "seconds": _r(s["gc_full_seconds"], 3),
                    "share_of_session": _r(share, 3),
                    "threshold": th["gc_full_share"],
                },
                "Full collections at module boundaries (SKPLT_TEST_GC in scikitplot/conftest.py). "
                "Compare a run with SKPLT_TEST_GC=off on the same selection before changing the policy.",
                weight=s["gc_full_seconds"],
            )


def _time_findings(row):
    """Apply the rules on where a test's time went: setup, child processes, waiting, sleeps."""
    th = THRESHOLDS
    if (
        row["setup"] >= th["setup_dominated_seconds"]
        and row["setup"] >= th["setup_dominated_ratio"] * row["call"]
    ):
        top = sorted(row["fixtures"], key=lambda f: (-f[2], f[0]))[:_FIXTURES_NAMED]
        yield _finding(
            "setup-dominated",
            row["nodeid"],
            {
                "setup": _r(row["setup"], 3),
                "call": _r(row["call"], 3),
                "slowest_fixtures": [[f[0], f[1], _r(f[2], 3), f[3]] for f in top],
            },
            "Profile this test without editing it: SKPLT_TEST_PERF_PROFILE='<regex of its node id>' "
            "with SKPLT_TEST_PERF_DIR set, then: ci_test_perf.py profile <its .prof> --callees '<fixture name>'.",
            weight=row["setup"],
        )
    if (
        row["total"] >= th["subprocess_min_seconds"]
        and row["child_cpu"] >= th["subprocess_share"] * row["total"]
    ):
        yield _finding(
            "subprocess-dominated",
            row["nodeid"],
            {
                "wall": _r(row["total"], 3),
                "child_cpu": _r(row["child_cpu"], 3),
                "programs": row["programs"],
            },
            "The time is in child processes; profile the child command itself, or reduce how many are started.",
            weight=row["child_cpu"],
        )
    elif (
        row["total"] >= th["waiting_min_seconds"]
        and (row["cpu"] + row["child_cpu"]) < th["waiting_cpu_share"] * row["total"]
    ):
        yield _finding(
            "waiting",
            row["nodeid"],
            {
                "wall": _r(row["total"], 3),
                "cpu": _r(row["cpu"], 3),
                "child_cpu": _r(row["child_cpu"], 3),
                "sleep_requested": _r(row["sleep_requested"], 3),
                "events": {
                    k: row["events"][k] for k in _WAITING_EVENTS if k in row["events"]
                },
            },
            "Less than half of the wall time was CPU: the test waited (sleep, I/O, a lock, a child "
            "process still running, the network). The counters beside it say which.",
            weight=row["total"],
        )
    if (
        row["sleep_requested"] is not None
        and row["sleep_requested"] >= th["sleep_seconds"]
    ):
        yield _finding(
            "sleeps",
            row["nodeid"],
            {"sleep_requested": _r(row["sleep_requested"], 3)},
            "Replace fixed sleeps by waiting on the condition itself.",
            weight=row["sleep_requested"],
        )


def _resource_findings(row):
    """Apply the rules on what a test used besides time: the network, memory."""
    th = THRESHOLDS
    network = {k: row["events"][k] for k in _NETWORK_EVENTS if row["events"].get(k)}
    if network:
        yield _finding(
            "network",
            row["nodeid"],
            network,
            "The test opened a connection or resolved a name; in CI that is slow and can fail for "
            "reasons outside the code (check whether it is meant to).",
            weight=sum(network.values()),
        )
    if (row["rss_retained"] or 0) >= th["rss_retained_bytes"] or (
        row["peak_rss_growth"] or 0
    ) >= th["rss_peak_growth_bytes"]:
        yield _finding(
            "memory",
            row["nodeid"],
            {
                "rss_retained": row["rss_retained"],
                "peak_rss_growth": row["peak_rss_growth"],
                "tracemalloc_peak": row["tracemalloc_peak"],
            },
            "Run it alone with SKPLT_TEST_PERF_TRACEMALLOC=1 to see the Python allocation peak; "
            "retained growth that later tests do not reuse is a leak candidate.",
            weight=max(row["rss_retained"] or 0, row["peak_rss_growth"] or 0),
        )


def _fixture_finding(f):
    th = THRESHOLDS
    if (
        "function" in f["scope"].split("/")
        and f["setups"] >= th["repeated_fixture_setups"]
        and f["mean"] >= th["repeated_fixture_mean_seconds"]
    ):
        return _finding(
            "repeated-fixture",
            f"{f['name']} ({f['location']})",
            {"setups": f["setups"], "mean": f["mean"], "seconds": f["seconds"]},
            "An expensive function-scoped fixture: a wider scope (module, session) shares it, "
            "if no test changes what it returns; copy what tests change instead of rebuilding it.",
            weight=f["seconds"],
        )
    return None


def _collection_finding(c):
    if (c["seconds"] or 0.0) >= THRESHOLDS["collect_file_seconds"]:
        return _finding(
            "slow-collection",
            c["path"],
            {"seconds": c["seconds"], "module_exec": c["module_exec"]},
            "Importing this test file is slow: python -X importtime -c 'import <module>' shows "
            "which import; move heavy imports into the tests or fixtures that need them.",
            weight=c["seconds"] or 0.0,
        )
    return None


def _reexecution_finding(entry):
    return _finding(
        "module-re-executed",
        entry["module"],
        {
            "executions": entry["executions"],
            "collected_by": entry["collected_by"],
            "collected_by_count": entry["collected_by_count"],
            "tests": entry["tests"],
            "tests_count": entry["tests_count"],
        },
        "The same file ran more than once in one process (imported under two names, loaded by "
        "path, or a failed import retried). Its import cost is paid each time, its module state is "
        "not shared, and cProfile's cumulative times for its functions cover one copy only.",
        weight=entry["executions"],
    )


def _unattributed_is_material(unattributed, seconds):
    """Return True when lost edges hide at least the threshold share of ``seconds``."""
    return bool(unattributed) and unattributed >= THRESHOLDS[
        "unattributed_share"
    ] * max(seconds, 0.0)


def _profile_finding(name, prof):
    lost = [
        [fx["fixture"], fx["unattributed"], fx["seconds"]]
        for fx in prof["fixture_breakdown"]
        if fx["unattributed"] is not None
        and fx["unattributed"] >= THRESHOLDS["unattributed_seconds"]
        and _unattributed_is_material(fx["unattributed"], fx["seconds"])
    ]
    if not (prof["inconsistent"] or lost):
        return None
    return _finding(
        "profile-records-incomplete",
        name,
        {
            "functions_with_extra_edges": [
                i["function"] for i in prof["inconsistent"][:_EVIDENCE_ITEMS]
            ],
            "fixtures_with_unattributed_seconds": lost[:_EVIDENCE_ITEMS],
        },
        "A module was executed more than once in the profiled process: use the callee-edge "
        "totals of the first list, and profile the fixtures of the second alone "
        "(SKPLT_TEST_PERF_PROFILE on one test) to see their callees.",
        weight=sum(x[1] for x in lost) + len(prof["inconsistent"]),
    )


def _floor_finding(rows):
    th = THRESHOLDS
    totals = [row["total"] for row in rows]
    if len(totals) < th["floor_min_tests"]:
        return None
    floor = _quantile(totals, th["floor_quantile"])
    if floor < th["floor_seconds"]:
        return None
    return _finding(
        "cost-per-test",
        "all tests",
        {
            "floor_seconds": _r(floor, 4),
            "quantile": th["floor_quantile"],
            "tests": len(totals),
        },
        "Every test pays this before it does anything: an autouse fixture or a per-test hook "
        "became expensive (see the developer notes of scikitplot/conftest.py).",
        weight=floor,
    )


def _findings(*, sessions, rows, fixtures, collections, reexecuted, profiles):
    """Apply every rule of :data:`_RULES`; return the findings, in rule order."""
    out = [f for s in sessions for f in _session_findings(s)]
    for row in rows:
        out.extend(_time_findings(row))
        out.extend(_resource_findings(row))
    out.extend(_fixture_finding(f) for f in fixtures)
    out.extend(_collection_finding(c) for c in collections)
    out.extend(_reexecution_finding(entry) for entry in reexecuted)
    out.extend(_profile_finding(name, prof) for name, prof in sorted(profiles.items()))
    out.append(_floor_finding(rows))
    return sorted(
        (f for f in out if f is not None),
        key=lambda f: (_RULE_ORDER[f["rule"]], -f["weight"], f["subject"]),
    )


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

_KINDS_C = (
    (
        "subprocess wait",
        re.compile(
            r"<method 'poll' of 'select\.(poll|epoll)' objects>"
            r"|<built-in method (posix\.waitpid|_posixsubprocess\.fork_exec|posix\.posix_spawn)>"
        ),
    ),
    (
        "filesystem",
        re.compile(
            r"<built-in method (posix\.(stat|lstat|mkdir|rmdir|unlink|rename|replace|scandir"
            r"|listdir|fsync|sendfile|utime|chmod|open|close)|io\.open|io\.open_code|nt\.\w+)>"
            r"|<method '(read|write|flush|readline|readlines|__exit__)' of "
            r"'_io\.(BufferedReader|BufferedWriter|TextIOWrapper|FileIO|_IOBase)' objects>"
        ),
    ),
    ("garbage collection", re.compile(r"<built-in method gc\.collect>")),
    (
        "cryptography",
        re.compile(
            r"<method '(verify|sign)' of 'cryptography\..*' objects>|<built-in method _hashlib\..*>"
        ),
    ),
    ("sleep", re.compile(r"<built-in method time\.sleep>")),
    (
        "compile",
        re.compile(r"<built-in method (builtins\.compile|_imp\.create_dynamic)>"),
    ),
)
_KINDS_PY = (
    ("json", re.compile(r"/json/(__init__|encoder|decoder|scanner)\.py$")),
    ("pathlib", re.compile(r"/pathlib(/__init__)?\.py$")),
    ("pytest + pluggy", re.compile(r"/(_pytest|pluggy)/")),
)


def _kind(key):
    filename, _, name = key
    if filename == "~":
        for label, pattern in _KINDS_C:
            if pattern.fullmatch(name):
                return label
        return None
    norm = filename.replace("\\", "/")
    for label, pattern in _KINDS_PY:
        if pattern.search(norm):
            return label
    return None


def _label(key, package):
    filename, line, name = key
    if filename == "~":
        return name
    norm = filename.replace("\\", "/")
    marker = f"/{package}/"
    if marker in norm:
        norm = package + "/" + norm.split(marker, 1)[1]
    elif "site-packages/" in norm:
        norm = norm.split("site-packages/", 1)[1]
    else:
        norm = norm.rsplit("/", 1)[-1]
    return f"{norm}:{line}({name})"


def _edges_from(table, caller_pred):
    out = {}
    for callee, value in table.items():
        seconds, calls = 0.0, 0
        for caller, edge in value[4].items():
            if caller_pred(caller):
                seconds += edge[3]
                calls += edge[1]
        if calls:
            out[callee] = (seconds, calls)
    return out


#: pytest's own functions whose cumulative time is a phase of the session.
_PHASE_FUNCTIONS = (
    ("collection", "_pytest/main.py", "pytest_collection"),
    ("setup", "_pytest/runner.py", "pytest_runtest_setup"),
    ("call", "_pytest/runner.py", "pytest_runtest_call"),
    ("teardown", "_pytest/runner.py", "pytest_runtest_teardown"),
)


def _is_pytest_function(key, suffix, name):
    return key[2] == name and key[0].replace("\\", "/").endswith(suffix)


def _profile_phases(table):
    return {
        phase: _r(
            sum(v[3] for k, v in table.items() if _is_pytest_function(k, suffix, name))
        )
        for phase, suffix, name in _PHASE_FUNCTIONS
    }


def _profile_kinds(table):
    kinds = {}
    for key, value in table.items():
        label = _kind(key)
        if label:
            k = kinds.setdefault(label, [0.0, 0])
            k[0] += value[2]
            k[1] += value[1]
    return [
        [k, _r(v[0]), v[1]]
        for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ]


def _ranked_edges(edges, package, top):
    rows = sorted(
        ((_label(k, package), s, c) for k, (s, c) in edges.items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [[n, _r(s), c] for n, s, c in rows[:top]]


def _callees_of(table, key, package, top):
    rows = sorted(
        (
            (_label(callee, package), value[4][key][3], value[4][key][1])
            for callee, value in table.items()
            if callee != key and key in value[4]
        ),
        key=lambda r: (-r[1], r[0]),
    )
    return [[n, _r(s), c] for n, s, c in rows[:top]]


def _callers_of(table, key, package, top):
    rows = sorted(
        ((_label(c, package), e[3], e[1]) for c, e in table[key][4].items()),
        key=lambda r: (-r[1], r[0]),
    )
    return [[n, _r(s), c] for n, s, c in rows[:top]]


def _unattributed(table, key):
    """
    Return the cumulative seconds of ``key`` that neither its own time nor its callees explain.

    Notes
    -----
    **Developer notes.** A function's cumulative time is its own time plus
    its callees' edges, exactly, unless an edge was lost: cProfile stores the
    edges *into* a function in that function's record and keeps one record
    per (file, line, name), so a callee in a module executed twice loses the
    edges recorded on the other copy.
    """
    value = table[key]
    inside = sum(v[4][key][3] for c, v in table.items() if c != key and key in v[4])
    return max(value[3] - value[2] - inside, 0.0)


def _inconsistent(table, package, limit):
    """Project functions whose callee edges exceed their recorded cumulative time."""
    outgoing = {}
    for callee, value in table.items():
        for caller, edge in value[4].items():
            if caller != callee:
                outgoing[caller] = outgoing.get(caller, 0.0) + edge[3]
    found = []
    for key, out_seconds in outgoing.items():
        if key not in table or key[0] == "~":
            continue
        label = _label(key, package)
        if out_seconds - table[key][3] > limit and label.startswith(package + "/"):
            found.append(
                {
                    "function": label,
                    "recorded_cumulative": _r(table[key][3]),
                    "callee_edges": _r(out_seconds),
                    "calls": table[key][1],
                    "direct_recursion": key in table[key][4],
                }
            )
    found.sort(
        key=lambda i: (-(i["callee_edges"] - i["recorded_cumulative"]), i["function"])
    )
    return found


def _profile_queries(table, package, top, *, callers, callees):
    queries = []
    for direction, patterns in (("callers", callers), ("callees", callees)):
        for text in patterns:
            pattern = re.compile(text)
            matched = [
                k
                for k in table
                if pattern.search(_label(k, package))
                or pattern.search(f"{k[0]}:{k[1]}({k[2]})")
            ]
            for key in sorted(
                matched, key=lambda k: (-table[k][3], _label(k, package))
            )[:top]:
                query = {
                    "pattern": text,
                    "direction": direction,
                    "function": _label(key, package),
                    "own": _r(table[key][2]),
                    "cumulative": _r(table[key][3]),
                    "calls": table[key][1],
                }
                if direction == "callers":
                    query["edges"] = _callers_of(table, key, package, top)
                else:
                    query["edges"] = _callees_of(table, key, package, top)
                    query["unattributed"] = _r(_unattributed(table, key))
                queries.append(query)
    return queries


def analyze_profile(
    table,
    package=DEFAULT_PACKAGE,
    top=DEFAULT_TOP,
    *,
    callers=(),
    callees=(),
    min_excess=None,
):
    """
    Summarise one cProfile table.

    Parameters
    ----------
    table : dict
        From :func:`load_profile`.
    package : str
        Package directory, to name project functions.
    top : int
        Rows per table.
    callers, callees : iterable of str
        Regular expressions; matching functions are listed with their
        direct callers or callees (edge-exact).
    min_excess : float, optional
        Seconds by which a function's callee edges must exceed its recorded
        cumulative time to be listed as inconsistent;
        ``THRESHOLDS["profile_excess_seconds"]`` by default.

    Returns
    -------
    dict
        Phases, kinds of work (own time, non-overlapping), slowest test
        functions and fixtures (edge-exact from pytest's own frames), each
        fixture's slowest callees, hot spots, inconsistent records, queries.

    Raises
    ------
    re.error
        If a query is not a valid regular expression.
    """
    tests = _edges_from(
        table,
        lambda c: _is_pytest_function(c, "_pytest/python.py", "pytest_pyfunc_call"),
    )
    fixtures = _edges_from(
        table,
        lambda c: _is_pytest_function(c, "_pytest/fixtures.py", "call_fixture_func"),
    )
    fixture_breakdown = [
        {
            "fixture": _label(key, package),
            "seconds": _r(seconds),
            "calls": calls,
            "callees": _callees_of(table, key, package, top),
            "unattributed": _r(_unattributed(table, key)),
        }
        for key, (seconds, calls) in sorted(
            fixtures.items(), key=lambda kv: (-kv[1][0], _label(kv[0], package))
        )[:top]
    ]
    rows = [(_label(k, package), v[2], v[3], v[1]) for k, v in table.items()]
    project = [
        r for r in rows if r[0].startswith(package + "/") and "/tests/" not in r[0]
    ]
    limit = (
        THRESHOLDS["profile_excess_seconds"]
        if min_excess is None
        else float(min_excess)
    )
    return {
        "total_own_seconds": _r(sum(v[2] for v in table.values())),
        "functions": len(table),
        "phases": _profile_phases(table),
        "kinds": _profile_kinds(table),
        "tests": _ranked_edges(tests, package, top),
        "fixtures": _ranked_edges(fixtures, package, top),
        "fixture_breakdown": fixture_breakdown,
        "hot_own": [
            [n, _r(o), _r(c), k]
            for n, o, c, k in sorted(rows, key=lambda r: (-r[1], r[0]))[:top]
        ],
        "hot_cumulative": [
            [n, _r(o), _r(c), k]
            for n, o, c, k in sorted(project, key=lambda r: (-r[2], r[0]))[:top]
        ],
        "inconsistent": _inconsistent(table, package, limit)[:top],
        "queries": _profile_queries(
            table, package, top, callers=callers, callees=callees
        ),
    }


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def analyze(dataset, *, top=DEFAULT_TOP):
    """
    Build the summary of a dataset.

    Parameters
    ----------
    dataset : dict
        From :func:`build_dataset`.
    top : int
        Rows per ranked table.

    Returns
    -------
    dict
        The content of ``summary.json``: inputs, sessions, totals, units,
        files, ranked tests, fixtures, collection, re-executed modules,
        coverage, JUnit-only data, profiles, findings, and every test in a
        compact form (``all_tests``) for a later comparison.

    Raises
    ------
    PerfError
        If ``top`` is not positive.
    """
    if not isinstance(top, int) or top < 1:
        raise PerfError(f"--top must be a positive integer, got {top!r}")
    rows = _test_rows(dataset)
    sessions = _session_rows(dataset)
    units = _aggregate(rows, "unit")
    files = _aggregate(rows, "path")
    fixtures = _fixtures(rows)
    collections = _collections(dataset)
    reexecuted = _reexecuted(sessions, rows, collections)
    profiles = {
        name: analyze_profile(table, dataset["package"], top)
        for name, table in sorted(dataset["profiles"].items())
    }
    findings = _findings(
        sessions=sessions,
        rows=rows,
        fixtures=fixtures,
        collections=collections,
        reexecuted=reexecuted,
        profiles=profiles,
    )
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    totals = {
        "tests": len(rows),
        "outcomes": dict(sorted(outcomes.items())),
        "setup": _r(sum(r["setup"] for r in rows)),
        "call": _r(sum(r["call"] for r in rows)),
        "teardown": _r(sum(r["teardown"] for r in rows)),
        "total": _r(sum(r["total"] for r in rows)),
        "cpu": _r(sum(r["cpu"] for r in rows)),
        "child_cpu": _r(sum(r["child_cpu"] for r in rows)),
        "gc_seconds": [_r(sum(r["gc_seconds"][i] for r in rows)) for i in range(3)],
        "collection_seconds": _r(sum(c["seconds"] or 0.0 for c in collections)),
        "floor_seconds": (
            _r(_quantile([r["total"] for r in rows], THRESHOLDS["floor_quantile"]))
            if len(rows) >= THRESHOLDS["floor_min_tests"]
            else None
        ),
    }

    def compact(row):
        return [
            row["nodeid"],
            row["unit"],
            row["outcome"],
            _r(row["setup"]),
            _r(row["call"]),
            _r(row["teardown"]),
            _r(row["cpu"]),
            _r(row["child_cpu"]),
            row["peak_rss_growth"],
            row["rss_retained"],
        ]

    def ranked(key, keep=lambda r: True):
        chosen = sorted(
            (r for r in rows if keep(r)), key=lambda r: (-(r[key] or 0), r["nodeid"])
        )[:top]
        return [
            {
                "nodeid": r["nodeid"],
                "unit": r["unit"],
                "outcome": r["outcome"],
                "total": _r(r["total"]),
                "setup": _r(r["setup"]),
                "call": _r(r["call"]),
                "teardown": _r(r["teardown"]),
                "cpu": _r(r["cpu"]),
                "child_cpu": _r(r["child_cpu"]),
                "peak_rss_growth": r["peak_rss_growth"],
                "rss_retained": r["rss_retained"],
                "tracemalloc_peak": r["tracemalloc_peak"],
                "programs": r["programs"],
                "profile": r["profile"],
            }
            for r in chosen
        ]

    junit_rows = _junit_rows(dataset)
    junit_units = {}
    for r in junit_rows:
        u = junit_units.setdefault(
            r["unit"], {"unit": r["unit"], "tests": 0, "seconds": 0.0, "outcomes": {}}
        )
        u["tests"] += 1
        u["seconds"] += r["seconds"]
        u["outcomes"][r["outcome"]] = u["outcomes"].get(r["outcome"], 0) + 1
    estimates = {}
    for name, doc in dataset["durations"]:
        for unit, value in sorted((doc.get("units") or {}).items()):
            estimates[unit] = {
                "tests": value.get("tests"),
                "work_seconds": value.get("work_seconds"),
                "source": doc.get("source") or name,
            }
    coverage = _coverage_by_unit(dataset)
    for unit in units:
        if unit["unit"] in coverage:
            unit["coverage"] = coverage[unit["unit"]]
        if unit["unit"] in estimates:
            unit["planner_estimate_seconds"] = estimates[unit["unit"]]["work_seconds"]
    return {
        "schema": SUMMARY_SCHEMA,
        "version": SUMMARY_VERSION,
        "tool_version": TOOL_VERSION,
        "inputs": sorted(dataset["manifest"], key=lambda e: e["name"]),
        "ignored_inputs": sorted(dataset["ignored"]),
        "skipped_inputs": list(dataset["skipped"]),
        "unit_rule": dataset["unit_rule"],
        "package": dataset["package"],
        "thresholds": dict(THRESHOLDS),
        "sessions": sessions,
        "totals": totals,
        "units": units,
        "files": files[:top],
        "slowest_tests": ranked("total"),
        "slowest_setups": ranked("setup", lambda r: r["setup"] > 0),
        "child_process_tests": ranked("child_cpu", lambda r: r["child_cpu"] > 0),
        "memory_tests": ranked(
            "peak_rss_growth", lambda r: (r["peak_rss_growth"] or 0) > 0
        ),
        "fixtures": fixtures[:top],
        "collection": collections[:top],
        "reexecuted_modules": reexecuted,
        "findings": findings,
        "junit": {
            "tests": len(junit_rows),
            "seconds": _r(sum(r["seconds"] for r in junit_rows)),
            "units": sorted(
                (
                    {
                        **u,
                        "seconds": _r(u["seconds"]),
                        "outcomes": dict(sorted(u["outcomes"].items())),
                    }
                    for u in junit_units.values()
                ),
                key=lambda u: (-u["seconds"], u["unit"]),
            ),
            "slowest": [
                [r["nodeid"], r["unit"], _r(r["seconds"]), r["outcome"]]
                for r in sorted(junit_rows, key=lambda r: (-r["seconds"], r["nodeid"]))[
                    :top
                ]
            ],
            "all": [
                [r["nodeid"], r["unit"], _r(r["seconds"]), r["outcome"]]
                for r in junit_rows
            ],
        },
        "planner_estimates": estimates,
        "coverage": coverage,
        "profiles": profiles,
        "all_tests_columns": [
            "nodeid",
            "unit",
            "outcome",
            "setup",
            "call",
            "teardown",
            "cpu",
            "child_cpu",
            "peak_rss_growth",
            "rss_retained",
        ],
        "all_tests": [compact(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _one_line(text):
    return re.sub(r"[\r\n\t\v\f\x00-\x1f\x7f]+", " ", str(text))


def _code(text):
    """Return ``text`` as inline code that can neither end its table cell nor its line."""
    text = _one_line(text).replace("|", "\\|")
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _cell(text):
    """Return ``text`` (from an input: untrusted) as plain table-cell text."""
    text = _one_line(text).replace("\\", "\\\\").replace("|", "\\|")
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
        text = text.replace(char, entity)
    return re.sub(r"([`*_\[\]#])", r"\\\1", text)


def _s(value, digits=1):
    if value is None:
        return "-"
    return f"{float(value):,.{digits}f}"


def _mb(value):
    return "-" if value is None else f"{value / 2**20:,.1f}"


def _pct(part, whole):
    return "-" if not whole or part is None else f"{100.0 * part / whole:.1f}%"


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return lines


def _scope_lines(summary):
    t = summary["totals"]
    sessions = summary["sessions"]
    lines = []
    if sessions:
        complete = sum(1 for s in sessions if s["complete"])
        lines.append(
            f"- **{t['tests']:,} tests** in {len(sessions)} recorded session(s) ({complete} complete); "
            f"outcomes: {', '.join(f'{k} {v:,}' for k, v in t['outcomes'].items()) or 'none'}."
        )
        lines.append(
            f"- Test time **{_s(t['total'])} s**: setup {_s(t['setup'])} s, call {_s(t['call'])} s, "
            f"teardown {_s(t['teardown'])} s; CPU {_s(t['cpu'])} s, child processes {_s(t['child_cpu'])} s; "
            f"garbage collection young {_s((t['gc_seconds'][0] or 0) + (t['gc_seconds'][1] or 0))} s, full {_s(t['gc_seconds'][2])} s; "
            f"collection {_s(t['collection_seconds'])} s."
        )
        if t["floor_seconds"] is not None:
            lines.append(
                f"- Cost every test pays (5% quantile): {_s(t['floor_seconds'], 4)} s."
            )
    if summary["junit"]["tests"]:
        lines.append(
            f"- JUnit: {summary['junit']['tests']:,} test cases, {_s(summary['junit']['seconds'])} s."
        )
    if summary["profiles"]:
        lines.append(f"- cProfile files: {len(summary['profiles'])}.")
    lines.append(
        f"- Units by: **{summary['unit_rule']}** rule (package {_code(summary['package'])})."
    )
    return lines


def _machines(summary):
    rows = []
    for s in summary["sessions"]:
        m, p = s["machine"], s.get("speed_probe") or {}
        rows.append(
            [
                _cell(s["label"] or "-"),
                _cell(s["worker"] or "-"),
                _code(s["run_id"][:12]),
                _cell(s["python"] or "-"),
                _cell(f"{m.get('system') or '-'} {m.get('machine') or ''}".strip()),
                _cell(m.get("cpu_count") or "-"),
                _s(p.get("python_loop_us"), 0),
                _s(p.get("stat_us"), 2),
                _s(p.get("mkdir_rmdir_us"), 1),
                _s(p.get("write_fsync_us"), 0),
                s["tests"],
                _s(s["seconds"]),
                "yes" if s["complete"] else "**no**",
            ]
        )
    return _table(
        [
            "label",
            "worker",
            "run",
            "python",
            "system",
            "cpus",
            "loop us",
            "stat us",
            "mkdir us",
            "fsync us",
            "tests",
            "seconds",
            "complete",
        ],
        rows,
    )


def _finding_lines(findings, per_rule=10, heading="###"):
    """Group findings by rule: the advice once, then the heaviest subjects with their evidence."""
    lines, groups = [], {}
    for f in findings:
        groups.setdefault(f["rule"], []).append(f)
    for rule, items in groups.items():
        lines += [f"{heading} {rule} ({len(items)})", "", items[0]["next_step"], ""]
        for f in items[:per_rule]:
            evidence = json.dumps(f["evidence"], sort_keys=True, ensure_ascii=True)
            lines.append(f"- {_code(f['subject'])}: {_code(evidence)}")
        if len(items) > per_rule:
            lines.append(
                f"- ... {len(items) - per_rule} more in summary.json -> findings"
            )
        lines.append("")
    return lines


def _profile_lines(name, prof, top):
    lines = [f"### {_code(name)}", ""]
    lines.append(
        "phases (cumulative s): "
        + ", ".join(f"{k} {_s(v)}" for k, v in prof["phases"].items())
        + f"; own time recorded {_s(prof['total_own_seconds'])} s over {prof['functions']:,} functions."
    )
    lines.append("")
    lines += _table(
        ["kind of work (own time)", "seconds", "calls"],
        ([k, _s(s), f"{c:,}"] for k, s, c in prof["kinds"]),
    )
    if prof["fixture_breakdown"]:
        lines += ["", "Fixtures and what they called (edge-exact):", ""]
        for fx in prof["fixture_breakdown"][: min(top, _FIXTURES_SHOWN)]:
            lines.append(
                f"- {_code(fx['fixture'])}: {_s(fx['seconds'], 2)} s over {fx['calls']} call(s)"
            )
            lines.extend(
                f"  - {_code(callee)}: {_s(seconds, 2)} s, {calls} call(s)"
                for callee, seconds, calls in fx["callees"][:_CALLEES_SHOWN]
            )
            if _unattributed_is_material(fx["unattributed"], fx["seconds"]):
                lines.append(
                    f"  - **{_s(fx['unattributed'], 2)} s not attributed**: callees in a module executed more "
                    "than once lost their edges; profile this test alone, or fix the double import"
                )
    if prof["tests"]:
        lines += ["", "Test functions (edge-exact):", ""]
        lines += _table(
            ["test function", "seconds", "calls"],
            ([_code(n), _s(s, 2), c] for n, s, c in prof["tests"][:top]),
        )
    lines += ["", "Hot spots, own time:", ""]
    lines += _table(
        ["function", "own s", "cum s", "calls"],
        (
            [_code(n), _s(o, 2), _s(c, 2), f"{k:,}"]
            for n, o, c, k in prof["hot_own"][:top]
        ),
    )
    if prof["inconsistent"]:
        lines += [
            "",
            "Records that do not add up (module executed more than once; use the edge total):",
            "",
        ]
        lines += _table(
            ["function", "recorded cum s", "callee edges s", "calls"],
            (
                [
                    _code(i["function"]),
                    _s(i["recorded_cumulative"], 2),
                    _s(i["callee_edges"], 2),
                    i["calls"],
                ]
                for i in prof["inconsistent"]
            ),
        )
    for q in prof["queries"]:
        lines += [
            "",
            f"{q['direction']} of {_code(q['function'])} (own {_s(q['own'], 2)} s, cum {_s(q['cumulative'], 2)} s, {q['calls']} calls):",
            "",
        ]
        lines.extend(
            f"- {_code(n)}: {_s(s, 2)} s, {c} call(s)" for n, s, c in q["edges"]
        )
        if _unattributed_is_material(q.get("unattributed"), q["cumulative"]):
            lines.append(
                f"- **{_s(q['unattributed'], 2)} s not attributed** (lost edges of a module executed more than once)"
            )
    return lines


def render_report(summary, top=DEFAULT_TOP):
    """
    Render ``report.md``.

    Parameters
    ----------
    summary : dict
        From :func:`analyze`.
    top : int
        Rows per table.

    Returns
    -------
    str
        GitHub-flavoured Markdown.
    """
    lines = ["# Test performance report", "", *_scope_lines(summary)]
    lines += _section(
        "Sessions and machines", _machines(summary) if summary["sessions"] else []
    )
    lines += [
        "",
        "## Findings",
        "",
        *(_finding_lines(summary["findings"]) or ["None of the fixed rules fired."]),
    ]
    lines += _section("Units", _units_table(summary, top))
    lines += _section("Test files", _files_table(summary["files"][:top]))
    for title, key in _TEST_TABLES:
        lines += _section(title, _tests_table(summary[key][:top]))
    lines += _section(
        "Fixtures (setups of 1 ms or more)", _fixtures_table(summary["fixtures"][:top])
    )
    lines += _section(
        "Collection, per test file", _collection_table(summary["collection"][:top])
    )
    lines += _section(
        "Modules executed more than once in one process",
        _reexecuted_table(summary["reexecuted_modules"][:top]),
    )
    lines += _section("From JUnit", _junit_tables(summary["junit"], top))
    lines += _section(
        "cProfile files",
        [
            line
            for name, prof in summary["profiles"].items()
            for line in [*_profile_lines(name, prof, top), ""]
        ],
    )
    lines += _section(
        "Inputs not read", [f"- {_code(s)}" for s in summary["skipped_inputs"]]
    )
    lines += [
        "",
        f"_ci_test_perf.py {TOOL_VERSION}, from {len(summary['inputs'])} input file(s); thresholds in summary.json._",
        "",
    ]
    return "\n".join(lines)


#: The ranked test tables of report.md: title, key of the summary.
_TEST_TABLES = (
    ("Slowest tests", "slowest_tests"),
    ("Slowest setups", "slowest_setups"),
    ("Tests whose child processes use CPU", "child_process_tests"),
    ("Tests that raise the memory peak", "memory_tests"),
)


def _section(title, body):
    """Return a section's heading and body, or nothing when the body is empty."""
    return ["", f"## {title}", "", *body] if body else []


def _units_table(summary, top):
    total = summary["totals"]["total"]
    rows = []
    for u in summary["units"][:top]:
        cov = u.get("coverage") or {}
        rows.append(
            [
                _code(u["unit"]),
                u["tests"],
                _s(u["total"]),
                _pct(u["total"], total),
                _s(u["setup"]),
                _s(u["call"]),
                _s(u["teardown"]),
                _s(u["child_cpu"]),
                _s(u["gc_seconds"]),
                _mb(u["max_peak_rss_growth"]),
                u["events"].get("subprocess", 0),
                u["events"].get("module_exec", 0),
                _s(u.get("planner_estimate_seconds")),
                (
                    "-"
                    if cov.get("line_rate") is None
                    else f"{100 * cov['line_rate']:.1f}%"
                ),
            ]
        )
    if not rows:
        return []
    return _table(
        [
            "unit",
            "tests",
            "total s",
            "share",
            "setup s",
            "call s",
            "teardown s",
            "children s",
            "gc s",
            "max peak growth MB",
            "subprocesses",
            "module runs",
            "planner estimate s",
            "line coverage",
        ],
        rows,
    )


def _files_table(files):
    if not files:
        return []
    return _table(
        ["file", "tests", "total s", "setup s", "call s", "children s"],
        (
            [
                _code(f["path"]),
                f["tests"],
                _s(f["total"]),
                _s(f["setup"]),
                _s(f["call"]),
                _s(f["child_cpu"]),
            ]
            for f in files
        ),
    )


def _tests_table(tests):
    if not tests:
        return []
    return _table(
        [
            "test",
            "outcome",
            "total s",
            "setup s",
            "call s",
            "teardown s",
            "cpu s",
            "children s",
            "peak growth MB",
            "programs",
        ],
        (
            [
                _code(r["nodeid"]),
                _cell(r["outcome"]),
                _s(r["total"], 2),
                _s(r["setup"], 2),
                _s(r["call"], 2),
                _s(r["teardown"], 2),
                _s(r["cpu"], 2),
                _s(r["child_cpu"], 2),
                _mb(r["peak_rss_growth"]),
                _cell(
                    ", ".join(f"{k}x{v}" for k, v in sorted(r["programs"].items()))
                    or "-"
                ),
            ]
            for r in tests
        ),
    )


def _fixtures_table(fixtures):
    if not fixtures:
        return []
    return _table(
        ["fixture", "where", "scope", "setups", "total s", "mean s", "max s"],
        (
            [
                _code(f["name"]),
                _code(f["location"]),
                _cell(f["scope"]),
                f["setups"],
                _s(f["seconds"], 2),
                _s(f["mean"], 3),
                _s(f["max"], 2),
            ]
            for f in fixtures
        ),
    )


def _collection_table(collections):
    if not collections:
        return []
    return _table(
        ["file", "seconds", "module runs", "items"],
        (
            [_code(c["path"]), _s(c["seconds"], 2), c["module_exec"], c["items"]]
            for c in collections
        ),
    )


def _some_names(names, count):
    """Return the first names in code style, and how many of ``count`` are not shown."""
    text = ", ".join(_code(n) for n in names[:_NAMES_SHOWN])
    if count > _NAMES_SHOWN:
        text += f" (+{count - _NAMES_SHOWN})"
    return text or "-"


def _reexecuted_table(entries):
    if not entries:
        return []
    return _table(
        [
            "module",
            "executions",
            "sessions",
            "test files whose import ran it again",
            "tests during which it ran again",
        ],
        (
            [
                _code(e["module"]),
                e["executions"],
                e["sessions"],
                _some_names(e["collected_by"], e["collected_by_count"]),
                _some_names(e["tests"], e["tests_count"]),
            ]
            for e in entries
        ),
    )


def _junit_tables(junit, top):
    if not junit["tests"]:
        return []
    lines = _table(
        ["unit", "tests", "seconds", "outcomes"],
        (
            [
                _code(u["unit"]),
                u["tests"],
                _s(u["seconds"]),
                ", ".join(f"{k} {v}" for k, v in u["outcomes"].items()),
            ]
            for u in junit["units"][:top]
        ),
    )
    lines.append("")
    lines += _table(
        ["slowest test case", "unit", "seconds", "outcome"],
        ([_code(n), _code(u), _s(s, 2), o] for n, u, s, o in junit["slowest"][:top]),
    )
    return lines


def render_step_summary(summary, top=10):
    """Return a short Markdown summary for ``$GITHUB_STEP_SUMMARY``."""
    lines = ["## Test performance", ""]
    lines += _scope_lines(summary)
    if summary["units"]:
        lines += [""]
        lines += _table(
            ["unit", "tests", "total s", "setup s", "children s"],
            (
                [
                    _code(u["unit"]),
                    u["tests"],
                    _s(u["total"]),
                    _s(u["setup"]),
                    _s(u["child_cpu"]),
                ]
                for u in summary["units"][:top]
            ),
        )
    if summary["slowest_tests"]:
        lines += [""]
        lines += _table(
            ["slowest test", "total s", "setup s"],
            (
                [_code(r["nodeid"]), _s(r["total"], 2), _s(r["setup"], 2)]
                for r in summary["slowest_tests"][:top]
            ),
        )
    if summary["findings"]:
        counts = {}
        for f in summary["findings"]:
            counts[f["rule"]] = counts.get(f["rule"], 0) + 1
        lines += [
            "",
            "Findings: "
            + ", ".join(f"{k} x{v}" for k, v in counts.items())
            + ". Details: the `test-perf` artifact (report.md, ANALYZE_ME.md).",
        ]
    return "\n".join(lines) + "\n"


_FIELDS = """\
| field | meaning |
|---|---|
| `setup`, `call`, `teardown` | wall seconds of each pytest phase, from the start to the end of its hook |
| `cpu` | CPU seconds of the test process in those phases (`time.process_time`, all threads) |
| `child_cpu` | CPU seconds of child processes that ended during the phase (`RUSAGE_CHILDREN`) |
| `gc` | garbage collections per generation (0, 1, 2) and their seconds; 2 is a full collection |
| `rss` | resident memory at the start and end of the test (Linux `/proc/self/statm`) |
| `peak_rss_growth` | how much the process's peak resident memory rose during the test (`ru_maxrss`) |
| `tracemalloc` | Python allocations at the start, the end, and the peak (only with `SKPLT_TEST_PERF_TRACEMALLOC=1`) |
| `events` | counts of audit events: `open`, `open_write`, `mkdir`, `remove`, `subprocess`, `module_exec`, `socket_connect`, `dns_lookup`, ... |
| `programs` | the programs the test's subprocesses ran, by name |
| `reexecuted` | module files whose body ran during the test although it had already run in this process |
| `fixtures` | each fixture set up for the test that took 1 ms or more: name, scope, seconds, where it is defined |
| unit | the planner's unit (`.github/scripts/ci_test_plan.py`), the same grouping as `test_durations.json` |
"""


def render_brief(summary, top=10):
    """
    Render ``ANALYZE_ME.md``: the self-contained brief for a person or an AI.

    Parameters
    ----------
    summary : dict
        From :func:`analyze`.
    top : int
        Rows per table.

    Returns
    -------
    str
        Markdown that needs nothing else to be understood.
    """
    total = summary["totals"]["total"]
    lines = [*_BRIEF_HEAD, *_scope_lines(summary)]
    lines += _section(
        "Sessions and machines", _machines(summary) if summary["sessions"] else []
    )
    lines += [
        "",
        f"## Findings ({len(summary['findings'])})",
        "",
        "Each finding is a fixed rule; its threshold is in `summary.json` -> `thresholds`.",
        "",
        *(
            _finding_lines(summary["findings"], per_rule=8)
            or ["None of the fixed rules fired."]
        ),
    ]
    units = [
        [
            _code(u["unit"]),
            u["tests"],
            _s(u["total"]),
            _pct(u["total"], total),
            _s(u["setup"]),
            _s(u["call"]),
            _s(u["child_cpu"]),
        ]
        for u in summary["units"][:top]
    ]
    lines += _section(
        "Units (largest first)",
        (
            _table(
                [
                    "unit",
                    "tests",
                    "total s",
                    "share",
                    "setup s",
                    "call s",
                    "children s",
                ],
                units,
            )
            if units
            else []
        ),
    )
    slowest = [
        [
            _code(r["nodeid"]),
            _s(r["total"], 2),
            _s(r["setup"], 2),
            _s(r["call"], 2),
            _s(r["child_cpu"], 2),
        ]
        for r in summary["slowest_tests"][:top]
    ]
    lines += _section(
        "Slowest tests",
        (
            _table(["test", "total s", "setup s", "call s", "children s"], slowest)
            if slowest
            else []
        ),
    )
    lines += _section("Costliest fixtures", _fixtures_table(summary["fixtures"][:top]))
    for name, prof in summary["profiles"].items():
        lines += _section(
            f"Profile {_code(name)}: fixtures and what they called",
            _brief_fixture_lines(prof),
        )
    lines += [
        *_BRIEF_TAIL,
        _FIELDS,
        "## Files in this archive",
        "",
        (
            "- `ANALYZE_ME.md`: this brief. `report.md`: every table. `summary.json`: all of it as data "
            f"(schema `{SUMMARY_SCHEMA}` version {SUMMARY_VERSION}; `all_tests` has one row per test)."
        ),
        "- `MANIFEST.json`: every input with its size and sha256. `raw/`: the inputs. `tool/ci_test_perf.py`: the analyzer.",
        "",
    ]
    return "\n".join(lines)


def _brief_fixture_lines(prof):
    lines = []
    for fx in prof["fixture_breakdown"][:_BRIEF_FIXTURES_SHOWN]:
        lines.append(f"- {_code(fx['fixture'])}: {_s(fx['seconds'], 2)} s")
        lines.extend(
            f"  - {_code(callee)}: {_s(seconds, 2)} s, {calls} call(s)"
            for callee, seconds, calls in fx["callees"][:_BRIEF_CALLEES_SHOWN]
        )
        if _unattributed_is_material(fx["unattributed"], fx["seconds"]):
            lines.append(
                f"  - {_s(fx['unattributed'], 2)} s not attributed (module executed more than once)"
            )
    return lines


_BRIEF_HEAD = (
    "# ANALYZE ME: test performance evidence",
    "",
    (
        "This archive describes where a pytest run of scikit-plots spent its time and memory. It is "
        "complete on its own: no chat history, repository or tool is needed to read it."
    ),
    "",
    "## How to analyse it",
    "",
    "If you are an AI assistant, follow these rules:",
    "",
    (
        "1. State only what the numbers in this archive show. Name the field and the value you rely on "
        "(for example `setup = 15.6 s` of a test). If the data cannot answer a question, say so and "
        "name the measurement that would (see *What to run next*)."
    ),
    (
        "2. Do not add up times that overlap: `cpu` and `child_cpu` happen inside `setup`/`call`/`teardown`; "
        "garbage-collection seconds happen inside the phase that triggered them."
    ),
    (
        "3. Machines differ: compare runs only together with their *Sessions and machines* row "
        "(`loop us` is a fixed CPU workload, `stat us` / `mkdir us` / `fsync us` the filesystem)."
    ),
    (
        "4. Treat a cProfile function listed under *records that do not add up* as unreliable "
        "(its module was executed more than once; use its callee-edge total)."
    ),
    (
        "5. Rank proposals by the seconds they can save, measured here, and say how to verify each: "
        "record again with `SKPLT_TEST_PERF_DIR` and run `compare`."
    ),
    (
        "6. Lines counted by the `invalid-lines` finding are in no total, and an incomplete session "
        "ends at its last recorded test: say so when either applies."
    ),
    "",
    "Questions worth answering, in order:",
    "",
    "1. Which units and test files take the most time, and in which phase (setup, call, teardown)?",
    "2. Which findings below explain most of that time?",
    "3. For setup-dominated tests: which fixtures, and what do the profiles say they call?",
    "4. What would remove the cost at its root (without weakening what the tests check)?",
    "",
    "## Scope",
    "",
)

_BRIEF_TAIL = (
    "",
    "## What to run next",
    "",
    "From a checkout of scikit-plots (Python and pytest installed):",
    "",
    "```bash",
    "# Record a selection; one line per test is written as it finishes.",
    "SKPLT_TEST_PERF_DIR=build/test-perf python -m pytest <paths>",
    "# Profile chosen tests from setup to teardown, without editing them.",
    "SKPLT_TEST_PERF_DIR=build/test-perf SKPLT_TEST_PERF_PROFILE='<regex of node ids>' python -m pytest <paths>",
    "# Python allocation peaks per test (slower).",
    "SKPLT_TEST_PERF_DIR=build/test-perf SKPLT_TEST_PERF_TRACEMALLOC=1 python -m pytest <paths>",
    "# Analyse, compare.",
    "python .github/scripts/ci_test_perf.py report build/test-perf --out build/test-perf/report --bundle",
    "python .github/scripts/ci_test_perf.py compare --base <before> --head <after> --out build/test-perf/compare",
    "```",
    "",
    (
        "Without a checkout, this archive carries the analyzer: "
        "`python tool/ci_test_perf.py report raw --out again` analyses `raw/` again (the same numbers; input names then start with `raw/`)."
    ),
    "",
    "## Fields",
    "",
)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def _zip_add(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    # ZipInfo takes the creating system from the running platform; fixed,
    # the bundle is the same bytes on every platform.
    info.create_system = 3
    archive.writestr(info, data)


def _raw_name(name):
    clean = re.sub(r"[^A-Za-z0-9._/-]+", "_", name.replace("!", "/")).lstrip("/.")
    parts = [p for p in PurePosixPath(clean).parts if p not in ("..", ".")]
    return "raw/" + "/".join(parts)


def _unique_entry(name, used, folders):
    """
    Return ``name``, changed where needed so that the archive still unpacks.

    A component that is already a file of the archive (``raw/x`` beside
    ``raw/x/a``) becomes ``x.2``; a file where a folder is needed, or a name
    taken, gets the same suffix on its last component.
    """
    parts = list(PurePosixPath(name).parts)
    for index in range(len(parts)):
        prefix = "/".join(parts[: index + 1])
        last = index == len(parts) - 1
        taken = prefix in used or (last and prefix in folders)
        n = 1
        while taken:
            n += 1
            candidate = "/".join([*parts[:index], f"{parts[index]}.{n}"])
            taken = candidate in used or (last and candidate in folders)
            if not taken:
                parts[index] = f"{parts[index]}.{n}"
    return "/".join(parts)


def write_bundle(path, texts, inputs, include_raw=True):
    """
    Write the reproducible archive.

    Parameters
    ----------
    path : pathlib.Path
        The zip to write.
    texts : dict of str to str
        Generated files by name (``ANALYZE_ME.md``, ``report.md``, ``summary.json``).
    inputs : list of Input
        The evidence, stored under ``raw/`` when ``include_raw``.
    include_raw : bool
        Store the inputs.
    """
    entries = {
        name: text.encode("utf-8", "backslashreplace") for name, text in texts.items()
    }
    manifest = []
    if include_raw:
        used, folders = set(), set()
        for item in inputs:
            name = _unique_entry(_raw_name(item.name), used, folders)
            used.add(name)
            folders.update(str(p) for p in PurePosixPath(name).parents)
            entries[name] = item.data
            manifest.append(
                {
                    "source": item.name,
                    "stored_as": name,
                    "bytes": len(item.data),
                    "sha256": _sha256(item.data),
                }
            )
    entries["MANIFEST.json"] = (
        json.dumps(
            {"tool_version": TOOL_VERSION, "inputs": manifest}, indent=1, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")
    entries["tool/ci_test_perf.py"] = Path(__file__).read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(entries):
            _zip_add(archive, name, entries[name])


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _side(paths, repo_root):
    """
    Return the summary of one side of a comparison.

    Raw evidence is analysed when there is any (a bundle carries both its
    summary and its inputs: the inputs decide); otherwise exactly one
    ``summary.json`` must be given.
    """
    dataset = build_dataset(paths, repo_root=repo_root)
    if dataset["telemetry"] or dataset["junit"]:
        return analyze(dataset)
    if len(dataset["summaries"]) == 1:
        name, summary = dataset["summaries"][0]
        problem = _summary_problem(summary)
        if problem:
            raise PerfError(f"{name}: {problem}")
        return summary
    if dataset["summaries"]:
        raise PerfError(
            "one side holds more than one summary.json and no raw evidence; give one"
        )
    raise PerfError(f"no test evidence in {', '.join(map(str, paths))}")


#: Columns of ``all_tests`` that :func:`compare` reads.
_COMPARE_COLUMNS = ("nodeid", "unit", "outcome", "setup", "call", "teardown")


def _junit_row_ok(row):
    return (
        isinstance(row, list)
        and len(row) == len(("nodeid", "unit", "seconds", "outcome"))
        and _text(row[0])
        and _text(row[1])
        and _number(row[2])
        and _text(row[3])
    )


def _summary_problem(summary):
    """Return why a loaded ``summary.json`` cannot be compared, or None."""
    if summary.get("version") != SUMMARY_VERSION:
        return (
            f"summary version {summary.get('version')!r}; this analyzer reads "
            f"{SUMMARY_VERSION}"
        )
    columns, rows = summary.get("all_tests_columns"), summary.get("all_tests")
    if not (_list_of(_text)(columns) and set(_COMPARE_COLUMNS) <= set(columns)):
        return f"all_tests_columns must name {', '.join(_COMPARE_COLUMNS)}"
    if not isinstance(rows, list):
        return "all_tests must be a list"
    at = {name: index for index, name in enumerate(columns)}
    for row in rows:
        if not (
            isinstance(row, list)
            and len(row) == len(columns)
            and _text(row[at["nodeid"]])
            and _text(row[at["unit"]])
            and _text(row[at["outcome"]])
            and all(_number(row[at[c]]) for c in ("setup", "call", "teardown"))
        ):
            return "a row of all_tests does not match all_tests_columns"
    junit = summary.get("junit")
    if not (isinstance(junit, dict) and _list_of(_junit_row_ok)(junit.get("all"))):
        return "junit.all must be a list of [nodeid, unit, seconds, outcome]"
    return None


def _per_test(summary):
    if summary["all_tests"]:
        cols = summary["all_tests_columns"]
        out = {}
        for row in summary["all_tests"]:
            d = dict(zip(cols, row))
            d["total"] = (d["setup"] or 0) + (d["call"] or 0) + (d["teardown"] or 0)
            out[d["nodeid"]] = d
        return out, "recorder"
    out = {}
    for nodeid, unit, seconds, outcome in summary["junit"]["all"]:
        out[nodeid] = {
            "nodeid": nodeid,
            "unit": unit,
            "outcome": outcome,
            "total": seconds,
        }
    return out, "junit"


def compare(base, head, *, factor=1.2, min_seconds=0.5, top=DEFAULT_TOP):
    """
    Compare two summaries, test by test and unit by unit.

    Parameters
    ----------
    base, head : dict
        Summaries (from :func:`analyze`, or loaded ``summary.json``).
    factor : float
        A time changed when ``head / base`` is at least ``factor`` or at
        most ``1 / factor`` (asv's default is 1.1; test times are noisier).
    min_seconds : float
        ... and when it moved by at least this many seconds.
    top : int
        Rows per table.

    Returns
    -------
    dict
        Counts, changed tests and units (``+`` slower, ``-`` faster), outcome
        changes (``!``), tests only on one side, and the totals.

    Raises
    ------
    PerfError
        If ``factor`` is not above 1, ``min_seconds`` is negative, or
        ``top`` is not a positive integer.
    """
    if not isinstance(top, int) or top < 1:
        raise PerfError(f"--top must be a positive integer, got {top!r}")
    if not (
        isinstance(factor, (int, float)) and math.isfinite(factor) and factor > 1.0
    ):
        raise PerfError(f"--factor must be a number above 1, got {factor!r}")
    if not (
        isinstance(min_seconds, (int, float))
        and math.isfinite(min_seconds)
        and min_seconds >= 0
    ):
        raise PerfError(
            f"--min-seconds must be a number of seconds, got {min_seconds!r}"
        )
    b, b_source = _per_test(base)
    h, h_source = _per_test(head)
    if b_source != h_source:
        raise PerfError(
            f"the base has {b_source} times and the head {h_source} times; their test "
            "names differ (pkg.mod::t against pkg/mod.py::t), so no test would be compared. "
            "Give both sides the same kind of evidence."
        )
    if b and h and not set(b) & set(h):
        raise PerfError("no test is in both sides: they are runs of different tests")
    rows, outcome_changes = [], []
    for nodeid in sorted(set(b) & set(h)):
        old, new = b[nodeid]["total"], h[nodeid]["total"]
        mark = _mark(old, new, factor, min_seconds)
        if mark != "~":
            rows.append(
                [mark, nodeid, _r(old), _r(new), _r(new / old, 3) if old > 0 else None]
            )
        if b[nodeid]["outcome"] != h[nodeid]["outcome"]:
            outcome_changes.append(
                ["!", nodeid, b[nodeid]["outcome"], h[nodeid]["outcome"]]
            )
    rows.sort(key=lambda r: (-abs((r[3] or 0) - (r[2] or 0)), r[1]))
    units_b, units_h = {}, {}
    for side, tests in ((units_b, b), (units_h, h)):
        for d in tests.values():
            side[d["unit"]] = side.get(d["unit"], 0.0) + d["total"]
    unit_rows = []
    for unit in sorted(set(units_b) | set(units_h)):
        old, new = units_b.get(unit), units_h.get(unit)
        mark = (
            "?" if old is None or new is None else _mark(old, new, factor, min_seconds)
        )
        unit_rows.append(
            [
                mark,
                unit,
                _r(old),
                _r(new),
                _r(new / old, 3) if old and new is not None else None,
            ]
        )
    unit_rows.sort(key=lambda r: (-abs((r[3] or 0) - (r[2] or 0)), r[1]))
    only_base = sorted(set(b) - set(h))
    only_head = sorted(set(h) - set(b))
    return {
        "schema": SUMMARY_SCHEMA + "-compare",
        "version": SUMMARY_VERSION,
        "tool_version": TOOL_VERSION,
        "factor": factor,
        "min_seconds": min_seconds,
        "sources": [b_source, h_source],
        "tests": {"base": len(b), "head": len(h), "both": len(set(b) & set(h))},
        "total": {
            "base": _r(sum(d["total"] for d in b.values())),
            "head": _r(sum(d["total"] for d in h.values())),
        },
        "slower": sum(1 for r in rows if r[0] == "+"),
        "faster": sum(1 for r in rows if r[0] == "-"),
        "changed_tests": rows[: top * 4],
        "units": unit_rows,
        "outcome_changes": outcome_changes,
        "only_in_base": only_base[: top * 4],
        "only_in_head": only_head[: top * 4],
        "only_counts": [len(only_base), len(only_head)],
    }


def _mark(old, new, factor, min_seconds):
    if abs(new - old) < min_seconds:
        return "~"
    if old <= 0:
        return "+" if new > 0 else "~"
    ratio = new / old
    if ratio >= factor:
        return "+"
    if ratio <= 1.0 / factor:
        return "-"
    return "~"


def render_compare(result, top=DEFAULT_TOP):
    """Render a comparison as Markdown (``+`` slower, ``-`` faster, ``!`` outcome changed)."""
    tests, total, only = result["tests"], result["total"], result["only_counts"]
    lines = [
        "# Test performance: comparison",
        "",
        (
            f"- Changed when the ratio is at least {result['factor']} (or at most "
            f"1/{result['factor']}) and the time moved by at least {result['min_seconds']} s."
        ),
        (
            f"- Tests: base {tests['base']:,}, head {tests['head']:,}, in both {tests['both']:,}; "
            f"total {_s(total['base'])} s -> {_s(total['head'])} s."
        ),
        (
            f"- **{result['slower']} slower, {result['faster']} faster**, "
            f"{len(result['outcome_changes'])} outcome change(s); "
            f"only in base {only[0]}, only in head {only[1]}."
        ),
        "",
        "## Units",
        "",
    ]
    lines += _table(
        ["", "unit", "base s", "head s", "ratio"],
        (
            [m, _code(u), _s(o), _s(n), "-" if r is None else f"{r:.3f}"]
            for m, u, o, n, r in result["units"]
        ),
    )
    if result["changed_tests"]:
        lines += ["", "## Tests that changed", ""]
        lines += _table(
            ["", "test", "base s", "head s", "ratio"],
            (
                [m, _code(t), _s(o, 2), _s(n, 2), "-" if r is None else f"{r:.3f}"]
                for m, t, o, n, r in result["changed_tests"][: top * 2]
            ),
        )
    if result["outcome_changes"]:
        lines += ["", "## Outcomes that changed", ""]
        lines += _table(
            ["", "test", "base", "head"],
            (
                [m, _code(t), o, n]
                for m, t, o, n in result["outcome_changes"][: top * 2]
            ),
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _write_text(path, text):
    """Write UTF-8 text with Unix line ends (``Path.write_text`` has no ``newline`` before 3.10)."""
    with open(
        path, "w", encoding="utf-8", newline="\n", errors="backslashreplace"
    ) as handle:
        handle.write(text)


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def _parser():
    parser = argparse.ArgumentParser(
        prog="ci_test_perf.py", description=__doc__.split("\n\n")[0].strip()
    )
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser(
        "report",
        help="analyse evidence and write report.md, summary.json, ANALYZE_ME.md",
    )
    report.add_argument("inputs", nargs="+", help="files, directories or zip archives")
    report.add_argument(
        "--out",
        default="test-perf-report",
        help="output directory (default: %(default)s)",
    )
    report.add_argument(
        "--bundle",
        nargs="?",
        const="",
        default=None,
        metavar="ZIP",
        help="also write a reproducible zip (default name: <out>/test-perf-bundle.zip)",
    )
    report.add_argument(
        "--no-raw", action="store_true", help="leave the inputs out of the bundle"
    )
    report.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="rows per table (default: %(default)s)",
    )
    report.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ID",
        help="keep only this recorded run (id prefix, or 'latest'); repeatable",
    )
    report.add_argument(
        "--repo-root",
        default=".",
        help="checkout whose .github/ci/test_plan.json gives the units when no recorder line does",
    )
    report.add_argument(
        "--step-summary",
        default=None,
        metavar="FILE",
        help="append a short summary to FILE (for $GITHUB_STEP_SUMMARY)",
    )

    cmp_ = sub.add_parser(
        "compare", help="compare two runs test by test and unit by unit"
    )
    cmp_.add_argument(
        "--base",
        nargs="+",
        required=True,
        help="evidence or a summary.json of the earlier run",
    )
    cmp_.add_argument(
        "--head",
        nargs="+",
        required=True,
        help="evidence or a summary.json of the later run",
    )
    cmp_.add_argument(
        "--factor",
        type=float,
        default=1.2,
        help="ratio that counts as a change (default: %(default)s)",
    )
    cmp_.add_argument(
        "--min-seconds",
        type=float,
        default=0.5,
        help="smallest change in seconds that counts (default: %(default)s)",
    )
    cmp_.add_argument(
        "--out",
        default=None,
        help="write compare.md and compare.json here; else print the Markdown",
    )
    cmp_.add_argument(
        "--fail-on-slower",
        action="store_true",
        help="exit 1 when any test or unit got slower",
    )
    cmp_.add_argument("--repo-root", default=".")
    cmp_.add_argument("--top", type=int, default=DEFAULT_TOP)

    prof = sub.add_parser("profile", help="callers and callees in cProfile files")
    prof.add_argument(
        "profiles", nargs="+", help=".prof files, or directories and zips holding them"
    )
    prof.add_argument("--callers", action="append", default=[], metavar="REGEX")
    prof.add_argument("--callees", action="append", default=[], metavar="REGEX")
    prof.add_argument("--package", default=DEFAULT_PACKAGE)
    prof.add_argument("--top", type=int, default=DEFAULT_TOP)
    return parser


def _report(args):
    out = Path(args.out)
    target = (
        None
        if args.bundle is None
        else (Path(args.bundle) if args.bundle else out / "test-perf-bundle.zip")
    )
    # This run's outputs are never its inputs, also when they lie inside the
    # evidence directory: a second run of the same command reads the same.
    dataset = build_dataset(
        args.inputs,
        runs=args.run or None,
        repo_root=args.repo_root,
        exclude=[p for p in (out, target) if p is not None],
    )
    summary = analyze(dataset, top=args.top)
    out.mkdir(parents=True, exist_ok=True)
    texts = {
        "ANALYZE_ME.md": render_brief(summary, top=min(args.top, 15)),
        "report.md": render_report(summary, top=args.top),
        "summary.json": _dump(summary),
    }
    for name, text in texts.items():
        _write_text(out / name, text)
    if target is not None:
        write_bundle(target, texts, dataset["inputs"], include_raw=not args.no_raw)
    if args.step_summary:
        with open(
            args.step_summary,
            "a",
            encoding="utf-8",
            newline="\n",
            errors="backslashreplace",
        ) as handle:
            handle.write(render_step_summary(summary))
    t = summary["totals"]
    sys.stdout.write(
        f"ci_test_perf: {t['tests']} tests, {len(summary['sessions'])} session(s), "
        f"{len(summary['findings'])} finding(s); wrote {out}\n"
    )
    return 0


def _compare(args):
    base = _side(args.base, args.repo_root)
    head = _side(args.head, args.repo_root)
    result = compare(
        base, head, factor=args.factor, min_seconds=args.min_seconds, top=args.top
    )
    text = render_compare(result, top=args.top)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write_text(out / "compare.md", text)
        _write_text(out / "compare.json", _dump(result))
    else:
        sys.stdout.write(text)
    slower = result["slower"] or any(u[0] == "+" for u in result["units"])
    return 1 if args.fail_on_slower and slower else 0


def _profile(args):
    dataset = build_dataset(args.profiles, check_runs=False)
    if not dataset["profiles"]:
        raise PerfError("no cProfile file among the inputs")
    lines = []
    for name, table in sorted(dataset["profiles"].items()):
        prof = analyze_profile(
            table, args.package, args.top, callers=args.callers, callees=args.callees
        )
        lines += [*_profile_lines(name, prof, args.top), ""]
    sys.stdout.write("\n".join(lines))
    return 0


def main(argv=None):
    """
    Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``compare --fail-on-slower`` found a
        slower test or unit; ``2`` when an input or an option is unusable.
    """
    args = _parser().parse_args(argv)
    try:
        if getattr(args, "top", DEFAULT_TOP) < 1:
            raise PerfError(f"--top must be a positive integer, got {args.top}")
        if args.command == "report":
            return _report(args)
        if args.command == "compare":
            return _compare(args)
        return _profile(args)
    except (PerfError, re.error, OSError) as exc:
        sys.stderr.write(f"ci_test_perf: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
