# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Run the package benchmarks of ``benchmarks/`` without asv, and compare runs.

Notes
-----
**User notes.** The benchmarks are written for asv, as SciPy's and
scikit-learn's are; this runner reads the same files and needs nothing but
Python and the scikit-plots installed in the environment it runs::

    # What there is.
    python .github/scripts/ci_bench.py list

    # Run all of them, or those whose name matches (repeatable).
    python .github/scripts/ci_bench.py run --out build/bench/head.json
    python .github/scripts/ci_bench.py run --bench "Plots" --quick --out quick.json

    # Another environment, e.g. a virtualenv with the main branch installed.
    python .github/scripts/ci_bench.py run --python base-venv/bin/python --out base.json

    # Before and after, like ``asv compare``.
    python .github/scripts/ci_bench.py compare base.json head.json --out build/bench/compare

``compare`` marks a timing slower (``+``) or faster (``-``) only when the
ratio of the medians reaches ``--factor`` *and* every round of one side
(a worker process; ``--processes``, 4 by default) has a slower median than
every round of the other. A ratio that reaches the factor without that is
``x``, inconclusive: run again, or with more ``--processes``. Memory peaks
and ``track_`` values, which are not noisy, change by the ratio.

**Developer notes.**

* *Standard library only*: the CI job runs it before anything else is
  installed, and with ``--python`` it drives any interpreter.
* *The asv subset* (also described in ``benchmarks/benchmarks/__init__.py``):
  functions and methods named ``time_*``, ``peakmem_*``, ``track_*`` and
  ``timeraw_*``; ``params``, ``param_names``, ``timeout``; ``setup`` and
  ``teardown``, and ``NotImplementedError`` from ``setup`` to skip.
* *Isolation*: every benchmark runs in its own process, a ``peakmem_``
  combination of parameters too (the peak of a process only rises), and
  each ``timeraw_`` sample in a fresh interpreter. Workers start in an empty
  temporary directory, so ``import scikitplot`` finds the installed
  package, never a source tree.
* *Timing*: after one call that is not counted, the number of calls per
  sample is chosen so that a sample lasts at least ``--sample-time``; each
  of ``--repeat`` samples is divided by that number. The result keeps every
  sample, their median, minimum, maximum and quartiles.
* *Results are untrusted input to* ``compare``: size-limited and checked
  field by field before use.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import inspect
import itertools
import json
import math
import numbers
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

__all__ = ["PREFIXES", "BenchError", "compare", "discover", "main", "render_compare"]

#: Version of this runner; written into every result.
TOOL_VERSION = "1"

#: The format of a result file.
RESULT_SCHEMA = "skplt-bench"
RESULT_VERSION = 1

#: Benchmark kinds by name prefix, with the unit of their value.
PREFIXES = {
    "time_": "seconds",
    "timeraw_": "seconds",
    "peakmem_": "bytes",
    "track_": "unit",
}

#: Default suite, relative to the repository root.
DEFAULT_DIR = "benchmarks/benchmarks"

#: Seconds a benchmark may run (each parameter combination), unless it says.
DEFAULT_TIMEOUT = 60.0

#: Timing defaults: samples per process, the shortest sample, worker
#: processes per benchmark (their samples are pooled), and ``--quick``.
DEFAULT_REPEAT = 5
DEFAULT_PROCESSES = 4
DEFAULT_SAMPLE_TIME = 0.01
QUICK_REPEAT = 1

#: The most calls one sample may take, however fast the benchmark.
MAX_NUMBER = 1_000_000

#: Default thresholds of ``compare`` (asv's default factor is 1.1).
DEFAULT_FACTOR = 1.1

#: The largest result file ``compare`` reads.
MAX_RESULT_BYTES = 64 << 20

#: Packages whose versions a result records.
PACKAGES_RECORDED = (
    "scikit-plots",
    "numpy",
    "scipy",
    "matplotlib",
    "scikit-learn",
    "pandas",
)


class BenchError(Exception):
    """A suite, a result file or an option that cannot be used; the message says why."""


# ---------------------------------------------------------------------------
# Discovery and execution (inside a worker process)
# ---------------------------------------------------------------------------


def _kind(name):
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return prefix[:-1]
    return None


def _combinations(params):
    """Return the parameter combinations of ``params`` (asv's rule), as tuples."""
    if params is None or (isinstance(params, (list, tuple)) and not params):
        return [()]
    if not isinstance(params, (list, tuple)):
        raise BenchError(f"params must be a list, got {type(params).__name__}")
    if not isinstance(params[0], (list, tuple)):
        params = [params]
    return list(itertools.product(*params))


def _import_suite(directory):
    """Import every module of the suite package; return (package name, modules)."""
    directory = Path(directory).resolve()
    if not (directory / "__init__.py").is_file():
        raise BenchError(f"{directory}: not a package (no __init__.py)")
    sys.path.insert(0, str(directory.parent))
    package = directory.name
    modules = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        modules.append(importlib.import_module(f"{package}.{path.stem}"))
    return package, modules


def _benchmarks_of(module, package):
    """Yield ``(name, owner class or None, function name)`` defined in ``module``."""
    short = module.__name__[len(package) + 1 :]
    for name, obj in sorted(vars(module).items()):
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj) and _kind(name):
            yield f"{short}.{name}", None, name
        elif inspect.isclass(obj):
            # dir(): inherited benchmark methods count, as in asv.
            for attr in sorted(dir(obj)):
                if _kind(attr) and callable(getattr(obj, attr, None)):
                    yield f"{short}.{name}.{attr}", obj, attr


def _sources(module, owner, func_name):
    """Where a benchmark's attributes are looked up, most specific first (asv's order)."""
    function = getattr(owner if owner is not None else module, func_name)
    return [s for s in (function, owner, module) if s is not None]


def _first_attr(sources, name, default):
    for source in sources:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _describe(name, module, owner, func_name):
    sources = _sources(module, owner, func_name)
    combos = _combinations(_first_attr(sources, "params", None))
    if not combos:
        raise BenchError(f"{name}: a parameter in params has no values")
    names = list(_first_attr(sources, "param_names", None) or [])
    if combos != [()] and len(names) != len(combos[0]):
        names = [f"param{i + 1}" for i in range(len(combos[0]))]
    timeout = _first_attr(sources, "timeout", DEFAULT_TIMEOUT)
    kind = _kind(func_name)
    description = {
        "kind": kind,
        "unit": _first_attr(sources, "unit", PREFIXES[kind + "_"]),
        "param_names": names,
        "combinations": [[repr(v) for v in c] for c in combos],
        "timeout": float(timeout),
    }
    if any(hasattr(source, "setup_cache") for source in sources[1:]):
        description["unsupported"] = (
            "setup_cache is not supported here; run it with asv"
        )
    return description


def discover(directory):
    """
    Import the suite and describe every benchmark in it.

    Parameters
    ----------
    directory : str or pathlib.Path
        The suite package (``benchmarks/benchmarks``).

    Returns
    -------
    dict
        Benchmark name to ``{"kind", "unit", "param_names", "combinations",
        "timeout"}``; parameter values are their ``repr``.

    Raises
    ------
    BenchError
        If the directory is not a package or a ``params`` is not a list.
    """
    package, modules = _import_suite(directory)
    found = {}
    for module in modules:
        for name, owner, func_name in _benchmarks_of(module, package):
            found[name] = _describe(name, module, owner, func_name)
    return dict(sorted(found.items()))


def _locate(directory, name):
    package, modules = _import_suite(directory)
    for module in modules:
        for found, owner, func_name in _benchmarks_of(module, package):
            if found == name:
                return module, owner, func_name
    raise BenchError(f"no benchmark named {name}")


class _Instance:
    """
    One parameter combination of a benchmark, set up and torn down.

    Notes
    -----
    **Developer notes.** As in asv, every ``setup`` found runs, module first,
    then the class, then the function's own, and the ``teardown`` functions
    in the reverse order.
    """

    def __init__(self, module, owner, func_name, values):
        self.values = values
        self.obj = owner() if owner is not None else None
        self.func = getattr(self.obj if owner is not None else module, func_name)
        levels = (
            [module, self.obj, self.func] if owner is not None else [module, self.func]
        )
        found = {"setup": [], "teardown": []}
        for hook, order in (("setup", levels), ("teardown", list(reversed(levels)))):
            for level in order:
                function = getattr(level, hook, None)
                # A function benchmark's module setup is not also its own.
                if callable(function) and function not in found[hook]:
                    found[hook].append(function)
        self.setups, self.teardowns = found["setup"], found["teardown"]

    def setup(self):
        for setup in self.setups:
            setup(*self.values)

    def teardown(self):
        for teardown in self.teardowns:
            teardown(*self.values)

    def __call__(self):
        return self.func(*self.values)


def _quartiles(samples):
    ordered = sorted(samples)
    if len(ordered) < 2:  # noqa: PLR2004 - quantiles need two samples
        return ordered[0], ordered[0]
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    return q[0], q[2]


def _statistics(samples, number):
    q25, q75 = _quartiles(samples)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "q25": q25,
        "q75": q75,
        "samples": samples,
        "number": number,
    }


def _time(call, repeat, sample_time, clock=time.perf_counter):
    """
    Time ``call``; return ``repeat`` per-call samples and the calls per sample.

    The garbage collector is off while samples are taken, as in ``timeit``
    and asv, so a collection the benchmark did not cause is not in its time.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = clock()
        call()  # not counted: caches, lazy imports, first allocations
        first = clock() - start
        number = (
            1
            if first >= sample_time
            else min(MAX_NUMBER, max(1, math.ceil(sample_time / max(first, 1e-9))))
        )
        samples = []
        for _ in range(repeat):
            start = clock()
            for _ in range(number):
                call()
            samples.append((clock() - start) / number)
    finally:
        if enabled:
            gc.enable()
    return samples, number


_RAW_WRAPPER = """\
import sys, time
setup, code = sys.argv[1], sys.argv[2]
namespace = {}
exec(compile(setup, "<setup>", "exec"), namespace)
start = time.perf_counter()
exec(compile(code, "<benchmark>", "exec"), namespace)
sys.stdout.write("%r\\n" % (time.perf_counter() - start))
"""


def _time_raw(instance, repeat, timeout):
    """Run the code a ``timeraw_`` benchmark returns, each sample in a new interpreter."""
    code = instance()
    setup = ""
    if isinstance(code, tuple):
        code, setup = code
    if not isinstance(code, str) or not isinstance(setup, str):
        raise BenchError("a timeraw_ benchmark returns code, or (code, setup), as str")
    code, setup = textwrap.dedent(code), textwrap.dedent(setup)
    samples = []
    with tempfile.TemporaryDirectory() as empty:
        for _ in range(repeat):
            done = subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
                [sys.executable, "-c", _RAW_WRAPPER, setup, code],
                cwd=empty,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if done.returncode != 0:
                last = done.stderr.strip().splitlines()[-1:]
                raise BenchError(last[0] if last else f"exited with {done.returncode}")
            samples.append(float(done.stdout.strip().splitlines()[-1]))
    return samples, 1


def _peak_rss():
    import resource  # noqa: PLC0415 - not on Windows

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _run_one(instance, kind, settings):
    """Measure one set-up combination; return its samples and the calls per sample."""
    if kind == "time":
        return _time(instance, settings["repeat"], settings["sample_time"])
    if kind == "timeraw":
        return _time_raw(instance, settings["repeat"], settings["timeout"])
    if kind == "peakmem":
        instance()
        return [_peak_rss()], 1
    value = instance()
    # numpy scalars are numbers.Real too; stored as plain numbers.
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
    if not _number(value):
        raise BenchError(f"a track_ benchmark returns a finite number, got {value!r}")
    return [value], 1


def _worker(directory, name, combination_index, settings):
    """Run combinations of one benchmark; return their results, in order."""
    module, owner, func_name = _locate(directory, name)
    description = _describe(name, module, owner, func_name)
    if "unsupported" in description:
        return [
            {"params": params, "error": description["unsupported"]}
            for params in description["combinations"]
        ]
    combos = _combinations(
        _first_attr(_sources(module, owner, func_name), "params", None)
    )
    indices = range(len(combos)) if combination_index is None else [combination_index]
    results = []
    for index in indices:
        entry = {"params": description["combinations"][index]}
        instance = _Instance(module, owner, func_name, combos[index])
        try:
            instance.setup()
        except NotImplementedError:
            entry["skipped"] = "setup raised NotImplementedError"
            results.append(entry)
            continue
        except Exception as exc:  # ruff: ignore[blind-except] - reported, run goes on
            entry["error"] = _error(exc)
            results.append(entry)
            continue
        try:
            entry["samples"], entry["number"] = _run_one(
                instance, description["kind"], settings
            )
        except Exception as exc:  # ruff: ignore[blind-except] - reported, run goes on
            entry["error"] = _error(exc)
        finally:
            with contextlib.suppress(Exception):
                instance.teardown()
        results.append(entry)
    return results


def _error(exc):
    last = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return last[:500]


# ---------------------------------------------------------------------------
# Running (in the parent process)
# ---------------------------------------------------------------------------


def _call_worker(python, args, timeout):
    """Run this file as a worker with ``python``; return its JSON answer."""
    with tempfile.TemporaryDirectory() as empty:
        answer = Path(empty) / "answer.json"
        command = [python, str(Path(__file__).resolve()), "_worker", *args]
        command += ["--answer", str(answer)]
        try:
            done = subprocess.run(  # noqa: S603 - the chosen interpreter and this file
                command,
                cwd=empty,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:g} s"
        if not answer.is_file():
            tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
            return None, " | ".join(tail) or f"worker exited with {done.returncode}"
        return json.loads(answer.read_text(encoding="utf-8")), None


def _environment(python):
    code = (
        "import json, platform, sys\n"
        "from importlib import metadata\n"
        "names = json.loads(sys.argv[1])\n"
        "versions = {}\n"
        "for n in names:\n"
        "    try:\n"
        "        versions[n] = metadata.version(n)\n"
        "    except Exception:\n"
        "        versions[n] = None\n"
        "print(json.dumps({'python': platform.python_version(),\n"
        "  'implementation': platform.python_implementation(),\n"
        "  'executable': sys.executable, 'packages': versions}))\n"
    )
    with tempfile.TemporaryDirectory() as empty:
        done = subprocess.run(  # noqa: S603 - the chosen interpreter, fixed code
            [python, "-c", code, json.dumps(PACKAGES_RECORDED)],
            cwd=empty,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    if done.returncode != 0:
        raise BenchError(f"{python}: cannot run it ({done.stderr.strip()[-300:]})")
    info = json.loads(done.stdout)
    info["executable"] = Path(info["executable"]).name
    return info


def _machine():
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or None,
    }


def _interpreter(python):
    """
    Return ``python`` as an absolute command that works from any directory.

    Workers start in an empty temporary directory, where a relative path
    such as ``.venv/bin/python`` would name nothing. The path is made
    absolute without resolving links: a virtual environment's ``python`` is
    a link, and resolved it would leave the environment.
    """
    if python is None:
        return sys.executable
    if os.sep in python or (os.altsep and os.altsep in python):
        return os.path.abspath(python)
    found = shutil.which(python)
    if found is None:
        raise BenchError(f"{python}: no such interpreter on PATH")
    return found


def _select(names, patterns):
    if not patterns:
        return list(names)
    compiled = [re.compile(p) for p in patterns]
    return [n for n in names if any(p.search(n) for p in compiled)]


def _settings(settings):
    out = {
        "repeat": DEFAULT_REPEAT,
        "sample_time": DEFAULT_SAMPLE_TIME,
        "processes": DEFAULT_PROCESSES,
        **(settings or {}),
    }
    if not (
        isinstance(out["repeat"], int)
        and out["repeat"] >= 1
        and isinstance(out["processes"], int)
        and out["processes"] >= 1
        and _number(out["sample_time"])
        and out["sample_time"] > 0
    ):
        raise BenchError(
            "repeat and processes must be positive integers and sample_time above 0"
        )
    return out


def _listing(python, directory, patterns):
    listing, problem = _call_worker(python, ["discover", directory], timeout=600)
    if listing is None:
        raise BenchError(f"cannot list the benchmarks of {directory}: {problem}")
    if "error" in listing:
        raise BenchError(listing["error"])
    names = _select(listing["benchmarks"], patterns)
    if not names:
        raise BenchError(f"no benchmark matches {', '.join(patterns)}")
    return listing["benchmarks"], names


def _document(label, python, settings, benchmarks):
    return {
        "schema": RESULT_SCHEMA,
        "version": RESULT_VERSION,
        "tool_version": TOOL_VERSION,
        "label": label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": _machine(),
        "environment": _environment(python),
        "settings": settings,
        "benchmarks": benchmarks,
    }


def run(directory, *, python=None, patterns=(), settings=None, label=None, echo=None):
    """
    Run the benchmarks of a suite, each in worker processes of its own.

    Parameters
    ----------
    directory : str or pathlib.Path
        The suite package.
    python : str, optional
        Interpreter of the environment to measure; this one by default.
    patterns : iterable of str
        Regular expressions; a benchmark runs when one matches its name.
    settings : dict, optional
        ``repeat``, ``sample_time`` and ``processes`` (worker processes per
        benchmark, whose samples are pooled).
    label : str, optional
        Free text stored with the result (a branch, a commit).
    echo : callable, optional
        Called with one line of progress per benchmark.

    Returns
    -------
    dict
        The result document (schema ``skplt-bench``).

    Raises
    ------
    BenchError
        If the suite cannot be listed, nothing matches ``patterns``, or a
        setting is out of range.
    """
    python = _interpreter(python)
    settings = _settings(settings)
    directory = str(Path(directory).resolve())
    described, names = _listing(python, directory, patterns)
    benchmarks = {}
    for name in names:
        rounds = [
            _measure(python, directory, name, described[name], settings)
            for _ in range(settings["processes"])
        ]
        benchmarks[name] = _pooled(described[name], rounds)
        if echo is not None:
            echo(_progress_line(name, benchmarks[name]))
    return _document(label, python, settings, benchmarks)


def run_pair(
    directory,
    *,
    base_python,
    head_python,
    patterns=(),
    settings=None,
    labels=(None, None),
    echo=None,
):
    """
    Run every benchmark in two environments, alternating them, and return both results.

    Parameters
    ----------
    directory : str or pathlib.Path
        The suite package; both environments run these same files.
    base_python, head_python : str
        Interpreters of the two environments.
    patterns, settings, echo
        As for :func:`run`; ``processes`` is the number of rounds.
    labels : tuple of str
        Labels of the base and head results.

    Returns
    -------
    tuple of dict
        The base and head result documents.

    Notes
    -----
    **Developer notes.** Round ``r`` runs base then head when ``r`` is even
    and head then base when it is odd (ABBA), so a machine that slows down
    or speeds up during the run affects both sides alike, instead of
    showing as a change of one of them. The suite is listed with the head
    environment.
    """
    settings = _settings(settings)
    base_python, head_python = _interpreter(base_python), _interpreter(head_python)
    directory = str(Path(directory).resolve())
    described, names = _listing(head_python, directory, patterns)
    sides = {"base": {}, "head": {}}
    pythons = {"base": base_python, "head": head_python}
    for name in names:
        rounds = {"base": [], "head": []}
        for index in range(settings["processes"]):
            order = ("base", "head") if index % 2 == 0 else ("head", "base")
            for side in order:
                rounds[side].append(
                    _measure(pythons[side], directory, name, described[name], settings)
                )
        for side, results in sides.items():
            results[name] = _pooled(described[name], rounds[side])
        if echo is not None:
            echo(
                f"{_progress_line(name, sides['base'][name])}  ->  "
                f"{_progress_line(name, sides['head'][name]).split(': ', 1)[1]}"
            )
    return (
        _document(labels[0], base_python, settings, sides["base"]),
        _document(labels[1], head_python, settings, sides["head"]),
    )


def _measure(python, directory, name, description, settings):
    """Run one round of a benchmark; return one raw entry per parameter combination."""
    combos = description["combinations"]
    if "unsupported" in description:
        return [{"params": c, "error": description["unsupported"]} for c in combos]
    timeout = description["timeout"]
    worker_settings = {**settings, "timeout": timeout}
    base = ["run", directory, name, "--settings", json.dumps(worker_settings)]
    # A process per combination where the measurement needs one: the peak
    # of a process only rises.
    if description["kind"] == "peakmem":
        calls = [([*base, "--combination", str(i)], [i]) for i in range(len(combos))]
    else:
        calls = [(base, list(range(len(combos))))]
    entries = []
    for args, indices in calls:
        # Each sample of a timeraw_ benchmark may take the whole timeout.
        budget = timeout * len(indices) * (settings["repeat"] + 2)
        answer, problem = _call_worker(python, args, timeout=budget)
        if answer is None or "results" not in answer:
            reason = problem or answer["error"]
            entries += [{"params": combos[i], "error": reason} for i in indices]
        else:
            entries += answer["results"]
    return entries


def _pooled(description, rounds):
    """Pool the samples of every round, combination by combination."""
    results = []
    for entries in zip(*rounds):
        failed = next((e for e in entries if "error" in e or "skipped" in e), None)
        if failed is not None:
            results.append(failed)
            continue
        samples = [s for e in entries for s in e["samples"]]
        stats = _statistics(samples, max(e["number"] for e in entries))
        stats["round_medians"] = [statistics.median(e["samples"]) for e in entries]
        results.append({"params": entries[0]["params"], **stats})
    return {
        **{k: description[k] for k in ("kind", "unit", "param_names")},
        "results": results,
    }


def _progress_line(name, bench):
    parts = []
    for entry in bench["results"]:
        if "error" in entry:
            parts.append("error")
        elif "skipped" in entry:
            parts.append("skipped")
        else:
            parts.append(_format(entry["median"], bench["unit"]))
    return f"{name}: {', '.join(parts)}"


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


#: The largest magnitude of an integer read from a result file.
_MAX_INTEGER = 2**63


def _number(value):
    if isinstance(value, float):
        return math.isfinite(value)
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -_MAX_INTEGER <= value <= _MAX_INTEGER
    )


#: The most samples one result may hold in a file that is read back.
_MAX_SAMPLES = 100_000


def _entry_ok(entry):
    if not (isinstance(entry, dict) and isinstance(entry.get("params"), list)):
        return False
    if not all(isinstance(p, str) for p in entry["params"]):
        return False
    if "error" in entry or "skipped" in entry:
        return isinstance(entry.get("error", entry.get("skipped")), str)
    samples, rounds = entry.get("samples"), entry.get("round_medians", [0])
    return (
        all(_number(entry.get(k)) for k in ("median", "min", "max"))
        and isinstance(samples, list)
        and 0 < len(samples) <= _MAX_SAMPLES
        and all(_number(v) for v in samples)
        and isinstance(rounds, list)
        and 0 < len(rounds) <= _MAX_SAMPLES
        and all(_number(v) for v in rounds)
    )


def _optional_mapping(doc, key):
    return doc.get(key) is None or isinstance(doc.get(key), dict)


def load_result(path):
    """
    Read a result file and check every field that :func:`compare` uses.

    Raises
    ------
    BenchError
        If the file is missing, too large, not a result of a version this
        runner reads, or a field has an unexpected type.
    """
    path = Path(path)
    if not path.is_file():
        raise BenchError(f"{path}: no such file")
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise BenchError(f"{path}: larger than {MAX_RESULT_BYTES} bytes")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, RecursionError) as exc:
        raise BenchError(f"{path}: not JSON ({type(exc).__name__})") from None
    if not isinstance(doc, dict) or doc.get("schema") != RESULT_SCHEMA:
        raise BenchError(f"{path}: not a {RESULT_SCHEMA} result")
    if doc.get("version") != RESULT_VERSION:
        raise BenchError(
            f"{path}: result version {doc.get('version')!r}; this runner reads {RESULT_VERSION}"
        )
    if not all(
        _optional_mapping(doc, k) for k in ("machine", "environment", "settings")
    ):
        raise BenchError(f"{path}: machine, environment and settings must be objects")
    if not (doc.get("label") is None or isinstance(doc.get("label"), str)):
        raise BenchError(f"{path}: label must be text")
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise BenchError(f"{path}: benchmarks must be an object")
    for name, bench in benchmarks.items():
        if not (
            isinstance(bench, dict)
            and bench.get("kind") in {p[:-1] for p in PREFIXES}
            and isinstance(bench.get("unit"), str)
            and isinstance(bench.get("results"), list)
            and all(_entry_ok(e) for e in bench["results"])
        ):
            raise BenchError(f"{path}: benchmark {name!r} has an unexpected shape")
    return doc


#: The fewest rounds (worker processes) per side for a timing to be marked
#: changed: with 4, "every round of head slower than every round of base"
#: happens by chance in 2 of 70 orderings (p = 0.029, the exact rank test).
MIN_ROUNDS = 4

#: Kinds whose values are not timings: compared by their ratio alone.
_EXACT_KINDS = ("peakmem", "track")


def _rounds(entry):
    return entry.get("round_medians") or [entry["median"]]


def _mark(old, new, factor, kind):
    """
    Return ``+`` slower, ``-`` faster, ``~`` unchanged or ``x`` inconclusive.

    Notes
    -----
    **Developer notes.** A timing changed when the ratio of the medians
    reaches ``factor`` and every round of one side is slower than every
    round of the other, comparing each round's median. The round, not the
    sample, is the unit: samples in one process share its state (memory
    layout, caches, the machine's load at the time), so they vary less than
    processes do, and a test that treats them as independent finds changes
    between two runs of the same code (measured: 6 of 25 with a rank test
    on the samples). A round's median ignores one outlier sample, which a
    comparison of the extreme samples does not. Memory peaks and ``track_``
    values change by the ratio.
    """
    a, b = old["median"], new["median"]
    if a <= 0 or b <= 0:
        # Not a ratio (a zero or negative track_ value): any difference counts.
        return "~" if a == b else ("+" if b > a else "-")
    direction = "+" if b / a >= factor else ("-" if b / a <= 1.0 / factor else "~")
    if direction == "~" or kind in _EXACT_KINDS:
        return direction
    base, head = _rounds(old), _rounds(new)
    if min(len(base), len(head)) < MIN_ROUNDS:
        return "x"
    separated = min(head) > max(base) if direction == "+" else max(head) < min(base)
    return direction if separated else "x"


def compare(base, head, *, factor=DEFAULT_FACTOR):
    """
    Compare two result documents, result by result.

    Parameters
    ----------
    base, head : dict
        From :func:`load_result` (or :func:`run`).
    factor : float
        Ratio of medians that counts as a change, above 1.

    Returns
    -------
    dict
        ``rows`` of ``[mark, benchmark, params, base, head, ratio, unit]``:
        ``+`` slower, ``-`` faster, ``~`` unchanged, ``x`` a ratio that
        reached the factor without enough evidence (see :func:`_mark`),
        ``!`` failed or skipped on a side, ``?`` on one side only; the
        counts, ``new_failures`` (failing in head, measured in base), and
        whether the two runs used the same machine and Python.

    Raises
    ------
    BenchError
        If ``factor`` is not a number above 1.
    """
    if not (_number(factor) and factor > 1.0):
        raise BenchError(f"--factor must be a number above 1, got {factor!r}")
    rows, new_failures, fewest = [], 0, None
    names = sorted(set(base["benchmarks"]) | set(head["benchmarks"]))
    for name in names:
        bench = head["benchmarks"].get(name) or base["benchmarks"][name]
        b = {
            tuple(e["params"]): e
            for e in base["benchmarks"].get(name, {}).get("results", [])
        }
        h = {
            tuple(e["params"]): e
            for e in head["benchmarks"].get(name, {}).get("results", [])
        }
        for params in sorted(set(b) | set(h)):
            old, new = b.get(params), h.get(params)
            rows.append(_row(name, params, old, new, factor=factor, bench=bench))
            if (
                old is not None
                and new is not None
                and "median" in old
                and "error" in new
            ):
                new_failures += 1
            for entry in (old, new):
                if (
                    entry is not None
                    and "median" in entry
                    and bench["kind"] not in _EXACT_KINDS
                ):
                    count = len(_rounds(entry))
                    fewest = count if fewest is None else min(fewest, count)
    counts = {}
    for row in rows:
        counts[row[0]] = counts.get(row[0], 0) + 1
    same = {
        "machine": base.get("machine") == head.get("machine"),
        "python": (
            (base.get("environment") or {}).get("python")
            == (head.get("environment") or {}).get("python")
        ),
    }
    return {
        "schema": RESULT_SCHEMA + "-compare",
        "version": RESULT_VERSION,
        "tool_version": TOOL_VERSION,
        "factor": factor,
        "min_rounds": MIN_ROUNDS,
        "fewest_rounds": fewest,
        "labels": [base.get("label"), head.get("label")],
        "same": same,
        "counts": dict(sorted(counts.items())),
        "new_failures": new_failures,
        "rows": sorted(rows, key=lambda r: (_MARK_ORDER[r[0]], r[1], r[2])),
    }


#: Order of the rows of a comparison: what needs attention first.
_MARK_ORDER = {"+": 0, "!": 1, "x": 2, "-": 3, "?": 4, "~": 5}


def _value(entry):
    if entry is None:
        return None
    return entry.get("median")


def _row(name, params, old, new, *, factor, bench):
    label = ", ".join(params)
    unit = bench["unit"]
    if old is None or new is None:
        return ["?", name, label, _value(old), _value(new), None, unit]
    a, b = _value(old), _value(new)
    if a is None or b is None:
        both_skipped = "skipped" in old and "skipped" in new
        return ["~" if both_skipped else "!", name, label, a, b, None, unit]
    ratio = b / a if a else None
    return [_mark(old, new, factor, bench["kind"]), name, label, a, b, ratio, unit]


def _format(value, unit):
    if value is None:
        return "-"
    if unit == "seconds":
        for scale, suffix in ((1.0, "s"), (1e-3, "ms"), (1e-6, "us")):
            if abs(value) >= scale:
                return f"{value / scale:.3g} {suffix}"
        return f"{value / 1e-9:.3g} ns"
    if unit == "bytes":
        return f"{value / 2**20:.1f} MiB"
    return f"{value:g}"


def _code(text):
    """Return ``text`` (from a result file: untrusted) as inline code in a table cell."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text)).replace("|", "\\|")
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def render_compare(result):
    """Render a comparison as Markdown, what needs attention first."""
    counts = result["counts"]
    labels = [_code(label) for label in result["labels"]]
    lines = [
        "# Benchmarks: comparison",
        "",
        (
            f"- Base {labels[0]}, head {labels[1]}. A timing changed when the medians "
            f"differ by {result['factor']}x and every round (worker process) of one side "
            "is slower than every round of the other; memory peaks and track values "
            "change by the ratio alone."
        ),
        (
            f"- **{counts.get('+', 0)} slower, {counts.get('-', 0)} faster**, "
            f"{counts.get('x', 0)} inconclusive, {counts.get('~', 0)} unchanged, "
            f"{counts.get('!', 0)} failed or skipped on a side "
            f"({result['new_failures']} newly failing in head), "
            f"{counts.get('?', 0)} on one side only."
        ),
    ]
    fewest = result["fewest_rounds"]
    if fewest is not None and fewest < result["min_rounds"]:
        lines.append(
            f"- **A timing has only {fewest} round(s)** (`--quick`, or few `--processes`): "
            f"{result['min_rounds']} a side are needed to mark a change, so what reached "
            "the factor is `x`."
        )
    if not all(result["same"].values()):
        lines.append(
            "- **The runs differ in machine or Python**: a difference may come from that, "
            "not from the code. Compare runs made on one machine."
        )
    lines += [
        "",
        "| | benchmark | params | base | head | ratio |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {m} | {_code(n)} | {_code(p) if p else '-'} | {_format(a, u)} | "
        f"{_format(b, u)} | {'-' if r is None else f'{r:.3f}'} |"
        for m, n, p, a, b, r, u in result["rows"]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(
        prog="ci_bench.py", description=__doc__.split("\n\n")[0].strip()
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list the benchmarks and their parameters")
    listing.add_argument(
        "--dir", default=DEFAULT_DIR, help="suite (default: %(default)s)"
    )
    listing.add_argument("--bench", action="append", default=[], metavar="REGEX")
    listing.add_argument("--python", default=None, help="interpreter to list them with")

    running = sub.add_parser("run", help="run benchmarks and write a result file")
    running.add_argument(
        "--dir", default=DEFAULT_DIR, help="suite (default: %(default)s)"
    )
    running.add_argument("--bench", action="append", default=[], metavar="REGEX")
    running.add_argument(
        "--python", default=None, help="interpreter of the environment to measure"
    )
    running.add_argument("--out", required=True, help="result file to write")
    running.add_argument("--label", default=None, help="text stored with the result")
    _timing_options(running)
    running.add_argument(
        "--fail-on-error", action="store_true", help="exit 1 when a benchmark failed"
    )

    comparing = sub.add_parser("compare", help="compare two result files")
    comparing.add_argument("base")
    comparing.add_argument("head")
    comparing.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    comparing.add_argument(
        "--out", default=None, help="write compare.md and compare.json here"
    )
    comparing.add_argument(
        "--fail-on-slower",
        action="store_true",
        help="exit 1 when a result got slower or a benchmark measured in base fails in head",
    )

    pair = sub.add_parser(
        "pair", help="run in two environments, alternating them, and compare"
    )
    pair.add_argument("--dir", default=DEFAULT_DIR, help="suite (default: %(default)s)")
    pair.add_argument("--bench", action="append", default=[], metavar="REGEX")
    pair.add_argument(
        "--base-python", required=True, help="interpreter of the base environment"
    )
    pair.add_argument(
        "--head-python", required=True, help="interpreter of the head environment"
    )
    pair.add_argument("--base-label", default="base")
    pair.add_argument("--head-label", default="head")
    pair.add_argument(
        "--out",
        required=True,
        help="write base.json, head.json, compare.md, compare.json here",
    )
    pair.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    pair.add_argument(
        "--fail-on-slower",
        action="store_true",
        help="exit 1 when a result got slower or a benchmark measured in base fails in head",
    )
    _timing_options(pair)

    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("action", choices=("discover", "run"))
    worker.add_argument("directory")
    worker.add_argument("name", nargs="?")
    worker.add_argument("--settings", default="{}")
    worker.add_argument("--combination", type=int, default=None)
    worker.add_argument("--answer", required=True)
    return parser


def _timing_options(parser):
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEAT, help="samples per process"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help="worker processes (pair: rounds) per benchmark, samples pooled",
    )
    parser.add_argument("--sample-time", type=float, default=DEFAULT_SAMPLE_TIME)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"{QUICK_REPEAT} sample in 1 process per result: a smoke test, not a measurement",
    )


def _cli_settings(args):
    return {
        "repeat": QUICK_REPEAT if args.quick else args.repeat,
        "processes": 1 if args.quick else args.processes,
        "sample_time": args.sample_time,
    }


def _write_text(path, text):
    """Write UTF-8 text with Unix line ends (``Path.write_text`` has no ``newline`` before 3.10)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(
        path, json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    )


def _main_worker(args):
    try:
        if args.action == "discover":
            answer = {"benchmarks": discover(args.directory)}
        else:
            settings = json.loads(args.settings)
            answer = {
                "results": _worker(
                    args.directory, args.name, args.combination, settings
                )
            }
    except Exception as exc:  # ruff: ignore[blind-except] - sent to the parent
        answer = {"error": _error(exc)}
    _write_json(args.answer, answer)
    return 0


def _main_list(args):
    listing, problem = _call_worker(
        _interpreter(args.python),
        ["discover", str(Path(args.dir).resolve())],
        timeout=600,
    )
    if listing is None or "error" in listing:
        raise BenchError(problem or listing["error"])
    for name in _select(listing["benchmarks"], args.bench):
        d = listing["benchmarks"][name]
        params = (
            f" [{', '.join(d['param_names'])}] x{len(d['combinations'])}"
            if d["param_names"]
            else ""
        )
        sys.stdout.write(f"{name} ({d['kind']}){params}\n")
    return 0


def _main_run(args):
    settings = _cli_settings(args)
    result = run(
        args.dir,
        python=args.python,
        patterns=args.bench,
        settings=settings,
        label=args.label,
        echo=lambda line: sys.stdout.write(line + "\n"),
    )
    _write_json(args.out, result)
    errors = sum(
        1 for b in result["benchmarks"].values() for e in b["results"] if "error" in e
    )
    sys.stdout.write(
        f"ci_bench: {len(result['benchmarks'])} benchmark(s), {errors} error(s); wrote {args.out}\n"
    )
    return 1 if args.fail_on_error and errors else 0


def _got_worse(result):
    """Return True when a result got slower, or a benchmark measured in base fails in head."""
    return bool(result["counts"].get("+") or result["new_failures"])


def _main_pair(args):
    base, head = run_pair(
        args.dir,
        base_python=args.base_python,
        head_python=args.head_python,
        patterns=args.bench,
        settings=_cli_settings(args),
        labels=(args.base_label, args.head_label),
        echo=lambda line: sys.stdout.write(line + "\n"),
    )
    out = Path(args.out)
    _write_json(out / "base.json", base)
    _write_json(out / "head.json", head)
    result = compare(base, head, factor=args.factor)
    text = render_compare(result)
    _write_text(out / "compare.md", text)
    _write_json(out / "compare.json", result)
    sys.stdout.write(text)
    return 1 if args.fail_on_slower and _got_worse(result) else 0


def _main_compare(args):
    result = compare(load_result(args.base), load_result(args.head), factor=args.factor)
    text = render_compare(result)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write_text(out / "compare.md", text)
        _write_json(out / "compare.json", result)
    sys.stdout.write(text)
    return 1 if args.fail_on_slower and _got_worse(result) else 0


def main(argv=None):
    """
    Run the command line.

    Returns
    -------
    int
        ``0`` on success; ``1`` when ``compare --fail-on-slower`` found a
        slower result or ``run --fail-on-error`` a failed benchmark; ``2``
        when the suite, a file or an option is unusable.
    """
    args = _parser().parse_args(argv)
    if args.command == "_worker":
        return _main_worker(args)
    try:
        return {
            "list": _main_list,
            "run": _main_run,
            "pair": _main_pair,
            "compare": _main_compare,
        }[args.command](args)
    except (BenchError, re.error, OSError) as exc:
        sys.stderr.write(f"ci_bench: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
