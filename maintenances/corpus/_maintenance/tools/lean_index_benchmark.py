#!/usr/bin/env python3
r"""
Time and size every LeanIndex stage on a real Lean project, per graph backend.

Runs, on one folder: :meth:`LeanIndex.from_folder`, ``summary``,
``discover``, a ``dossier`` on the first headline result and ``export``,
and reports the seconds of each, the memory of each, the index's size,
and the count of every discovery section. The output is the evidence
a pull request quotes for "a run on a real large project".

Usage
-----
::

    python lean_index_benchmark.py FOLDER [--out HELPERS] [--top 20] [--json]
    python lean_index_benchmark.py FOLDER --graph numpy --workers 4
    python lean_index_benchmark.py FOLDER --compare [--workers 4] [--json]

``--compare`` runs every graph backend (``python``, ``numpy``, ``scipy``)
in a process of its own, so each peak memory is its own, and checks that
their reports are identical (``report_sha``: the digest of the summary,
discovery and dossier JSON). Exit status 1 when two backends disagree,
when a backend fails, or when none could run; a backend whose optional
dependency is missing (exit status 3 when run alone) is listed as
unavailable and does not fail the comparison.

Memory (``memory_mb``, Linux): the peak, per stage and in all, of the
proportional set size (PSS) summed over the benchmark and every process it
starts (file-reading and name-resolving workers). PSS counts a page shared
by forked workers once, split between them, so the sum is what the run
really takes; resident sizes (``peak_mb``, ``workers_peak_mb``) count a
shared page in every process. A sampling process (``--no-pss`` turns it
off) reads ``/proc`` every half second: a stage shorter than that may
have no sample (``None``). Elsewhere ``memory_mb`` is ``None``.

Index the project's root (the folder holding the top-level ``Mathlib.lean``
for mathlib4), not a subfolder: imports are matched against paths from the
root, and a subfolder leaves every import unresolved.

Notes
-----
**Developer.** The package is imported in :func:`main`, never at import
time, so the tests can pass a stand-in class without a build. Peak memory
comes from :func:`resource.getrusage` where it exists (``None`` on
Windows). Record counts only; never commit the helper documents of a
user's project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


#: Exit status of a single run whose graph backend's dependency is missing.
UNAVAILABLE = 3


def _peak_mb(who: str = "RUSAGE_SELF") -> int | None:
    """Peak resident memory of this process, or of its largest finished child."""
    try:
        import resource  # noqa: PLC0415  (POSIX only)
    except ImportError:
        return None
    peak = resource.getrusage(getattr(resource, who)).ru_maxrss
    return int(peak // (1024 * 1024 if sys.platform == "darwin" else 1024))


#: Seconds between two memory samples.
PSS_INTERVAL = 0.5


def _pss_available(proc: str = "/proc") -> bool:
    """Return whether this system reports each process's PSS (Linux)."""
    return Path(proc, "self", "smaps_rollup").is_file()


def _parents(proc: str = "/proc") -> dict[int, int]:
    """Return every process's parent: ``{pid: ppid}``, read from ``/proc/*/stat``."""
    parents = {}
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            text = Path(proc, entry, "stat").read_text()
        except OSError:  # gone since listed
            continue
        # "pid (comm) state ppid ...": comm may hold spaces and parentheses.
        fields = text[text.rfind(")") + 2 :].split()
        if len(fields) > 1 and fields[1].isdigit():
            parents[int(entry)] = int(fields[1])
    return parents


def _tree(parents: dict[int, int], root: int) -> set[int]:
    """Return *root* and every process descended from it."""
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    found, todo = {root}, [root]
    while todo:
        for child in children.get(todo.pop(), ()):
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


def _pss_kb(pid: int, proc: str = "/proc") -> int:
    """Return a process's PSS in kB, or 0 once it is gone."""
    try:
        with open(Path(proc, str(pid), "smaps_rollup")) as stream:
            for line in stream:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _alive(pid: int, proc: str = "/proc") -> bool:
    """Return whether *pid* runs: it exists and is not a zombie (exited, not reaped)."""
    try:
        text = Path(proc, str(pid), "stat").read_text()
    except OSError:
        return False
    fields = text[text.rfind(")") + 2 :].split()
    return bool(fields) and fields[0] not in ("Z", "X")


def sample_pss(root: int, out: str | Path, *, proc: str = "/proc") -> None:
    """
    Append ``time kB`` lines, the PSS of *root*'s tree, until *root* is gone.

    The sampling process itself is left out of the sum.
    """
    me = os.getpid()
    with open(out, "a") as stream:
        while _alive(root, proc):
            pids = _tree(_parents(proc), root) - {me}
            stream.write(f"{time.time()} {sum(_pss_kb(p, proc) for p in pids)}\n")
            stream.flush()
            time.sleep(PSS_INTERVAL)


def stage_peaks(
    timeline: list[tuple[float, int]], windows: dict[str, tuple[float, float]]
) -> dict[str, int | None]:
    """
    Return the peak PSS in MB within each stage's window, and ``"all"``.

    A stage with no sample inside its window gets ``None``.
    """
    peaks: dict[str, int | None] = {}
    for stage, (start, end) in windows.items():
        inside = [kb for at, kb in timeline if start <= at <= end]
        peaks[stage] = max(inside) // 1024 if inside else None
    peaks["all"] = max((kb for _at, kb in timeline), default=0) // 1024 or None
    return peaks


def _read_timeline(path: str | Path) -> list[tuple[float, int]]:
    timeline = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            timeline.append((float(parts[0]), int(parts[1])))
    return timeline


def _digest(*reports: Any) -> str:
    """First 16 hex digits of the SHA-256 of *reports* as canonical JSON."""

    def plain(value: Any) -> Any:
        if isinstance(value, (set, frozenset)):  # no hash-seed-dependent order
            return sorted(value, key=repr)
        return str(value)

    text = json.dumps(list(reports), sort_keys=True, default=plain)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def measure(
    index_class: Any,
    folder: str | Path,
    *,
    out: str | Path,
    top: int,
    options: dict[str, Any] | None = None,
    pss: bool = False,
) -> dict[str, Any]:
    """
    Run every stage on *folder* and return the timings and counts.

    Parameters
    ----------
    index_class : type
        :class:`~scikitplot.lean.LeanIndex`, or a stand-in with the same
        methods.
    folder : str or pathlib.Path
        The project's root.
    out : str or pathlib.Path
        Where ``export`` writes.
    top : int
        Items per section.
    options : dict or None, optional
        Passed to ``from_folder`` (``graph``, ``workers``). Default: none.
    pss : bool, optional
        Sample the PSS of this process and its workers (Linux; ignored
        elsewhere). Default: ``False``.

    Returns
    -------
    dict
        ``seconds`` per stage, ``peak_mb`` (this process), ``workers_peak_mb``
        (the largest worker process, when ``workers`` > 1), ``nodes``, ``files``,
        ``skipped``, ``sections`` (count per discovery section),
        ``files_written``, ``options``, ``report_sha`` and ``memory_mb``
        (peak PSS per stage and ``"all"``, or ``None``).
    """
    seconds: dict[str, float] = {}
    windows: dict[str, tuple[float, float]] = {}
    sampler = timeline_file = None
    if pss and _pss_available():
        # A process, not a thread: forked workers are used only while this
        # process runs a single thread.
        handle, timeline_file = tempfile.mkstemp(prefix="lean_pss_", suffix=".txt")
        os.close(handle)
        sampler = subprocess.Popen(  # noqa: S603
            [sys.executable, "-B", str(Path(__file__).resolve()),
             "--sample-pss", str(os.getpid()), timeline_file],
        )  # fmt: skip

    def timed(stage: str, call: Any) -> Any:
        start, wall = time.perf_counter(), time.time()
        result = call()
        seconds[stage] = round(time.perf_counter() - start, 2)
        windows[stage] = (wall, time.time())
        return result

    options = dict(options or {})
    index = timed("index", lambda: index_class.from_folder(folder, **options))
    summary = timed("summary", lambda: index.summary(max_items=top))
    report = timed("discover", lambda: index.discover(top=top))
    headline = report["headline_results"]["items"]
    dossier = None
    if headline:
        dossier = timed(
            "dossier", lambda: index.dossier(headline[0]["declaration"], max_items=top)
        )
    digest = _digest(summary, report, dossier)
    written = timed("export", lambda: index.export(out, top=top))
    memory = None
    if sampler is not None:
        time.sleep(PSS_INTERVAL)  # one sample after the last stage
        sampler.terminate()
        sampler.wait()
        memory = stage_peaks(_read_timeline(timeline_file), windows)
        os.unlink(timeline_file)
    sections: dict[str, int] = {}
    for group in ("rules", "patterns", "claims", "anomalies"):
        for key, listing in report[group].items():
            sections[f"{group}.{key}"] = listing["count"]
    sections["headline_results"] = report["headline_results"]["count"]
    return {
        "folder": Path(folder).name,
        "seconds": seconds,
        "peak_mb": _peak_mb(),
        "workers_peak_mb": _peak_mb("RUSAGE_CHILDREN"),
        "nodes": len(index._name),
        "files": len(index._labels),
        "skipped": len(index._skipped),
        "sections": sections,
        "files_written": len(written),
        "options": options,
        "report_sha": digest,
        "memory_mb": memory,
    }


#: The graph backends ``--compare`` runs, in this order.
GRAPHS = ("python", "numpy", "scipy")


def compare(run: Any, graphs: Sequence[str] = GRAPHS) -> tuple[int, list[dict[str, Any]]]:
    """
    Run *run(graph)* for each backend; return the exit status and the results.

    *run* returns :func:`measure`'s dict, ``{"unavailable": reason}`` (an
    optional dependency is missing) or ``{"error": reason}``. The status is
    1 when a backend failed, when none was measured, or when two measured
    backends give different ``report_sha``; 0 otherwise.
    """
    results = []
    for graph in graphs:
        result = dict(run(graph))
        result["graph"] = graph
        results.append(result)
    digests = {r["report_sha"] for r in results if "report_sha" in r}
    failed = any("error" in r for r in results)
    return (1 if failed or len(digests) != 1 else 0), results


def _run_subprocess(
    folder: Path, graph: str, workers: int, top: int, pss: bool = True
) -> dict[str, Any]:
    """Measure one backend in a fresh interpreter (its own peak memory)."""
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), str(folder),
        "--graph", graph, "--workers", str(workers), "--top", str(top), "--json",
    ] + ([] if pss else ["--no-pss"])
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        reason = lines[-1][-200:] if lines else f"exit {done.returncode}"
        key = "unavailable" if done.returncode == UNAVAILABLE else "error"
        return {key: reason}
    return json.loads(done.stdout)


def _compare_text(status: int, results: list[dict[str, Any]]) -> str:
    lines = [
        f"{'graph':<7} {'index':>8} {'summary':>8} {'discover':>9} {'dossier':>8}"
        f" {'peak MB':>8} {'PSS MB':>7}  report"
    ]
    for r in results:
        if "unavailable" in r or "error" in r:
            key = "unavailable" if "unavailable" in r else "error"
            lines.append(f"{r['graph']:<7} {key}: {r[key]}")
            continue
        s = r["seconds"]
        lines.append(
            f"{r['graph']:<7} {s.get('index', 0):>8} {s.get('summary', 0):>8}"
            f" {s.get('discover', 0):>9} {s.get('dossier', 0):>8} {r['peak_mb']!s:>8}"
            f" {(r.get('memory_mb') or {}).get('all')!s:>7}  {r['report_sha']}"
        )
    if any("error" in r for r in results):
        lines.append("A BACKEND FAILED")
    elif not any("report_sha" in r for r in results):
        lines.append("NO BACKEND COULD RUN")
    else:
        lines.append("reports identical" if status == 0 else "REPORTS DIFFER between backends")
    return "\n".join(lines)


def _text(result: dict[str, Any]) -> str:
    lines = [
        f"{result['folder']}: {result['files']:,} files, {result['nodes']:,} declarations,"
        f" {result['skipped']} skipped; peak {result['peak_mb']} MB",
        "seconds: " + ", ".join(f"{k} {v}" for k, v in result["seconds"].items()),
    ]
    memory = result.get("memory_mb")
    if memory:
        lines.append("PSS MB: " + ", ".join(f"{k} {v}" for k, v in memory.items()))
    lines += [f"  {key}: {count:,}" for key, count in result["sections"].items()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, index_class: Any = None) -> int:
    """Command line; see the module docstring."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--sample-pss"]:  # the sampling process measure() starts
        sample_pss(int(argv[1]), argv[2])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="export folder (default: temporary)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--graph", default="python", choices=GRAPHS)
    parser.add_argument("--workers", type=int, default=1, help="0: every CPU")
    parser.add_argument(
        "--no-pss", action="store_true", help="do not sample memory (PSS, Linux)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run every backend (ignores --graph and --out); exit 1 if reports differ",
    )
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.workers < 0:
        parser.error("--workers must be 0 (every CPU) or more")
    if args.compare:
        status, results = compare(
            lambda graph: _run_subprocess(
                args.folder, graph, args.workers, args.top, not args.no_pss
            )
        )
        print(json.dumps(results, indent=1, sort_keys=True) if args.json else _compare_text(status, results))  # noqa: T201
        return status
    if index_class is None:
        from scikitplot.lean import LeanIndex as index_class  # noqa: PLC0415, N813
    with tempfile.TemporaryDirectory() as scratch:
        options = {"graph": args.graph, "workers": None if args.workers == 0 else args.workers}
        try:
            result = measure(
                index_class,
                args.folder,
                out=args.out or scratch,
                top=args.top,
                options=options,
                pss=not args.no_pss,
            )
        except ImportError as exc:  # the backend's optional dependency
            print(f"unavailable: {exc}", file=sys.stderr)  # noqa: T201
            return UNAVAILABLE
    print(json.dumps(result, indent=1, sort_keys=True) if args.json else _text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
