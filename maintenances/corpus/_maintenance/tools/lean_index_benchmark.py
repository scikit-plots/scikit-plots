#!/usr/bin/env python3
r"""
Time and size every LeanIndex stage on a real Lean project, per graph backend.

Runs, on one folder: :meth:`LeanIndex.from_folder`, ``summary``,
``discover``, a ``dossier`` on the first headline result and ``export``,
and reports the seconds of each, the peak resident memory, the index's
size, and the count of every discovery section. The output is the evidence
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

    Returns
    -------
    dict
        ``seconds`` per stage, ``peak_mb`` (this process), ``workers_peak_mb``
        (the largest worker process, when ``workers`` > 1), ``nodes``, ``files``,
        ``skipped``, ``sections`` (count per discovery section),
        ``files_written``, ``options`` and ``report_sha``.
    """
    seconds: dict[str, float] = {}

    def timed(stage: str, call: Any) -> Any:
        start = time.perf_counter()
        result = call()
        seconds[stage] = round(time.perf_counter() - start, 2)
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


def _run_subprocess(folder: Path, graph: str, workers: int, top: int) -> dict[str, Any]:
    """Measure one backend in a fresh interpreter (its own peak memory)."""
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), str(folder),
        "--graph", graph, "--workers", str(workers), "--top", str(top), "--json",
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        reason = lines[-1][-200:] if lines else f"exit {done.returncode}"
        key = "unavailable" if done.returncode == UNAVAILABLE else "error"
        return {key: reason}
    return json.loads(done.stdout)


def _compare_text(status: int, results: list[dict[str, Any]]) -> str:
    lines = [f"{'graph':<7} {'index':>8} {'summary':>8} {'discover':>9} {'dossier':>8} {'peak MB':>8}  report"]
    for r in results:
        if "unavailable" in r or "error" in r:
            key = "unavailable" if "unavailable" in r else "error"
            lines.append(f"{r['graph']:<7} {key}: {r[key]}")
            continue
        s = r["seconds"]
        lines.append(
            f"{r['graph']:<7} {s.get('index', 0):>8} {s.get('summary', 0):>8}"
            f" {s.get('discover', 0):>9} {s.get('dossier', 0):>8} {r['peak_mb']!s:>8}  {r['report_sha']}"
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
    lines += [f"  {key}: {count:,}" for key, count in result["sections"].items()]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, *, index_class: Any = None) -> int:
    """Command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="export folder (default: temporary)")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--graph", default="python", choices=GRAPHS)
    parser.add_argument("--workers", type=int, default=1, help="0: every CPU")
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
            lambda graph: _run_subprocess(args.folder, graph, args.workers, args.top)
        )
        print(json.dumps(results, indent=1, sort_keys=True) if args.json else _compare_text(status, results))  # noqa: T201
        return status
    if index_class is None:
        from scikitplot.lean import LeanIndex as index_class  # noqa: PLC0415, N813
    with tempfile.TemporaryDirectory() as scratch:
        options = {"graph": args.graph, "workers": None if args.workers == 0 else args.workers}
        try:
            result = measure(
                index_class, args.folder, out=args.out or scratch, top=args.top, options=options
            )
        except ImportError as exc:  # the backend's optional dependency
            print(f"unavailable: {exc}", file=sys.stderr)  # noqa: T201
            return UNAVAILABLE
    print(json.dumps(result, indent=1, sort_keys=True) if args.json else _text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
