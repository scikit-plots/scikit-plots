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
discovery and dossier JSON). Exit status 1 when two backends disagree; a
backend whose optional dependency is missing is listed as unavailable.

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


def _peak_mb() -> int | None:
    try:
        import resource  # noqa: PLC0415  (POSIX only)
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak // (1024 * 1024 if sys.platform == "darwin" else 1024))


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
        ``seconds`` per stage, ``peak_mb``, ``nodes``, ``files``,
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
    digest = hashlib.sha256(
        json.dumps([summary, report, dossier], sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
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

    *run* returns :func:`measure`'s dict, or ``{"unavailable": reason}``.
    The status is 1 when two available backends give different
    ``report_sha``, 0 otherwise (also when only one is available).
    """
    results = []
    for graph in graphs:
        result = dict(run(graph))
        result["graph"] = graph
        results.append(result)
    digests = {r["report_sha"] for r in results if "report_sha" in r}
    return (1 if len(digests) > 1 else 0), results


def _run_subprocess(folder: Path, graph: str, workers: int, top: int) -> dict[str, Any]:
    """Measure one backend in a fresh interpreter (its own peak memory)."""
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), str(folder),
        "--graph", graph, "--workers", str(workers), "--top", str(top), "--json",
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        return {"unavailable": lines[-1][-200:] if lines else f"exit {done.returncode}"}
    return json.loads(done.stdout)


def _compare_text(status: int, results: list[dict[str, Any]]) -> str:
    lines = [f"{'graph':<7} {'index':>8} {'summary':>8} {'discover':>9} {'dossier':>8} {'peak MB':>8}  report"]
    for r in results:
        if "unavailable" in r:
            lines.append(f"{r['graph']:<7} unavailable: {r['unavailable']}")
            continue
        s = r["seconds"]
        lines.append(
            f"{r['graph']:<7} {s.get('index', 0):>8} {s.get('summary', 0):>8}"
            f" {s.get('discover', 0):>9} {s.get('dossier', 0):>8} {r['peak_mb']!s:>8}  {r['report_sha']}"
        )
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
        "--compare", action="store_true", help="run every backend; exit 1 if reports differ"
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
        result = measure(
            index_class, args.folder, out=args.out or scratch, top=args.top, options=options
        )
    print(json.dumps(result, indent=1, sort_keys=True) if args.json else _text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
