#!/usr/bin/env python3
r"""
Time and size every LeanIndex stage on a real Lean project.

Runs, on one folder: :meth:`LeanIndex.from_folder`, ``summary``,
``discover``, a ``dossier`` on the first headline result and ``export``,
and reports the seconds of each, the peak resident memory, the index's
size, and the count of every discovery section. The output is the evidence
a pull request quotes for "a run on a real large project".

Usage
-----
::

    python lean_index_benchmark.py FOLDER [--out HELPERS] [--top 20] [--json]

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
import json
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


def measure(index_class: Any, folder: str | Path, *, out: str | Path, top: int) -> dict[str, Any]:
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

    Returns
    -------
    dict
        ``seconds`` per stage, ``peak_mb``, ``nodes``, ``files``,
        ``skipped``, ``sections`` (count per discovery section),
        ``files_written``.
    """
    seconds: dict[str, float] = {}

    def timed(stage: str, call: Any) -> Any:
        start = time.perf_counter()
        result = call()
        seconds[stage] = round(time.perf_counter() - start, 2)
        return result

    index = timed("index", lambda: index_class.from_folder(folder))
    timed("summary", lambda: index.summary(max_items=top))
    report = timed("discover", lambda: index.discover(top=top))
    headline = report["headline_results"]["items"]
    if headline:
        timed("dossier", lambda: index.dossier(headline[0]["declaration"], max_items=top))
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
    }


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
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if index_class is None:
        from scikitplot.lean import LeanIndex as index_class  # noqa: PLC0415, N813
    with tempfile.TemporaryDirectory() as scratch:
        result = measure(index_class, args.folder, out=args.out or scratch, top=args.top)
    print(json.dumps(result, indent=1, sort_keys=True) if args.json else _text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
