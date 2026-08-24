#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Re-derive the physical tracker and check the annoy-family contract.

Usage
-----
.. code-block:: console

    $ python <submodule>/_maintenance/check_trackers.py
    $ python <submodule>/_maintenance/check_trackers.py --update

Notes
-----
**Why this is a script.**  The physical inventory drifts *silently*: a file
grows, a header is added, nothing breaks.  A document describing it is stale the
week after it is written.

**What is specific to this family.**  Three submodules reach into
``cexternals/_annoy/src/`` through a *relative path written into Cython source*,
not through ``include_directories``.  That coupling is invisible to any tool
reading ``meson.build`` alone, so it is checked here: every
``cdef extern from "../../cexternals/_annoy/src/X.h"`` must name a header that
actually exists.  A rename upstream otherwise surfaces as a Cython compile error
in a *different* submodule, naming a path rather than a contract.

Exit codes
----------
0
    Tracker matches the tree and every cross-submodule reference resolves.
1
    Drift, a crossed tripwire, or a dangling shared-header reference.
2
    Usage error.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

#: Relative tolerance on LOC before drift is reported.
LOC_TOLERANCE = 0.10

#: Source extensions that count toward the inventory.
SOURCE_SUFFIXES = (
    ".py",
    ".pyx",
    ".pxd",
    ".pxi",
    ".in",
    ".h",
    ".hpp",
    ".cc",
    ".cpp",
)

#: The shared C++ source every consumer in this family depends on.
SHARED_SRC = "cexternals/_annoy/src"

#: ``cdef extern from "…/cexternals/_annoy/src/<header>"``
_EXTERN_RE = re.compile(r'extern\s+from\s+"([^"]*cexternals/_annoy/src/[^"]+)"')


def submodule_root() -> pathlib.Path:
    """Return the submodule directory containing this script."""
    return pathlib.Path(__file__).resolve().parent.parent


def package_root(root: pathlib.Path) -> pathlib.Path:
    """Return the ``scikitplot/`` directory above this submodule."""
    here = root
    while here.name != "scikitplot" and here.parent != here:
        here = here.parent
    return here


def scan(root: pathlib.Path) -> dict:
    """Walk the tree and return the physical inventory."""
    files: list[dict] = []
    subpackages: dict = collections.defaultdict(
        lambda: {"src": 0, "src_loc": 0, "test": 0, "test_loc": 0}
    )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        rel = str(path.relative_to(root))
        kind = (
            "test" if "/tests/" in "/" + rel or rel.startswith("tests/") else "source"
        )
        loc = len(path.read_text(errors="replace").splitlines())
        files.append({"path": rel, "kind": kind, "loc": loc})
        top = rel.split("/", maxsplit=1)[0] if "/" in rel else "(root)"
        key = "src" if kind == "source" else "test"
        subpackages[top][key] += 1
        subpackages[top][key + "_loc"] += loc

    source = [f for f in files if f["kind"] == "source"]
    tests = [f for f in files if f["kind"] == "test"]
    return {
        "totals": {
            "source_files": len(source),
            "source_loc": sum(f["loc"] for f in source),
            "test_files": len(tests),
            "test_loc": sum(f["loc"] for f in tests),
            "markdown_files": len(list(root.rglob("*.md"))),
        },
        "subpackages": {k: dict(v) for k, v in subpackages.items()},
        "largest_source": [
            {"loc": f["loc"], "path": f["path"]}
            for f in sorted(source, key=lambda f: -f["loc"])[:10]
        ],
    }


def compare(recorded: dict, actual: dict) -> list[str]:
    """Return one message per drift found."""
    problems: list[str] = []
    for name, was in recorded["totals"].items():
        now = actual["totals"].get(name, 0)
        if was and abs(now - was) / was > LOC_TOLERANCE:
            problems.append(
                f"totals.{name}: recorded {was}, actual {now} "
                f"({(now - was) / was:+.0%})"
            )
    recorded_subs = set(recorded["subpackages"])
    actual_subs = set(actual["subpackages"])
    for added in sorted(actual_subs - recorded_subs):
        problems.append(
            f"new area {added!r}: run the submodule review in "
            "SUBMODULE_STRUCTURE.md before merging"
        )
    for removed in sorted(recorded_subs - actual_subs):
        problems.append(f"area {removed!r} disappeared")
    return problems


def check_shared_headers(root: pathlib.Path) -> list[str]:
    """
    Verify every reference to the shared C++ source resolves.

    Notes
    -----
    **Developer.**  This is the family's defining coupling and the one nothing
    else checks.  Consumers name headers by relative path *inside Cython
    source*, and in two of them ``'src'`` is commented out of
    ``include_directories`` -- so the path string is the whole mechanism.  A
    rename upstream produces a compile error in a different submodule that names
    a file, not a contract.
    """
    problems: list[str] = []
    package = package_root(root)
    shared = package / SHARED_SRC
    if not shared.is_dir():
        return [
            f"the shared source {SHARED_SRC!r} was not found under {package}; "
            "this family cannot build without it"
        ]

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".pyx", ".pxd", ".pxi", ".in"):
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            match = _EXTERN_RE.search(line)
            if not match:
                continue
            header = match.group(1).rsplit("/", 1)[-1]
            if not (shared / header).is_file():
                problems.append(
                    f"{path.relative_to(root)}:{number} declares an ABI from "
                    f"{header!r}, which does not exist in {SHARED_SRC}/ -- the "
                    "header was renamed, moved or removed upstream"
                )
    return problems


def check_no_upward_imports(root: pathlib.Path) -> list[str]:
    """`cexternals` must not import the Python layers built on top of it."""
    if root.name != "cexternals":
        return []
    problems: list[str] = []
    forbidden = ("scikitplot.annoy", "scikitplot.memmap", "scikitplot.random")
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "_backup" in path.parts:
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if line[:1] in (" ", "\t", "#") or not line:
                continue
            for name in forbidden:
                if re.match(rf"(import|from)\s+{re.escape(name)}\b", line.strip()):
                    problems.append(
                        f"{path.relative_to(root)}:{number} imports {name!r}; "
                        "cexternals is upstream and must stay standalone"
                    )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update", action="store_true", help="rewrite the physical section"
    )
    args = parser.parse_args(argv)

    root = submodule_root()
    tracker_path = root / "_maintenance" / "TRACKER.json"
    if not tracker_path.is_file():
        print(f"error: {tracker_path} not found", file=sys.stderr)
        return 2

    tracker = json.loads(tracker_path.read_text())
    actual = scan(root)

    if args.update:
        tracker["physical"] = actual
        tracker_path.write_text(json.dumps(tracker, indent=2) + "\n")
        print(f"updated {tracker_path.name}; regenerate TRACKER_PHYSICAL.md to match")
        return 0

    drift = compare(tracker["physical"], actual)
    shared = check_shared_headers(root)
    upward = check_no_upward_imports(root)

    for label, items in (
        ("DRIFT", drift),
        ("SHARED-SOURCE", shared),
        ("FAMILY", upward),
    ):
        for item in items:
            print(f"{label}: {item}")

    if not (drift or shared or upward):
        totals = actual["totals"]
        print(
            f"{tracker['module']}: tracker matches the tree "
            f"({totals['source_files']} source / {totals['test_files']} test files, "
            f"{totals['source_loc']} / {totals['test_loc']} LOC)"
        )
        return 0

    print(
        "\nReconcile before continuing. In this family a shared-source problem "
        "is never local: cexternals/_annoy/src is upstream of three submodules. "
        "See _maintenance/FAMILY.md."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
