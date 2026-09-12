#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Re-derive the physical tracker and check this submodule's independence.

Usage
-----
.. code-block:: console

    $ python scikitplot/cython/_maintenance/check_trackers.py
    $ python scikitplot/cython/_maintenance/check_trackers.py --update

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


#: Sibling submodules this one must not import.  Its independence is a design
#: property, not an accident: an edge from here would be an edge into a
#: subprocess-invoking, filesystem-locking service.
_SIBLING_RE = re.compile(r"^\s*(?:from|import)\s+((?:scikitplot\.|\.\.)[\w.]+)")


def check_independence(root: pathlib.Path) -> list[str]:
    """
    Verify no sibling ``scikitplot`` submodule is imported.

    Notes
    -----
    **Developer.**  ``scikitplot.cython`` has zero edges in either direction,
    which is what lets it be reviewed and released without coordinating with
    another campaign.  An import added here would be the project's first
    dependency on a service that shells out to a compiler and holds filesystem
    locks -- a different shape from an optional dependency on a library, and one
    that needs its own review run rather than a convenience import.
    """
    package = root
    while package.name != "scikitplot" and package.parent != package:
        package = package.parent
    siblings = (
        {
            p.name
            for p in package.iterdir()
            if p.is_dir() and not p.name.startswith((".", "__")) and p.name != root.name
        }
        if package.name == "scikitplot"
        else set()
    )

    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            match = _SIBLING_RE.match(line)
            if not match:
                continue
            target = match.group(1).lstrip(".")
            head = (
                target.split(".")[1]
                if target.startswith("scikitplot.")
                else target.split(".")[0]
            )
            if head in siblings:
                problems.append(
                    f"{rel}:{number} imports sibling submodule {head!r}; this "
                    "submodule has no edges by design -- see DEPENDENCY_MAP.md"
                )
    return problems


def check_pycache(root: pathlib.Path) -> list[str]:
    """Byte-compiled files must not ship in a source tree (observation O-6)."""
    return [
        f"__pycache__ present at {p.relative_to(root)}; byte-compiled files are "
        "stale by definition and must not ship in a source archive"
        for p in sorted(root.rglob("__pycache__"))
        if p.is_dir()
    ]


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
        print(  # ruff: ignore[print]
            f"error: {tracker_path} not found", file=sys.stderr
        )
        return 2

    tracker = json.loads(tracker_path.read_text())
    actual = scan(root)

    if args.update:
        tracker["physical"] = actual
        tracker_path.write_text(json.dumps(tracker, indent=2) + "\n")
        print(  # ruff: ignore[print]
            f"updated {tracker_path.name}; regenerate TRACKER_PHYSICAL.md to match"
        )
        return 0

    drift = compare(tracker["physical"], actual)
    independence = check_independence(root)
    stale = check_pycache(root)

    for label, items in (
        ("DRIFT", drift),
        ("INDEPENDENCE", independence),
        ("STALE", stale),
    ):
        for item in items:
            print(f"{label}: {item}")  # ruff: ignore[print]

    if not (drift or independence or stale):
        totals = actual["totals"]
        print(  # ruff: ignore[print]
            f"{tracker['module']}: tracker matches the tree "
            f"({totals['source_files']} source / {totals['test_files']} test files, "
            f"{totals['source_loc']} / {totals['test_loc']} LOC)"
        )
        return 0

    print(  # ruff: ignore[print]
        "\nReconcile before continuing. This submodule compiles caller-supplied "
        "source; a drift here is a drift in a trust boundary."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
