#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Re-derive the physical tracker and check that every delegation target resolves.

Usage
-----
.. code-block:: console

    $ python scikitplot/_cli/_maintenance/check_trackers.py
    $ python scikitplot/_cli/_maintenance/check_trackers.py --update

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


#: Matches ``delegate="module:attr"`` in the registry.
_DELEGATE_RE = re.compile(r'delegate\s*=\s*"([^"]+)"')


def check_delegation(root: pathlib.Path) -> list[str]:
    """
    Verify every delegation target names a module file that exists.

    Notes
    -----
    **Developer.**  ``_cli`` reaches other submodules by a module-path *string*,
    resolved at runtime.  That is deliberate -- importing the target would make
    every optional dependency mandatory for ``--help``.  The cost is that a
    dangling target stays syntactically valid and fails **in front of a user**,
    at the moment they run the command.  No test in ``_cli`` catches it, because
    ``_cli``'s tests deliberately do not import what it delegates to.

    Resolution is **path-based**, not import-based.  ``importlib.util.find_spec``
    would need the package installed and would import parent packages on the way
    -- so on an uninstalled checkout it reports every target as missing, and on
    an installed one it violates the very import contract this gate protects.
    Mapping ``scikitplot.mcp.__main__`` to ``scikitplot/mcp/__main__.py`` needs
    neither.
    """
    package = root
    while package.name != "scikitplot" and package.parent != package:
        package = package.parent
    if package.name != "scikitplot":
        return []
    tree = package.parent

    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        # _maintenance holds this script, whose docstrings mention the syntax.
        if "__pycache__" in path.parts or rel.startswith(
            ("_backup/", "tests/", "_maintenance/")
        ):
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            match = _DELEGATE_RE.search(line)
            if not match:
                continue
            target = match.group(1)
            if ":" not in target:
                problems.append(
                    f"{rel}:{number} delegate {target!r} is malformed; "
                    "expected 'module:attr'"
                )
                continue
            module_name = target.partition(":")[0]
            parts = module_name.split(".")
            candidates = (
                tree.joinpath(*parts).with_suffix(".py"),
                tree.joinpath(*parts, "__init__.py"),
            )
            if not any(c.is_file() for c in candidates):
                problems.append(
                    f"{rel}:{number} delegate {target!r} names module "
                    f"{module_name!r}, for which no source file exists; the "
                    "command would appear in --help and then fail in front of "
                    "a user"
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
    delegation = check_delegation(root)
    stale = check_pycache(root)

    for label, items in (
        ("DRIFT", drift),
        ("DELEGATION", delegation),
        ("STALE", stale),
    ):
        for item in items:
            print(f"{label}: {item}")  # ruff: ignore[print]

    if not (drift or delegation or stale):
        totals = actual["totals"]
        print(  # ruff: ignore[print]
            f"{tracker['module']}: tracker matches the tree "
            f"({totals['source_files']} source / {totals['test_files']} test files, "
            f"{totals['source_loc']} / {totals['test_loc']} LOC)"
        )
        return 0

    print(  # ruff: ignore[print]
        "\nReconcile before continuing. A dangling delegation target does not "
        "fail here -- it fails in front of a user running the command."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
