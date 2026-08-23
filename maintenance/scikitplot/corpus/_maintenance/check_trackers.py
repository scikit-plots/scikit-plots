#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Re-derive the physical tracker from the tree and fail on drift.

Usage
-----
.. code-block:: console

    $ python scikitplot/corpus/_maintenance/check_trackers.py
    $ python scikitplot/corpus/_maintenance/check_trackers.py --update

Notes
-----
**Why this is a script and not a document.**  The physical inventory drifts
*silently*: a module grows, a subpackage doubles, a file acquires a fifth
responsibility, and nothing breaks.  A document describing it is stale the week
after it is written.  A gate is not.

The rule this encodes: **a document that describes what a script can check
should be replaced by the script.**  A document that records *why* cannot be,
and stays a document -- which is what ``TRACKER_LOGICAL.md`` is for.

Exit codes
----------
0
    Physical tracker matches the tree, and every logical contract names a
    module that exists.
1
    Drift detected, or a contract points at a missing module.
2
    Usage error.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

#: Relative tolerance on LOC before drift is reported.  Small edits are
#: expected; a subpackage moving by more than this is a structural change.
LOC_TOLERANCE = 0.10

#: Tripwires from TRACKER_PHYSICAL.md, checked here so they are enforced rather
#: than merely written down.
MAX_MODULE_LOC = 3500
MAX_ROOT_LOC_SHARE = 0.45
MIN_TEST_SOURCE_RATIO = 0.40


def corpus_root() -> pathlib.Path:
    """Return the ``scikitplot/corpus`` directory containing this script."""
    return pathlib.Path(__file__).resolve().parent.parent


def scan(root: pathlib.Path) -> dict:
    """Walk the tree and return the physical inventory."""
    files = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        kind = "test" if "/tests/" in "/" + rel else "source"
        files.append(
            {
                "path": rel,
                "kind": kind,
                "loc": len(path.read_text(errors="replace").splitlines()),
            }
        )

    subpackages: dict = collections.defaultdict(
        lambda: {"src": 0, "src_loc": 0, "test": 0, "test_loc": 0}
    )
    for entry in files:
        top = entry["path"].split("/")[0] if "/" in entry["path"] else "(root)"
        key = "src" if entry["kind"] == "source" else "test"
        subpackages[top][key] += 1
        subpackages[top][key + "_loc"] += entry["loc"]

    source = [f for f in files if f["kind"] == "source"]
    tests = [f for f in files if f["kind"] == "test"]
    return {
        "totals": {
            "source_files": len(source),
            "source_loc": sum(f["loc"] for f in source),
            "test_files": len(tests),
            "test_loc": sum(f["loc"] for f in tests),
        },
        "subpackages": {k: dict(v) for k, v in subpackages.items()},
        "largest_source": [
            {"loc": f["loc"], "path": f["path"]}
            for f in sorted(source, key=lambda f: -f["loc"])[:12]
        ],
    }


def compare(recorded: dict, actual: dict) -> list[str]:
    """Return one message per drift found."""
    problems: list[str] = []

    for name, was in recorded["totals"].items():
        now = actual["totals"][name]
        if was and abs(now - was) / was > LOC_TOLERANCE:
            problems.append(
                f"totals.{name}: recorded {was}, actual {now} "
                f"({(now - was) / was:+.0%})"
            )

    recorded_subs = set(recorded["subpackages"])
    actual_subs = set(actual["subpackages"])
    for added in sorted(actual_subs - recorded_subs):
        problems.append(
            f"new subpackage {added!r}: run the submodule review in "
            "SUBMODULE_STRUCTURE.md before merging"
        )
    for removed in sorted(recorded_subs - actual_subs):
        problems.append(f"subpackage {removed!r} disappeared")

    for name in sorted(recorded_subs & actual_subs):
        was = recorded["subpackages"][name].get("src_loc", 0)
        now = actual["subpackages"][name].get("src_loc", 0)
        if was and abs(now - was) / was > LOC_TOLERANCE:
            problems.append(
                f"subpackage {name!r} source LOC: recorded {was}, actual {now} "
                f"({(now - was) / was:+.0%})"
            )

    return problems


def tripwires(actual: dict) -> list[str]:
    """Return one message per tripwire crossed."""
    crossed: list[str] = []
    totals = actual["totals"]

    biggest = actual["largest_source"][0] if actual["largest_source"] else None
    if biggest and biggest["loc"] > MAX_MODULE_LOC:
        crossed.append(
            f"{biggest['path']} is {biggest['loc']} lines (> {MAX_MODULE_LOC}); "
            "single-responsibility is already lost at this size"
        )

    root_loc = actual["subpackages"].get("(root)", {}).get("src_loc", 0)
    if totals["source_loc"]:
        share = root_loc / totals["source_loc"]
        if share > MAX_ROOT_LOC_SHARE:
            crossed.append(
                f"root-level source is {share:.0%} of the module "
                f"(> {MAX_ROOT_LOC_SHARE:.0%}); structure is dissolving into a "
                "flat namespace"
            )

    if totals["source_loc"]:
        ratio = totals["test_loc"] / totals["source_loc"]
        if ratio < MIN_TEST_SOURCE_RATIO:
            crossed.append(
                f"test:source LOC ratio is {ratio:.2f} "
                f"(< {MIN_TEST_SOURCE_RATIO}); contracts are losing their pins"
            )

    for name, counts in sorted(actual["subpackages"].items()):
        if name in ("(root)", "_maintenance", "tests"):
            continue
        if counts.get("src", 0) and not counts.get("test", 0):
            crossed.append(f"subpackage {name!r} has source but no tests")

    return crossed


def check_logical(root: pathlib.Path, tracker: dict) -> list[str]:
    """Verify every logical contract names a module that exists."""
    problems: list[str] = []
    for name, spec in tracker.get("logical", {}).get("contracts", {}).items():
        module = spec.get("module", "")
        if module and not (root / module).is_file():
            problems.append(
                f"contract {name!r} names module {module!r}, which does not exist"
            )
        if not spec.get("invariant"):
            problems.append(f"contract {name!r} records no invariant")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the gate."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite TRACKER.json's physical section from the tree",
    )
    args = parser.parse_args(argv)

    root = corpus_root()
    tracker_path = root / "_maintenance" / "TRACKER.json"
    if not tracker_path.is_file():
        print(  # ruff: ignore[print]
            f"error: {tracker_path} not found", file=sys.stderr
        )
        return 2

    tracker = json.loads(tracker_path.read_text())
    actual = scan(root)

    if args.update:
        tracker["physical"].update(actual)
        tracker_path.write_text(json.dumps(tracker, indent=2) + "\n")
        print(  # ruff: ignore[print]
            f"updated {tracker_path.name}; regenerate TRACKER_PHYSICAL.md to match"
        )
        return 0

    drift = compare(tracker["physical"], actual)
    crossed = tripwires(actual)
    logical = check_logical(root, tracker)

    for label, items in (
        ("DRIFT", drift),
        ("TRIPWIRE", crossed),
        ("LOGICAL", logical),
    ):
        for item in items:
            print(f"{label}: {item}")  # ruff: ignore[print]

    if not (drift or crossed or logical):
        totals = actual["totals"]
        print(  # ruff: ignore[print]
            "physical tracker matches the tree "
            f"({totals['source_files']} source / {totals['test_files']} test files, "
            f"{totals['source_loc']} / {totals['test_loc']} LOC)"
        )
        return 0

    print(  # ruff: ignore[print]
        "\nReconcile before continuing. Structural drift is not a merge "
        "conflict to resolve later: it is the signal that TRACKER_PHYSICAL.md "
        "and SUBMODULE_STRUCTURE.md need reading."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
