#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Turn JUnit reports of a test run into per-submodule duration estimates.

Notes
-----
**User notes.** The report job runs this over the JUnit files of every shard
and publishes the result as an artifact. To make the planner balance shards
with fresh figures, copy that file over ``.github/ci/test_durations.json``
and commit it::

    python .github/scripts/ci_test_durations.py reports/*.xml --output durations.json

**Developer notes.** The planner only needs relative sizes, so the estimate
is the plain sum of each test's recorded time per unit. A unit is found the
same way the planner finds it, from the dotted class name JUnit records; a
test whose unit is not a directory of this checkout is counted under
``unassigned`` and reported, never dropped silently.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# reports produced by our own test run
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_test_plan import PlanError, Planner, load_config  # noqa: E402

__all__ = ["collect", "main"]

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "ci" / "test_plan.json"


def collect(planner, reports):
    """
    Sum recorded test time per unit.

    Parameters
    ----------
    planner : ci_test_plan.Planner
        Supplies the unit a test belongs to.
    reports : iterable of str or pathlib.Path
        JUnit XML files.

    Returns
    -------
    units : dict
        Unit path to ``{"tests": n, "work_seconds": s}``.
    unassigned : int
        Tests whose unit could not be determined.

    Raises
    ------
    PlanError
        If a report cannot be read or is not JUnit XML.
    """
    totals, unassigned = {}, 0
    known = set(planner.units())
    for report in reports:
        try:
            root = ET.parse(str(report)).getroot()  # noqa: S314
        except (OSError, ET.ParseError) as exc:
            raise PlanError(f"cannot read the JUnit report {report}: {exc}") from exc
        for case in root.iter("testcase"):
            classname = case.get("classname") or ""
            try:
                seconds = max(float(case.get("time") or 0.0), 0.0)
            except ValueError as exc:
                raise PlanError(
                    f"{report}: a testcase has a time that is not a number",
                ) from exc
            kind, unit = planner.owner(classname.replace(".", "/") + "/x")
            if kind != "unit" or unit not in known:
                unassigned += 1
                continue
            entry = totals.setdefault(unit, {"tests": 0, "work_seconds": 0.0})
            entry["tests"] += 1
            entry["work_seconds"] += seconds
    for entry in totals.values():
        entry["work_seconds"] = round(entry["work_seconds"], 1)
    return dict(sorted(totals.items())), unassigned


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
        ``0`` on success, ``2`` when a report or the configuration is unusable.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0].strip(),
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="JUnit XML files",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repository checkout",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="planner configuration",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="file to write, or - for standard output",
    )
    parser.add_argument(
        "--source",
        default="",
        help="a line describing the run, stored in the file",
    )
    args = parser.parse_args(argv)
    try:
        planner = Planner(args.root, load_config(args.config))
        units, unassigned = collect(planner, args.reports)
    except PlanError as exc:
        sys.stderr.write(f"ci_test_durations: {exc}\n")
        return 2
    document = {
        "schema": 1,
        "source": (
            args.source or f"JUnit reports of {datetime.date.today().isoformat()}"
        ),
        "unassigned_tests": unassigned,
        "units": units,
    }
    text = json.dumps(document, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
