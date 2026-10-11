#!/usr/bin/env python3
"""Reviewer for ``scikitplot.cleanprompt``.

Reads the maintenance plane and reports the review posture: which findings are
recorded, which are closed and with what gate, and which verification lanes are
green. It makes no claim the documents do not support.

Notes
-----
**Developer notes.** The reviewer is deliberately separate from the checker.
The checker asks "does the code still have the shape the documents claim?"; the
reviewer asks "what do the documents currently claim?". Conflating the two lets
a green checker be read as a green release.

The reverse confusion is just as real: the documents' *recorded* state can
describe a tree that no longer exists. The internal review of 2026-10-10 found
the checker failing on a stale evidence fingerprint while this reviewer
printed maintenance and runtime ``PASS`` and exited 0. So the report also says
whether the recorded evidence describes the tree on disk
(``evidence_current``), and a stale record exits 3 — never 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_contract import (  # noqa: E402
    MAINTENANCE,
    discover_repo,
    read_json,
    tree_fingerprint,
)

#: Exit status when the recorded state describes another tree.
EXIT_STALE = 3


def review(root: Path) -> dict:
    """Return the review report."""
    base = root.joinpath(*MAINTENANCE)
    findings = read_json(base / "REVIEW.json")
    state = read_json(base / "_maintenance" / "STATE.json")
    evidence = read_json(base / "_maintenance" / "EVIDENCE.json")

    by_status: dict = {}
    for finding in findings.get("findings", []):
        by_status.setdefault(finding.get("status", "unknown"), []).append(
            finding.get("id")
        )
    lanes: dict = {}
    for lane in evidence.get("lanes", []):
        lanes.setdefault(lane.get("status", "UNKNOWN"), []).append(lane.get("id"))

    recorded = evidence.get("runtime_tree_fingerprint")
    current = tree_fingerprint(root)
    evidence_current = recorded is not None and recorded == current

    open_findings = [
        identifier
        for status, ids in by_status.items()
        if status != "closed"
        for identifier in ids
    ]
    return {
        "subsystem": findings.get("subsystem"),
        "updated_at": findings.get("updated_at"),
        "review_status": findings.get("status"),
        "findings_by_status": {k: sorted(v) for k, v in sorted(by_status.items())},
        "open_findings": sorted(open_findings),
        "investigated_and_rejected": [
            item.get("id") for item in findings.get("investigated_and_rejected", [])
        ],
        "lanes_by_status": {k: sorted(v) for k, v in sorted(lanes.items())},
        "unavailable_lanes": sorted(lanes.get("UNAVAILABLE", [])),
        "state": {
            "maintenance_status": state.get("maintenance_status"),
            "runtime_status": state.get("runtime_status"),
            "release_status": state.get("release_status"),
        },
        # The state above is what the documents recorded. It describes the
        # tree on disk only when the fingerprints agree.
        "evidence_current": evidence_current,
        "recorded_fingerprint": recorded,
        "current_fingerprint": current,
        "next_actions": state.get("next_actions", []),
    }


def main(argv=None) -> int:
    """Run the reviewer."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, help="wide checkout root")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)
    root = args.repo.resolve() if args.repo else discover_repo(Path.cwd())
    report = review(root)

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(
            "{0}  review={1}  release={2}\n".format(
                report["subsystem"],
                report["review_status"],
                report["state"]["release_status"],
            )
        )
        sys.stdout.write(
            "  closed: {0}  open: {1}\n".format(
                len(report["findings_by_status"].get("closed", [])),
                len(report["open_findings"]),
            )
        )
        for lane in report["unavailable_lanes"]:
            sys.stdout.write("  UNAVAILABLE lane: {0}\n".format(lane))
        if not report["evidence_current"]:
            sys.stdout.write(
                "  STALE: the recorded state describes another tree; re-run the "
                "verification lanes before relying on it\n"
            )
    if report["open_findings"]:
        return 2
    return 0 if report["evidence_current"] else EXIT_STALE


if __name__ == "__main__":
    raise SystemExit(main())
