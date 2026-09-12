"""Stable CLI exits: 0 pass, 1 failed/blocked/broken output, 2 invalid input."""
from __future__ import annotations
import argparse
import json
import os
import sys
import tokenize

from maintenance import (ContractError, architecture, inspect, load_contract,
                         repository_root)


def emit(text, code):
    try:
        print(text)
        sys.stdout.flush()
    except BrokenPipeError:
        # A closed consumer must not produce a second exception at shutdown.
        sys.stdout = open(os.devnull, "w")
        return 1
    return code


def main(*, allow_update=False, graph_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", "--repo", dest="repo_root", help="wide repository path; default: discover from script")
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    parser.add_argument("--inventory", action="store_true", help="include runtime inventory")
    if allow_update:
        parser.add_argument("--update", action="store_true", help="refresh derived files after structural checks")
    if not graph_only:
        parser.add_argument("--release", action="store_true", help="require current passing evidence for every release gate")
    args = parser.parse_args()
    try:
        root = repository_root(args.repo_root)
        if graph_only:
            manifest, _ = load_contract(root)
            graph, findings = architecture(root, manifest)
            return emit(json.dumps({"graph": graph, "findings": findings}, indent=2, sort_keys=True), int(any(findings.values())))
        report = inspect(root, refresh=getattr(args, "update", False))
        code = int(report["maintenance_status"] != "PASS" or (args.release and report["release_status"] != "PASS"))
        if args.json:
            return emit(json.dumps(report, indent=2, sort_keys=True), code)
        lines = [f"{report['subsystem']}: maintenance {report['maintenance_status']}; release {report['release_status']}"]
        for lane in report["lanes"]:
            lines.append(f"{lane['id']}: {lane['status']}")
            for name, errors in lane["checks"].items():
                lines.extend(f"  {name}: {error}" for error in errors)
        lines.extend(f"{name}: {status}" for name, status in sorted(report["gates"].items()))
        if args.inventory:
            lines.append("runtime inventory: " + json.dumps(report["runtime_inventory"], sort_keys=True))
        return emit("\n".join(lines), code)
    except (ContractError, OSError, ValueError, TypeError, SyntaxError, tokenize.TokenError, RecursionError) as exc:
        if args.json:
            return emit(json.dumps({"status": "ERROR", "detail": str(exc)}, sort_keys=True), 2)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
