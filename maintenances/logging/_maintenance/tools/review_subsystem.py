#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path


def load_checker(here: Path):
    path = here / "check_contract.py"
    spec = importlib.util.spec_from_file_location("logging_contract", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)
    checker = load_checker(Path(__file__).resolve().parent)
    root = args.repo.resolve() if args.repo else checker.discover_repo(Path.cwd())
    out = checker.payload(root)
    report = {
        "schema_version": 1,
        "subsystem": "scikitplot.logging",
        "maintenance_status": out["maintenance_status"],
        "runtime_status": out["runtime_status"],
        "integration_status": "UNAVAILABLE",
        "release_status": "BLOCKED",
        "maintenance_errors": out["maintenance_errors"],
        "runtime_findings": out["runtime_findings"],
        "inventory": out["inventory"],
        "runtime_tree_fingerprint": out["runtime_tree_fingerprint"],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if out["maintenance_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
