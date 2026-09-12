#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cli_contract", HERE / "check_contract.py")
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(contract)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    root = Path(args.repo).resolve() if args.repo else contract.discover_repo(Path(__file__))
    result = contract.inspect(root)
    state = contract.load_json(root / "maintenances/_cli/_maintenance/STATE.json")
    out = {
        "subsystem": result["subsystem"],
        "maintenance_status": result["maintenance_status"],
        "runtime_status": result["runtime_status"],
        "release_status": result["release_status"],
        "findings": state.get("findings", []),
        "lanes": result["lanes"],
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{out['subsystem']}: maintenance={out['maintenance_status']} runtime={out['runtime_status']} release={out['release_status']}")
        for finding in out["findings"]:
            print(f"{finding['id']} [{finding['status']}]: {finding['summary']}")
    return 0 if result["maintenance_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
