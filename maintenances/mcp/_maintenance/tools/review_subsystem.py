#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

TOOL = Path(__file__).resolve().with_name("check_contract.py")
spec = importlib.util.spec_from_file_location("mcp_contract", TOOL)
contract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(contract)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--repo"); p.add_argument("--json", action="store_true"); a=p.parse_args()
    root=Path(a.repo).resolve() if a.repo else contract.discover_repo(Path(__file__))
    result=contract.inspect(root)
    review={
        "subsystem": result["subsystem"],
        "maintenance_status": result["maintenance_status"],
        "runtime_status": result["runtime_status"],
        "release_status": result["release_status"],
        "lanes": result["lanes"],
    }
    if a.json: print(json.dumps(review, indent=2))
    else:
        print(f"{review['subsystem']}: maintenance={review['maintenance_status']} runtime={review['runtime_status']} release={review['release_status']}")
        for lane in review["lanes"]:
            count=sum(len(v) for v in lane["checks"].values())
            print(f"{lane['id']}: {'PASS' if count == 0 else 'FAIL'} ({count} issue(s))")
    return 0 if result["maintenance_status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
