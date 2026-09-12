#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, runpy
from pathlib import Path
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path); ap.add_argument("--json",action="store_true"); a=ap.parse_args(argv)
    here=Path(__file__).resolve().parent; ns=runpy.run_path(str(here/"check_contract.py")); root=a.repo.resolve() if a.repo else ns["discover_repo"](Path.cwd()); out=ns["check"](root)
    review=json.loads((root/"maintenances/_brand/REVIEW.json").read_text(encoding="utf-8")); out["integration_status"]=review.get("integration_status","UNAVAILABLE"); out["declared_release_status"]=review.get("release_status","BLOCKED")
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if out["maintenance_status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
