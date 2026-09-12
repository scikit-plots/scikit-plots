#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

def load_checker():
    p=Path(__file__).resolve().with_name('check_contract.py')
    spec=importlib.util.spec_from_file_location('cython_maintenance_contract',p)
    mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod); return mod

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); args=ap.parse_args(argv)
    c=load_checker(); root=c.discover_repo(Path(__file__)); m,fp,inv,me,re=c.run_checks(root)
    ev=c.load_json(root/m['maintenance_root']/'_maintenance/EVIDENCE.json')
    gates=ev.get('gates',{})
    release_green=all(gates.get(g,{}).get('status')=='GREEN' for g in c.load_json(root/m['maintenance_root']/'REVIEW.json').get('release_gates',[]))
    payload={'subsystem':'scikitplot.cython','maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','release_status':'PASS' if release_green and not me and not re else 'BLOCKED','runtime_fingerprint':fp,'inventory':inv,'maintenance_errors':me,'runtime_errors':re,'evidence_gates':{k:v.get('status') for k,v in gates.items()}}
    if args.json: print(json.dumps(payload,indent=2))
    else:
        for k in ('maintenance_status','runtime_status','release_status'): print(f"{k}: {payload[k]}")
    return 0 if payload['maintenance_status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())
