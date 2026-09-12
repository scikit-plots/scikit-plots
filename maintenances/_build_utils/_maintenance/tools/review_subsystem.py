#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path

def load_checker(here):
    p=here/'check_contract.py'; spec=importlib.util.spec_from_file_location('_build_utils_contract',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path); ap.add_argument('--write',type=Path); a=ap.parse_args(argv)
    c=load_checker(Path(__file__).resolve().parent)
    root=a.repo.resolve() if a.repo else c.discover_repo(Path.cwd())
    out=c.check(root)
    payload={'schema_version':1,'subsystem':'scikitplot._build_utils','maintenance_status':out['maintenance_status'],'runtime_status':out['runtime_status'],'release_status':out['release_status'],'maintenance_errors':out['maintenance_errors'],'runtime_errors':out['runtime_errors'],'inventory':out['inventory']}
    text=json.dumps(payload,indent=2,sort_keys=True)+'\n'
    if a.write: a.write.write_text(text,encoding='utf-8')
    print(text,end=''); return 0 if out['maintenance_status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
