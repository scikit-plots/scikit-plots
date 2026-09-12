#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
HERE = Path(__file__).resolve(); TOOL = HERE.parent/'check_contract.py'
spec = importlib.util.spec_from_file_location('visualkeras_contract', TOOL); c = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def review(root: Path):
    m, fp, inv, me, re = c.run_checks(root)
    ev = c.load_json(root/m['maintenance_root']/'_maintenance/EVIDENCE.json'); gates = ev.get('gates',{})
    integration = 'PASS' if gates.get('complete_package',{}).get('status') == 'GREEN' else 'UNAVAILABLE'
    release = 'PASS' if (not me and not re and integration == 'PASS' and gates.get('release_matrix',{}).get('status') == 'GREEN') else 'BLOCKED'
    return {'subsystem':m['subsystem'],'maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','integration_status':integration,'release_status':release,'findings':re,'evidence':{k:v.get('status') for k,v in gates.items()},'inventory':inv,'runtime_fingerprint':fp}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); a=ap.parse_args(argv)
    root=c.discover_repo(Path(__file__)); d=review(root); print(json.dumps(d,indent=2) if a.json else '\n'.join(f'{k}: {v}' for k,v in d.items())); return 0 if d['maintenance_status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
