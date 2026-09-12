\
#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
HERE=Path(__file__).resolve(); TOOL=HERE.parent/'check_contract.py'
spec=importlib.util.spec_from_file_location('impute_contract',TOOL); mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); a=ap.parse_args(argv)
    root=mod.discover_repo(HERE); m,fp,inv,me,re=mod.run_checks(root)
    ev=mod.load_json(root/m['maintenance_root']/'_maintenance/EVIDENCE.json')
    gates=ev.get('gates',{})
    integration=gates.get('real_integration',{}).get('status','UNAVAILABLE')
    release='PASS' if not me and not re and integration=='GREEN' and gates.get('release_matrix',{}).get('status')=='GREEN' else 'BLOCKED'
    payload={'subsystem':'scikitplot.impute','maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','integration_status':integration,'release_status':release,'findings':re,'evidence':{k:v.get('status') for k,v in gates.items()},'inventory':inv,'runtime_fingerprint':fp}
    print(json.dumps(payload,indent=2) if a.json else '\n'.join(f'{k}: {v}' for k,v in payload.items() if k.endswith('_status')))
    return 0 if not me else 2
if __name__=='__main__': raise SystemExit(main())
