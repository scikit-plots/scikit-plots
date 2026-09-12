#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
SRC_EXT={'.py','.pyx','.pxd','.pxi','.pyi','.c','.cc','.cpp','.cxx','.h','.hpp'}

def load_json(path: Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    try: obj=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as e: raise ContractError(f"cannot load {path}: {e}") from e
    return obj

def reject_command_surface(obj, where='metadata'):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS: raise ContractError(f'{where} contains unsupported executable field {k!r}')
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def discover_repo(start: Path) -> Path:
    p=start.resolve()
    for c in [p,*p.parents]:
        if (c/'scikitplot').is_dir() and (c/'maintenances').is_dir() and (c/'skills').is_dir(): return c
    raise ContractError('could not locate wide repository root containing scikitplot/, maintenances/, and skills/')

def safe_rel(value: str) -> str:
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value: raise ContractError(f'unsafe repository path {value!r}')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or '//' in value: raise ContractError(f'unsafe repository path {value!r}')
    return value

def safe_path(root: Path, value: str, exists=True) -> Path:
    value=safe_rel(value); p=root/value
    if exists and not p.exists(): raise ContractError(f'missing required path: {value}')
    q=p if p.exists() else p.parent
    try: q.resolve().relative_to(root.resolve())
    except ValueError as e: raise ContractError(f'path escapes repository: {value}') from e
    return p

def runtime_fingerprint(root: Path, runtime: str):
    base=safe_path(root,runtime)
    h=hashlib.sha256(); entries=[]
    for p in sorted(x for x in base.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        rel=p.relative_to(root).as_posix(); data=p.read_bytes(); h.update(rel.encode()+b'\0'+data+b'\0')
        entries.append({'path':rel,'bytes':len(data),'loc':data.count(b'\n')+(0 if not data or data.endswith(b'\n') else 1)})
    return h.hexdigest(),entries

def inventory(entries):
    src=[e for e in entries if PurePosixPath(e['path']).suffix in SRC_EXT and '/tests/' not in e['path']]
    tests=[e for e in entries if '/tests/' in e['path'] and PurePosixPath(e['path']).suffix in SRC_EXT]
    return {'runtime_files':len(entries),'runtime_bytes':sum(e['bytes'] for e in entries),'source_files':len(src),'source_loc':sum(e['loc'] for e in src),'test_files':len(tests),'test_loc':sum(e['loc'] for e in tests)}

def validate_manifest(root,m):
    reject_command_surface(m,'MAINTENANCE.json'); errors=[]
    if type(m.get('schema_version')) is not int or m.get('schema_version')!=3: errors.append('MAINTENANCE.json schema_version must be integer 3')
    required=['subsystem','runtime_root','maintenance_root','skill','ownership','runtime_contract','cross_module_boundaries','read_order']
    for k in required:
        if k not in m: errors.append(f'MAINTENANCE.json missing {k}')
    for k in ('runtime_root','maintenance_root','skill'):
        if k in m:
            try: safe_path(root,m[k])
            except ContractError as e: errors.append(str(e))
    rc=m.get('runtime_contract',{})
    if not isinstance(rc,dict) or not isinstance(rc.get('required_contracts'),list) or not rc.get('required_contracts'):
        errors.append('runtime_contract.required_contracts must be a non-empty list')
    return errors

def runtime_presence_checks(root,m,inv):
    errors=[]; rc=m['runtime_contract']
    if inv['source_files'] < rc.get('minimum_source_files',1): errors.append(f"runtime source incomplete: {inv['source_files']} source files < required floor {rc.get('minimum_source_files',1)}")
    if inv['test_files'] < rc.get('minimum_test_files',1): errors.append(f"runtime tests incomplete: {inv['test_files']} test files < required floor {rc.get('minimum_test_files',1)}")
    for c in rc['required_contracts']:
        p=root/c['path']
        if not p.is_file(): errors.append(f"missing contract module {c['name']}: {c['path']}")
    return errors

def contract_checks(root,m):
    errors=[]
    for c in m['runtime_contract']['required_contracts']:
        p=root/c['path']
        if not p.is_file(): continue
        text=p.read_text(encoding='utf-8',errors='replace')
        for symbol in c.get('symbols',[]):
            if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(symbol)+r'(?![A-Za-z0-9_])',text): errors.append(f"{c['path']}: contract {c['name']} missing symbol {symbol}")
    return errors

def plane_checks(root,m):
    errors=[]; runtime=safe_path(root,m['runtime_root'])
    for p in runtime.rglob('*'):
        if not p.is_file() or p.suffix not in {'.py','.pyx','.pxd','.pxi','.pyi'}: continue
        for i,line in enumerate(p.read_text(encoding='utf-8',errors='replace').splitlines(),1):
            s=line.strip()
            if s.startswith('#'): continue
            if re.search(r'(^|\s)(from|import)\s+(maintenances|skills)(\b|\.)',s): errors.append(f'{p.relative_to(root)}:{i}: runtime imports maintenance/skill plane')
    return errors

def handoff_checks(root,m):
    errors=[]
    for item in m['read_order']:
        try: safe_path(root,item)
        except ContractError as e: errors.append(str(e))
    hand=safe_path(root,m['maintenance_root']+'/_maintenance/FRESH_CHAT_HANDOFF.md'); text=hand.read_text(encoding='utf-8',errors='replace')
    for token in ('check_trackers.py','review_subsystem.py','STATE.json','EVIDENCE.json','runtime `FAIL`'):
        if token not in text: errors.append(f'fresh-chat handoff missing {token}')
    return errors

def hygiene_checks(root,m):
    errors=[]
    for rel in (m['maintenance_root'],str(PurePosixPath(m['skill']).parent)):
        base=safe_path(root,rel)
        bad=[p.relative_to(root).as_posix() for p in base.rglob('*') if p.is_file() and ('__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'})]
        if bad: errors.append(f'tooling residue under {rel}: {bad}')
    skill=safe_path(root,m['skill']); text=skill.read_text(encoding='utf-8',errors='replace')
    if not text.startswith('---\n') or '\nname:' not in text or '\ndescription:' not in text or len(text.splitlines())<20: errors.append('skill must be a substantive SKILL.md with YAML frontmatter')
    return errors

def evidence_checks(root,m,fp):
    errors=[]; e=load_json(safe_path(root,m['maintenance_root']+'/_maintenance/EVIDENCE.json')); reject_command_surface(e,'EVIDENCE.json')
    if e.get('runtime_fingerprint')!=fp: errors.append('EVIDENCE.json runtime_fingerprint is stale')
    gates=e.get('gates')
    if not isinstance(gates,dict): return errors+['EVIDENCE.json gates must be an object']
    allowed={'GREEN','RED','UNAVAILABLE'}
    for name,g in gates.items():
        if not isinstance(g,dict) or g.get('status') not in allowed: errors.append(f'evidence gate {name} has invalid status'); continue
        log=g.get('log'); sha=g.get('sha256')
        if log is None:
            if sha is not None: errors.append(f'evidence gate {name}: sha256 requires log')
        else:
            try: lp=safe_path(root,log)
            except ContractError as ex: errors.append(str(ex)); continue
            actual=hashlib.sha256(lp.read_bytes()).hexdigest()
            if actual!=sha: errors.append(f'evidence gate {name}: log hash mismatch')
    return errors

def review_profile(root,m):
    r=load_json(safe_path(root,m['maintenance_root']+'/REVIEW.json')); reject_command_surface(r,'REVIEW.json'); errors=[]
    if type(r.get('schema_version')) is not int or r.get('schema_version')!=1: errors.append('REVIEW.json schema_version must be integer 1')
    if set(r)-{'schema_version','subsystem','lanes','release_gates'}: errors.append('REVIEW.json contains unsupported extra fields')
    allowed={'runtime_presence','contracts','planes','handoff','inventory','hygiene','evidence'}; seen=set()
    for lane in r.get('lanes',[]):
        if not isinstance(lane,dict) or set(lane)-{'id','checks'}: errors.append('review lane contains unsupported fields'); continue
        for c in lane.get('checks',[]):
            if c not in allowed: errors.append(f'unknown review check {c}')
            seen.add(c)
    if seen != allowed: errors.append(f'review checks must be exactly {sorted(allowed)}')
    if not isinstance(r.get('release_gates'),list) or not r['release_gates']: errors.append('review release_gates must be non-empty list')
    if errors: raise ContractError('; '.join(errors))
    return r

def inspect(root: Path, refresh=False):
    maint=root/'maintenances'/'corpus'; manifest=load_json(maint/'MAINTENANCE.json'); manifest_errors=validate_manifest(root,manifest); review=review_profile(root,manifest)
    fp,entries=runtime_fingerprint(root,manifest['runtime_root']); inv=inventory(entries)
    errors={
      'manifest':manifest_errors,
      'runtime_presence':runtime_presence_checks(root,manifest,inv),
      'contracts':contract_checks(root,manifest),
      'planes':plane_checks(root,manifest),
      'handoff':handoff_checks(root,manifest),
      'hygiene':hygiene_checks(root,manifest),
      'evidence':evidence_checks(root,manifest,fp),
    }
    tracker=load_json(safe_path(root,manifest['maintenance_root']+'/_maintenance/TRACKER.json')); reject_command_surface(tracker,'TRACKER.json'); inv_errors=[]
    if tracker.get('runtime_fingerprint')!=fp: inv_errors.append('TRACKER.json runtime_fingerprint is stale')
    if tracker.get('inventory')!=inv: inv_errors.append(f'TRACKER.json inventory is stale: recorded={tracker.get("inventory")} actual={inv}')
    errors['inventory']=inv_errors
    maintenance_keys=['manifest','handoff','hygiene','evidence','inventory']
    runtime_keys=['runtime_presence','contracts','planes']
    maintenance='PASS' if not any(errors[k] for k in maintenance_keys) else 'FAIL'
    runtime='PASS' if not any(errors[k] for k in runtime_keys) else 'FAIL'
    if refresh:
        blockers=[x for k in maintenance_keys+runtime_keys for x in errors[k] if k not in {'evidence','inventory'}]
        if blockers: raise ContractError('refusing --update while maintenance/runtime checks fail: '+'; '.join(blockers))
        tracker.update({'schema_version':2,'subsystem':manifest['subsystem'],'runtime_root':manifest['runtime_root'],'runtime_fingerprint':fp,'inventory':inv})
        tp=safe_path(root,manifest['maintenance_root']+'/_maintenance/TRACKER.json'); tp.write_text(json.dumps(tracker,indent=2)+'\n',encoding='utf-8'); errors['inventory']=[]
    ev=load_json(safe_path(root,manifest['maintenance_root']+'/_maintenance/EVIDENCE.json'))
    release='PASS' if maintenance=='PASS' and runtime=='PASS' and all(ev.get('gates',{}).get(g,{}).get('status')=='GREEN' for g in review['release_gates']) else 'BLOCKED'
    lanes=[{'id':lane['id'],'checks':{c:errors[c] for c in lane['checks']}} for lane in review['lanes']]
    return {'subsystem':manifest['subsystem'],'maintenance_status':maintenance,'runtime_status':runtime,'release_status':release,'runtime_fingerprint':fp,'inventory':inv,'lanes':lanes,'errors':errors}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo'); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); ap.add_argument('--release',action='store_true'); ap.add_argument('--inventory',action='store_true'); a=ap.parse_args()
    try:
        root=Path(a.repo).resolve() if a.repo else discover_repo(Path(__file__)); result=inspect(root,refresh=a.update)
        out={'subsystem':result['subsystem'],'runtime_inventory':result['inventory'],'runtime_fingerprint':result['runtime_fingerprint']} if a.inventory else result
        if a.json: print(json.dumps(out,indent=2))
        else:
            print(f"{result['subsystem']}: maintenance={result['maintenance_status']} runtime={result['runtime_status']} release={result['release_status']}")
            for name,vals in result['errors'].items():
                for v in vals: print(f'{name.upper()}: {v}')
        if a.release: return 0 if result['release_status']=='PASS' else 1
        return 0 if result['maintenance_status']=='PASS' else 1
    except ContractError as e:
        print(f'contract error: {e}',file=sys.stderr); return 2
    except BrokenPipeError:
        return 1
if __name__=='__main__': raise SystemExit(main())
