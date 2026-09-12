#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, re, sys
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass

def load_json(path: Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    try: return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as e: raise ContractError(f"cannot load {path}: {e}") from e

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
    # Existing path components must not escape by symlink.
    q=p if p.exists() else p.parent
    try: q.resolve().relative_to(root.resolve())
    except ValueError as e: raise ContractError(f'path escapes repository: {value}') from e
    return p

def runtime_fingerprint(root: Path, runtime: str):
    base=safe_path(root,runtime)
    h=hashlib.sha256(); entries=[]
    for p in sorted(x for x in base.rglob('*') if x.is_file() and '__pycache__' not in x.parts):
        rel=p.relative_to(root).as_posix(); data=p.read_bytes()
        h.update(rel.encode()+b'\0'+data+b'\0')
        entries.append({'path':rel,'bytes':len(data),'loc':data.count(b'\n')+(0 if not data or data.endswith(b'\n') else 1)})
    return h.hexdigest(),entries

def inventory(entries):
    src_ext={'.py','.pyx','.pxd','.pxi','.pyi','.c','.cc','.cpp','.cxx','.h','.hpp'}
    src=[e for e in entries if PurePosixPath(e['path']).suffix in src_ext and '/tests/' not in e['path']]
    tests=[e for e in entries if '/tests/' in e['path'] and PurePosixPath(e['path']).suffix in src_ext]
    return {'runtime_files':len(entries),'runtime_bytes':sum(e['bytes'] for e in entries),'source_files':len(src),'source_loc':sum(e['loc'] for e in src),'test_files':len(tests),'test_loc':sum(e['loc'] for e in tests)}

def validate_manifest(root: Path, m):
    errors=[]
    if type(m.get('schema_version')) is not int or m.get('schema_version')!=3: errors.append('MAINTENANCE.json schema_version must be integer 3')
    required=['subsystem','runtime_root','maintenance_root','skill','upstream','public_contract','build','tests','read_order']
    for k in required:
        if k not in m: errors.append(f'MAINTENANCE.json missing {k}')
    for k in ('runtime_root','maintenance_root','skill'):
        if k in m:
            try: safe_path(root,m[k])
            except ContractError as e: errors.append(str(e))
    up=m.get('upstream',{})
    if not isinstance(up,dict) or not isinstance(up.get('headers'),list) or len(up.get('headers',[]))!=1: errors.append('upstream.headers must contain exactly one header contract')
    else:
        h=up['headers'][0]
        for k in ('name','path','extern_sites'):
            if k not in h: errors.append(f'upstream header missing {k}')
    return errors

def dependency_checks(root,m):
    errors=[]; h=m['upstream']['headers'][0]
    header=safe_path(root,h['path']) if not errors else None
    expected='../../cexternals/_annoy/src/'+h['name']
    for site in h['extern_sites']:
        p=safe_path(root,site); text=p.read_text(encoding='utf-8',errors='replace')
        # Only active cdef extern statements count; comments/docstrings are ignored conservatively line-by-line.
        hits=[]
        for i,line in enumerate(text.splitlines(),1):
            stripped=line.lstrip()
            if stripped.startswith('#'): continue
            match=re.search(r'cdef\s+extern\s+from\s+[\"\']([^\"\']+)[\"\']', line)
            if match and h['name'] in match.group(1): hits.append((i,match.group(1)))
        if not hits: errors.append(f'{site}: missing active extern for {h["name"]}')
        elif not any(path == expected for _,path in hits): errors.append(f'{site}: {h["name"]} extern must use {expected!r}; found {[path for _,path in hits]}')
    runtime=safe_path(root,m['runtime_root'])
    duplicates=[p.relative_to(root).as_posix() for p in runtime.rglob(h['name']) if p.is_file()]
    if duplicates: errors.append(f'runtime vendors upstream header {h["name"]}: {duplicates}')
    if not header.is_file(): errors.append(f'upstream header is not a file: {h["path"]}')
    return errors

def public_checks(root,m):
    errors=[]; p=m['public_contract']; impl=safe_path(root,p['implementation']); stub=safe_path(root,p['stub'])
    it=impl.read_text(encoding='utf-8',errors='replace'); st=stub.read_text(encoding='utf-8',errors='replace')
    for name in p.get('required_symbols',[]):
        # Presence check is intentionally lexical across Cython+stub: exact API parity remains a compiled/runtime gate.
        if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',it): errors.append(f'implementation missing required symbol {name}')
        if not re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',st): errors.append(f'stub missing required symbol {name}')
    return errors

def build_checks(root,m):
    errors=[]; b=m['build']; p=safe_path(root,b['meson']); text=p.read_text(encoding='utf-8',errors='replace')
    if 'py.extension_module' not in text: errors.append(f'{b["meson"]}: missing py.extension_module')
    if b['extension_name'] not in text: errors.append(f'{b["meson"]}: extension name {b["extension_name"]!r} not found')
    if 'cython_language=cpp' not in text: errors.append(f'{b["meson"]}: Cython extension must compile as C++')
    if b.get('install_subdir') and b['install_subdir'] not in text: errors.append(f'{b["meson"]}: install subdir {b["install_subdir"]!r} not found')
    for test in m['tests']:
        safe_path(root,test)
    return errors

def plane_checks(root,m):
    errors=[]; runtime=safe_path(root,m['runtime_root'])
    forbidden=('maintenances','skills')
    for p in runtime.rglob('*'):
        if not p.is_file() or p.suffix not in {'.py','.pyx','.pxd','.pxi','.pyi'}: continue
        text=p.read_text(encoding='utf-8',errors='replace')
        for i,line in enumerate(text.splitlines(),1):
            s=line.strip()
            if s.startswith('#'): continue
            if re.search(r'(^|\s)(from|import)\s+(maintenances|skills)(\b|\.)',s): errors.append(f'{p.relative_to(root)}:{i}: runtime imports maintenance/skill plane')
    return errors

def handoff_checks(root,m):
    errors=[]
    for item in m['read_order']:
        try: safe_path(root,item)
        except ContractError as e: errors.append(str(e))
    hand=safe_path(root,m['maintenance_root']+'/_maintenance/FRESH_CHAT_HANDOFF.md')
    text=hand.read_text(encoding='utf-8',errors='replace')
    for token in ('check_trackers.py','review_subsystem.py','STATE.json','EVIDENCE.json'):
        if token not in text: errors.append(f'fresh-chat handoff missing {token}')
    return errors

def hygiene_checks(root,m):
    errors=[]
    for rel in (m['maintenance_root'],str(PurePosixPath(m['skill']).parent)):
        base=safe_path(root,rel)
        bad=[p.relative_to(root).as_posix() for p in base.rglob('*') if p.is_file() and ('__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'})]
        if bad: errors.append(f'tooling residue under {rel}: {bad}')
    skill=safe_path(root,m['skill']); text=skill.read_text(encoding='utf-8',errors='replace')
    if not text.startswith('---\n') or '\nname:' not in text or '\ndescription:' not in text: errors.append('skill must have YAML frontmatter with name and description')
    return errors

def evidence_checks(root,m,fp):
    errors=[]; p=safe_path(root,m['maintenance_root']+'/_maintenance/EVIDENCE.json'); e=load_json(p)
    if e.get('runtime_fingerprint')!=fp: errors.append('EVIDENCE.json runtime_fingerprint is stale')
    gates=e.get('gates')
    if not isinstance(gates,dict): return errors+['EVIDENCE.json gates must be an object']
    allowed={'GREEN','RED','UNAVAILABLE'}
    for name,g in gates.items():
        if not isinstance(g,dict) or g.get('status') not in allowed: errors.append(f'evidence gate {name} has invalid status') ; continue
        log=g.get('log'); sha=g.get('sha256')
        if log is None:
            if sha is not None: errors.append(f'evidence gate {name}: sha256 requires log')
        else:
            lp=safe_path(root,log); actual=hashlib.sha256(lp.read_bytes()).hexdigest()
            if actual!=sha: errors.append(f'evidence gate {name}: log hash mismatch')
    return errors

def review_profile(root,m):
    p=safe_path(root,m['maintenance_root']+'/REVIEW.json'); r=load_json(p); errors=[]
    if type(r.get('schema_version')) is not int or r.get('schema_version')!=1: errors.append('REVIEW.json schema_version must be integer 1')
    if set(r)-{'schema_version','subsystem','lanes','release_gates'}: errors.append('REVIEW.json contains unsupported executable/extra fields')
    allowed={'dependencies','public_surface','build','planes','handoff','inventory','hygiene','evidence'}
    seen=set()
    for lane in r.get('lanes',[]):
        if not isinstance(lane,dict) or set(lane)-{'id','checks'}: errors.append('review lane contains unsupported fields'); continue
        for c in lane.get('checks',[]):
            if c not in allowed: errors.append(f'unknown review check {c}')
            seen.add(c)
    if seen != allowed: errors.append(f'review checks must be exactly {sorted(allowed)}')
    if not isinstance(r.get('release_gates'),list) or not r['release_gates']: errors.append('review release_gates must be non-empty list')
    if errors: raise ContractError('; '.join(errors))
    return r,[]

def inspect(root: Path, refresh=False):
    # Resolve the module name from this tool, but load authority from the selected repository.
    module_name=Path(__file__).resolve().parents[2].name
    maint=root/'maintenances'/module_name
    manifest=load_json(maint/'MAINTENANCE.json')
    errors={'manifest':validate_manifest(root,manifest)}
    review,rev_err=review_profile(root,manifest); errors['manifest']+=rev_err
    fp,entries=runtime_fingerprint(root,manifest['runtime_root']); inv=inventory(entries)
    errors.update({
      'dependencies':dependency_checks(root,manifest),
      'public_surface':public_checks(root,manifest),
      'build':build_checks(root,manifest),
      'planes':plane_checks(root,manifest),
      'handoff':handoff_checks(root,manifest),
      'hygiene':hygiene_checks(root,manifest),
      'evidence':evidence_checks(root,manifest,fp),
    })
    tracker=load_json(safe_path(root,manifest['maintenance_root']+'/_maintenance/TRACKER.json'))
    inv_errors=[]
    if tracker.get('runtime_fingerprint')!=fp: inv_errors.append('TRACKER.json runtime_fingerprint is stale')
    if tracker.get('inventory')!=inv: inv_errors.append(f'TRACKER.json inventory is stale: recorded={tracker.get("inventory")} actual={inv}')
    errors['inventory']=inv_errors
    architectural=['manifest','dependencies','public_surface','build','planes','handoff','hygiene']
    if refresh:
        blockers=[x for k in architectural for x in errors[k]]
        if blockers: raise ContractError('refusing --update while architecture/hygiene checks fail: '+ '; '.join(blockers))
        tracker.update({'schema_version':2,'subsystem':manifest['subsystem'],'runtime_root':manifest['runtime_root'],'runtime_fingerprint':fp,'inventory':inv})
        tp=safe_path(root,manifest['maintenance_root']+'/_maintenance/TRACKER.json'); tp.write_text(json.dumps(tracker,indent=2)+'\n',encoding='utf-8')
        errors['inventory']=[]
    lanes=[]
    for lane in review['lanes']:
        lanes.append({'id':lane['id'],'checks':{c:errors[c] for c in lane['checks']}})
    status='PASS' if not any(errors.values()) else 'FAIL'
    ev=load_json(safe_path(root,manifest['maintenance_root']+'/_maintenance/EVIDENCE.json'))
    release='PASS' if all(ev.get('gates',{}).get(g,{}).get('status')=='GREEN' for g in review['release_gates']) else 'BLOCKED'
    return {'subsystem':manifest['subsystem'],'maintenance_status':status,'release_status':release,'runtime_fingerprint':fp,'inventory':inv,'lanes':lanes,'errors':errors}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo'); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); ap.add_argument('--release',action='store_true'); ap.add_argument('--inventory',action='store_true'); ap.add_argument('--dependencies',action='store_true')
    a=ap.parse_args()
    try:
        root=Path(a.repo).resolve() if a.repo else discover_repo(Path(__file__))
        result=inspect(root,refresh=a.update)
        if a.inventory:
            out={'subsystem':result['subsystem'],'runtime_inventory':result['inventory'],'runtime_fingerprint':result['runtime_fingerprint']}
        elif a.dependencies:
            out={'subsystem':result['subsystem'],'dependencies':result['errors']['dependencies']}
        else: out=result
        if a.json: print(json.dumps(out,indent=2))
        else:
            print(f"{result['subsystem']}: maintenance={result['maintenance_status']} release={result['release_status']}")
            for name,vals in result['errors'].items():
                for v in vals: print(f'{name.upper()}: {v}')
        if a.release: return 0 if result['maintenance_status']=='PASS' and result['release_status']=='PASS' else 1
        return 0 if result['maintenance_status']=='PASS' else 1
    except ContractError as e:
        print(f'contract error: {e}',file=sys.stderr); return 2
    except BrokenPipeError:
        try: sys.stdout.close()
        except Exception: pass
        return 1
if __name__=='__main__': raise SystemExit(main())
