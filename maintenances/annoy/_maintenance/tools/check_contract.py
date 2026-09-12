#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
SRC_EXT={'.py','.pyx','.pxd','.pxi','.pyi','.c','.cc','.cpp','.cxx','.h','.hpp','.in'}

def load_json(path: Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    try: return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as e: raise ContractError(f"cannot load {path}: {e}") from e

def reject_command_surface(obj,where='metadata'):
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

def safe_path(root: Path,value: str,exists=True) -> Path:
    value=safe_rel(value); p=root/value
    if exists and not p.exists(): raise ContractError(f'missing required path: {value}')
    q=p if p.exists() else p.parent
    try: q.resolve().relative_to(root.resolve())
    except ValueError as e: raise ContractError(f'path escapes repository: {value}') from e
    return p

def runtime_fingerprint(root: Path,runtime: str):
    base=safe_path(root,runtime); h=hashlib.sha256(); entries=[]
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
    if type(m.get('schema_version')) is not int or m.get('schema_version')!=4: errors.append('MAINTENANCE.json schema_version must be integer 4')
    req=['subsystem','runtime_root','maintenance_root','skill','ownership','compiled_layers','upstream','generation','public_contract','build','read_order']
    for k in req:
        if k not in m: errors.append(f'MAINTENANCE.json missing {k}')
    for k in ('runtime_root','maintenance_root','skill'):
        if k in m:
            try: safe_path(root,m[k])
            except ContractError as e: errors.append(str(e))
    layers=m.get('compiled_layers',{})
    if set(layers) != {'native_backend','cython_backend'}: errors.append('compiled_layers must define native_backend and cython_backend exactly')
    headers=m.get('upstream',{}).get('headers')
    if not isinstance(headers,list) or {h.get('name') for h in headers if isinstance(h,dict)} != {'annoylib.h','kissrandom.h','annoy_type_support.h'}:
        errors.append('upstream.headers must define annoylib.h, kissrandom.h, and annoy_type_support.h exactly')
    return errors

def active_extern_paths(text: str,header: str):
    hits=[]
    for i,line in enumerate(text.splitlines(),1):
        if line.lstrip().startswith('#'): continue
        m=re.search(r'cdef\s+extern\s+from\s+["\']([^"\']+)["\']',line)
        if m and header in m.group(1): hits.append((i,m.group(1)))
    return hits

def dependency_checks(root,m):
    errors=[]; runtime=safe_path(root,m['runtime_root'])
    for h in m['upstream']['headers']:
        hp=safe_path(root,h['path'],exists=False)
        if not hp.is_file(): errors.append(f'missing upstream header: {h["path"]}'); continue
        expected='../../cexternals/_annoy/src/'+h['name']
        for site in h['extern_sites']:
            sp=safe_path(root,site); hits=active_extern_paths(sp.read_text(encoding='utf-8',errors='replace'),h['name'])
            if not hits: errors.append(f'{site}: missing active cdef extern for {h["name"]}')
            elif not any(v==expected for _,v in hits): errors.append(f'{site}: {h["name"]} extern must use {expected!r}; found {[v for _,v in hits]}')
        dup=[p.relative_to(root).as_posix() for p in runtime.rglob(h['name']) if p.is_file()]
        if dup: errors.append(f'annoy runtime vendors upstream header {h["name"]}: {dup}')
    # Separate Python native-backend dependency.
    init=safe_path(root,m['public_contract']['entrypoint']).read_text(encoding='utf-8',errors='replace')
    base=safe_path(root,m['public_contract']['high_level_index']).read_text(encoding='utf-8',errors='replace')
    if not re.search(r'from\s+\.\.cexternals\._annoy\s+import',init): errors.append('public annoy __init__.py no longer imports native backend from ..cexternals._annoy')
    if not re.search(r'from\s+\.\.cexternals\._annoy\s+import\s+Annoy\b',base): errors.append('_base.py no longer imports native Annoy owner')
    if not re.search(r'class\s+Index\s*\([^)]*\bAnnoy\b',base,re.S): errors.append('_base.Index no longer inherits the native cexternals Annoy backend')
    return errors

def generation_checks(root,m):
    errors=[]; g=m['generation']; meson=safe_path(root,g['meson']); text=meson.read_text(encoding='utf-8',errors='replace')
    tool=safe_path(root,g['template_tool'],exists=False)
    if not tool.is_file(): errors.append(f'missing template generator required by Meson: {g["template_tool"]}')
    if '../../_build_utils/tempita.py' not in text: errors.append(f'{g["meson"]}: expected repository Tempita helper path not found')
    for item in g['templates']:
        inp=safe_path(root,item['input']); name=PurePosixPath(item['input']).name
        if f"input   : '{name}'" not in text and f"input: '{name}'" not in text: errors.append(f'{g["meson"]}: template input {name} not wired')
        if f"output  : '{item['output']}'" not in text and f"output: '{item['output']}'" not in text: errors.append(f'{g["meson"]}: generated output {item["output"]} not wired')
        src_output=inp.parent/item['output']
        if src_output.exists(): errors.append(f'build-generated output must not be checked into runtime source: {src_output.relative_to(root)}')
    legacy=safe_path(root,g['legacy_inactive_cpp'],exists=False)
    if legacy.exists():
        # It may remain as provenance, but must not silently become the Cython target source.
        if re.search(r"['\"]annoymodule\.cpp['\"]",text): errors.append(f'{g["meson"]}: inactive legacy annoymodule.cpp is now a build source; architecture review required')
    return errors

def public_checks(root,m):
    errors=[]; p=m['public_contract']; init=safe_path(root,p['entrypoint']).read_text(encoding='utf-8',errors='replace'); stub=safe_path(root,p['stub']).read_text(encoding='utf-8',errors='replace')
    for name in p['required_exports']:
        pat=r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])'
        if not re.search(pat,init): errors.append(f'{p["entrypoint"]}: missing required export {name}')
        if not re.search(pat,stub): errors.append(f'{p["stub"]}: missing required export {name}')
    cy=safe_path(root,p['cython_template']).read_text(encoding='utf-8',errors='replace')
    patterns={'FloatType':r'cpdef\s+enum\s+FloatType\b','supported_dtypes':r'def\s+supported_dtypes\s*\(','BaseIndex':r'cdef\s+class\s+BaseIndex\b','Index':r'cdef\s+class\s+Index\b'}
    for name in p['required_cython_symbols']:
        if not re.search(patterns[name],cy): errors.append(f'{p["cython_template"]}: missing Cython contract symbol {name}')
    cinit=safe_path(root,'scikitplot/annoy/_annoy/__init__.py').read_text(encoding='utf-8',errors='replace')
    if 'from .annoylib import *' not in cinit or not re.search(r'AnnoyIndex\s*=\s*Index',cinit): errors.append('Cython package init must export annoylib surface and AnnoyIndex = Index compatibility alias')
    return errors

def build_checks(root,m):
    errors=[]; b=m['build']; pkg=safe_path(root,b['package_meson']).read_text(encoding='utf-8',errors='replace'); ext=safe_path(root,b['extension_meson']).read_text(encoding='utf-8',errors='replace'); project=safe_path(root,b['project_meson']).read_text(encoding='utf-8',errors='replace')
    if b.get('minimum_cpp_standard')=='c++17' and not re.search(r"cpp_std\s*=\s*c\+\+17|['\"]cpp_std=c\+\+17['\"]",project): errors.append(f"{b['project_meson']}: active project must require C++17 for vendored Annoy headers")
    if "subdir('_annoy')" not in pkg: errors.append(f'{b["package_meson"]}: must include _annoy subdir')
    if 'py.extension_module' not in ext: errors.append(f'{b["extension_meson"]}: missing py.extension_module')
    if f"'{b['extension_name']}'" not in ext: errors.append(f'{b["extension_meson"]}: extension name {b["extension_name"]!r} not found')
    if 'cython_language=cpp' not in ext: errors.append(f'{b["extension_meson"]}: Cython extension must compile as C++')
    if b['install_subdir'] not in ext: errors.append(f'{b["extension_meson"]}: install subdir {b["install_subdir"]!r} not found')
    if '_annoylib_pyx' not in ext: errors.append(f'{b["extension_meson"]}: generated annoylib.pyx target is not the extension source')
    if '_annoylib_pxd' not in ext: errors.append(f'{b["extension_meson"]}: generated annoylib.pxd target is not part of Cython tree')
    for tr in b['test_roots']:
        d=safe_path(root,tr)
        if not any(x.is_file() and x.suffix in {'.py','.cpp','.cc','.cxx'} for x in d.rglob('*')): errors.append(f'{tr}: no executable test source found')
    return errors

def plane_checks(root,m):
    errors=[]; runtime=safe_path(root,m['runtime_root'])
    for p in runtime.rglob('*'):
        if not p.is_file() or p.suffix not in {'.py','.pyx','.pxd','.pxi','.pyi','.in'}: continue
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
    hand=safe_path(root,m['maintenance_root']+'/_maintenance/FRESH_CHAT_HANDOFF.md').read_text(encoding='utf-8',errors='replace')
    for token in ('check_trackers.py','review_subsystem.py','STATE.json','EVIDENCE.json','runtime `FAIL`','cexternals/_annoy','Cython'):
        if token not in hand: errors.append(f'fresh-chat handoff missing {token}')
    return errors

def hygiene_checks(root,m):
    errors=[]
    for rel in (m['maintenance_root'],str(PurePosixPath(m['skill']).parent)):
        base=safe_path(root,rel)
        bad=[p.relative_to(root).as_posix() for p in base.rglob('*') if p.is_file() and ('__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'})]
        if bad: errors.append(f'tooling residue under {rel}: {bad}')
    skill=safe_path(root,m['skill']); text=skill.read_text(encoding='utf-8',errors='replace')
    if not text.startswith('---\n') or '\nname:' not in text or '\ndescription:' not in text or len(text.splitlines())<30: errors.append('skill must be a substantive SKILL.md with YAML frontmatter')
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
            if hashlib.sha256(lp.read_bytes()).hexdigest()!=sha: errors.append(f'evidence gate {name}: log hash mismatch')
    return errors

def review_profile(root,m):
    r=load_json(safe_path(root,m['maintenance_root']+'/REVIEW.json')); reject_command_surface(r,'REVIEW.json'); errors=[]
    if type(r.get('schema_version')) is not int or r.get('schema_version')!=1: errors.append('REVIEW.json schema_version must be integer 1')
    if set(r)-{'schema_version','subsystem','lanes','release_gates'}: errors.append('REVIEW.json contains unsupported extra fields')
    allowed={'dependencies','generation','public_surface','build','planes','handoff','inventory','hygiene','evidence'}; seen=set()
    for lane in r.get('lanes',[]):
        if not isinstance(lane,dict) or set(lane)-{'id','checks'}: errors.append('review lane contains unsupported fields'); continue
        for c in lane.get('checks',[]):
            if c not in allowed: errors.append(f'unknown review check {c}')
            seen.add(c)
    if seen!=allowed: errors.append(f'review checks must be exactly {sorted(allowed)}')
    if not isinstance(r.get('release_gates'),list) or not r['release_gates']: errors.append('review release_gates must be non-empty list')
    if errors: raise ContractError('; '.join(errors))
    return r

def inspect(root: Path,refresh=False):
    maint=root/'maintenances'/'annoy'; manifest=load_json(maint/'MAINTENANCE.json'); manifest_errors=validate_manifest(root,manifest); review=review_profile(root,manifest)
    fp,entries=runtime_fingerprint(root,manifest['runtime_root']); inv=inventory(entries)
    errors={'manifest':manifest_errors,'dependencies':dependency_checks(root,manifest),'generation':generation_checks(root,manifest),'public_surface':public_checks(root,manifest),'build':build_checks(root,manifest),'planes':plane_checks(root,manifest),'handoff':handoff_checks(root,manifest),'hygiene':hygiene_checks(root,manifest),'evidence':evidence_checks(root,manifest,fp)}
    tracker=load_json(safe_path(root,manifest['maintenance_root']+'/_maintenance/TRACKER.json')); reject_command_surface(tracker,'TRACKER.json'); inv_errors=[]
    if tracker.get('runtime_fingerprint')!=fp: inv_errors.append('TRACKER.json runtime_fingerprint is stale')
    if tracker.get('inventory')!=inv: inv_errors.append(f'TRACKER.json inventory is stale: recorded={tracker.get("inventory")} actual={inv}')
    errors['inventory']=inv_errors
    maintenance_keys=['manifest','handoff','hygiene','evidence','inventory']
    runtime_keys=['dependencies','generation','public_surface','build','planes']
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
    ap=argparse.ArgumentParser(); ap.add_argument('--repo'); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); ap.add_argument('--release',action='store_true'); ap.add_argument('--inventory',action='store_true'); ap.add_argument('--dependencies',action='store_true'); a=ap.parse_args()
    try:
        root=Path(a.repo).resolve() if a.repo else discover_repo(Path(__file__)); result=inspect(root,refresh=a.update)
        if a.inventory: out={'subsystem':result['subsystem'],'runtime_inventory':result['inventory'],'runtime_fingerprint':result['runtime_fingerprint']}
        elif a.dependencies: out={'subsystem':result['subsystem'],'dependencies':result['errors']['dependencies']}
        else: out=result
        if a.json: print(json.dumps(out,indent=2))
        else:
            print(f"{result['subsystem']}: maintenance={result['maintenance_status']} runtime={result['runtime_status']} release={result['release_status']}")
            for name,vals in result['errors'].items():
                for v in vals: print(f'{name.upper()}: {v}')
        if a.release: return 0 if result['release_status']=='PASS' else 1
        return 0 if result['maintenance_status']=='PASS' else 1
    except ContractError as e:
        print(f'contract error: {e}',file=sys.stderr); return 2
    except BrokenPipeError: return 1
if __name__=='__main__': raise SystemExit(main())
