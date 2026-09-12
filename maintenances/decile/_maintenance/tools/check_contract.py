#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from pathlib import Path, PurePosixPath
class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
TRACKED_SUFFIXES={'.py','.pyi','.json','.md','.toml','.yaml','.typed'}
ALLOWED_REVIEW_CHECKS={'runtime_presence','api_surface','kds_argument_contract','legacy_rng_contract','planes','handoff','inventory','evidence','tests','integration','release'}

def load_json(path:Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f'duplicate JSON key {k!r} in {path}')
            out[k]=v
        return out
    try: return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as exc: raise ContractError(f'cannot load {path}: {exc}') from exc

def reject_command_surface(obj,where='metadata'):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS: raise ContractError(f'{where} contains unsupported executable field {k!r}')
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def discover_repo(start:Path)->Path:
    r=start.resolve()
    for c in [r,*r.parents]:
        if all((c/n).is_dir() for n in ('scikitplot','maintenances','skills')): return c
    raise ContractError('could not locate wide repository root containing scikitplot/, maintenances/, and skills/')

def safe_rel(value:str)->str:
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value: raise ContractError(f'unsafe repository path {value!r}')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or '//' in value: raise ContractError(f'unsafe repository path {value!r}')
    return value

def parse_python(path:Path):
    try: return ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    except (OSError,SyntaxError) as exc: raise ContractError(f'cannot parse {path}: {exc}') from exc

def top_symbols(path:Path):
    out=set()
    for n in parse_python(path).body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): out.add(n.name)
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            ts=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in ts:
                if isinstance(t,ast.Name): out.add(t.id)
    return out

def imports(path:Path):
    out=[]
    for n in parse_python(path).body:
        if isinstance(n,ast.Import): out += [(n.lineno,a.name) for a in n.names]
        elif isinstance(n,ast.ImportFrom): out.append((n.lineno,'.'*n.level+(n.module or '')))
    return out

def tracked(root:Path,rr:str):
    base=root/rr
    return [p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and (p.suffix in TRACKED_SUFFIXES or p.name=='py.typed')]

def fingerprint(root:Path,rr:str):
    base=root/rr; h=hashlib.sha256(); entries=[]
    for p in tracked(root,rr):
        rel=p.relative_to(base).as_posix(); d=hashlib.sha256(p.read_bytes()).hexdigest(); h.update(rel.encode()+b'\0'+d.encode()+b'\n'); entries.append(rel)
    return h.hexdigest(),entries

def inventory(entries):
    def is_test(x): return 'tests' in PurePosixPath(x).parts
    return {'tracked_files':len(entries),'production_python_files':sum(x.endswith('.py') and not is_test(x) for x in entries),'test_python_files':sum(x.endswith('.py') and is_test(x) and PurePosixPath(x).name.startswith('test') for x in entries),'test_support_python_files':sum(x.endswith('.py') and is_test(x) and not PurePosixPath(x).name.startswith('test') for x in entries)}

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,'MAINTENANCE.json')
    except ContractError as x: e.append(str(x))
    if m.get('schema_version')!=4: e.append('MAINTENANCE.json schema_version must be integer 4')
    if m.get('subsystem')!='scikitplot.decile': e.append('MAINTENANCE.json subsystem must be scikitplot.decile')
    for k in ('runtime_root','maintenance_root','skill'):
        try:
            p=root/safe_rel(m[k])
            if not p.exists(): e.append(f'required repository path does not exist: {m[k]}')
        except (KeyError,ContractError) as x: e.append(str(x))
    return e

def runtime_presence(root,m,inv):
    e=[]
    for rel in m['runtime_contract']['required_files']:
        if not (root/safe_rel(rel)).is_file(): e.append(f'missing required runtime file {rel}')
    if inv['tracked_files']!=12: e.append('decile tracked runtime/test surface changed; review ownership tiers before blessing inventory')
    if inv['production_python_files']!=6: e.append('decile production Python surface changed; expected current + KDS + legacy modules')
    if inv['test_python_files']!=3: e.append('decile test surface changed; expected one focused test module per owned surface')
    for rel in m['runtime_contract']['test_files']:
        if not (root/rel).is_file(): e.append(f'missing focused regression test {rel}')
    return e

def api_surface(root,m):
    e=[]
    for rel,syms in m['runtime_contract']['required_symbols'].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f'{rel} is missing required top-level symbol {s}')
    init=(root/m['runtime_root']/'__init__.py').read_text(encoding='utf-8')
    for marker in ('from . import kds','from . import modelplotpy','from . import _decile_modelplotpy as _dmpy','from ._decile_modelplotpy import *','__all__ += _dmpy.__all__'):
        if marker not in init: e.append(f'top-level API aggregation marker {marker!r} disappeared')
    return e

def _func(tree,name):
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name: return n
    return None

def _calls(fn,target):
    if fn is None: return []
    out=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Call):
            f=n.func
            if isinstance(f,ast.Name) and f.id==target: out.append(n)
            elif isinstance(f,ast.Attribute) and f.attr==target: out.append(n)
    return out

def kds_argument_contract(root,m):
    p=root/m['runtime_root']/'kds/_kds.py'; e=[]
    if not p.is_file(): return e
    tree=parse_python(p)
    for name in ('plot_lift','plot_lift_decile_wise','plot_cumulative_gain','plot_ks_statistic'):
        fn=_func(tree,name); calls=_calls(fn,'decile_table')
        if not calls: e.append(f'KDS {name} no longer calls decile_table'); continue
        kws={k.arg for k in calls[0].keywords if k.arg}
        missing={'class_index','pos_label'}-kws
        if missing: e.append(f'KDS {name} drops decile_table selection arguments: missing {sorted(missing)}')
    fn=_func(tree,'report'); calls=_calls(fn,'decile_table')
    if not calls: e.append('KDS report no longer calls decile_table')
    else:
        kws={k.arg for k in calls[0].keywords if k.arg}
        missing={'class_index','pos_label','digits','feature_infos'}-kws
        if missing: e.append(f'KDS report drops/renames decile_table arguments: missing {sorted(missing)}')
        if 'round_decimal' in kws: e.append('KDS report passes digits as round_decimal, which decile_table does not consume')
    for child in ('plot_lift','plot_lift_decile_wise','plot_cumulative_gain','plot_ks_statistic'):
        calls=_calls(fn,child)
        if not calls: e.append(f'KDS report no longer composes {child}'); continue
        kws={k.arg for k in calls[0].keywords if k.arg}
        missing={'class_index','pos_label'}-kws
        if missing: e.append(f'KDS report drops selection arguments when calling {child}: missing {sorted(missing)}')
    return e

def legacy_rng_contract(root,m):
    p=root/m['runtime_root']/'modelplotpy/_modelplotpy.py'; e=[]
    if not p.is_file(): return e
    tree=parse_python(p)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='seed':
            v=n.func.value
            if isinstance(v,ast.Attribute) and isinstance(v.value,ast.Name) and v.value.id=='np' and v.attr=='random':
                e.append(f'legacy modelplotpy mutates process-global NumPy RNG via np.random.seed at line {n.lineno}')
    return e

def plane_checks(root,m):
    maint=[]; runtime=[]; rr=root/m['runtime_root']
    for p in sorted(rr.rglob('*.py')):
        if 'tests' in p.parts: continue
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='maintenances' or plain.startswith('maintenances.') or plain=='skills' or plain.startswith('skills.'):
                runtime.append(f'runtime plane violation {p.relative_to(rr)}:{line}: {name}')
    live=root/m['maintenance_root']/'_maintenance'
    for p in sorted(live.rglob('*.py')):
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='scikitplot.decile' or plain.startswith('scikitplot.decile.'):
                maint.append(f'maintenance plane imports runtime {p.relative_to(root)}:{line}: {name}')
    return maint,runtime

def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        if not (root/safe_rel(rel)).is_file(): e.append(f'read-order file is missing: {rel}')
    skill=root/safe_rel(m['skill']); text=skill.read_text(encoding='utf-8') if skill.is_file() else ''
    if not text.startswith('---\n') or len(text.splitlines())<70: e.append('skill must be a substantive SKILL.md with YAML frontmatter')
    for marker in ('three surfaces','KDS argument propagation','global NumPy RNG','financial metrics','Shared decorators','Release remains blocked'):
        if marker not in text: e.append(f'skill is missing maintainer marker {marker!r}')
    return e

def review_metadata(root,m):
    e=[]; p=root/m['maintenance_root']/'REVIEW.json'
    try: r=load_json(p); reject_command_surface(r,'REVIEW.json')
    except ContractError as x: return [str(x)]
    if r.get('subsystem')!='scikitplot.decile': e.append('REVIEW.json subsystem mismatch')
    for lane in r.get('lanes',[]):
        for c in lane.get('checks',[]):
            if c not in ALLOWED_REVIEW_CHECKS: e.append(f'REVIEW.json uses unknown check {c!r}')
    ids={x.get('id') for x in r.get('known_findings',[])}
    for need in ('DEC-KDS-001','DEC-RNG-001'):
        if need not in ids: e.append(f'REVIEW.json lost known finding {need}')
    return e

def evidence(root,m,fp):
    e=[]; p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'
    try: ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as x: return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp,'PENDING'): e.append(f'evidence runtime fingerprint drift: recorded {ev.get("runtime_fingerprint")}, actual {fp}')
    for gate,row in ev.get('gates',{}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}: e.append(f'evidence gate {gate!r} has invalid status')
        log=row.get('log')
        if log:
            lp=root/safe_rel(log)
            if not lp.is_file(): e.append(f'evidence log missing for {gate}: {log}')
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=row['sha256']: e.append(f'evidence log hash mismatch for {gate}')
    return e

def run_checks(root:Path):
    m=load_json(root/'maintenances/decile/MAINTENANCE.json'); fp,entries=fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maint=[]; runtime=[]
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp,rp=plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += api_surface(root,m); runtime += kds_argument_contract(root,m); runtime += legacy_rng_contract(root,m)
    maint += evidence(root,m,fp)
    return m,fp,inv,maint,runtime

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); a=ap.parse_args(argv)
    try:
        root=discover_repo(Path(__file__)); m,fp,inv,me,re=run_checks(root)
        if a.update:
            if me or re: raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            ep=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'; ev=load_json(ep); ev['runtime_fingerprint']=fp; ep.write_text(json.dumps(ev,indent=2)+'\n')
        payload={'subsystem':'scikitplot.decile','maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','runtime_fingerprint':fp,'inventory':inv,'maintenance_errors':me,'runtime_errors':re}
        print(json.dumps(payload,indent=2) if a.json else f"maintenance: {payload['maintenance_status']}\nruntime: {payload['runtime_status']}\n"+'\n'.join('FAIL: '+x for x in me+re))
        return 0 if not me and not re else 1
    except ContractError as x:
        if a.json: print(json.dumps({'subsystem':'scikitplot.decile','maintenance_status':'FAIL','runtime_status':'UNKNOWN','error':str(x)},indent=2))
        else: print(f'error: {x}',file=sys.stderr)
        return 2
if __name__=='__main__': raise SystemExit(main())
