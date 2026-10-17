#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from pathlib import Path, PurePosixPath
class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
TRACKED_SUFFIXES={'.py','.pyi','.json','.md','.toml','.yaml','.typed'}
ALLOWED_REVIEW_CHECKS={'runtime_presence','public_api','security_boundary','optional_dependencies','typing','process_boundary','planes','handoff','inventory','evidence','tests','integration','release'}

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
    return {'tracked_files':len(entries),'production_python_files':sum(x.endswith('.py') and not x.startswith('tests/') for x in entries),'stub_files':sum(x.endswith('.pyi') for x in entries),'test_python_files':sum(x.endswith('.py') and x.startswith('tests/') for x in entries)}

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,'MAINTENANCE.json')
    except ContractError as x: e.append(str(x))
    if m.get('schema_version')!=4: e.append('MAINTENANCE.json schema_version must be integer 4')
    if m.get('subsystem')!='scikitplot.mlflow': e.append('MAINTENANCE.json subsystem must be scikitplot.mlflow')
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
    if inv['production_python_files']!=18: e.append('mlflow production Python surface changed; review ownership tiers before blessing inventory')
    if inv['stub_files']!=3: e.append('mlflow stub surface changed; review typing contract before blessing inventory')
    tests=root/m['runtime_contract']['tests_root']
    if not tests.is_dir() or len(list(tests.glob('test*.py')))<20: e.append('mlflow regression test surface is unexpectedly small/missing')
    return e

def public_api(root,m):
    e=[]; rr=root/m['runtime_root']
    for rel,syms in m['runtime_contract']['required_symbols'].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f'{rel} is missing required top-level symbol {s}')
    init=(rr/'__init__.py').read_text(encoding='utf-8')
    for marker in ('from ._security import *','from ._session import *','from ._project import *','__all__ += _security.__all__','__all__ += _session.__all__'):
        if marker not in init: e.append(f'public API aggregation marker {marker!r} disappeared')
    if not (rr/'py.typed').is_file(): e.append('py.typed marker disappeared')
    return e

def security_boundary(root,m):
    rr=root/m['runtime_root']; e=[]
    sp=rr/'_security.py'
    if not sp.is_file(): return e
    sec=sp.read_text(encoding='utf-8')
    for marker in ('Policies are enforced only when explicitly set','session() / spawn_server() calls here are guarded','def get_security_policy','def validate_server_config','def validate_session_config'):
        if marker not in sec: e.append(f'security policy contract marker {marker!r} disappeared')
    consumers=[]
    for name in ('_server.py','_session.py','_project.py','_env.py','_workflow.py'):
        cp=rr/name
        if not cp.is_file(): continue
        txt=cp.read_text(encoding='utf-8')
        if 'get_security_policy' in txt or '._security import' in txt: consumers.append(name)
    if not consumers: e.append('SecurityPolicy is not consulted by session/server/project/env/workflow runtime entrypoints despite enforcement documentation')
    return e

def optional_dependencies(root,m):
    rr=root/m['runtime_root']; e=[]
    for name in ('_project.py','_workflow.py'):
        p=rr/name; tree=parse_python(p)
        lines=p.read_text(encoding='utf-8').splitlines()
        for n in ast.walk(tree):
            if isinstance(n,ast.Try):
                imports_yaml=any(isinstance(x,(ast.Import,ast.ImportFrom)) and ((isinstance(x,ast.Import) and any(a.name=='yaml' for a in x.names)) or (isinstance(x,ast.ImportFrom) and x.module=='yaml')) for x in n.body)
                if imports_yaml:
                    for h in n.handlers:
                        if isinstance(h.type,ast.Name) and h.type.id=='Exception': e.append(f'{name}:{h.lineno} catches Exception around optional PyYAML import; broken dependency can be mislabeled as missing')
    # importing the package itself must remain MLflow-optional: no module-scope `import mlflow` outside TYPE_CHECKING.
    for p in sorted(rr.glob('*.py')):
        for line,name in imports(p):
            if name=='mlflow' or name.startswith('mlflow.'):
                if p.name!='_facade.py': e.append(f'module-scope upstream MLflow import threatens optional import contract: {p.name}:{line}: {name}')
    return e

def typing_checks(root,m):
    rr=root/m['runtime_root']; e=[]
    p=rr/'_session.pyi'
    if not p.is_file(): return e
    txt=p.read_text(encoding='utf-8')
    if 'from _facade import ArtifactsFacade, ModelsFacade' in txt: e.append('_session.pyi uses absolute `from _facade import ...` instead of package-relative `from ._facade import ...`')
    for stub in ('_config.pyi','_project.pyi','_session.pyi'):
        try: parse_python(rr/stub)
        except ContractError as x: e.append(str(x))
    return e

def process_boundary(root,m):
    rr=root/m['runtime_root']; e=[]
    if not (rr/'_server.py').is_file() or not (rr/'_cli_caps.py').is_file() or not (rr/'_readiness.py').is_file(): return e
    server=(rr/'_server.py').read_text(encoding='utf-8'); caps=(rr/'_cli_caps.py').read_text(encoding='utf-8')
    for marker in ('[sys.executable, "-m", "mlflow", "server", *args]','subprocess.Popen(cmd','start_new_session','CREATE_NEW_PROCESS_GROUP','proc.wait(timeout=0.2)'):
        if marker not in server: e.append(f'managed-server lifecycle marker {marker!r} disappeared')
    if 'shell=True' in server or 'shell=True' in caps: e.append('mlflow subprocess path must not use shell=True')
    ready=(rr/'_readiness.py').read_text(encoding='utf-8')
    for marker in ('timeout=request_timeout_s','deadline = time.monotonic() + timeout_s','server.process.poll()'):
        if marker not in ready: e.append(f'readiness bounding marker {marker!r} disappeared')
    return e

def plane_checks(root,m):
    runtime=[]; maintenance=[]; rr=root/m['runtime_root']
    for p in sorted(rr.glob('*.py')):
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='maintenances' or plain.startswith('maintenances.') or plain=='skills' or plain.startswith('skills.'):
                runtime.append(f'runtime plane violation {p.relative_to(rr)}:{line}: {name}')
    live=root/m['maintenance_root']/'_maintenance'
    for p in sorted(live.rglob('*.py')):
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='scikitplot.mlflow' or plain.startswith('scikitplot.mlflow.'):
                maintenance.append(f'maintenance plane imports runtime {p.relative_to(root)}:{line}: {name}')
    return maintenance,runtime

def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        if not (root/safe_rel(rel)).is_file(): e.append(f'read-order file is missing: {rel}')
    skill=root/safe_rel(m['skill']); text=skill.read_text(encoding='utf-8') if skill.is_file() else ''
    if not text.startswith('---\n') or len(text.splitlines())<80: e.append('skill must be a substantive SKILL.md with YAML frontmatter')
    for marker in ('Security policy must be connected','optional dependencies','subprocess','Typing is a shipped contract','Evidence ladder','upstream `mlflow`'):
        if marker not in text: e.append(f'skill is missing maintainer marker {marker!r}')
    return e

def review_metadata(root,m):
    e=[]; p=root/m['maintenance_root']/'REVIEW.json'
    try: r=load_json(p); reject_command_surface(r,'REVIEW.json')
    except ContractError as x: return [str(x)]
    if r.get('subsystem')!='scikitplot.mlflow': e.append('REVIEW.json subsystem mismatch')
    for lane in r.get('lanes',[]):
        for c in lane.get('checks',[]):
            if c not in ALLOWED_REVIEW_CHECKS: e.append(f'REVIEW.json uses unknown check {c!r}')
    ids={x.get('id') for x in r.get('known_findings',[])}
    for need in ('MLF-SEC-001','MLF-OPT-001','MLF-TYP-001','MLF-TEST-001'):
        if need not in ids: e.append(f'REVIEW.json lost known finding {need}')
    return e

def evidence(root,m,fp):
    e=[]; p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'
    try: ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as x: return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp,'PENDING'): e.append(f"evidence runtime fingerprint drift: recorded {ev.get('runtime_fingerprint')}, actual {fp}")
    for gate,row in ev.get('gates',{}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}: e.append(f'evidence gate {gate!r} has invalid status')
        log=row.get('log')
        if log:
            lp=root/safe_rel(log)
            if not lp.is_file(): e.append(f'evidence log missing for {gate}: {log}')
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=row['sha256']: e.append(f'evidence log hash mismatch for {gate}')
    return e

def run_checks(root:Path):
    m=load_json(root/'maintenances/mlflow/MAINTENANCE.json'); fp,entries=fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maint=[]; runtime=[]
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp,rp=plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += public_api(root,m); runtime += security_boundary(root,m); runtime += optional_dependencies(root,m); runtime += typing_checks(root,m); runtime += process_boundary(root,m)
    maint += evidence(root,m,fp)
    return m,fp,inv,maint,runtime

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); a=ap.parse_args(argv)
    try:
        root=discover_repo(Path(__file__)); m,fp,inv,me,re=run_checks(root)
        if a.update:
            if me or re: raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            ep=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'; ev=load_json(ep); ev['runtime_fingerprint']=fp; ep.write_text(json.dumps(ev,indent=2)+'\n')
        payload={'subsystem':'scikitplot.mlflow','maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','runtime_fingerprint':fp,'inventory':inv,'maintenance_errors':me,'runtime_errors':re}
        print(json.dumps(payload,indent=2) if a.json else f"maintenance: {payload['maintenance_status']}\nruntime: {payload['runtime_status']}\n"+'\n'.join('FAIL: '+x for x in me+re))
        return 0 if not me and not re else 1
    except ContractError as x:
        if a.json: print(json.dumps({'subsystem':'scikitplot.mlflow','maintenance_status':'FAIL','runtime_status':'UNKNOWN','error':str(x)},indent=2))
        else: print(f'error: {x}',file=sys.stderr)
        return 2
if __name__=='__main__': raise SystemExit(main())
