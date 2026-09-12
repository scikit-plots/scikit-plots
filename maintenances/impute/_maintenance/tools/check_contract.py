\
#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={"command","commands","cmd","shell","exec","executable"}
TRACKED_SUFFIXES={".py",".pyi",".json",".md"}
ALLOWED_REVIEW_CHECKS={"runtime_presence","public_api","backend_boundary","privacy","planes","handoff","inventory","evidence","tests","integration","release"}

def load_json(path:Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    try: return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as exc: raise ContractError(f"cannot load {path}: {exc}") from exc

def reject_command_surface(obj,where="metadata"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS: raise ContractError(f"{where} contains unsupported executable field {k!r}")
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def discover_repo(start:Path)->Path:
    r=start.resolve()
    for c in [r,*r.parents]:
        if all((c/n).is_dir() for n in ("scikitplot","maintenances","skills")): return c
    raise ContractError("could not locate wide repository root containing scikitplot/, maintenances/, and skills/")

def safe_rel(value:str)->str:
    if not isinstance(value,str) or not value or "\\" in value or "\x00" in value: raise ContractError(f"unsafe repository path {value!r}")
    p=PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "." in p.parts or "//" in value: raise ContractError(f"unsafe repository path {value!r}")
    return value

def parse_python(path:Path):
    try: return ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    except (OSError,SyntaxError) as exc: raise ContractError(f"cannot parse {path}: {exc}") from exc

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
        elif isinstance(n,ast.ImportFrom): out.append((n.lineno,"."*n.level+(n.module or "")))
    return out

def tracked(root:Path,rr:str):
    base=root/rr
    return [p for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix in TRACKED_SUFFIXES]

def fingerprint(root:Path,rr:str):
    base=root/rr; h=hashlib.sha256(); entries=[]
    for p in tracked(root,rr):
        rel=p.relative_to(base).as_posix(); d=hashlib.sha256(p.read_bytes()).hexdigest(); h.update(rel.encode()+b"\0"+d.encode()+b"\n"); entries.append(rel)
    return h.hexdigest(),entries

def inventory(entries):
    return {"tracked_files":len(entries),"production_python_files":sum(x.endswith('.py') and not x.startswith('tests/') for x in entries),"test_python_files":sum(x.endswith('.py') and x.startswith('tests/') for x in entries)}

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,"MAINTENANCE.json")
    except ContractError as x: e.append(str(x))
    if m.get("schema_version")!=4: e.append("MAINTENANCE.json schema_version must be integer 4")
    if m.get("subsystem")!="scikitplot.impute": e.append("MAINTENANCE.json subsystem must be scikitplot.impute")
    for k in ("runtime_root","maintenance_root","skill"):
        try:
            p=root/safe_rel(m[k])
            if not p.exists(): e.append(f"required repository path does not exist: {m[k]}")
        except (KeyError,ContractError) as x: e.append(str(x))
    return e

def runtime_presence(root,m,inv):
    e=[]
    for rel in m["runtime_contract"]["required_files"]:
        if not (root/safe_rel(rel)).is_file(): e.append(f"missing required runtime file {rel}")
    if inv["production_python_files"]!=4: e.append("impute production Python surface changed; review ownership tiers before blessing inventory")
    tests=root/m["runtime_contract"]["tests_root"]
    if not tests.is_dir() or len(list(tests.glob('test*.py')))<4: e.append("impute regression test surface is unexpectedly small/missing")
    return e

def public_api(root,m):
    e=[]; rr=root/m["runtime_root"]
    for rel,syms in m["runtime_contract"]["required_symbols"].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f"{rel} is missing required top-level symbol {s}")
    init=(rr/'__init__.py').read_text(encoding='utf-8')
    for marker in ('MissingIndicator, SimpleImputer','"MissingIndicator"','"SimpleImputer"','def __getattr__','enable_ann_imputer','ANNImputer'):
        if marker not in init: e.append(f"public/experimental gate marker {marker!r} disappeared")
    ann=(rr/'_ann.py').read_text(encoding='utf-8')
    if 'AnnoyKNNImputer = ANNImputer' not in ann: e.append('AnnoyKNNImputer compatibility alias drifted')
    if 'class VoyagerKNNImputer(ANNImputer)' not in ann: e.append('VoyagerKNNImputer specialization drifted')
    return e

def backend_boundary(root,m):
    rr=root/m["runtime_root"]; p=rr/'_ann.py'; text=p.read_text(encoding='utf-8'); e=[]
    if 'from ..annoy._annoy import Index as AnnoyIndex' not in text: e.append('ANN backend no longer points at scikitplot.annoy._annoy owner')
    if 'except Exception' in text and 'fallback to external annoy' in text: e.append('in-tree Annoy import catches Exception and can silently fall back to external annoy')
    # pandas is only a documentation reference in current source; import at module scope hardens optionality.
    tree=parse_python(p); module_pd=False; runtime_pd_use=False
    for n in tree.body:
        if isinstance(n,ast.Import):
            if any(a.name=='pandas' and a.asname=='pd' for a in n.names): module_pd=True
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='pd': runtime_pd_use=True
    if module_pd and not runtime_pd_use: e.append("_ann.py imports pandas as pd at module scope but has no runtime pd usage")
    if 'except ImportError as e' not in text or '_VOYAGER_IMPORT_ERROR' not in text: e.append('Voyager optional-import failure contract drifted')
    return e

def privacy_checks(root,m):
    p=root/m["runtime_root"]/'_privacy.py'; e=[]
    if not p.is_file():
        return e
    text=p.read_text(encoding='utf-8')
    for marker in ('hard security boundary', "mode in {\"public\", \"private\"}", 'if mode == "external"', 'loader is None', 'external index file not found', 'def delete_external_index'):
        if marker not in text: e.append(f"index privacy/persistence marker {marker!r} disappeared")
    annp=root/m["runtime_root"]/'_ann.py'
    if not annp.is_file():
        return e
    ann=annp.read_text(encoding='utf-8')
    for marker in ('self._store_index(', 'self._get_index_for_runtime(', 'def train_index_', 'index_access="external"'):
        if marker not in ann: e.append(f"ANN index-access integration marker {marker!r} disappeared")
    return e

def plane_checks(root,m):
    runtime=[]; maintenance=[]; rr=root/m["runtime_root"]
    for p in sorted(rr.glob('*.py')):
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='maintenances' or plain.startswith('maintenances.') or plain=='skills' or plain.startswith('skills.'):
                runtime.append(f"runtime plane violation {p.relative_to(rr)}:{line}: {name}")
    live=root/m["maintenance_root"]/'_maintenance'
    for p in sorted(live.rglob('*.py')):
        if 'history' in p.parts: continue
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='scikitplot.impute' or plain.startswith('scikitplot.impute.'):
                maintenance.append(f"maintenance plane imports runtime {p.relative_to(root)}:{line}: {name}")
    return maintenance,runtime

def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        if not (root/safe_rel(rel)).is_file(): e.append(f"read-order file is missing: {rel}")
    skill=root/safe_rel(m['skill']); text=skill.read_text(encoding='utf-8') if skill.is_file() else ''
    if not text.startswith('---\n') or len(text.splitlines())<65: e.append('skill must be a substantive SKILL.md with YAML frontmatter')
    for marker in ('backend ownership','PrivateIndexMixin','_base.py','experimental','test double','release remains blocked'):
        if marker not in text: e.append(f"skill is missing maintainer marker {marker!r}")
    return e

def review_metadata(root,m):
    e=[]; p=root/m['maintenance_root']/'REVIEW.json'
    try: r=load_json(p); reject_command_surface(r,'REVIEW.json')
    except ContractError as x: return [str(x)]
    if r.get('subsystem')!='scikitplot.impute': e.append('REVIEW.json subsystem mismatch')
    for lane in r.get('lanes',[]):
        for c in lane.get('checks',[]):
            if c not in ALLOWED_REVIEW_CHECKS: e.append(f"REVIEW.json uses unknown check {c!r}")
    ids={x.get('id') for x in r.get('known_findings',[])}
    for need in ('IMP-ANN-001','IMP-OPT-001','IMP-BASE-001'):
        if need not in ids: e.append(f"REVIEW.json lost known finding {need}")
    return e

def evidence(root,m,fp):
    e=[]; p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'
    try: ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as x: return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp,'PENDING'): e.append(f"evidence runtime fingerprint drift: recorded {ev.get('runtime_fingerprint')}, actual {fp}")
    for gate,row in ev.get('gates',{}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}: e.append(f"evidence gate {gate!r} has invalid status")
        log=row.get('log')
        if log:
            lp=root/safe_rel(log)
            if not lp.is_file(): e.append(f"evidence log missing for {gate}: {log}")
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=row['sha256']: e.append(f"evidence log hash mismatch for {gate}")
    return e

def run_checks(root:Path):
    m=load_json(root/'maintenances/impute/MAINTENANCE.json'); fp,entries=fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maint=[]; runtime=[]
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp,rp=plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += public_api(root,m); runtime += backend_boundary(root,m); runtime += privacy_checks(root,m)
    maint += evidence(root,m,fp)
    return m,fp,inv,maint,runtime

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); a=ap.parse_args(argv)
    try:
        root=discover_repo(Path(__file__)); m,fp,inv,me,re=run_checks(root)
        if a.update:
            if me or re: raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            ep=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'; ev=load_json(ep); ev['runtime_fingerprint']=fp; ep.write_text(json.dumps(ev,indent=2)+'\n')
        payload={'subsystem':'scikitplot.impute','maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','runtime_fingerprint':fp,'inventory':inv,'maintenance_errors':me,'runtime_errors':re}
        print(json.dumps(payload,indent=2) if a.json else f"maintenance: {payload['maintenance_status']}\nruntime: {payload['runtime_status']}\n"+'\n'.join('FAIL: '+x for x in me+re))
        return 0 if not me and not re else 1
    except ContractError as x:
        if a.json: print(json.dumps({'subsystem':'scikitplot.impute','maintenance_status':'FAIL','runtime_status':'UNKNOWN','error':str(x)},indent=2))
        else: print(f'error: {x}',file=sys.stderr)
        return 2
if __name__=='__main__': raise SystemExit(main())
