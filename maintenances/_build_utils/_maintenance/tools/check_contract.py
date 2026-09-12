#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={"command","commands","cmd","shell","exec","executable"}
ALLOWED_REVIEW_CHECKS={"runtime_presence","api_surface","build_only_boundary","cython_validation","git_error_contract","git_safe_directory","meson_freshness","copy_semantics","test_fidelity","planes","handoff","inventory","evidence","release"}

def load_json(path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f"duplicate JSON key {k!r} in {path}")
            out[k]=v
        return out
    try: return json.loads(Path(path).read_text(encoding="utf-8"),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as exc: raise ContractError(f"cannot load {path}: {exc}") from exc

def reject_command_surface(obj,where="metadata"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS: raise ContractError(f"{where} contains unsupported executable field {k!r}")
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def discover_repo(start):
    r=Path(start).resolve()
    for c in [r,*r.parents]:
        if all((c/n).is_dir() for n in ("scikitplot","maintenances","skills")): return c
    raise ContractError("could not locate wide repository root containing scikitplot/, maintenances/, and skills/")

def safe_rel(v):
    if not isinstance(v,str) or not v or "\\" in v or "\x00" in v: raise ContractError(f"unsafe repository path {v!r}")
    p=PurePosixPath(v)
    if p.is_absolute() or ".." in p.parts or "." in p.parts or "//" in v: raise ContractError(f"unsafe repository path {v!r}")
    return v

def parse(path):
    try: return ast.parse(Path(path).read_text(encoding="utf-8"),filename=str(path))
    except (OSError,SyntaxError) as exc: raise ContractError(f"cannot parse {path}: {exc}") from exc

def top_symbols(path):
    out=set()
    for n in parse(path).body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): out.add(n.name)
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t,ast.Name): out.add(t.id)
    return out

def module_imports(path):
    out=[]
    for n in parse(path).body:
        if isinstance(n,ast.Import): out += [(n.lineno,a.name) for a in n.names]
        elif isinstance(n,ast.ImportFrom): out.append((n.lineno,"."*n.level+(n.module or "")))
    return out

def tracked(root,rr):
    base=root/rr
    return [p for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]

def inventory(root,rr):
    base=root/rr; fs=tracked(root,rr); rel=[p.relative_to(base).as_posix() for p in fs]
    def is_test(p): return "tests" in p.relative_to(base).parts
    import hashlib
    h=hashlib.sha256()
    for p,r in zip(fs,rel):
        sha=hashlib.sha256(p.read_bytes()).hexdigest(); h.update(f"{r}\0{sha}\0{p.stat().st_size}\n".encode())
    return {"tracked_files":len(rel),"production_python_files":sum(p.suffix==".py" and not is_test(p) for p in fs),"test_python_files":sum(p.suffix==".py" and is_test(p) and p.name.startswith("test") for p in fs),"test_support_python_files":sum(p.suffix==".py" and is_test(p) and not p.name.startswith("test") for p in fs),"paths":rel,"baseline_sha256":h.hexdigest()}

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,"MAINTENANCE.json")
    except ContractError as x: e.append(str(x))
    if m.get("schema_version") != 4: e.append("MAINTENANCE.json schema_version must be integer 4")
    if m.get("subsystem") != "scikitplot._build_utils": e.append("MAINTENANCE.json subsystem must be scikitplot._build_utils")
    for k in ("runtime_root","maintenance_root","skill"):
        try:
            p=root/safe_rel(m[k])
            if not p.exists(): e.append(f"required repository path does not exist: {m[k]}")
        except (KeyError,ContractError) as x: e.append(str(x))
    return e

def runtime_presence(root,m,inv):
    e=[]
    for rel in m["runtime_contract"]["required_files"]:
        try:
            if not (root/safe_rel(rel)).is_file(): e.append(f"missing required runtime file {rel}")
        except ContractError as x: e.append(str(x))
    exp=m["runtime_contract"]["inventory"]
    for k in ("tracked_files","production_python_files","test_python_files","test_support_python_files"):
        if inv[k] != exp[k]: e.append(f"_build_utils {k} changed: expected {exp[k]}, got {inv[k]}")
    if inv["paths"] != exp["paths"]: e.append("_build_utils tracked source/test inventory changed; review ownership before blessing")
    return e

def api_surface(root,m):
    e=[]
    for rel,syms in m["runtime_contract"]["required_symbols"].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f"{rel} is missing required top-level symbol {s}")
    return e

def build_only_boundary(root,m):
    e=[]; own=(root/m["runtime_root"]).resolve()
    for p in (root/'scikitplot').rglob('*.py'):
        if own in p.resolve().parents or p.resolve()==own: continue
        if '__pycache__' in p.parts: continue
        try: imports=module_imports(p)
        except ContractError: continue
        for line,name in imports:
            n=name.lstrip('.')
            if n=='scikitplot._build_utils' or n.startswith('scikitplot._build_utils.') or n=='_build_utils' or n.startswith('_build_utils.'):
                e.append(f"build-only boundary violation: {p.relative_to(root)}:{line} imports {name}")
    return e

def cython_validation(root,m):
    p=root/m['runtime_root']/'cython_generate.py'
    if not p.is_file(): return []
    src=p.read_text(encoding='utf-8'); e=[]
    if ('"{{" in content' in src or "'{{' in content" in src) and ('"}}" in content' in src or "'}}' in content" in src):
        e.append('cython_generate validates generated files with raw doubled-brace substring checks; legitimate CSS/JavaScript braces are false positives')
    if 'Output is written atomically' in src and 'os.replace(' not in src and '.replace(' not in src:
        e.append('cython_generate documentation promises atomic output but generator writes directly without atomic replacement')
    return e

def git_error_contract(root,m):
    p=root/m['runtime_root']/'gitversion.py'
    if not p.is_file(): return []
    src=p.read_text(encoding='utf-8'); start=src.find('def add_safe_directory'); end=src.find('## Version Extraction',start); seg=src[start:end if end!=-1 else None]
    e=[]
    if 'except ValueError as ve:' in seg and 'return (0, "", str(e))' in seg: e.append('add_safe_directory ValueError/error fallback references e outside its exception scope and can raise UnboundLocalError')
    return e

def git_safe_directory(root,m):
    p=root/m['runtime_root']/'gitversion.py'
    if not p.is_file(): return []
    src=p.read_text(encoding='utf-8'); e=[]
    if '"--global"' in src and '"safe.directory"' in src: e.append('git safe-directory recovery mutates persistent --global Git configuration')
    if 'add_safe_directory(repo_path=git_dir)' in src: e.append('git_version marks the _build_utils working subdirectory safe instead of resolving the Git repository top level')
    if 'if returncode == 128:' in src and 'dubious ownership' not in src[src.find('if returncode == 128:'):src.find('if returncode == 128:')+350].lower(): e.append('git_version triggers safe-directory mutation for every return code 128 rather than a verified dubious-ownership diagnostic')
    return e

def meson_freshness(root,m):
    p=root/m['runtime_root']/'install_meson_features.py'
    if not p.is_file(): return []
    src=p.read_text(encoding='utf-8'); start=src.find('def _needs_update'); end=src.find('\ndef main',start); seg=src[start:end]
    e=[]
    if 'getmtime' in seg and 'hashlib' not in seg and 'filecmp' not in seg: e.append('_needs_update uses mtimes rather than content identity, so a newer but different destination can be accepted')
    if 'os.listdir(dst_dir)' not in seg and 'set(' not in seg: e.append('_needs_update does not detect destination-only stale files removed from the source feature package')
    return e

def copy_semantics(root,m):
    p=root/m['runtime_root']/'copyfiles.py'
    if not p.is_file(): return []
    src=p.read_text(encoding='utf-8'); start=src.find('def copy_file'); end=src.find('\ndef copy_directory',start); seg=src[start:end]
    if 'os.makedirs(os.path.dirname(dest), exist_ok=True)' in seg: return ['copy_file creates os.path.dirname(dest) unconditionally; a bare destination filename yields os.makedirs(\"\")']
    return []

def test_fidelity(root,m):
    p=root/m['runtime_root']/'tests/test_gitversion.py'
    if not p.is_file(): return []
    try: tree=parse(p)
    except ContractError as x: return [str(x)]
    calls=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            if isinstance(n.func,ast.Name): calls.add(n.func.id)
            elif isinstance(n.func,ast.Attribute): calls.add(n.func.attr)
    expected={'git_version','GitVersionInfo','generate_version_template','add_safe_directory'}
    if not calls.intersection(expected): return ['test_gitversion imports production version APIs but never calls them; simulated tests do not exercise real error paths']
    return []

def planes(root,m):
    me=[]; re=[]
    for p in (root/m['runtime_root']).rglob('*.py'):
        if '__pycache__' in p.parts: continue
        for line,name in module_imports(p):
            n=name.lstrip('.')
            if n.startswith('maintenances') or n.startswith('skills'): re.append(f"runtime plane violation: {p.relative_to(root)}:{line} imports {name}")
    for p in (root/m['maintenance_root']).rglob('*.py'):
        if '__pycache__' in p.parts: continue
        for line,name in module_imports(p):
            n=name.lstrip('.')
            if n=='scikitplot._build_utils' or n.startswith('scikitplot._build_utils.'):
                me.append(f"maintenance plane violation: {p.relative_to(root)}:{line} imports runtime build package instead of inspecting source")
    return me,re

def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        try:
            if not (root/safe_rel(rel)).is_file(): e.append(f"read-order file is missing: {rel}")
        except ContractError as x: e.append(str(x))
    sp=root/m['skill']
    if not sp.is_file(): e.append('_build_utils maintainer skill is missing')
    elif len(sp.read_text(encoding='utf-8').split())<500: e.append('_build_utils maintainer skill is not substantive enough for build-tool ownership handoff')
    return e

def metadata(root,m):
    e=[]
    for rel in ['maintenances/_build_utils/REVIEW.json','maintenances/_build_utils/_maintenance/STATE.json','maintenances/_build_utils/_maintenance/EVIDENCE.json']:
        try:
            o=load_json(root/rel); reject_command_surface(o,rel)
            if o.get('subsystem')!='scikitplot._build_utils': e.append(f"{rel} subsystem must be scikitplot._build_utils")
        except ContractError as x: e.append(str(x))
    try:
        r=load_json(root/'maintenances/_build_utils/REVIEW.json')
        unknown=set(r.get('checks',[]))-ALLOWED_REVIEW_CHECKS
        if unknown: e.append(f"REVIEW.json contains unknown check names: {sorted(unknown)}")
        if r.get('maintenance_status')!='PASS': e.append('REVIEW.json maintenance_status must be PASS for a healthy maintenance plane')
        if r.get('release_status')=='PASS' and r.get('runtime_status')!='PASS': e.append('release cannot PASS while runtime_status is not PASS')
    except ContractError: pass
    return e

def check(root):
    root=Path(root)
    try: m=load_json(root/'maintenances/_build_utils/MAINTENANCE.json')
    except ContractError as x: return {'maintenance_status':'FAIL','runtime_status':'UNKNOWN','release_status':'BLOCKED','maintenance_errors':[str(x)],'runtime_errors':[]}
    maintenance_errors=validate_manifest(root,m)+handoff(root,m)+metadata(root,m)
    inv=inventory(root,m.get('runtime_root','scikitplot/_build_utils')); runtime_errors=[]
    runtime_errors+=runtime_presence(root,m,inv); runtime_errors+=api_surface(root,m)
    for fn in (build_only_boundary,cython_validation,git_error_contract,git_safe_directory,meson_freshness,copy_semantics,test_fidelity): runtime_errors+=fn(root,m)
    me,re=planes(root,m); maintenance_errors+=me; runtime_errors+=re
    return {'subsystem':'scikitplot._build_utils','maintenance_status':'PASS' if not maintenance_errors else 'FAIL','runtime_status':'PASS' if not runtime_errors else 'FAIL','release_status':'PASS' if not maintenance_errors and not runtime_errors else 'BLOCKED','inventory':inv,'maintenance_errors':maintenance_errors,'runtime_errors':runtime_errors}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path); ap.add_argument('--json',action='store_true'); a=ap.parse_args(argv)
    try:
        if a.repo:
            root=a.repo.resolve()
        else:
            try: root=discover_repo(Path.cwd())
            except ContractError: root=discover_repo(Path(__file__).resolve())
    except ContractError as x: out={'maintenance_status':'FAIL','runtime_status':'UNKNOWN','release_status':'BLOCKED','maintenance_errors':[str(x)],'runtime_errors':[]}
    else: out=check(root)
    print(json.dumps(out,indent=2,sort_keys=True) if a.json else f"maintenance={out['maintenance_status']} runtime={out['runtime_status']} release={out['release_status']}\n"+'\n'.join(['M: '+x for x in out.get('maintenance_errors',[])]+['R: '+x for x in out.get('runtime_errors',[])]))
    return 0 if out.get('maintenance_status')=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
