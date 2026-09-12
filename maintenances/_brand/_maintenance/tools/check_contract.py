#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError): pass
FORBIDDEN_META_KEYS={"command","commands","cmd","shell","exec","executable"}
ALLOWED_REVIEW_CHECKS={"runtime_presence","api_surface","export_hygiene","cli_surface","module_execution","banner_failure_semantics","determinism","external_tool_boundary","planes","handoff","inventory","evidence","release"}

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
    return [p for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.suffix in {".py",".pyi",".md",".json",".toml",".yaml",".typed"}]

def inventory(root,rr):
    base=root/rr; rel=[p.relative_to(base).as_posix() for p in tracked(root,rr)]
    def is_test(x): return "tests" in PurePosixPath(x).parts
    return {"tracked_files":len(rel),"production_python_files":sum(x.endswith(".py") and not is_test(x) for x in rel),"test_python_files":sum(x.endswith(".py") and is_test(x) and PurePosixPath(x).name.startswith("test") for x in rel),"test_support_python_files":sum(x.endswith(".py") and is_test(x) and not PurePosixPath(x).name.startswith("test") for x in rel),"paths":rel}

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,"MAINTENANCE.json")
    except ContractError as x: e.append(str(x))
    if m.get("schema_version") != 4: e.append("MAINTENANCE.json schema_version must be integer 4")
    if m.get("subsystem") != "scikitplot._brand": e.append("MAINTENANCE.json subsystem must be scikitplot._brand")
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
    exp=m["runtime_contract"]["inventory"]
    for k in ("tracked_files","production_python_files","test_python_files","test_support_python_files"):
        if inv[k] != exp[k]: e.append(f"_brand {k} changed: expected {exp[k]}, got {inv[k]}")
    if inv["paths"] != exp["paths"]: e.append("_brand tracked runtime/test path inventory changed; review ownership before blessing")
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

def export_hygiene(root,m):
    init=root/m["runtime_root"] / "__init__.py"; logo=root/m["runtime_root"] / "_logo.py"; e=[]
    if not init.is_file() or not logo.is_file(): return e
    isrc=init.read_text(encoding="utf-8")
    if "from ._logo import *" in isrc and "__all__" not in top_symbols(logo): e.append("_brand star-imports _logo but _logo defines no __all__; dependency/helper names leak into package namespace")
    if "from ._logo import *" in isrc and "__all__ += _logo.__all__" not in isrc: e.append("_brand package __all__ does not include the logo surface imported into the package namespace")
    return e

def cli_surface(root,m):
    p=root/m["runtime_root"] / "_logo.py"
    if not p.is_file(): return []
    src=p.read_text(encoding="utf-8")
    return ["logo CLI advertises python -m scikitplot.logo but scikitplot/logo.py does not exist"] if "python -m scikitplot.logo" in src and not (root/"scikitplot/logo.py").is_file() else []

def module_execution(root,m):
    init=root/m["runtime_root"] / "__init__.py"; e=[]
    if not init.is_file(): return e
    src=init.read_text(encoding="utf-8")
    bp=root/m["runtime_root"] / "_banner.py"; lp=root/m["runtime_root"] / "_logo.py"
    b=bp.read_text(encoding="utf-8") if bp.is_file() else ""; l=lp.read_text(encoding="utf-8") if lp.is_file() else ""
    if "_banner" in src and "python3 -m scikitplot._brand._banner" in b: e.append("_brand eagerly imports executable _banner; python -m scikitplot._brand._banner is preloaded before runpy execution")
    if "_logo" in src and 'if __name__ == "__main__"' in l: e.append("_brand eagerly imports executable _logo; direct python -m execution is preloaded before runpy execution")
    return e

def banner_failure_semantics(root,m):
    p=root/m["runtime_root"] / "_banner.py"
    if not p.is_file(): return []
    src=p.read_text(encoding="utf-8"); main=src[src.find("def main("):] if "def main(" in src else ""
    return ["banner main has no zero-result failure guard even though generate_all skips per-banner generation failures"] if "except (BannerGenerationError, ValueError)" in src and "if not results" not in main and "if len(results) == 0" not in main else []

def determinism(root,m):
    p=root/m["runtime_root"] / "_logo.py"
    if not p.is_file(): return []
    src=p.read_text(encoding="utf-8"); e=[]
    if "np.random.seed(" in src: e.append("logo runtime mutates NumPy global RNG via np.random.seed")
    if "np.random.default_rng(seed)" not in src: e.append("random-dot mode lost local seeded RNG construction")
    if "_FIXED_DOTS" not in src: e.append("fixed-dot deterministic brand layout is missing")
    return e

def external_tool_boundary(root,m):
    p=root/m["runtime_root"] / "_banner.py"
    if not p.is_file(): return []
    src=p.read_text(encoding="utf-8"); e=[]
    for token,msg in [('shutil.which(\"figlet\")','banner figlet availability check is missing'),('subprocess.run(','banner subprocess execution is missing'),('timeout=10','banner subprocess timeout guard is missing')]:
        if token not in src: e.append(msg)
    try:
        tree=parse(p)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(k.arg == "shell" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords):
                e.append("banner subprocess boundary must not use shell=True")
                break
    except ContractError as x:
        e.append(str(x))
    return e

def planes(root,m):
    me=[]; re=[]
    for p in (root/m["runtime_root"]).rglob("*.py"):
        if "__pycache__" in p.parts: continue
        for line,name in module_imports(p):
            n=name.lstrip(".")
            if n.startswith("maintenances") or n.startswith("skills"): re.append(f"runtime plane violation: {p.relative_to(root)}:{line} imports {name}")
    for p in (root/m["maintenance_root"]).rglob("*.py"):
        if "__pycache__" in p.parts: continue
        for line,name in module_imports(p):
            n=name.lstrip(".")
            if n=="scikitplot._brand" or n.startswith("scikitplot._brand."): me.append(f"maintenance plane violation: {p.relative_to(root)}:{line} imports runtime package instead of inspecting source")
    return me,re

def handoff(root,m):
    e=[]
    for rel in m.get("read_order",[]):
        try:
            if not (root/safe_rel(rel)).is_file(): e.append(f"read-order file is missing: {rel}")
        except ContractError as x: e.append(str(x))
    sp=root/m["skill"]
    if not sp.is_file(): e.append("_brand maintainer skill is missing")
    elif len(sp.read_text(encoding="utf-8").split()) < 300: e.append("_brand maintainer skill is not substantive enough to hand off ownership safely")
    return e

def metadata(root,m):
    e=[]
    for rel in ["maintenances/_brand/REVIEW.json","maintenances/_brand/_maintenance/STATE.json","maintenances/_brand/_maintenance/EVIDENCE.json"]:
        try:
            o=load_json(root/rel); reject_command_surface(o,rel)
            if o.get("subsystem") != "scikitplot._brand": e.append(f"{rel} subsystem must be scikitplot._brand")
        except ContractError as x: e.append(str(x))
    try:
        r=load_json(root/"maintenances/_brand/REVIEW.json")
        unknown=set(r.get("checks",[]))-ALLOWED_REVIEW_CHECKS
        if unknown: e.append(f"REVIEW.json contains unknown check names: {sorted(unknown)}")
        if r.get("maintenance_status") != "PASS": e.append("REVIEW.json maintenance_status must be PASS for a healthy maintenance plane")
        if r.get("release_status") == "PASS" and r.get("runtime_status") != "PASS": e.append("release cannot PASS while runtime_status is not PASS")
    except ContractError: pass
    return e

def check(root):
    root=Path(root)
    try: m=load_json(root/"maintenances/_brand/MAINTENANCE.json")
    except ContractError as x: return {"maintenance_status":"FAIL","runtime_status":"UNKNOWN","release_status":"BLOCKED","maintenance_errors":[str(x)],"runtime_errors":[]}
    maintenance_errors=validate_manifest(root,m)+handoff(root,m)+metadata(root,m)
    inv=inventory(root,m.get("runtime_root","scikitplot/_brand")); runtime_errors=[]
    for fn in (runtime_presence,): runtime_errors += fn(root,m,inv)
    for fn in (api_surface,export_hygiene,cli_surface,module_execution,banner_failure_semantics,determinism,external_tool_boundary): runtime_errors += fn(root,m)
    me,re=planes(root,m); maintenance_errors += me; runtime_errors += re
    return {"subsystem":"scikitplot._brand","maintenance_status":"PASS" if not maintenance_errors else "FAIL","runtime_status":"PASS" if not runtime_errors else "FAIL","release_status":"PASS" if not maintenance_errors and not runtime_errors else "BLOCKED","inventory":inv,"maintenance_errors":maintenance_errors,"runtime_errors":runtime_errors}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path); ap.add_argument("--json",action="store_true"); a=ap.parse_args(argv)
    try: root=a.repo.resolve() if a.repo else discover_repo(Path.cwd())
    except ContractError as x: out={"maintenance_status":"FAIL","runtime_status":"UNKNOWN","release_status":"BLOCKED","maintenance_errors":[str(x)],"runtime_errors":[]}
    else: out=check(root)
    if a.json: print(json.dumps(out,indent=2,sort_keys=True))
    else:
        print(f"maintenance={out['maintenance_status']} runtime={out['runtime_status']} release={out['release_status']}")
        for x in out.get("maintenance_errors",[]): print("M:",x)
        for x in out.get("runtime_errors",[]): print("R:",x)
    return 0 if out.get("maintenance_status")=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
