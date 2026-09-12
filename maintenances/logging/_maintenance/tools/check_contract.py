#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path

class ContractError(RuntimeError):
    pass

FORBIDDEN_META_KEYS={"command","commands","cmd","shell","exec","executable","script","scripts","argv"}
REQUIRED_SYMBOLS={"get_logger","GoogleLogFormatter","AlwaysStdErrHandler","setLevel","set_verbosity",
                  "error_log","log_every_n","log_first_n","sanitize_log_message"}

def discover_repo(start: Path) -> Path:
    for origin in (start.resolve(), Path(__file__).resolve()):
        for c in (origin,*origin.parents):
            if all((c/n).is_dir() for n in ("scikitplot","maintenances","skills")):
                return c
    raise ContractError("could not locate wide repository root")

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def reject_command_surface(obj, where="metadata"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f"{where} contains unsupported executable field {k!r}")
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def parse_py(p: Path):
    return ast.parse(p.read_text(encoding="utf-8"), filename=str(p))

def symbols(tree):
    out=set()
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): out.add(n.name)
    return out

def func_src(src,tree,name):
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name:
            return ast.get_source_segment(src,n) or ""
    return ""

def class_src(src,tree,name):
    for n in ast.walk(tree):
        if isinstance(n,ast.ClassDef) and n.name==name:
            return ast.get_source_segment(src,n) or ""
    return ""

def runtime_files(root: Path):
    pkg=root/"scikitplot"/"logging"
    if not pkg.is_dir(): return []
    return sorted(p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts)

def tree_fingerprint(root: Path):
    pkg=root/"scikitplot"/"logging"; h=hashlib.sha256()
    for p in runtime_files(root):
        rel=p.relative_to(pkg).as_posix().encode()
        h.update(rel+b"\0"+p.read_bytes()+b"\0")
    return h.hexdigest() if pkg.is_dir() else None

def plane_violations(root: Path):
    out=[]
    for p in runtime_files(root):
        if p.suffix!=".py": continue
        try: tree=parse_py(p)
        except Exception: continue
        for n in ast.walk(tree):
            mod=None
            if isinstance(n,ast.Import):
                mods=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom):
                mods=[n.module or ""]
            else: continue
            if any(x.startswith(("maintenances","skills")) for x in mods):
                out.append(str(p.relative_to(root)))
    return out

def runtime_findings(root: Path):
    pkg=root/"scikitplot"/"logging"
    facade=pkg/"__init__.py"; core=pkg/"_logging.py"; test=pkg/"tests"/"test__logging.py"
    findings=[]
    if not pkg.is_dir(): return ["scikitplot/logging package is missing"]
    if not facade.is_file(): findings.append("scikitplot/logging/__init__.py is missing")
    if not core.is_file(): return findings+["scikitplot/logging/_logging.py is missing"]
    src=core.read_text(encoding="utf-8"); tree=parse_py(core)
    missing=sorted(REQUIRED_SYMBOLS-symbols(tree))
    if missing: findings.append("required logging symbols missing: "+", ".join(missing))
    v=plane_violations(root)
    if v: findings.append("runtime imports maintenance/skill plane: "+", ".join(v))

    # Public package facade must preserve former module-level stdlib compatibility.
    if facade.is_file():
        fs=facade.read_text(encoding="utf-8")
        if "__getattr__" not in fs or "__dir__" not in fs:
            findings.append("LOG-PKG-001: package facade does not forward the former stdlib logging __getattr__/__dir__ compatibility surface")

    err=func_src(src,tree,"error_log")
    if "del error_msg" in err or ("get_logger(" not in err and "log(" not in err):
        findings.append("LOG-ERR-001: error_log does not emit the supplied message")

    gl=func_src(src,tree,"get_logger")
    glnode=next((n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="get_logger"),None)
    applies=False
    if glnode:
        applies=any(isinstance(c.func,ast.Name) and c.func.id=="_default_logging_level"
                    for c in ast.walk(glnode) if isinstance(c,ast.Call))
    if not applies:
        findings.append("LOG-ENV-001: get_logger does not apply documented SKPLT logging environment level policy")
    if "_ENV_AUTO_CONFIG" in src and "def configure(" not in src:
        findings.append("LOG-ENV-001: SKPLT_LOGGING_AUTO_CONFIG references a configure() path that does not exist")

    hs=class_src(src,tree,"AlwaysStdErrHandler")
    if "def stream" in hs and "self._stream = value" not in hs:
        findings.append("LOG-HDL-001: AlwaysStdErrHandler stream setter cannot switch to the requested valid stream")

    cli=root/"scikitplot"/"_cli"/"logging.py"
    if cli.is_file():
        cs=cli.read_text(encoding="utf-8")
        if 'logging.getLogger("scikitplot")' in cs and "logger.addHandler(_HANDLER)" in cs and \
           "logger.addHandler(_handler)" in gl and "if not logger.handlers" not in gl and "_HANDLER_MARKER" not in gl:
            findings.append("LOG-CLI-001: shared logger creation can stack a second handler after CLI logging configuration")

    modern=src.split("def _logger_find_caller(stack_info=False, stacklevel=1):",1)
    if len(modern)==2:
        chunk=modern[1].split("elif (",1)[0]
        if "_get_caller(4)" in chunk:
            findings.append("LOG-CALL-001: custom findCaller ignores stacklevel and breaks documented direct Logger call-sites")

    ga=func_src(src,tree,"__getattr__")
    if "get_logger()" in ga:
        findings.append("LOG-ATTR-001: private core __getattr__ initializes project logging on missing stdlib attributes")

    fm=func_src(src,tree,"_make_default_formatter")
    if "Unknown formatter" not in fm and "Unsupported formatter" not in fm and "raise ValueError" not in fm:
        findings.append("LOG-FMT-001: unknown formatter selections can fall through and return None")

    if not test.is_file():
        findings.append("LOG-TEST-001: focused logging runtime tests are missing")
    else:
        ts=test.read_text(encoding="utf-8")
        if "from .. import logging as splog" in ts:
            findings.append("LOG-TEST-001: focused tests still import the pre-move module shape and fail collection")
        if 'test_error_log_is_noop' in ts or 'test_unknown_string_returns_none' in ts:
            findings.append("LOG-TEST-002: focused tests explicitly preserve known defective behavior")

    return list(dict.fromkeys(findings))

def maintenance_errors(root: Path):
    errors=[]; base=root/"maintenances"/"logging"
    req=["MAINTENANCE.json","REVIEW.json","_maintenance/STATE.json","_maintenance/EVIDENCE.json",
         "_maintenance/FRESH_CHAT_HANDOFF.md","_maintenance/VERIFICATION.md","_maintenance/tests/test_contract.py"]
    for rel in req:
        if not (base/rel).is_file(): errors.append("missing maintenance file: "+rel)
    skill=root/"skills"/"logging"/"SKILL.md"
    if not skill.is_file(): errors.append("missing skills/logging/SKILL.md")
    else:
        s=skill.read_text(encoding="utf-8")
        if len(s.splitlines())<80: errors.append("logging skill is too shallow")
        for marker in ("LOG-PKG-001","LOG-TEST-001","_logging.py","verification"):
            if marker.lower() not in s.lower(): errors.append("logging skill missing marker "+marker)
    for name in ("MAINTENANCE.json","REVIEW.json"):
        p=base/name
        if p.is_file():
            try: reject_command_surface(read_json(p),name)
            except Exception as e: errors.append(str(e))
    ev=base/"_maintenance"/"EVIDENCE.json"
    if ev.is_file():
        try:
            e=read_json(ev)
            if e.get("runtime_tree_fingerprint")!=tree_fingerprint(root):
                errors.append("EVIDENCE runtime_tree_fingerprint does not match scikitplot/logging tree")
        except Exception as exc: errors.append(str(exc))
    return errors

def payload(root: Path):
    f=runtime_findings(root); e=maintenance_errors(root)
    return {"subsystem":"scikitplot.logging","maintenance_status":"PASS" if not e else "FAIL",
            "runtime_status":"PASS" if not f else "FAIL","release_status":"BLOCKED" if e or f else "UNVERIFIED",
            "maintenance_errors":e,"runtime_findings":f,
            "inventory":{"runtime_files":len(runtime_files(root)),
                         "focused_runtime_test_files":1 if (root/"scikitplot/logging/tests/test__logging.py").is_file() else 0},
            "runtime_tree_fingerprint":tree_fingerprint(root)}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path); ap.add_argument("--json",action="store_true"); ap.add_argument("--update",action="store_true")
    a=ap.parse_args(argv); root=a.repo.resolve() if a.repo else discover_repo(Path.cwd()); out=payload(root)
    if a.update:
        if out["runtime_status"]!="PASS": raise SystemExit("refusing --update while runtime contract is FAIL")
        ep=root/"maintenances/logging/_maintenance/EVIDENCE.json"; ev=read_json(ep); ev["runtime_tree_fingerprint"]=out["runtime_tree_fingerprint"]; ep.write_text(json.dumps(ev,indent=2)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True) if a.json else f"maintenance={out['maintenance_status']} runtime={out['runtime_status']} release={out['release_status']}")
    return 0 if out["maintenance_status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
