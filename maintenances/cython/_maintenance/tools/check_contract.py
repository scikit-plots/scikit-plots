#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


class ContractError(RuntimeError):
    pass


FORBIDDEN_META_KEYS = {"command", "commands", "cmd", "shell", "exec", "executable"}
TRACKED_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".pxi", ".json", ".md", ".h", ".c", ".cc", ".cpp", ".cxx"}
ALLOWED_REVIEW_CHECKS = {
    "runtime_presence", "contracts", "optionality", "security", "cache",
    "locking", "templates", "planes", "independence", "handoff",
    "inventory", "hygiene", "evidence",
}


def load_json(path: Path):
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc


def reject_command_surface(obj, where="metadata"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f"{where} contains unsupported executable field {key!r}")
            reject_command_surface(value, where)
    elif isinstance(obj, list):
        for value in obj:
            reject_command_surface(value, where)


def discover_repo(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in [resolved, *resolved.parents]:
        if all((candidate / name).is_dir() for name in ("scikitplot", "maintenances", "skills")):
            return candidate
    raise ContractError("could not locate wide repository root containing scikitplot/, maintenances/, and skills/")


def safe_rel(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ContractError(f"unsafe repository path {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "." in p.parts or "//" in value:
        raise ContractError(f"unsafe repository path {value!r}")
    return value


def safe_path(root: Path, value: str, exists=True) -> Path:
    p = root / safe_rel(value)
    if exists and not p.exists():
        raise ContractError(f"required repository path does not exist: {value}")
    return p


def parse_python(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ContractError(f"cannot parse {path}: {exc}") from exc


def top_level_symbols(path: Path) -> set[str]:
    found: set[str] = set()
    for node in parse_python(path).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
    return found


def import_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return ["." * node.level + (node.module or "")]
    return []


def module_scope_imports(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in parse_python(path).body:
        for name in import_names(node):
            out.append((getattr(node, "lineno", 0), name))
    return out


def production_python(root: Path, m: dict) -> list[Path]:
    rr = root / m["runtime_root"]
    out = []
    for p in sorted(rr.glob("*.py")):
        if p.name == "_operations_examples.py":
            # Examples are not imported by the facade and can contain self-imports.
            continue
        out.append(p)
    return out


def tracked_runtime_files(root: Path, runtime_root: str) -> list[Path]:
    rr = safe_path(root, runtime_root)
    return [p for p in sorted(rr.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix in TRACKED_SUFFIXES]


def runtime_fingerprint(root: Path, runtime_root: str) -> tuple[str, list[tuple[str, Path, bytes]]]:
    rr = safe_path(root, runtime_root)
    h = hashlib.sha256(); entries = []
    for p in tracked_runtime_files(root, runtime_root):
        rel = p.relative_to(rr).as_posix(); data = p.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
        entries.append((rel, p, data))
    return h.hexdigest(), entries


def inventory(entries) -> dict[str, int]:
    out = {"tracked_files": len(entries), "production_python_files": 0, "test_python_files": 0, "template_files": 0, "template_sources": 0, "template_metadata": 0, "markdown_files": 0}
    for rel, _, _ in entries:
        if rel.endswith(".md"): out["markdown_files"] += 1
        if rel.startswith("tests/") and rel.endswith(".py"): out["test_python_files"] += 1
        elif "/tests/" not in rel and rel.endswith(".py") and not rel.startswith("_templates/"): out["production_python_files"] += 1
        if rel.startswith("_templates/"):
            out["template_files"] += 1
            if rel.endswith((".py", ".pyx")): out["template_sources"] += 1
            if rel.endswith(".meta.json"): out["template_metadata"] += 1
    return out


def validate_manifest(root: Path, m: dict) -> list[str]:
    errors = []
    try: reject_command_surface(m, "MAINTENANCE.json")
    except ContractError as exc: errors.append(str(exc))
    if m.get("schema_version") != 4: errors.append("MAINTENANCE.json schema_version must be integer 4")
    if m.get("subsystem") != "scikitplot.cython": errors.append("MAINTENANCE.json subsystem must be scikitplot.cython")
    for key in ("runtime_root", "maintenance_root", "skill"):
        try: safe_path(root, m[key])
        except (KeyError, ContractError) as exc: errors.append(str(exc))
    return errors


def runtime_presence_checks(root: Path, m: dict, inv: dict) -> list[str]:
    errors=[]
    for rel in m["runtime_contract"]["required_files"]:
        if not (root / safe_rel(rel)).is_file(): errors.append(f"missing required runtime file {rel}")
    tests = root / safe_rel(m["runtime_contract"]["tests_root"])
    if not tests.is_dir() or len(list(tests.glob("test__*.py"))) < 35: errors.append("Cython runtime regression suite is unexpectedly small/missing")
    tr = root / safe_rel(m["runtime_contract"]["templates_root"])
    families = [p for p in tr.iterdir() if p.is_dir()] if tr.is_dir() else []
    if len(families) < 20: errors.append("template/probe family surface is unexpectedly small/missing")
    if inv["template_sources"] < 80 or inv["template_metadata"] < 80: errors.append("template source/metadata corpus is unexpectedly small")
    return errors


def contract_checks(root: Path, m: dict) -> list[str]:
    errors=[]
    for rel, required in m["runtime_contract"]["required_symbols"].items():
        p=root/safe_rel(rel)
        if not p.is_file(): errors.append(f"cannot check symbols: missing {rel}"); continue
        syms=top_level_symbols(p)
        for s in required:
            if s not in syms: errors.append(f"{rel} is missing required top-level symbol {s}")
    init=(root/m["runtime_root"]/'__init__.py').read_text(encoding='utf-8')
    if "__all__ += _api.__all__" not in init: errors.append("public facade no longer merges the API-stability registry")
    stub=root/m["runtime_root"]/'__init__.pyi'
    if not stub.is_file() or "compile_and_load" not in stub.read_text(encoding='utf-8'): errors.append("public typing stub no longer declares compile_and_load")
    return errors


def optionality_checks(root: Path, m: dict) -> list[str]:
    errors=[]; blocked=set(m["boundary_contract"]["forbid_module_scope_optional"])
    for p in production_python(root,m):
        rel=p.relative_to(root/m["runtime_root"]).as_posix()
        for line,name in module_scope_imports(p):
            plain=name.lstrip('.'); head=plain.split('.')[0]
            if head in blocked: errors.append(f"{rel}:{line} imports optional toolchain dependency {head!r} at module scope")
    return errors


def independence_checks(root: Path, m: dict) -> list[str]:
    errors=[]
    for p in production_python(root,m):
        rel=p.relative_to(root/m["runtime_root"]).as_posix()
        for line,name in module_scope_imports(p):
            plain=name.lstrip('.')
            if plain.startswith("scikitplot.") and not (plain=="scikitplot.cython" or plain.startswith("scikitplot.cython.")):
                errors.append(f"{rel}:{line} imports sibling subsystem {plain!r}")
    return errors


def security_checks(root: Path, m: dict) -> list[str]:
    rr=root/m["runtime_root"]; errors=[]
    pubp=rr/'_public.py'; secp=rr/'_security.py'
    if not pubp.is_file() or not secp.is_file():
        return errors
    pub=pubp.read_text(encoding='utf-8'); sec=secp.read_text(encoding='utf-8')
    if pub.count("_validate_build_security(") < 4: errors.append("public build facade no longer visibly routes all major entry paths through _validate_build_security")
    for marker in ("DEFAULT_SECURITY_POLICY", "strict", "validate_build_inputs", "allow_shell_metacharacters"):
        if marker not in sec: errors.append(f"security policy marker {marker!r} disappeared")
    if "shell=True" in sec and "must not use``shell=True``" in sec.replace(' ', ''):
        pass
    return errors


def cache_checks(root: Path, m: dict) -> list[str]:
    text=(root/m["runtime_root"]/'_cache.py').read_text(encoding='utf-8'); errors=[]
    for marker in (
        "CACHE_SCHEMA_VERSION",
        '"python": platform.python_version()',
        '"python_impl": platform.python_implementation()',
        '"ext_suffix": get("EXT_SUFFIX")',
        '"soabi": get("SOABI")',
        '"cython": cython_version',
        'fp["resolved_compiler_type"] = rt.compiler_type',
        'fp["resolved_cc"] = rt.cc',
        'fp["resolved_cxx"] = rt.cxx',
    ):
        if marker not in text: errors.append(f"cache fingerprint/schema marker {marker!r} disappeared")
    builder=(root/m["runtime_root"]/'_builder.py').read_text(encoding='utf-8')
    for marker in (".staging-", "_publish_atomically", "build_lock("):
        if marker not in builder: errors.append(f"transactional builder marker {marker!r} disappeared")
    return errors


def locking_checks(root: Path, m: dict) -> list[str]:
    text=(root/m["runtime_root"]/'_lock.py').read_text(encoding='utf-8'); errors=[]
    for marker in ("_DEFAULT_STALE_AFTER_S", "effective_stale_after", "uuid.uuid4", "_owned_by", "lock_dir.mkdir(exist_ok=False)"):
        if marker not in text: errors.append(f"interprocess-lock invariant marker {marker!r} disappeared")
    if "max(timeout_s, _DEFAULT_STALE_AFTER_S)" not in text: errors.append("lock stale threshold appears coupled to short/zero wait timeout again")
    return errors


def template_checks(root: Path, m: dict) -> list[str]:
    tr=root/m["runtime_contract"]["templates_root"]; errors=[]
    for meta in sorted(tr.rglob("*.meta.json")):
        source_name=meta.name[:-len(".meta.json")]
        if source_name != "package" and not meta.with_name(source_name).is_file():
            errors.append(f"template metadata has no paired source: {meta.relative_to(root)}")
        try:
            data=load_json(meta)
            if not isinstance(data,dict): errors.append(f"template metadata is not an object: {meta.relative_to(root)}")
        except ContractError as exc: errors.append(str(exc))
    readme=tr/'probe/README.md'
    if not readme.is_file() or "repro_con001.py" not in readme.read_text(encoding='utf-8'): errors.append("probe catalog/inverted lock-probe note is missing")
    return errors


def plane_checks(root: Path, m: dict) -> list[str]:
    errors=[]
    for p in production_python(root,m):
        rel=p.relative_to(root/m["runtime_root"]).as_posix()
        for line,name in module_scope_imports(p):
            plain=name.lstrip('.')
            if plain=="maintenances" or plain.startswith("maintenances.") or plain=="skills" or plain.startswith("skills."):
                errors.append(f"runtime plane violation {rel}:{line}: {name}")
    live=root/m["maintenance_root"]/'_maintenance'
    for p in sorted(live.rglob('*.py')):
        if 'history' in p.parts: continue
        for line,name in module_scope_imports(p):
            plain=name.lstrip('.')
            if plain=="scikitplot.cython" or plain.startswith("scikitplot.cython."):
                errors.append(f"maintenance plane imports runtime {p.relative_to(root)}:{line}: {name}")
    return errors


def hygiene_checks(root: Path, m: dict) -> list[str]:
    rr=root/m["runtime_root"]; errors=[]
    caches=[p for p in rr.rglob('__pycache__') if p.is_dir()]
    pycs=[p for p in rr.rglob('*.pyc') if p.is_file()]
    if caches or pycs:
        names=sorted({p.relative_to(root).as_posix() for p in caches+pycs})
        errors.append("runtime source contains interpreter bytecode/cache artifacts: " + ", ".join(names[:6]))
    return errors


def handoff_checks(root: Path, m: dict) -> list[str]:
    errors=[]
    for rel in m.get('read_order',[]):
        if not (root/safe_rel(rel)).is_file(): errors.append(f"read-order file is missing: {rel}")
    skill=root/safe_rel(m['skill'])
    text=skill.read_text(encoding='utf-8') if skill.is_file() else ''
    if not text.startswith('---\n') or len(text.splitlines()) < 45: errors.append("skill must be a substantive SKILL.md with YAML frontmatter")
    for marker in ('runtime compiler service','scikitplot.annoy','security','skip reasons','__pycache__'):
        if marker not in text: errors.append(f"skill is missing maintainer marker {marker!r}")
    return errors


def review_metadata_checks(root: Path, m: dict) -> list[str]:
    errors=[]; p=root/m["maintenance_root"]/'REVIEW.json'
    try:
        review=load_json(p); reject_command_surface(review,'REVIEW.json')
    except ContractError as exc: return [str(exc)]
    if review.get('subsystem')!='scikitplot.cython': errors.append('REVIEW.json subsystem mismatch')
    for lane in review.get('lanes',[]):
        for check in lane.get('checks',[]):
            if check not in ALLOWED_REVIEW_CHECKS: errors.append(f"REVIEW.json uses unknown check {check!r}")
    return errors


def evidence_checks(root: Path, m: dict, fingerprint: str) -> list[str]:
    errors=[]; p=root/m["maintenance_root"]/'_maintenance/EVIDENCE.json'
    try:
        ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as exc: return [str(exc)]
    recorded=ev.get('runtime_fingerprint')
    if recorded not in (fingerprint, 'PENDING'):
        errors.append(f"evidence runtime fingerprint drift: recorded {recorded}, actual {fingerprint}")
    for gate, row in ev.get('gates',{}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}: errors.append(f"evidence gate {gate!r} has invalid status")
        log=row.get('log')
        if log:
            lp=root/safe_rel(log)
            if not lp.is_file(): errors.append(f"evidence log missing for {gate}: {log}")
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=row['sha256']:
                errors.append(f"evidence log hash mismatch for {gate}")
    return errors


def run_checks(root: Path):
    manifest_path=root/'maintenances/cython/MAINTENANCE.json'
    m=load_json(manifest_path)
    fingerprint,entries=runtime_fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maintenance=[]; runtime=[]
    maintenance += validate_manifest(root,m)
    maintenance += handoff_checks(root,m)
    maintenance += review_metadata_checks(root,m)
    # Keep plane violations categorized by origin.
    planes=plane_checks(root,m)
    maintenance += [e for e in planes if e.startswith('maintenance plane')]
    runtime += [e for e in planes if e.startswith('runtime plane')]
    runtime += runtime_presence_checks(root,m,inv)
    runtime += contract_checks(root,m)
    runtime += optionality_checks(root,m)
    runtime += independence_checks(root,m)
    runtime += security_checks(root,m)
    runtime += cache_checks(root,m)
    runtime += locking_checks(root,m)
    runtime += template_checks(root,m)
    runtime += hygiene_checks(root,m)
    maintenance += evidence_checks(root,m,fingerprint)
    return m,fingerprint,inv,maintenance,runtime


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='Validate the scikitplot.cython maintenance/runtime contract.')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--update',action='store_true',help='refresh evidence fingerprint only when runtime and maintenance checks are clean')
    args=parser.parse_args(argv)
    try:
        root=discover_repo(Path(__file__))
        m,fp,inv,maint_err,runtime_err=run_checks(root)
        if args.update:
            if maint_err or runtime_err:
                raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            ep=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'; ev=load_json(ep); ev['runtime_fingerprint']=fp; ep.write_text(json.dumps(ev,indent=2)+'\n',encoding='utf-8')
        payload={
            'subsystem':'scikitplot.cython', 'maintenance_status':'PASS' if not maint_err else 'FAIL',
            'runtime_status':'PASS' if not runtime_err else 'FAIL', 'runtime_fingerprint':fp,
            'inventory':inv, 'maintenance_errors':maint_err, 'runtime_errors':runtime_err,
        }
        if args.json: print(json.dumps(payload,indent=2))
        else:
            print(f"maintenance: {payload['maintenance_status']}")
            print(f"runtime: {payload['runtime_status']}")
            for e in maint_err+runtime_err: print(f"FAIL: {e}")
        return 0 if not maint_err and not runtime_err else 1
    except ContractError as exc:
        if args.json: print(json.dumps({'subsystem':'scikitplot.cython','maintenance_status':'FAIL','runtime_status':'UNKNOWN','error':str(exc)},indent=2))
        else: print(f"error: {exc}",file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
