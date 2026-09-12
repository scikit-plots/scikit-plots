#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError):
    pass

FORBIDDEN_META_KEYS = {'command','commands','cmd','shell','exec','executable'}
TRACKED_SUFFIXES = {'.py','.pyi','.json','.md','.toml','.yaml','.typed'}
ALLOWED_REVIEW_CHECKS = {
    'runtime_presence','api_surface','single_output_shapes','ellipsize_semantics',
    'optional_backend','deprecation_semantics','test_matrix','planes','handoff',
    'inventory','evidence','keras_harness','negative_probes','rendering_dependencies',
    'integration','release'
}

def load_json(path: Path):
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ContractError(f'duplicate JSON key {k!r} in {path}')
            out[k] = v
        return out
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=hook)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f'cannot load {path}: {exc}') from exc

def reject_command_surface(obj, where='metadata'):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f'{where} contains unsupported executable field {k!r}')
            reject_command_surface(v, where)
    elif isinstance(obj, list):
        for v in obj:
            reject_command_surface(v, where)

def discover_repo(start: Path) -> Path:
    r = start.resolve()
    for c in [r, *r.parents]:
        if all((c / n).is_dir() for n in ('scikitplot','maintenances','skills')):
            return c
    raise ContractError('could not locate wide repository root containing scikitplot/, maintenances/, and skills/')

def safe_rel(value: str) -> str:
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ContractError(f'unsafe repository path {value!r}')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or '//' in value:
        raise ContractError(f'unsafe repository path {value!r}')
    return value

def parse_python(path: Path):
    try:
        return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ContractError(f'cannot parse {path}: {exc}') from exc

def top_symbols(path: Path):
    out = set()
    for n in parse_python(path).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out

def imports(path: Path):
    out = []
    for n in parse_python(path).body:
        if isinstance(n, ast.Import):
            out += [(n.lineno, a.name) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.lineno, '.' * n.level + (n.module or '')))
    return out

def tracked(root: Path, rr: str):
    base = root / rr
    return [p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and (p.suffix in TRACKED_SUFFIXES or p.name == 'py.typed')]

def fingerprint(root: Path, rr: str):
    base = root / rr
    h = hashlib.sha256(); entries = []
    for p in tracked(root, rr):
        rel = p.relative_to(base).as_posix()
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(rel.encode() + b'\0' + d.encode() + b'\n')
        entries.append(rel)
    return h.hexdigest(), entries

def inventory(entries):
    def is_test(x): return 'tests' in PurePosixPath(x).parts
    return {
        'tracked_files': len(entries),
        'production_python_files': sum(x.endswith('.py') and not is_test(x) for x in entries),
        'test_python_files': sum(x.endswith('.py') and is_test(x) and PurePosixPath(x).name.startswith('test') for x in entries),
        'test_support_python_files': sum(x.endswith('.py') and is_test(x) and not PurePosixPath(x).name.startswith('test') for x in entries),
    }

def validate_manifest(root, m):
    e = []
    try:
        reject_command_surface(m, 'MAINTENANCE.json')
    except ContractError as x:
        e.append(str(x))
    if m.get('schema_version') != 4: e.append('MAINTENANCE.json schema_version must be integer 4')
    if m.get('subsystem') != 'scikitplot.visualkeras': e.append('MAINTENANCE.json subsystem must be scikitplot.visualkeras')
    for k in ('runtime_root','maintenance_root','skill'):
        try:
            p = root / safe_rel(m[k])
            if not p.exists(): e.append(f'required repository path does not exist: {m[k]}')
        except (KeyError, ContractError) as x:
            e.append(str(x))
    return e

def runtime_presence(root, m, inv):
    e = []
    for rel in m['runtime_contract']['required_files']:
        if not (root / safe_rel(rel)).is_file(): e.append(f'missing required runtime file {rel}')
    if inv['tracked_files'] != 11: e.append('visualkeras tracked runtime/test surface changed; review ownership before blessing inventory')
    if inv['production_python_files'] != 5: e.append('visualkeras production Python surface changed; expected __init__.py plus four implementation modules')
    if inv['test_python_files'] != 4: e.append('visualkeras focused test surface changed; expected four test modules')
    if inv['test_support_python_files'] != 2: e.append('visualkeras test support surface changed; expected tests/__init__.py plus conftest.py')
    return e

def api_surface(root, m):
    e = []
    for rel, syms in m['runtime_contract']['required_symbols'].items():
        p = root / rel
        if not p.is_file(): continue
        have = top_symbols(p)
        for s in syms:
            if s not in have: e.append(f'{rel} is missing required top-level symbol {s}')
    p = root / m['runtime_root'] / '__init__.py'
    if p.is_file():
        src = p.read_text(encoding='utf-8')
        for n in ('"SpacingDummyLayer"','"graph_view"','"layered_view"'):
            if n not in src: e.append(f'visualkeras.__all__ lost {n}')
    return e

def single_output_shapes(root, m):
    p = root / m['runtime_root'] / '_graph.py'
    if not p.is_file(): return []
    src = p.read_text(encoding='utf-8')
    return ['graph_view assumes model.output_shape is always a per-output sequence; single-output inout_as_tensor=False selects None instead of the output shape'] if 'self_multiply(model.output_shape[i])' in src else []

def ellipsize_semantics(root, m):
    p = root / m['runtime_root'] / '_graph.py'
    if not p.is_file(): return []
    src = p.read_text(encoding='utf-8')
    if 'Circle() if i != ellipsize_after - 2 else Ellipses()' in src and 'units > ellipsize_after' not in src:
        return ['graph_view ellipsis placement is index-only; layers at or below ellipsize_after can be falsely rendered as truncated']
    return []

def optional_backend(root, m):
    p = root / m['runtime_root'] / '_layer_utils.py'
    if not p.is_file(): return []
    src = p.read_text(encoding='utf-8')
    anchor = 'Layer = _lazy_import_tensorflow()'
    region = src[src.index('class SpacingDummyLayer'):] if 'class SpacingDummyLayer' in src else src
    if anchor in region:
        tail = region[region.index(anchor):region.index(anchor)+700]
        if 'if Layer is None' not in tail and 'if not Layer' not in tail:
            return ['SpacingDummyLayer does not guard a missing backend Layer before dynamic subclass construction; missing Keras/TensorFlow degrades to TypeError']
    return []

def deprecation_semantics(root, m):
    p = root / m['runtime_root'] / '_layered.py'
    if not p.is_file(): return []
    src = p.read_text(encoding='utf-8')
    if 'legend_text_spacing_offset=15' in src and 'if legend_text_spacing_offset != 0:' in src:
        return ['layered_view emits the legend_text_spacing_offset deprecation warning on default invocation because the default is 15 and the warning condition is nonzero']
    return []

def test_matrix(root, m):
    p = root / m['runtime_root'] / 'tests/conftest.py'
    if not p.is_file(): return []
    src = p.read_text(encoding='utf-8'); e = []
    if 'if not (HAS_TF or HAS_KERAS)' in src:
        if 'model = tf.keras.Sequential' in src:
            e.append('visualkeras tests claim TensorFlow-or-Keras support but test_dummy_model unconditionally references tf')
        if '["functional_model_tf", "functional_model_keras"]' in src or '"sequential_model_tf",\n                "sequential_model_keras"' in src:
            e.append('visualkeras tests schedule TensorFlow fixture variants even when HAS_TF is false; Keras-only environments error instead of skipping unavailable lanes')
    return e

def plane_checks(root, m):
    me = []; re = []; rr = root / m['runtime_root']
    for p in rr.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        for line, name in imports(p):
            normalized = name.lstrip('.')
            if normalized.startswith('maintenances') or normalized.startswith('skills'):
                re.append(f'runtime plane violation: {p.relative_to(root)}:{line} imports {name}')
    mr = root / m['maintenance_root']
    for p in mr.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        for line, name in imports(p):
            normalized = name.lstrip('.')
            if normalized == 'scikitplot.visualkeras' or normalized.startswith('scikitplot.visualkeras.'):
                me.append(f'maintenance plane violation: {p.relative_to(root)}:{line} imports runtime package instead of inspecting source')
    return me, re

def handoff(root, m):
    e = []
    for rel in m.get('read_order', []):
        try:
            p = root / safe_rel(rel)
            if not p.is_file(): e.append(f'read-order file is missing: {rel}')
        except ContractError as x:
            e.append(str(x))
    sp = root / m['skill']
    if not sp.is_file(): e.append('visualkeras maintainer skill is missing')
    elif len(sp.read_text(encoding='utf-8').split()) < 300: e.append('visualkeras maintainer skill is not substantive enough to hand off ownership safely')
    return e

def review_metadata(root, m):
    e = []; p = root / m['maintenance_root'] / 'REVIEW.json'
    try:
        d = load_json(p); reject_command_surface(d, 'REVIEW.json')
    except ContractError as x:
        return [str(x)]
    if d.get('schema_version') != 2: e.append('REVIEW.json schema_version must be 2')
    if d.get('subsystem') != m['subsystem']: e.append('REVIEW.json subsystem mismatch')
    for lane in d.get('lanes', []):
        for check in lane.get('checks', []):
            if check not in ALLOWED_REVIEW_CHECKS: e.append(f'REVIEW.json contains unknown check {check!r}')
    ids = {x.get('id') for x in d.get('known_findings', [])}
    for need in ('VKR-GRAPH-001','VKR-GRAPH-002','VKR-OPT-001','VKR-LAY-001','VKR-TEST-001'):
        if need not in ids: e.append(f'REVIEW.json lost known finding {need}')
    return e

def evidence(root, m, fp):
    e = []; p = root / m['maintenance_root'] / '_maintenance/EVIDENCE.json'
    try:
        ev = load_json(p); reject_command_surface(ev, 'EVIDENCE.json')
    except ContractError as x:
        return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp, 'PENDING'):
        e.append(f'evidence runtime fingerprint drift: recorded {ev.get("runtime_fingerprint")}, actual {fp}')
    for gate, row in ev.get('gates', {}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}:
            e.append(f'evidence gate {gate!r} has invalid status')
        log = row.get('log')
        if log:
            lp = root / safe_rel(log)
            if not lp.is_file(): e.append(f'evidence log missing for {gate}: {log}')
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest() != row['sha256']:
                e.append(f'evidence log hash mismatch for {gate}')
    return e

def run_checks(root: Path):
    m = load_json(root/'maintenances/visualkeras/MAINTENANCE.json')
    fp, entries = fingerprint(root, m['runtime_root']); inv = inventory(entries)
    maint = []; runtime = []
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp, rp = plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += api_surface(root,m); runtime += single_output_shapes(root,m); runtime += ellipsize_semantics(root,m); runtime += optional_backend(root,m); runtime += deprecation_semantics(root,m); runtime += test_matrix(root,m)
    maint += evidence(root,m,fp)
    return m, fp, inv, maint, runtime

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--json', action='store_true'); ap.add_argument('--update', action='store_true'); a = ap.parse_args(argv)
    try:
        root = discover_repo(Path(__file__)); m, fp, inv, me, re = run_checks(root)
        if a.update:
            if me or re: raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            p = root / m['maintenance_root'] / '_maintenance/EVIDENCE.json'; ev = load_json(p); ev['runtime_fingerprint'] = fp; p.write_text(json.dumps(ev,indent=2)+'\n',encoding='utf-8')
        payload = {'subsystem':m['subsystem'],'maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','maintenance_errors':me,'runtime_findings':re,'inventory':inv,'runtime_fingerprint':fp}
        print(json.dumps(payload,indent=2) if a.json else '\n'.join([f"maintenance: {payload['maintenance_status']}",f"runtime: {payload['runtime_status']}",*['- '+x for x in me+re]]))
        return 0 if not me else 2
    except ContractError as exc:
        payload={'maintenance_status':'FAIL','error':str(exc)}; print(json.dumps(payload,indent=2) if a.json else str(exc)); return 2

if __name__ == '__main__':
    raise SystemExit(main())
