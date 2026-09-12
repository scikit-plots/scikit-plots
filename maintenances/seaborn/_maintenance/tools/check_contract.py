#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError):
    pass

FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
TRACKED_SUFFIXES={'.py','.pyi','.json','.md','.toml','.yaml','.typed'}
ALLOWED_REVIEW_CHECKS={'runtime_presence','api_surface','seaborn_compat','model_semantics','decile_weights','hue_mapping','planes','handoff','inventory','evidence','native_tests','compatibility_harness','negative_probes','integration','release'}


def load_json(path:Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out:
                raise ContractError(f'duplicate JSON key {k!r} in {path}')
            out[k]=v
        return out
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as exc:
        raise ContractError(f'cannot load {path}: {exc}') from exc


def reject_command_surface(obj, where='metadata'):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f'{where} contains unsupported executable field {k!r}')
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj:
            reject_command_surface(v,where)


def discover_repo(start:Path)->Path:
    r=start.resolve()
    for c in [r,*r.parents]:
        if all((c/n).is_dir() for n in ('scikitplot','maintenances','skills')):
            return c
    raise ContractError('could not locate wide repository root containing scikitplot/, maintenances/, and skills/')


def safe_rel(value:str)->str:
    if not isinstance(value,str) or not value or '\\' in value or '\x00' in value:
        raise ContractError(f'unsafe repository path {value!r}')
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or '//' in value:
        raise ContractError(f'unsafe repository path {value!r}')
    return value


def parse_python(path:Path):
    try:
        return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError,SyntaxError) as exc:
        raise ContractError(f'cannot parse {path}: {exc}') from exc


def top_symbols(path:Path):
    out=set()
    for n in parse_python(path).body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            ts=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in ts:
                if isinstance(t,ast.Name): out.add(t.id)
    return out


def imports(path:Path):
    out=[]
    for n in parse_python(path).body:
        if isinstance(n,ast.Import):
            out += [(n.lineno,a.name) for a in n.names]
        elif isinstance(n,ast.ImportFrom):
            out.append((n.lineno,'.'*n.level+(n.module or '')))
    return out


def tracked(root:Path,rr:str):
    base=root/rr
    return [p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and (p.suffix in TRACKED_SUFFIXES or p.name=='py.typed')]


def fingerprint(root:Path,rr:str):
    base=root/rr; h=hashlib.sha256(); entries=[]
    for p in tracked(root,rr):
        rel=p.relative_to(base).as_posix(); d=hashlib.sha256(p.read_bytes()).hexdigest(); h.update(rel.encode()+b'\0'+d.encode()+b'\n'); entries.append(rel)
    return h.hexdigest(),entries


def inventory(entries):
    def is_test(x): return 'tests' in PurePosixPath(x).parts
    return {
        'tracked_files':len(entries),
        'production_python_files':sum(x.endswith('.py') and not is_test(x) for x in entries),
        'test_python_files':sum(x.endswith('.py') and is_test(x) and PurePosixPath(x).name.startswith('test') for x in entries),
        'test_support_python_files':sum(x.endswith('.py') and is_test(x) and not PurePosixPath(x).name.startswith('test') for x in entries),
    }


def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,'MAINTENANCE.json')
    except ContractError as x: e.append(str(x))
    if m.get('schema_version')!=4: e.append('MAINTENANCE.json schema_version must be integer 4')
    if m.get('subsystem')!='scikitplot.seaborn': e.append('MAINTENANCE.json subsystem must be scikitplot.seaborn')
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
    if inv['tracked_files']!=10: e.append('seaborn tracked runtime/test surface changed; review ownership before blessing inventory')
    if inv['production_python_files']!=5: e.append('seaborn production Python surface changed; expected __init__.py plus four implementation modules')
    if inv['test_python_files']!=4: e.append('seaborn focused test surface changed; expected four test modules')
    if inv['test_support_python_files']!=1: e.append('seaborn test support surface changed; expected tests/__init__.py only')
    return e


def api_surface(root,m):
    e=[]
    for rel,syms in m['runtime_contract']['required_symbols'].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f'{rel} is missing required top-level symbol {s}')
    init_path=root/m['runtime_root']/'__init__.py'
    if init_path.is_file():
        src=init_path.read_text(encoding='utf-8')
        for marker in ('from ._auc import aucplot','from ._confusion_matrix import evalplot','from ._decile import decileplot, print_labels','from ._model import modelplot'):
            if marker not in src: e.append(f'top-level seaborn API marker {marker!r} disappeared')
        for name in ('"aucplot"','"decileplot"','"evalplot"','"modelplot"','"print_labels"'):
            if name not in src: e.append(f'seaborn.__all__ lost {name}')
    return e


def seaborn_compat(root,m):
    e=[]
    rr=root/m['runtime_root']
    bad=[]
    for name in ('_auc.py','_confusion_matrix.py','_decile.py','_model.py'):
        p=rr/name
        if not p.is_file(): continue
        tree=parse_python(p)
        parent={}
        for n in ast.walk(tree):
            for child in ast.iter_child_nodes(n): parent[child]=n
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_default_color':
                cur=n; owner=None
                while cur in parent:
                    cur=parent[cur]
                    if isinstance(cur,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        owner=cur.name; break
                if owner not in {'_resolve_default_color','_compat_default_color'}:
                    bad.append(name); break
    if bad:
        e.append('direct seaborn private _default_color calls remain without a project-owned compatibility resolver in '+', '.join(bad)+'; current seaborn 0.13.2 / Matplotlib 3.10 path returns None for decorated Axes.plot')
    return e

def model_semantics(root,m):
    p=root/m['runtime_root']/'_model.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    # Require a real model-attribute consumption marker, not merely a signature parameter.
    if 'x_estimator.feature_importances_' not in src and 'getattr(x_estimator, "feature_importances_"' not in src and "getattr(x_estimator, 'feature_importances_'" not in src:
        e.append('modelplot advertises feature_importances but does not consume x_estimator.feature_importances_; the current implementation renders x/y confusion-matrix data instead')
    return e


def decile_weights(root,m):
    p=root/m['runtime_root']/'_decile.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    # Current broken shape explicitly names _sw then drops it and has no weighted table API.
    broken='_sw = self._prepare_subset(sub_data)' in src and 'compute_decile_table(y_true, y_score, n_deciles)' in src
    weighted_signature=('def compute_decile_table(' in src and ('sample_weight' in src[src.find('def compute_decile_table('):src.find('def compute_decile_table(')+300] or 'weights' in src[src.find('def compute_decile_table('):src.find('def compute_decile_table(')+300]))
    if broken or not weighted_signature:
        e.append('decileplot validates/extracts weights but does not forward them into decile aggregation; observation weights are silently ignored')
    return e


def hue_mapping(root,m):
    p=root/m['runtime_root']/'_decile.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    good='p.map_hue(palette=palette, order=hue_order, norm=hue_norm)' in src
    if not good:
        e.append('decileplot exposes palette/hue_order/hue_norm but does not forward them to map_hue; caller semantic mapping is silently ignored')
    return e


def plane_checks(root,m):
    maint=[]; runtime=[]; rr=root/m['runtime_root']
    for p in sorted(rr.rglob('*.py')):
        if '__pycache__' in p.parts: continue
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='maintenances' or plain.startswith('maintenances.') or plain=='skills' or plain.startswith('skills.'):
                runtime.append(f'runtime plane violation {p.relative_to(rr)}:{line}: {name}')
    live=root/m['maintenance_root']/'_maintenance'
    for p in sorted(live.rglob('*.py')):
        for line,name in imports(p):
            plain=name.lstrip('.')
            if plain=='scikitplot.seaborn' or plain.startswith('scikitplot.seaborn.'):
                maint.append(f'maintenance plane imports runtime {p.relative_to(root)}:{line}: {name}')
    return maint,runtime


def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        if not (root/safe_rel(rel)).is_file(): e.append(f'read-order file is missing: {rel}')
    skill=root/safe_rel(m['skill']); text=skill.read_text(encoding='utf-8') if skill.is_file() else ''
    if not text.startswith('---\n') or len(text.splitlines())<80: e.append('skill must be a substantive SKILL.md with YAML frontmatter')
    for marker in ('Treat seaborn private APIs as volatile','`modelplot` must be about the estimator','Weighted decile semantics must be real','Forward semantic mapping inputs','Evidence ladder','Release remains blocked'):
        if marker not in text: e.append(f'skill is missing maintainer marker {marker!r}')
    return e


def review_metadata(root,m):
    e=[]; p=root/m['maintenance_root']/'REVIEW.json'
    try: r=load_json(p); reject_command_surface(r,'REVIEW.json')
    except ContractError as x: return [str(x)]
    if r.get('subsystem')!='scikitplot.seaborn': e.append('REVIEW.json subsystem mismatch')
    for lane in r.get('lanes',[]):
        for c in lane.get('checks',[]):
            if c not in ALLOWED_REVIEW_CHECKS: e.append(f'REVIEW.json uses unknown check {c!r}')
    ids={x.get('id') for x in r.get('known_findings',[])}
    for need in ('SBN-COMPAT-001','SBN-MODEL-001','SBN-DEC-001','SBN-DEC-002','SBN-SNAPSHOT-001'):
        if need not in ids: e.append(f'REVIEW.json lost known finding {need}')
    return e


def evidence(root,m,fp):
    e=[]; p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'
    try: ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as x: return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp,'PENDING'):
        e.append(f'evidence runtime fingerprint drift: recorded {ev.get("runtime_fingerprint")}, actual {fp}')
    for gate,row in ev.get('gates',{}).items():
        if row.get('status') not in {'GREEN','RED','UNAVAILABLE','BLOCKED'}: e.append(f'evidence gate {gate!r} has invalid status')
        log=row.get('log')
        if log:
            lp=root/safe_rel(log)
            if not lp.is_file(): e.append(f'evidence log missing for {gate}: {log}')
            elif row.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=row['sha256']: e.append(f'evidence log hash mismatch for {gate}')
    return e


def run_checks(root:Path):
    m=load_json(root/'maintenances/seaborn/MAINTENANCE.json'); fp,entries=fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maint=[]; runtime=[]
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp,rp=plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += api_surface(root,m); runtime += seaborn_compat(root,m); runtime += model_semantics(root,m); runtime += decile_weights(root,m); runtime += hue_mapping(root,m)
    maint += evidence(root,m,fp)
    return m,fp,inv,maint,runtime


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); a=ap.parse_args(argv)
    try:
        root=discover_repo(Path(__file__)); m,fp,inv,me,re=run_checks(root)
        if a.update:
            if me or re: raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'; ev=load_json(p); ev['runtime_fingerprint']=fp; p.write_text(json.dumps(ev,indent=2)+'\n',encoding='utf-8')
        payload={'subsystem':m['subsystem'],'maintenance_status':'PASS' if not me else 'FAIL','runtime_status':'PASS' if not re else 'FAIL','maintenance_errors':me,'runtime_findings':re,'inventory':inv,'runtime_fingerprint':fp}
        print(json.dumps(payload,indent=2) if a.json else '\n'.join([f"maintenance: {payload['maintenance_status']}",f"runtime: {payload['runtime_status']}",*['- '+x for x in me+re]]))
        return 0 if not me else 2
    except ContractError as exc:
        payload={'maintenance_status':'FAIL','error':str(exc)}; print(json.dumps(payload,indent=2) if a.json else str(exc)); return 2

if __name__=='__main__':
    raise SystemExit(main())
