#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, re
from pathlib import Path, PurePosixPath

class ContractError(RuntimeError):
    pass

FORBIDDEN_META_KEYS={'command','commands','cmd','shell','exec','executable'}
TRACKED_SUFFIXES={'.py','.pyi','.json','.md','.toml','.yaml','.typed'}
ALLOWED_REVIEW_CHECKS={
    'runtime_presence','api_surface','dataset_postprocess','archive_loading',
    'upload_lifecycle','database_defaults','cli_examples','loader_test_coverage',
    'planes','handoff','inventory','evidence','tests','negative_probes','export_cli',
    'parquet','network','database_backends','integration','release'
}
KNOWN_FINDINGS={
    'DSET-TIPS-001','DSET-ZIP-001','DSET-UPL-001','DSET-UPL-002',
    'DSET-DB-001','DSET-CLI-001','DSET-TEST-001'
}

def load_json(path:Path):
    def hook(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ContractError(f'duplicate JSON key {k!r} in {path}')
            out[k]=v
        return out
    try:
        return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=hook)
    except (OSError,json.JSONDecodeError) as exc:
        raise ContractError(f'cannot load {path}: {exc}') from exc

def reject_command_surface(obj,where='metadata'):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if str(k).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f'{where} contains unsupported executable field {k!r}')
            reject_command_surface(v,where)
    elif isinstance(obj,list):
        for v in obj: reject_command_surface(v,where)

def discover_repo(start:Path)->Path:
    r=start.resolve()
    for c in [r,*r.parents]:
        if all((c/n).is_dir() for n in ('scikitplot','maintenances','skills')):
            return c
    # File-location fallback keeps tools usable from foreign working directories.
    here=Path(__file__).resolve()
    for c in [here,*here.parents]:
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
        return ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
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

def tracked(root:Path,rr:str):
    base=root/rr
    return [p for p in sorted(base.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and (p.suffix in TRACKED_SUFFIXES or p.name=='py.typed')]

def fingerprint(root:Path,rr:str):
    base=root/rr; h=hashlib.sha256(); entries=[]
    for p in tracked(root,rr):
        rel=p.relative_to(base).as_posix(); d=hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(rel.encode()+b'\0'+d.encode()+b'\n'); entries.append(rel)
    return h.hexdigest(),entries

def inventory(entries):
    def is_test(x): return 'tests' in PurePosixPath(x).parts
    return {
        'tracked_files':len(entries),
        'production_python_files':sum(x.endswith('.py') and not is_test(x) for x in entries),
        'stub_files':sum(x.endswith('.pyi') for x in entries),
        'documentation_files':sum(x.endswith('.md') for x in entries),
        'test_python_files':sum(x.endswith('.py') and is_test(x) and PurePosixPath(x).name.startswith('test') for x in entries),
        'test_support_python_files':sum(x.endswith('.py') and is_test(x) and not PurePosixPath(x).name.startswith('test') for x in entries),
    }

def validate_manifest(root,m):
    e=[]
    try: reject_command_surface(m,'MAINTENANCE.json')
    except ContractError as x: e.append(str(x))
    if m.get('schema_version')!=4: e.append('MAINTENANCE.json schema_version must be integer 4')
    if m.get('subsystem')!='scikitplot.datasets': e.append('MAINTENANCE.json subsystem must be scikitplot.datasets')
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
    expected={'tracked_files':10,'production_python_files':5,'stub_files':1,'documentation_files':1,'test_python_files':2,'test_support_python_files':1}
    for k,v in expected.items():
        if inv[k]!=v: e.append(f'datasets {k} changed: expected {v}, found {inv[k]}; review ownership before blessing inventory')
    return e

def api_surface(root,m):
    e=[]
    for rel,syms in m['runtime_contract']['required_symbols'].items():
        p=root/rel
        if not p.is_file(): continue
        have=top_symbols(p)
        for s in syms:
            if s not in have: e.append(f'{rel} is missing required top-level symbol {s}')
    initp=root/m['runtime_root']/'__init__.py'
    if initp.is_file() and 'from ._load_dataset import *' not in initp.read_text(encoding='utf-8'):
        e.append('datasets top-level curated-loader aggregation marker disappeared')
    lp=root/m['runtime_root']/'_load_dataset.py'
    if lp.is_file():
        src=lp.read_text(encoding='utf-8')
        for name in ('"get_data_home"','"get_dataset_names"','"load_dataset"'):
            if name not in src: e.append(f'_load_dataset.__all__ lost {name}')
    return e

def dataset_postprocess(root,m):
    p=root/m['runtime_root']/'_load_dataset.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    if re.search(r'\[\s*["\']Their["\']\s*,\s*["\']Fri["\']',src):
        e.append('tips categorical day levels contain "Their" instead of "Thur"; valid Thursday source values are coerced to missing (DSET-TIPS-001)')
    return e

def archive_loading(root,m):
    p=root/m['runtime_root']/'_data_loader.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    pat=r'elif\s+str\(path\)\.endswith\(["\']\.zip["\']\):\s*\n\s*get_file_from_zip\(path\)'
    if re.search(pat,src):
        e.append('ZIP branch discards get_file_from_zip(path) and falls through to reopen the archive container (DSET-ZIP-001)')
    return e

def upload_lifecycle(root,m):
    p=root/m['runtime_root']/'_data_loader.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    if 'result = load_data(tmp_path, query=clean_sql(query))' in src:
        e.append('upload_handler cleans query=None before loading, so default non-database uploads fail and are swallowed (DSET-UPL-001)')
    if 'if return_file:\n            return tmp_path' in src and 'if clean_tmp and "tmp_path" in locals() and os.path.exists(tmp_path):' in src and 'not return_file' not in src:
        e.append('upload_handler can return tmp_path and then delete it when clean_tmp=True (DSET-UPL-002)')
    return e

def database_defaults(root,m):
    p=root/m['runtime_root']/'_data_loader.py'; e=[]
    if not p.is_file(): return e
    src=p.read_text(encoding='utf-8')
    bad=('query=clean_sql(query) or "SELECT 1;"' in src) or ('else loader(path, query=clean_sql(query), **kwargs)' in src)
    if bad:
        e.append('database loader dispatch calls clean_sql(query) when query defaults to None, making the documented default query unreachable (DSET-DB-001)')
    return e

def cli_examples(root,m):
    e=[]
    for rel in ('scikitplot/datasets/_data_export.py','scikitplot/datasets/_autoscout24_tasks.py','scikitplot/datasets/data_export_recipes_autoscout24.md'):
        p=root/rel
        if not p.is_file(): continue
        txt=p.read_text(encoding='utf-8')
        if re.search(r'python\s+-m\s+scikitplot\.datasets\.[A-Za-z0-9_]+\.py\b',txt):
            e.append(f'{rel} contains python -m example with a trailing .py module suffix (DSET-CLI-001)')
    return e

def loader_test_coverage(root,m):
    for rel in m['runtime_contract']['test_files']:
        p=root/rel
        if not p.is_file():
            continue
        try:
            tree=parse_python(p)
        except ContractError:
            continue
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):
                mod=(('.'*n.level)+(n.module or ''))
                if '_data_loader' in mod:
                    return []
            elif isinstance(n,ast.Import):
                if any('_data_loader' in a.name for a in n.names):
                    return []
    return ['focused datasets tests do not import _data_loader archive/upload/database dispatch (DSET-TEST-001)']

def plane_checks(root,m):
    maint=[]; runtime=[]
    base=root/m['runtime_root']
    for p in base.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        try: tree=parse_python(p)
        except ContractError as x:
            runtime.append(str(x)); continue
        for n in ast.walk(tree):
            if isinstance(n,ast.Import): names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom): names=[('.'*n.level+(n.module or ''))]
            else: continue
            for name in names:
                stripped=name.lstrip('.')
                if stripped.startswith('maintenances') or stripped.startswith('skills'):
                    runtime.append(f'runtime plane violation in {p.relative_to(root)}:{getattr(n,"lineno",0)} imports {name}')
    return maint,runtime

def handoff(root,m):
    e=[]
    for rel in m.get('read_order',[]):
        try: p=root/safe_rel(rel)
        except ContractError as x: e.append(str(x)); continue
        if not p.is_file(): e.append(f'missing handoff/read-order file {rel}')
    skill=root/m['skill']
    if skill.is_file():
        text=skill.read_text(encoding='utf-8')
        if len(text.split())<350: e.append('datasets maintainer skill is not substantive enough')
        for marker in ('_load_dataset.py','_data_export.py','_data_loader.py','DSET-TIPS-001','DSET-ZIP-001','DSET-DB-001'):
            if marker not in text: e.append(f'datasets maintainer skill lost required topic {marker}')
    return e

def review_metadata(root,m):
    e=[]; p=root/m['maintenance_root']/'REVIEW.json'
    try:
        r=load_json(p); reject_command_surface(r,'REVIEW.json')
    except ContractError as x: return [str(x)]
    if r.get('subsystem')!='scikitplot.datasets': e.append('REVIEW.json subsystem mismatch')
    for lane in r.get('lanes',[]):
        for c in lane.get('checks',[]):
            if c not in ALLOWED_REVIEW_CHECKS: e.append(f'REVIEW.json uses unknown check {c!r}')
    ids={x.get('id') for x in r.get('known_findings',[])}
    for need in sorted(KNOWN_FINDINGS):
        if need not in ids: e.append(f'REVIEW.json lost known finding {need}')
    return e

def evidence(root,m,fp):
    e=[]; p=root/m['maintenance_root']/'_maintenance/EVIDENCE.json'
    try:
        ev=load_json(p); reject_command_surface(ev,'EVIDENCE.json')
    except ContractError as x: return [str(x)]
    if ev.get('runtime_fingerprint') not in (fp,'PENDING'):
        e.append(f'evidence runtime fingerprint drift: recorded {ev.get("runtime_fingerprint")}, actual {fp}')
    for lane in ev.get('lanes',[]):
        if lane.get('status') not in {'PASS','FAIL','UNAVAILABLE','BLOCKED'}:
            e.append(f'evidence lane {lane.get("id")!r} has invalid status')
        art=lane.get('artifact')
        if art:
            lp=root/m['maintenance_root']/'_maintenance'/safe_rel(art)
            if not lp.is_file(): e.append(f'evidence artifact missing for {lane.get("id")}: {art}')
            elif lane.get('sha256') and hashlib.sha256(lp.read_bytes()).hexdigest()!=lane['sha256']:
                e.append(f'evidence artifact hash mismatch for {lane.get("id")}')
    return e

def run_checks(root:Path):
    m=load_json(root/'maintenances/datasets/MAINTENANCE.json')
    fp,entries=fingerprint(root,m['runtime_root']); inv=inventory(entries)
    maint=[]; runtime=[]
    maint += validate_manifest(root,m); maint += handoff(root,m); maint += review_metadata(root,m)
    mp,rp=plane_checks(root,m); maint += mp; runtime += rp
    runtime += runtime_presence(root,m,inv); runtime += api_surface(root,m)
    runtime += dataset_postprocess(root,m); runtime += archive_loading(root,m)
    runtime += upload_lifecycle(root,m); runtime += database_defaults(root,m)
    runtime += cli_examples(root,m); runtime += loader_test_coverage(root,m)
    maint += evidence(root,m,fp)
    return m,fp,inv,maint,runtime

def payload(root:Path):
    m,fp,inv,me,re=run_checks(root)
    release='BLOCKED' if re or me else 'BLOCKED'  # integration evidence remains independently required
    return {
        'subsystem':m['subsystem'],
        'maintenance_status':'PASS' if not me else 'FAIL',
        'runtime_status':'PASS' if not re else 'FAIL',
        'release_status':release,
        'maintenance_errors':me,
        'runtime_findings':re,
        'inventory':inv,
        'runtime_fingerprint':fp,
    }

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); ap.add_argument('--update',action='store_true'); ap.add_argument('--repo',type=Path)
    a=ap.parse_args(argv)
    try:
        root=a.repo.resolve() if a.repo else discover_repo(Path.cwd())
        out=payload(root)
        if a.update:
            if out['maintenance_errors'] or out['runtime_findings']:
                raise ContractError('--update refuses to bless a failing maintenance/runtime contract')
            p=root/'maintenances/datasets/_maintenance/EVIDENCE.json'; ev=load_json(p); ev['runtime_fingerprint']=out['runtime_fingerprint']; p.write_text(json.dumps(ev,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(out,indent=2) if a.json else '\n'.join([f"maintenance: {out['maintenance_status']}",f"runtime: {out['runtime_status']}",f"release: {out['release_status']}",*['- '+x for x in out['maintenance_errors']+out['runtime_findings']]]))
        return 0 if out['maintenance_status']=='PASS' else 2
    except ContractError as exc:
        out={'maintenance_status':'FAIL','error':str(exc)}; print(json.dumps(out,indent=2) if a.json else str(exc)); return 2

if __name__=='__main__':
    raise SystemExit(main())
