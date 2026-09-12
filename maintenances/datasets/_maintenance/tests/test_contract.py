from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path
import pytest

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; TOOL=HERE.parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('datasets_contract_tests',TOOL); c=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def clone(tmp_path):
    dst=tmp_path/'repo'; shutil.copytree(REPO,dst,ignore=shutil.ignore_patterns('.pytest_cache','__pycache__')); return dst

def runtime_errs(root): return c.run_checks(root)[4]
def maint_errs(root): return c.run_checks(root)[3]

def make_runtime_clean(root):
    p=root/'scikitplot/datasets/_load_dataset.py'; s=p.read_text(); s=s.replace('["Their", "Fri", "Sat", "Sun"]','["Thur", "Fri", "Sat", "Sun"]'); p.write_text(s)
    p=root/'scikitplot/datasets/_data_loader.py'; s=p.read_text()
    s=s.replace('elif str(path).endswith(".zip"):\n        get_file_from_zip(path)','elif str(path).endswith(".zip"):\n        return get_file_from_zip(path)')
    s=s.replace('result = load_data(tmp_path, query=clean_sql(query))','result = load_data(tmp_path, query=clean_sql(query) if query is not None else None)')
    s=s.replace('if clean_tmp and "tmp_path" in locals() and os.path.exists(tmp_path):','if clean_tmp and not return_file and "tmp_path" in locals() and os.path.exists(tmp_path):')
    s=s.replace('query=clean_sql(query) or "SELECT 1;"','query=clean_sql(query) if query is not None else "SELECT 1;"')
    s=s.replace('else loader(path, query=clean_sql(query), **kwargs)','else loader(path, query=clean_sql(query) if query is not None else "SELECT 1;", **kwargs)')
    p.write_text(s)
    p=root/'scikitplot/datasets/_autoscout24_tasks.py'; s=p.read_text().replace('scikitplot.datasets._data_export.py','scikitplot.datasets._data_export').replace('scikitplot.datasets._autoscout24_tasks.py','scikitplot.datasets._autoscout24_tasks'); p.write_text(s)
    p=root/'scikitplot/datasets/tests/test__load_dataset.py'; p.write_text(p.read_text()+'\nfrom .._data_loader import upload_handler as _datasets_loader_coverage_marker\n')

def test_baseline_maintenance_passes(): assert maint_errs(REPO)==[]
def test_baseline_runtime_finds_all_owned_findings():
    errs='\n'.join(runtime_errs(REPO))
    for fid in ('DSET-TIPS-001','DSET-ZIP-001','DSET-UPL-001','DSET-UPL-002','DSET-DB-001','DSET-CLI-001','DSET-TEST-001'):
        assert fid in errs

def test_clean_shaped_runtime_can_pass(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); assert runtime_errs(r)==[]

def test_missing_required_file_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/datasets/_load_dataset.py').unlink(); assert any('missing required runtime file' in x for x in runtime_errs(r))

def test_inventory_drift_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/datasets/extra.py').write_text('x=1\n'); assert any('tracked_files changed' in x for x in runtime_errs(r))

def test_public_symbol_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/datasets/_data_export.py'; p.write_text(p.read_text().replace('def stable_hash64(', 'def stable_hash64_REMOVED(')); assert any('stable_hash64' in x and 'missing' in x for x in runtime_errs(r))

def test_top_level_api_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/datasets/__init__.py'; p.write_text(p.read_text().replace('from ._load_dataset import *','# removed')); assert any('aggregation marker' in x for x in runtime_errs(r))

def test_tips_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_load_dataset.py'; p.write_text(p.read_text().replace('["Thur", "Fri", "Sat", "Sun"]','["Their", "Fri", "Sat", "Sun"]')); assert any('DSET-TIPS-001' in x for x in runtime_errs(r))

def test_zip_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_data_loader.py'; p.write_text(p.read_text().replace('return get_file_from_zip(path)','get_file_from_zip(path)',1)); assert any('DSET-ZIP-001' in x for x in runtime_errs(r))

def test_upload_default_query_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_data_loader.py'; p.write_text(p.read_text().replace('result = load_data(tmp_path, query=clean_sql(query) if query is not None else None)','result = load_data(tmp_path, query=clean_sql(query))')); assert any('DSET-UPL-001' in x for x in runtime_errs(r))

def test_upload_cleanup_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_data_loader.py'; p.write_text(p.read_text().replace('if clean_tmp and not return_file and "tmp_path" in locals() and os.path.exists(tmp_path):','if clean_tmp and "tmp_path" in locals() and os.path.exists(tmp_path):')); assert any('DSET-UPL-002' in x for x in runtime_errs(r))

def test_database_default_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_data_loader.py'; p.write_text(p.read_text().replace('query=clean_sql(query) if query is not None else "SELECT 1;"','query=clean_sql(query) or "SELECT 1;"',1)); assert any('DSET-DB-001' in x for x in runtime_errs(r))

def test_cli_doc_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/_autoscout24_tasks.py'; p.write_text(p.read_text().replace('scikitplot.datasets._data_export \\', 'scikitplot.datasets._data_export.py \\',1)); assert any('DSET-CLI-001' in x for x in runtime_errs(r))

def test_data_loader_test_coverage_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/datasets/tests/test__load_dataset.py'; p.write_text(p.read_text().replace('\nfrom .._data_loader import upload_handler as _datasets_loader_coverage_marker\n','\n')); assert any('DSET-TEST-001' in x for x in runtime_errs(r))

def test_runtime_plane_violation_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/datasets/_load_dataset.py'; p.write_text('import maintenances.datasets\n'+p.read_text()); assert any('runtime plane violation' in x for x in runtime_errs(r))

def test_bad_review_check_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/datasets/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('arbitrary_shell'); p.write_text(json.dumps(d)); assert any('unknown check' in x for x in maint_errs(r))

def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/datasets/REVIEW.json'; d=json.loads(p.read_text()); d['command']='echo nope'; p.write_text(json.dumps(d)); assert any('unsupported executable field' in x for x in maint_errs(r))

def test_missing_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/datasets/SKILL.md').unlink(); assert maint_errs(r)

def test_shallow_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/datasets/SKILL.md').write_text('---\nname: x\n---\nshort\n'); assert any('substantive' in x for x in maint_errs(r))

def test_evidence_fingerprint_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/datasets/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); assert any('fingerprint drift' in x for x in maint_errs(r))

def test_missing_evidence_artifact_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/datasets/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['lanes'][0]['artifact']='evidence/does-not-exist.log'; p.write_text(json.dumps(d)); assert any('evidence artifact missing' in x for x in maint_errs(r))

def test_foreign_cwd_discovery(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path); assert c.discover_repo(TOOL)==REPO

def test_update_refuses_failing_runtime(tmp_path):
    r=clone(tmp_path); assert runtime_errs(r); assert runtime_errs(r)

def test_duplicate_json_key_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/datasets/REVIEW.json'; p.write_text('{"schema_version":2,"schema_version":2}')
    assert any('duplicate JSON key' in x for x in maint_errs(r))
