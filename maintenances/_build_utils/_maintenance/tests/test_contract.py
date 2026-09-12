from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
CHECK=HERE.parent/'tools/check_contract.py'
spec=importlib.util.spec_from_file_location('buildutils_contract',CHECK); C=importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
SOURCE=C.discover_repo(HERE)

def clone(tmp_path):
    dst=tmp_path/'repo'; shutil.copytree(SOURCE,dst,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache')); return dst

def write_json(p,obj): p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')

def test_baseline_reports_maintenance_pass_runtime_fail():
    out=C.check(SOURCE); assert out['maintenance_status']=='PASS'; assert out['runtime_status']=='FAIL'; assert out['release_status']=='BLOCKED'; assert len(out['runtime_errors'])>=6

def test_checker_runs_from_foreign_cwd(tmp_path):
    r=subprocess.run([sys.executable,str(CHECK),'--json'],cwd=tmp_path,text=True,capture_output=True); assert r.returncode==0; o=json.loads(r.stdout); assert o['maintenance_status']=='PASS'

def test_bad_manifest_subsystem_fails_maintenance(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/MAINTENANCE.json'; o=json.loads(p.read_text()); o['subsystem']='wrong'; write_json(p,o); assert C.check(r)['maintenance_status']=='FAIL'

def test_unsafe_manifest_path_fails_maintenance(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/MAINTENANCE.json'; o=json.loads(p.read_text()); o['skill']='../escape'; write_json(p,o); assert C.check(r)['maintenance_status']=='FAIL'

def test_duplicate_json_key_fails_maintenance(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/REVIEW.json'; p.write_text('{"subsystem":"scikitplot._build_utils","subsystem":"x"}',encoding='utf-8'); assert C.check(r)['maintenance_status']=='FAIL'

def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/_maintenance/STATE.json'; o=json.loads(p.read_text()); o['command']='rm -rf /'; write_json(p,o); assert C.check(r)['maintenance_status']=='FAIL'

def test_unknown_review_check_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/REVIEW.json'; o=json.loads(p.read_text()); o['checks'].append('magic'); write_json(p,o); assert C.check(r)['maintenance_status']=='FAIL'

def test_missing_skill_fails_maintenance(tmp_path):
    r=clone(tmp_path); (r/'skills/_build_utils/SKILL.md').unlink(); assert C.check(r)['maintenance_status']=='FAIL'

def test_shallow_skill_fails_maintenance(tmp_path):
    r=clone(tmp_path); (r/'skills/_build_utils/SKILL.md').write_text('tiny'); assert C.check(r)['maintenance_status']=='FAIL'

def test_missing_runtime_file_is_runtime_failure(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/_build_utils/tempita.py').unlink(); out=C.check(r); assert out['maintenance_status']=='PASS'; assert any('missing required runtime file' in x for x in out['runtime_errors'])

def test_inventory_drift_is_runtime_failure(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/_build_utils/new_tool.py').write_text('x=1\n'); assert any('inventory changed' in x or 'tracked_files changed' in x for x in C.check(r)['runtime_errors'])

def test_api_loss_is_runtime_failure(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/tempita.py'; s=p.read_text(); p.write_text(s.replace('def process_tempita(', 'def process_tempita_removed(')); assert any('process_tempita' in x for x in C.check(r)['runtime_errors'])

def test_runtime_to_maintenance_plane_violation(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/system_info.py'; p.write_text(p.read_text()+'\nimport maintenances\n'); assert any('runtime plane violation' in x for x in C.check(r)['runtime_errors'])

def test_maintenance_to_runtime_plane_violation(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/_build_utils/_maintenance/tests/injected.py'; p.write_text('import scikitplot._build_utils\n'); assert C.check(r)['maintenance_status']=='FAIL'

def test_build_only_boundary_violation(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/utils/injected_build_import.py'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('import scikitplot._build_utils\n'); assert any('build-only boundary violation' in x for x in C.check(r)['runtime_errors'])

def test_cython_validator_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/cython_generate.py'; s=p.read_text(); s=s.replace('elif "{{" in content or "}}" in content:', 'elif False:  # repaired: grammar-aware validator'); s=s.replace('* Output is written atomically to the same directory as the template.','* Output is written to the same directory as the template.'); p.write_text(s); errs=C.cython_validation(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_git_error_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/gitversion.py'; s=p.read_text().replace('return (0, "", str(e))','return (1, "", str(ve))'); p.write_text(s); errs=C.git_error_contract(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_git_safe_directory_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/gitversion.py'; s=p.read_text().replace('"--global",','"--local",').replace('add_safe_directory(repo_path=git_dir)','add_safe_directory(repo_path=os.path.dirname(os.path.dirname(git_dir)))').replace('if returncode == 128:','if returncode == 128 and "dubious ownership" in stderr_str.lower():'); p.write_text(s); errs=C.git_safe_directory(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_meson_freshness_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/install_meson_features.py'; s=p.read_text().replace('import os\n','import os\nimport filecmp\n',1); s=s.replace('if os.path.getmtime(src_file) > os.path.getmtime(dst_file):','if not filecmp.cmp(src_file, dst_file, shallow=False):'); s=s.replace('    return False\n\n\ndef main', '    if set(os.listdir(dst_dir)) - set(os.listdir(src_dir)):\n        return True\n    return False\n\n\ndef main'); p.write_text(s); errs=C.meson_freshness(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_copy_semantics_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/copyfiles.py'; s=p.read_text().replace('        os.makedirs(os.path.dirname(dest), exist_ok=True)','        parent = os.path.dirname(dest)\n        if parent:\n            os.makedirs(parent, exist_ok=True)'); p.write_text(s); errs=C.copy_semantics(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_git_test_fidelity_finding_is_guarded(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/_build_utils/tests/test_gitversion.py'; p.write_text(p.read_text()+'\n\ndef test_real_api_smoke():\n    info = GitVersionInfo("1.0.0")\n    assert info.full_version == "1.0.0"\n'); errs=C.test_fidelity(r,json.loads((r/'maintenances/_build_utils/MAINTENANCE.json').read_text())); assert not errs

def test_synthetic_repaired_runtime_can_pass(tmp_path):
    r=clone(tmp_path)
    # repair cython validation + atomic claim
    p=r/'scikitplot/_build_utils/cython_generate.py'; s=p.read_text().replace('elif "{{" in content or "}}" in content:', 'elif False:').replace('* Output is written atomically to the same directory as the template.','* Output is written to the same directory as the template.'); p.write_text(s)
    # repair git shape
    p=r/'scikitplot/_build_utils/gitversion.py'; s=p.read_text().replace('return (0, "", str(e))','return (1, "", str(ve))').replace('"--global",','"--local",').replace('add_safe_directory(repo_path=git_dir)','add_safe_directory(repo_path=os.path.dirname(os.path.dirname(git_dir)))').replace('if returncode == 128:','if returncode == 128 and "dubious ownership" in stderr_str.lower():'); p.write_text(s)
    # repair meson freshness shape
    p=r/'scikitplot/_build_utils/install_meson_features.py'; s=p.read_text().replace('import os\n','import os\nimport filecmp\n',1).replace('if os.path.getmtime(src_file) > os.path.getmtime(dst_file):','if not filecmp.cmp(src_file, dst_file, shallow=False):').replace('    return False\n\n\ndef main', '    if set(os.listdir(dst_dir)) - set(os.listdir(src_dir)):\n        return True\n    return False\n\n\ndef main'); p.write_text(s)
    # repair copy
    p=r/'scikitplot/_build_utils/copyfiles.py'; p.write_text(p.read_text().replace('        os.makedirs(os.path.dirname(dest), exist_ok=True)','        parent = os.path.dirname(dest)\n        if parent:\n            os.makedirs(parent, exist_ok=True)'))
    # repair test fidelity
    p=r/'scikitplot/_build_utils/tests/test_gitversion.py'; p.write_text(p.read_text()+'\n\ndef test_real_api_smoke():\n    info = GitVersionInfo("1.0.0")\n    assert info.full_version == "1.0.0"\n')
    # Bless synthetic inventory only for the inventory contract itself.
    mp=r/'maintenances/_build_utils/MAINTENANCE.json'; m=json.loads(mp.read_text()); inv=C.inventory(r,m['runtime_root']); m['runtime_contract']['inventory'].update({k:inv[k] for k in ('tracked_files','production_python_files','test_python_files','test_support_python_files','paths','baseline_sha256')}); write_json(mp,m)
    out=C.check(r); assert out['maintenance_status']=='PASS'; assert out['runtime_status']=='PASS', out['runtime_errors']; assert out['release_status']=='PASS'
