\
from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=next(p for p in HERE.parents if all((p/n).is_dir() for n in ('scikitplot','maintenances','skills')))
TOOL=REPO/'maintenances/impute/_maintenance/tools/check_contract.py'
spec=importlib.util.spec_from_file_location('impute_contract',TOOL); contract=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(contract)

def make_repo(tmp_path:Path, clean_findings=False):
    root=tmp_path/'repo'; (root/'scikitplot').mkdir(parents=True); (root/'maintenances').mkdir(); (root/'skills').mkdir()
    shutil.copytree(REPO/'scikitplot/impute',root/'scikitplot/impute')
    shutil.copytree(REPO/'maintenances/impute',root/'maintenances/impute')
    shutil.copytree(REPO/'skills/impute',root/'skills/impute')
    ev=root/'maintenances/impute/_maintenance/EVIDENCE.json'; ev.write_text(json.dumps({'schema_version':2,'subsystem':'scikitplot.impute','runtime_fingerprint':'PENDING','gates':{}},indent=2)+'\n')
    if clean_findings:
        p=root/'scikitplot/impute/_ann.py'; s=p.read_text(); s=s.replace('import pandas as pd\n',''); s=s.replace('except Exception:  # pragma: no cover - fallback to external annoy  # noqa: BLE001','except ImportError:  # pragma: no cover - fallback only when in-tree backend is unavailable')
        p.write_text(s)
    return root

def errs(root): return contract.run_checks(root)[3:]

def test_current_tree_maintenance_pass_runtime_fails_for_known_local_findings():
    _,_,_,m,r=contract.run_checks(REPO); assert m==[]; assert any('catches Exception' in x for x in r); assert any('imports pandas' in x for x in r)

def test_clean_synthetic_runtime_contract_passes(tmp_path):
    root=make_repo(tmp_path,clean_findings=True); m,r=errs(root); assert m==[]; assert r==[]

def test_missing_runtime_file_detected(tmp_path):
    root=make_repo(tmp_path,True); (root/'scikitplot/impute/_privacy.py').unlink(); _,r=errs(root); assert any('missing required runtime file' in x for x in r)

def test_annoy_owner_drift_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_ann.py'; p.write_text(p.read_text().replace('from ..annoy._annoy import Index as AnnoyIndex','from ..cexternals._annoy import AnnoyIndex')); _,r=errs(root); assert any('backend no longer points' in x for x in r)

def test_broad_annoy_fallback_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_ann.py'; p.write_text(p.read_text().replace('except ImportError:  # pragma: no cover - fallback only when in-tree backend is unavailable','except Exception:  # pragma: no cover - fallback to external annoy  # noqa: BLE001')); _,r=errs(root); assert any('catches Exception' in x for x in r)

def test_unused_module_scope_pandas_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_ann.py'; p.write_text(p.read_text().replace('import numpy as np\n','import numpy as np\nimport pandas as pd\n')); _,r=errs(root); assert any('imports pandas' in x for x in r)

def test_voyager_optional_contract_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_ann.py'; p.write_text(p.read_text().replace('except ImportError as e', 'except Exception as e')); _,r=errs(root); assert any('Voyager optional-import' in x for x in r)

def test_experimental_gate_drift_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/__init__.py'; p.write_text(p.read_text().replace('enable_ann_imputer','enable_removed')); _,r=errs(root); assert any('enable_ann_imputer' in x for x in r)

def test_privacy_boundary_marker_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_privacy.py'; p.write_text(p.read_text().replace('hard security boundary','absolute security boundary')); _,r=errs(root); assert any('security boundary' in x for x in r)

def test_runtime_to_maintenance_plane_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/impute/_privacy.py'; p.write_text('import maintenances.impute\n'+p.read_text()); _,r=errs(root); assert any('runtime plane violation' in x for x in r)

def test_maintenance_to_runtime_plane_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/impute/_maintenance/review_subsystem.py'; p.write_text('import scikitplot.impute\n'+p.read_text()); m,_=errs(root); assert any('maintenance plane imports runtime' in x for x in m)

def test_skill_must_be_substantive(tmp_path):
    root=make_repo(tmp_path,True); (root/'skills/impute/SKILL.md').write_text('---\nname: x\n---\n'); m,_=errs(root); assert any('substantive SKILL.md' in x for x in m)

def test_executable_metadata_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/impute/MAINTENANCE.json'; d=json.loads(p.read_text()); d['shell']='rm -rf /'; p.write_text(json.dumps(d)); m,_=errs(root); assert any('unsupported executable field' in x for x in m)

def test_unknown_review_check_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/impute/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('magic'); p.write_text(json.dumps(d)); m,_=errs(root); assert any("unknown check 'magic'" in x for x in m)

def test_known_findings_cannot_disappear_from_review_metadata(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/impute/REVIEW.json'; d=json.loads(p.read_text()); d['known_findings']=[]; p.write_text(json.dumps(d)); m,_=errs(root); assert any('lost known finding' in x for x in m)

def test_evidence_fingerprint_drift_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/impute/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); m,_=errs(root); assert any('fingerprint drift' in x for x in m)

def test_update_refuses_to_bless_current_failing_runtime():
    cp=subprocess.run([sys.executable,'-B',str(REPO/'maintenances/impute/_maintenance/check_trackers.py'),'--update'],cwd='/tmp',text=True,capture_output=True); assert cp.returncode!=0; assert 'refuses to bless' in cp.stdout+cp.stderr

def test_wrapper_discovers_repo_from_foreign_cwd():
    cp=subprocess.run([sys.executable,'-B',str(REPO/'maintenances/impute/_maintenance/check_trackers.py'),'--json'],cwd='/tmp',text=True,capture_output=True); assert cp.returncode==1; d=json.loads(cp.stdout); assert d['maintenance_status']=='PASS'; assert d['runtime_status']=='FAIL'; assert d['subsystem']=='scikitplot.impute'
