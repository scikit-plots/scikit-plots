from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=next(p for p in HERE.parents if all((p/n).is_dir() for n in ('scikitplot','maintenances','plugins')))
TOOL=REPO/'maintenances/mlflow/_maintenance/tools/check_contract.py'
spec=importlib.util.spec_from_file_location('mlflow_contract',TOOL); contract=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(contract)

def make_repo(tmp_path:Path, clean_findings=False):
    root=tmp_path/'repo'; (root/'scikitplot').mkdir(parents=True); (root/'maintenances').mkdir(); (root/'plugins').mkdir()
    shutil.copytree(REPO/'scikitplot/mlflow',root/'scikitplot/mlflow'); shutil.copytree(REPO/'maintenances/mlflow',root/'maintenances/mlflow'); shutil.copytree(REPO/'plugins/scikitplot-dev/skills/scikitplot-mlflow-dev',root/'plugins/scikitplot-dev/skills/scikitplot-mlflow-dev')
    ev=root/'maintenances/mlflow/_maintenance/EVIDENCE.json'; ev.write_text(json.dumps({'schema_version':2,'subsystem':'scikitplot.mlflow','runtime_fingerprint':'PENDING','gates':{}},indent=2)+'\n')
    if clean_findings:
        # Wire the active policy markers into server/session synthetically for gate behavior.
        for name in ('_server.py','_session.py'):
            p=root/'scikitplot/mlflow'/name; s=p.read_text(); s=s.replace('from __future__ import annotations\n', 'from __future__ import annotations\nfrom ._security import get_security_policy\n', 1); p.write_text(s)
        for name in ('_project.py','_workflow.py'):
            p=root/'scikitplot/mlflow'/name; p.write_text(p.read_text().replace('except Exception as e:', 'except ImportError as e:'))
        p=root/'scikitplot/mlflow/_session.pyi'; p.write_text(p.read_text().replace('from _facade import ArtifactsFacade, ModelsFacade','from ._facade import ArtifactsFacade, ModelsFacade'))
    return root

def errs(root): return contract.run_checks(root)[3:]

def test_current_tree_maintenance_pass_runtime_fails_for_known_findings():
    _,_,_,m,r=contract.run_checks(REPO); assert m==[]; assert any('SecurityPolicy is not consulted' in x for x in r); assert any('PyYAML import' in x for x in r); assert any('_session.pyi uses absolute' in x for x in r)

def test_clean_synthetic_runtime_contract_passes(tmp_path):
    root=make_repo(tmp_path,True); m,r=errs(root); assert m==[]; assert r==[]

def test_missing_runtime_file_detected(tmp_path):
    root=make_repo(tmp_path,True); (root/'scikitplot/mlflow/_server.py').unlink(); _,r=errs(root); assert any('missing required runtime file' in x for x in r)

def test_security_disconnection_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_server.py'; p.write_text(p.read_text().replace('from ._security import get_security_policy\n','')); p=root/'scikitplot/mlflow/_session.py'; p.write_text(p.read_text().replace('from ._security import get_security_policy\n','')); _,r=errs(root); assert any('SecurityPolicy is not consulted' in x for x in r)

def test_broad_yaml_import_exception_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_workflow.py'; p.write_text(p.read_text().replace('except ImportError as e:', 'except Exception as e:',1)); _,r=errs(root); assert any('PyYAML import' in x for x in r)

def test_stub_absolute_import_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_session.pyi'; p.write_text(p.read_text().replace('from ._facade import ArtifactsFacade, ModelsFacade','from _facade import ArtifactsFacade, ModelsFacade')); _,r=errs(root); assert any('_session.pyi uses absolute' in x for x in r)

def test_shell_true_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_server.py'; p.write_text(p.read_text()+'\n# shell=True\n'); _,r=errs(root); assert any('shell=True' in x for x in r)

def test_current_interpreter_server_command_marker_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_server.py'; p.write_text(p.read_text().replace('[sys.executable, "-m", "mlflow", "server", *args]','["mlflow", "server", *args]')); _,r=errs(root); assert any('sys.executable' in x for x in r)

def test_readiness_timeout_marker_detected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_readiness.py'; p.write_text(p.read_text().replace('timeout=request_timeout_s','timeout=None')); _,r=errs(root); assert any('readiness bounding marker' in x for x in r)

def test_pytyped_required(tmp_path):
    root=make_repo(tmp_path,True); (root/'scikitplot/mlflow/py.typed').unlink(); _,r=errs(root); assert any('py.typed' in x for x in r)

def test_runtime_to_maintenance_plane_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'scikitplot/mlflow/_env.py'; p.write_text('import maintenances.mlflow\n'+p.read_text()); _,r=errs(root); assert any('runtime plane violation' in x for x in r)

def test_maintenance_to_runtime_plane_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/mlflow/_maintenance/review_subsystem.py'; p.write_text('import scikitplot.mlflow\n'+p.read_text()); m,_=errs(root); assert any('maintenance plane imports runtime' in x for x in m)

def test_skill_must_be_substantive(tmp_path):
    root=make_repo(tmp_path,True); (root/'plugins/scikitplot-dev/skills/scikitplot-mlflow-dev/SKILL.md').write_text('---\nname: x\n---\n'); m,_=errs(root); assert any('substantive SKILL.md' in x for x in m)

def test_executable_metadata_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/mlflow/MAINTENANCE.json'; d=json.loads(p.read_text()); d['shell']='rm -rf /'; p.write_text(json.dumps(d)); m,_=errs(root); assert any('unsupported executable field' in x for x in m)

def test_unknown_review_check_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/mlflow/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('magic'); p.write_text(json.dumps(d)); m,_=errs(root); assert any("unknown check 'magic'" in x for x in m)

def test_known_findings_cannot_disappear(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/mlflow/REVIEW.json'; d=json.loads(p.read_text()); d['known_findings']=[]; p.write_text(json.dumps(d)); m,_=errs(root); assert any('lost known finding' in x for x in m)

def test_evidence_fingerprint_drift_rejected(tmp_path):
    root=make_repo(tmp_path,True); p=root/'maintenances/mlflow/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); m,_=errs(root); assert any('fingerprint drift' in x for x in m)

def test_update_refuses_to_bless_current_failing_runtime():
    cp=subprocess.run([sys.executable,'-B',str(REPO/'maintenances/mlflow/_maintenance/check_trackers.py'),'--update'],cwd='/tmp',text=True,capture_output=True); assert cp.returncode!=0; assert 'refuses to bless' in cp.stdout+cp.stderr

def test_wrapper_discovers_repo_from_foreign_cwd():
    cp=subprocess.run([sys.executable,'-B',str(REPO/'maintenances/mlflow/_maintenance/check_trackers.py'),'--json'],cwd='/tmp',text=True,capture_output=True); assert cp.returncode==1; d=json.loads(cp.stdout); assert d['maintenance_status']=='PASS'; assert d['runtime_status']=='FAIL'; assert d['subsystem']=='scikitplot.mlflow'
