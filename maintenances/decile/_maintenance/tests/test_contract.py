from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path
import pytest
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; TOOL=HERE.parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('decile_contract_tests',TOOL); c=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def clone(tmp_path):
    dst=tmp_path/'repo'; shutil.copytree(REPO,dst,ignore=shutil.ignore_patterns('.pytest_cache','__pycache__')); return dst

def errs(root): return c.run_checks(root)[3:]
def runtime_errs(root): return c.run_checks(root)[4]
def maint_errs(root): return c.run_checks(root)[3]

def make_runtime_clean(root):
    p=root/'scikitplot/decile/kds/_kds.py'; s=p.read_text()
    # four direct plot calls
    s=s.replace('pl = decile_table(y_true, y_score)','pl = decile_table(y_true, y_score, pos_label=pos_label, class_index=class_index)')
    s=s.replace('pldw = decile_table(y_true, y_score)','pldw = decile_table(y_true, y_score, pos_label=pos_label, class_index=class_index)')
    s=s.replace('pcg = decile_table(y_true, y_score)','pcg = decile_table(y_true, y_score, pos_label=pos_label, class_index=class_index)')
    s=s.replace('pks = decile_table(y_true, y_score)','pks = decile_table(y_true, y_score, pos_label=pos_label, class_index=class_index)')
    s=s.replace('round_decimal=digits,\n        feature_infos=feature_infos,','digits=digits,\n        feature_infos=feature_infos,\n        pos_label=pos_label,\n        class_index=class_index,')
    for child in ('plot_lift','plot_lift_decile_wise','plot_cumulative_gain','plot_ks_statistic'):
        needle=f'{child}(\n        y_true,\n        y_score,'
        repl=f'{child}(\n        y_true,\n        y_score,\n        pos_label=pos_label,\n        class_index=class_index,'
        s=s.replace(needle,repl)
    p.write_text(s)
    p=root/'scikitplot/decile/modelplotpy/_modelplotpy.py'; s=p.read_text(); s=s.replace('                    np.random.seed(self.seed)\n',''); p.write_text(s)

def test_baseline_maintenance_passes(): assert maint_errs(REPO)==[]
def test_baseline_runtime_finds_kds(): assert any('KDS' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_rng(): assert any('NumPy RNG' in x for x in runtime_errs(REPO))
def test_clean_shaped_runtime_can_pass(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); assert runtime_errs(r)==[]
def test_missing_required_file_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/decile/_decile_modelplotpy.py').unlink(); assert any('missing required runtime file' in x for x in runtime_errs(r))
def test_inventory_drift_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/decile/extra.py').write_text('x=1\n'); assert any('surface changed' in x for x in runtime_errs(r))
def test_public_symbol_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/decile/_decile_modelplotpy.py'; p.write_text(p.read_text().replace('def summarize_selection(', 'def summarize_selection_REMOVED(')); assert any('summarize_selection' in x for x in runtime_errs(r))
def test_top_level_api_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/decile/__init__.py'; p.write_text(p.read_text().replace('__all__ += _dmpy.__all__','# removed')); assert any('API aggregation' in x for x in runtime_errs(r))
def test_kds_plot_argument_drop_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/decile/kds/_kds.py'; p.write_text(p.read_text().replace('pl = decile_table(y_true, y_score, pos_label=pos_label, class_index=class_index)','pl = decile_table(y_true, y_score)')); assert any('plot_lift drops' in x for x in runtime_errs(r))
def test_kds_report_digits_alias_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/decile/kds/_kds.py'; p.write_text(p.read_text().replace('digits=digits,\n        feature_infos=feature_infos,','round_decimal=digits,\n        feature_infos=feature_infos,')); assert any('round_decimal' in x or 'report drops' in x for x in runtime_errs(r))
def test_kds_report_child_drop_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/decile/kds/_kds.py'; p.write_text(p.read_text().replace('plot_lift(\n        y_true,\n        y_score,\n        pos_label=pos_label,\n        class_index=class_index,','plot_lift(\n        y_true,\n        y_score,')); assert any('report drops selection' in x for x in runtime_errs(r))
def test_rng_seed_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/decile/modelplotpy/_modelplotpy.py'; p.write_text(p.read_text().replace('prob_plus_smallrandom = _range01(', 'np.random.seed(self.seed)\n                    prob_plus_smallrandom = _range01(',1)); assert any('NumPy RNG' in x for x in runtime_errs(r))
def test_runtime_plane_violation_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/decile/_decile_modelplotpy.py'; p.write_text('import maintenances.decile\n'+p.read_text()); assert any('runtime plane violation' in x for x in runtime_errs(r))
def test_bad_review_check_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/decile/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('arbitrary_shell'); p.write_text(json.dumps(d)); assert any('unknown check' in x for x in maint_errs(r))
def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/decile/REVIEW.json'; d=json.loads(p.read_text()); d['command']='echo nope'; p.write_text(json.dumps(d)); assert any('unsupported executable field' in x for x in maint_errs(r))
def test_missing_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/decile/SKILL.md').unlink(); assert maint_errs(r)
def test_shallow_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/decile/SKILL.md').write_text('---\nname: x\n---\nshort\n'); assert any('substantive' in x for x in maint_errs(r))
def test_foreign_cwd_discovery(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path); assert c.discover_repo(TOOL)==REPO
def test_update_refuses_failing_runtime(tmp_path):
    r=clone(tmp_path); # baseline is failing by design; contract itself must expose errors
    assert runtime_errs(r)
