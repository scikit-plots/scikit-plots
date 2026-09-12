from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; TOOL=HERE.parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('seaborn_contract_tests',TOOL); c=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def clone(tmp_path):
    dst=tmp_path/'repo'; shutil.copytree(REPO,dst,ignore=shutil.ignore_patterns('.pytest_cache','__pycache__')); return dst

def runtime_errs(root): return c.run_checks(root)[4]
def maint_errs(root): return c.run_checks(root)[3]

def make_runtime_clean(root):
    rr=root/'scikitplot/seaborn'
    # Harness-shaped compatibility seam: each implementation now resolves color locally.
    for name in ('_auc.py','_confusion_matrix.py','_decile.py','_model.py'):
        p=rr/name; s=p.read_text()
        insert='''\n\ndef _resolve_default_color(method, hue, color, kws):\n    out = _default_color(method, hue, color, kws)\n    return "C0" if out is None and hue is None else out\n'''
        anchor='\n# Define __all__'
        if anchor in s:
            s=s.replace(anchor,insert+anchor,1)
        s=s.replace('_default_color(', '_resolve_default_color(')
        s=s.replace('def _resolve_resolve_default_color', 'def _resolve_default_color')
        # restore the helper's own call to the upstream function
        s=s.replace('out = _resolve_default_color(method, hue, color, kws)', 'out = _default_color(method, hue, color, kws)',1)
        p.write_text(s)
    # Make modelplot consume estimator importances structurally.
    p=rr/'_model.py'; s=p.read_text(); marker='    kind = (kind and kind.lower().strip()) or "feature_importances"\n'
    s=s.replace(marker, marker+'    if x_estimator is not None:\n        _feature_importances = x_estimator.feature_importances_\n',1); p.write_text(s)
    # Make decile compute API weight-aware and forward the extracted vector.
    p=rr/'_decile.py'; s=p.read_text(); s=s.replace('        n_deciles,\n        **kws,\n', '        n_deciles,\n        sample_weight=None,\n        **kws,\n',1)
    s=s.replace('agg = self.compute_decile_table(y_true, y_score, n_deciles)', 'agg = self.compute_decile_table(y_true, y_score, n_deciles, sample_weight=_sw)')
    s=s.replace('p.map_hue()  # default mapping; callers can later set palette via more advanced API if needed','p.map_hue(palette=palette, order=hue_order, norm=hue_norm)')
    p.write_text(s)

def test_baseline_maintenance_passes(): assert maint_errs(REPO)==[]
def test_baseline_runtime_finds_private_color_compat(): assert any('private _default_color' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_model_semantics(): assert any('x_estimator.feature_importances_' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_ignored_weights(): assert any('weights' in x and 'ignored' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_hue_mapping(): assert any('palette/hue_order/hue_norm' in x for x in runtime_errs(REPO))
def test_clean_shaped_runtime_can_pass(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); assert runtime_errs(r)==[]
def test_missing_required_file_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/seaborn/_auc.py').unlink(); assert any('missing required runtime file' in x for x in runtime_errs(r))
def test_inventory_drift_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/seaborn/extra.py').write_text('x=1\n'); assert any('surface changed' in x for x in runtime_errs(r))
def test_public_symbol_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/seaborn/_auc.py'; p.write_text(p.read_text().replace('def aucplot(', 'def aucplot_REMOVED(')); assert any('aucplot' in x and 'missing' in x for x in runtime_errs(r))
def test_top_level_api_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/seaborn/__init__.py'; p.write_text(p.read_text().replace('from ._model import modelplot','# removed')); assert any('top-level seaborn API' in x for x in runtime_errs(r))
def test_private_color_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/seaborn/_auc.py'; s=p.read_text().replace('_resolve_default_color(ax.plot, None, None, {})','_default_color(ax.plot, None, None, {})'); p.write_text(s); assert any('private _default_color' in x for x in runtime_errs(r))
def test_model_estimator_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/seaborn/_model.py'; s=p.read_text().replace('    if x_estimator is not None:\n        _feature_importances = x_estimator.feature_importances_\n',''); p.write_text(s); assert any('x_estimator.feature_importances_' in x for x in runtime_errs(r))
def test_decile_weight_forwarding_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/seaborn/_decile.py'; s=p.read_text().replace('agg = self.compute_decile_table(y_true, y_score, n_deciles, sample_weight=_sw)','agg = self.compute_decile_table(y_true, y_score, n_deciles)'); s=s.replace('        sample_weight=None,\n',''); p.write_text(s); assert any('weights' in x and 'ignored' in x for x in runtime_errs(r))
def test_hue_mapping_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/seaborn/_decile.py'; s=p.read_text().replace('p.map_hue(palette=palette, order=hue_order, norm=hue_norm)','p.map_hue()'); p.write_text(s); assert any('palette/hue_order/hue_norm' in x for x in runtime_errs(r))
def test_runtime_plane_violation_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/seaborn/_auc.py'; p.write_text('import maintenances.seaborn\n'+p.read_text()); assert any('runtime plane violation' in x for x in runtime_errs(r))
def test_bad_review_check_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/seaborn/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('arbitrary_shell'); p.write_text(json.dumps(d)); assert any('unknown check' in x for x in maint_errs(r))
def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/seaborn/REVIEW.json'; d=json.loads(p.read_text()); d['command']='echo nope'; p.write_text(json.dumps(d)); assert any('unsupported executable field' in x for x in maint_errs(r))
def test_missing_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/seaborn/SKILL.md').unlink(); assert maint_errs(r)
def test_shallow_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/seaborn/SKILL.md').write_text('---\nname: x\n---\nshort\n'); assert any('substantive' in x for x in maint_errs(r))
def test_evidence_fingerprint_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/seaborn/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); assert any('fingerprint drift' in x for x in maint_errs(r))
def test_foreign_cwd_discovery(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path); assert c.discover_repo(TOOL)==REPO
def test_update_refuses_failing_runtime(tmp_path):
    r=clone(tmp_path); assert runtime_errs(r)
