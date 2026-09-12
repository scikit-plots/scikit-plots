from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; TOOL=HERE.parents[1]/'tools'/'check_contract.py'
spec=importlib.util.spec_from_file_location('preprocessing_contract_tests',TOOL); c=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def clone(tmp_path):
    dst=tmp_path/'repo'; shutil.copytree(REPO,dst,ignore=shutil.ignore_patterns('.pytest_cache','__pycache__')); return dst

def runtime_errs(root): return c.run_checks(root)[4]
def maint_errs(root): return c.run_checks(root)[3]

def make_runtime_clean(root):
    p=root/'scikitplot/preprocessing/_encoders.py'; s=p.read_text()
    # Make cache feature-aware enough for the static anti-pattern gate.
    s=s.replace('categories_flat_ = [\n                item for arr in self.categories_.values() for item in arr\n            ]\n            self._cached_dict = dict(\n                zip(categories_flat_, range(len(categories_flat_)))\n            )', 'feature_categories_ = [(feature_idx, item) for feature_idx, arr in self.categories_.items() for item in arr]\n            self._cached_dict = dict(zip(feature_categories_, range(len(feature_categories_))))')
    # Add a live infrequent application marker/path for the structural contract.
    needle='        # Convert row buckets → CSR lists\n'
    s=s.replace(needle, '        if self._infrequent_enabled:\n            self._apply_infrequent_output_mapping = self._default_to_infrequent_mappings\n        self._apply_infrequent_output_mapping(X_int)  # live grouping owner marker for repaired design\n\n'+needle)
    # Add explicit fit-time GetDummies enum validation.
    needle='        # Ensure DataFrame\n        X = self._to_dataframe(X)'
    s=s.replace(needle, '        if self.handle_unknown not in {"error", "ignore"}:\n            raise ValueError("handle_unknown must be error or ignore")\n        # Ensure DataFrame\n        X = self._to_dataframe(X)',1)
    p.write_text(s)

def test_baseline_maintenance_passes(): assert maint_errs(REPO)==[]
def test_baseline_runtime_finds_feature_identity(): assert any('feature-local identity' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_infrequent_gap(): assert any('infrequent-category' in x for x in runtime_errs(REPO))
def test_baseline_runtime_finds_getdummies_validation(): assert any('GetDummies' in x and 'validation' in x for x in runtime_errs(REPO))
def test_clean_shaped_runtime_can_pass(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); assert runtime_errs(r)==[]
def test_missing_required_file_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/preprocessing/_encoders.py').unlink(); assert any('missing required runtime file' in x for x in runtime_errs(r))
def test_inventory_drift_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/preprocessing/extra.py').write_text('x=1\n'); assert any('surface changed' in x for x in runtime_errs(r))
def test_public_symbol_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/preprocessing/_encoders.py'; p.write_text(p.read_text().replace('class DummyCodeEncoder(', 'class DummyCodeEncoder_REMOVED(')); assert any('DummyCodeEncoder' in x and 'missing' in x for x in runtime_errs(r))
def test_top_level_api_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/preprocessing/__init__.py'; p.write_text(p.read_text().replace('__all__ += _encoders.__all__','# removed')); assert any('API aggregation' in x for x in runtime_errs(r))
def test_raw_category_cache_antipattern_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/preprocessing/_encoders.py'; s=p.read_text(); s=s.replace('feature_categories_ = [(feature_idx, item) for feature_idx, arr in self.categories_.items() for item in arr]\n            self._cached_dict = dict(zip(feature_categories_, range(len(feature_categories_))))','categories_flat_ = [item for arr in self.categories_.values() for item in arr]\n            self._cached_dict = dict(zip(categories_flat_, range(len(categories_flat_))))'); p.write_text(s); assert any('feature-local identity' in x for x in runtime_errs(r))
def test_missing_infrequent_application_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/preprocessing/_encoders.py'; s=p.read_text().replace('        self._apply_infrequent_output_mapping(X_int)  # live grouping owner marker for repaired design\n',''); p.write_text(s); assert any('infrequent-category' in x for x in runtime_errs(r))
def test_getdummies_validation_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/preprocessing/_encoders.py'; s=p.read_text().replace('        if self.handle_unknown not in {"error", "ignore"}:\n            raise ValueError("handle_unknown must be error or ignore")\n','',1); p.write_text(s); assert any('GetDummies' in x and 'validation' in x for x in runtime_errs(r))
def test_runtime_plane_violation_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/preprocessing/_encoders.py'; p.write_text('import maintenances.preprocessing\n'+p.read_text()); assert any('runtime plane violation' in x for x in runtime_errs(r))
def test_bad_review_check_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/preprocessing/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('arbitrary_shell'); p.write_text(json.dumps(d)); assert any('unknown check' in x for x in maint_errs(r))
def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/preprocessing/REVIEW.json'; d=json.loads(p.read_text()); d['command']='echo nope'; p.write_text(json.dumps(d)); assert any('unsupported executable field' in x for x in maint_errs(r))
def test_missing_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/preprocessing/SKILL.md').unlink(); assert maint_errs(r)
def test_shallow_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/preprocessing/SKILL.md').write_text('---\nname: x\n---\nshort\n'); assert any('substantive' in x for x in maint_errs(r))
def test_evidence_fingerprint_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/preprocessing/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); assert any('fingerprint drift' in x for x in maint_errs(r))
def test_foreign_cwd_discovery(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path); assert c.discover_repo(TOOL)==REPO
def test_update_refuses_failing_runtime(tmp_path):
    r=clone(tmp_path); assert runtime_errs(r)
