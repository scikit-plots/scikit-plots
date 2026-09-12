from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
TOOL = HERE.parents[1] / 'tools' / 'check_contract.py'
spec = importlib.util.spec_from_file_location('visualkeras_contract_tests', TOOL)
c = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(c)

def clone(tmp_path):
    dst = tmp_path / 'repo'
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns('.pytest_cache','__pycache__'))
    return dst

def runtime_errs(root): return c.run_checks(root)[4]
def maint_errs(root): return c.run_checks(root)[3]

def make_runtime_clean(root):
    rr = root / 'scikitplot/visualkeras'
    p = rr / '_graph.py'; s = p.read_text()
    insert = "\n\ndef _shape_for_output(model, i):\n    shape = model.output_shape\n    if len(model.outputs) == 1 and isinstance(shape, tuple):\n        return shape\n    return shape[i]\n"
    s = s.replace('\n## Define __all__', insert + '\n## Define __all__', 1)
    s = s.replace('self_multiply(model.output_shape[i])', 'self_multiply(_shape_for_output(model, i))')
    s = s.replace('c = Circle() if i != ellipsize_after - 2 else Ellipses()', 'c = Ellipses() if units > ellipsize_after and i == ellipsize_after - 2 else Circle()')
    p.write_text(s)

    p = rr / '_layer_utils.py'; s = p.read_text()
    s = s.replace('        Layer = _lazy_import_tensorflow()\n\n        # Dynamically define', '        Layer = _lazy_import_tensorflow()\n        if Layer is None:\n            raise ImportError("Keras or TensorFlow is required for SpacingDummyLayer")\n\n        # Dynamically define', 1)
    p.write_text(s)

    p = rr / '_layered.py'; s = p.read_text()
    s = s.replace('    legend_text_spacing_offset=15,', '    legend_text_spacing_offset=None,', 1)
    s = s.replace('    # Deprecation warning for legend_text_spacing_offset\n    if legend_text_spacing_offset != 0:\n        logger.warning(', '    # Deprecation warning for explicitly supplied legend_text_spacing_offset\n    if legend_text_spacing_offset is not None:\n        logger.warning(', 1)
    s = s.replace('        )\n\n    boxes = list()', '        )\n    else:\n        legend_text_spacing_offset = 15\n\n    boxes = list()', 1)
    p.write_text(s)

    p = rr / 'tests/conftest.py'; s = p.read_text()
    s = s.replace('    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])', '    lib = tf.keras if HAS_TF else keras\n    model = lib.Sequential([lib.layers.Dense(1)])', 1)
    s = s.replace('["functional_model_tf", "functional_model_keras"]', '(["functional_model_tf"] if HAS_TF else []) + (["functional_model_keras"] if HAS_KERAS else [])')
    s = s.replace('["sequential_model_tf", "sequential_model_keras"]', '(["sequential_model_tf"] if HAS_TF else []) + (["sequential_model_keras"] if HAS_KERAS else [])')
    s = s.replace('["internal_functional_model_tf", "internal_sequential_model_tf"]', '["internal_functional_model_tf", "internal_sequential_model_tf"] if HAS_TF else []')
    old = '''            [
                "sequential_model_tf",
                "sequential_model_keras",
                "functional_model_tf",
                "functional_model_keras",
                "sequential_model_tf_with_nested",
                "sequential_model_keras_with_nested",
                "functional_model_tf_with_nested",
                "functional_model_keras_with_nested",
            ],'''
    new = '''            (["sequential_model_tf", "functional_model_tf", "sequential_model_tf_with_nested", "functional_model_tf_with_nested"] if HAS_TF else []) +
            (["sequential_model_keras", "functional_model_keras", "sequential_model_keras_with_nested", "functional_model_keras_with_nested"] if HAS_KERAS else []),'''
    s = s.replace(old, new)
    p.write_text(s)

def test_baseline_maintenance_passes(): assert maint_errs(REPO) == []
def test_baseline_finds_single_output_shape_bug(): assert any('single-output' in x for x in runtime_errs(REPO))
def test_baseline_finds_ellipsize_bug(): assert any('ellipsis placement' in x for x in runtime_errs(REPO))
def test_baseline_finds_optional_backend_bug(): assert any('SpacingDummyLayer' in x and 'TypeError' in x for x in runtime_errs(REPO))
def test_baseline_finds_default_deprecation_warning(): assert any('deprecation warning' in x for x in runtime_errs(REPO))
def test_baseline_finds_test_matrix_bug(): assert any('TensorFlow' in x and ('test_dummy_model' in x or 'fixture variants' in x) for x in runtime_errs(REPO))
def test_clean_shaped_runtime_can_pass(tmp_path):
    r = clone(tmp_path); make_runtime_clean(r); assert runtime_errs(r) == []
def test_missing_required_file_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/visualkeras/_graph.py').unlink(); assert any('missing required runtime file' in x for x in runtime_errs(r))
def test_inventory_drift_detected(tmp_path):
    r=clone(tmp_path); (r/'scikitplot/visualkeras/extra.py').write_text('x=1\n'); assert any('surface changed' in x for x in runtime_errs(r))
def test_public_symbol_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/visualkeras/_graph.py'; p.write_text(p.read_text().replace('def graph_view(', 'def graph_view_REMOVED(')); assert any('graph_view' in x and 'missing' in x for x in runtime_errs(r))
def test_top_level_api_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/visualkeras/__init__.py'; p.write_text(p.read_text().replace('"graph_view",','')); assert any('visualkeras.__all__ lost' in x for x in runtime_errs(r))
def test_single_output_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/visualkeras/_graph.py'; p.write_text(p.read_text().replace('self_multiply(_shape_for_output(model, i))','self_multiply(model.output_shape[i])')); assert any('single-output' in x for x in runtime_errs(r))
def test_ellipsize_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/visualkeras/_graph.py'; p.write_text(p.read_text().replace('c = Ellipses() if units > ellipsize_after and i == ellipsize_after - 2 else Circle()','c = Circle() if i != ellipsize_after - 2 else Ellipses()')); assert any('ellipsis placement' in x for x in runtime_errs(r))
def test_optional_backend_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/visualkeras/_layer_utils.py'; p.write_text(p.read_text().replace('        if Layer is None:\n            raise ImportError("Keras or TensorFlow is required for SpacingDummyLayer")\n','')); assert any('SpacingDummyLayer' in x and 'TypeError' in x for x in runtime_errs(r))
def test_deprecation_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/visualkeras/_layered.py'; s=p.read_text().replace('    legend_text_spacing_offset=None,','    legend_text_spacing_offset=15,').replace('    if legend_text_spacing_offset is not None:','    if legend_text_spacing_offset != 0:'); p.write_text(s); assert any('deprecation warning' in x for x in runtime_errs(r))
def test_test_matrix_regression_detected(tmp_path):
    r=clone(tmp_path); make_runtime_clean(r); p=r/'scikitplot/visualkeras/tests/conftest.py'; s=p.read_text().replace('    lib = tf.keras if HAS_TF else keras\n    model = lib.Sequential([lib.layers.Dense(1)])','    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])'); p.write_text(s); assert any('test_dummy_model' in x for x in runtime_errs(r))
def test_runtime_plane_violation_detected(tmp_path):
    r=clone(tmp_path); p=r/'scikitplot/visualkeras/_graph.py'; p.write_text('import maintenances.visualkeras\n'+p.read_text()); assert any('runtime plane violation' in x for x in runtime_errs(r))
def test_bad_review_check_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/visualkeras/REVIEW.json'; d=json.loads(p.read_text()); d['lanes'][0]['checks'].append('arbitrary_shell'); p.write_text(json.dumps(d)); assert any('unknown check' in x for x in maint_errs(r))
def test_executable_metadata_rejected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/visualkeras/REVIEW.json'; d=json.loads(p.read_text()); d['command']='echo nope'; p.write_text(json.dumps(d)); assert any('unsupported executable field' in x for x in maint_errs(r))
def test_missing_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/visualkeras/SKILL.md').unlink(); assert maint_errs(r)
def test_shallow_skill_detected(tmp_path):
    r=clone(tmp_path); (r/'skills/visualkeras/SKILL.md').write_text('---\nname: x\n---\nshort\n'); assert any('substantive' in x for x in maint_errs(r))
def test_evidence_fingerprint_drift_detected(tmp_path):
    r=clone(tmp_path); p=r/'maintenances/visualkeras/_maintenance/EVIDENCE.json'; d=json.loads(p.read_text()); d['runtime_fingerprint']='0'*64; p.write_text(json.dumps(d)); assert any('fingerprint drift' in x for x in maint_errs(r))
def test_foreign_cwd_discovery(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path); assert c.discover_repo(TOOL) == REPO
def test_update_refuses_failing_runtime(tmp_path):
    r=clone(tmp_path); assert runtime_errs(r)
