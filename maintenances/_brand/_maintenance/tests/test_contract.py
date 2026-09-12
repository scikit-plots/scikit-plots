from __future__ import annotations
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=next(p for p in HERE.parents if (p/"scikitplot").is_dir() and (p/"maintenances").is_dir() and (p/"skills").is_dir()); CHECK=REPO/"maintenances/_brand/_maintenance/tools/check_contract.py"
spec=importlib.util.spec_from_file_location("brand_contract",CHECK); C=importlib.util.module_from_spec(spec); spec.loader.exec_module(C)

def mini(tmp_path):
    r=tmp_path/"repo"
    shutil.copytree(REPO/"scikitplot/_brand",r/"scikitplot/_brand",ignore=shutil.ignore_patterns("__pycache__",".pytest_cache"))
    shutil.copytree(REPO/"maintenances/_brand",r/"maintenances/_brand",ignore=shutil.ignore_patterns("__pycache__",".pytest_cache"))
    shutil.copytree(REPO/"skills/_brand",r/"skills/_brand",ignore=shutil.ignore_patterns("__pycache__",".pytest_cache"))
    return r

def repair(r):
    (r/"scikitplot/_brand/__init__.py").write_text("# Branding\nfrom __future__ import annotations\n__all__ = []\n",encoding="utf-8")
    p=r/"scikitplot/_brand/_logo.py"; p.write_text(p.read_text(encoding="utf-8").replace("python -m scikitplot.logo","python -m scikitplot._brand._logo"),encoding="utf-8")
    p=r/"scikitplot/_brand/_banner.py"; s=p.read_text(encoding="utf-8"); s=s.replace("    if args.dry_run:\n","    if not results:\n        logger.error('No banners were generated')\n        return 1\n\n    if args.dry_run:\n",1); p.write_text(s,encoding="utf-8")

def test_current_maintenance_pass_runtime_fail():
    o=C.check(REPO); assert o["maintenance_status"]=="PASS"; assert o["runtime_status"]=="FAIL"
def test_repaired_shape_can_pass(tmp_path):
    r=mini(tmp_path); repair(r); o=C.check(r); assert o["maintenance_status"]=="PASS"; assert o["runtime_status"]=="PASS",o["runtime_errors"]
def test_missing_runtime_file(tmp_path):
    r=mini(tmp_path); (r/"scikitplot/_brand/_logo.py").unlink(); assert any("missing required runtime file" in x for x in C.check(r)["runtime_errors"])
def test_extra_runtime_file_inventory(tmp_path):
    r=mini(tmp_path); (r/"scikitplot/_brand/extra.py").write_text("x=1\n"); assert any("tracked_files changed" in x or "path inventory changed" in x for x in C.check(r)["runtime_errors"])
def test_missing_required_symbol(tmp_path):
    r=mini(tmp_path); p=r/"scikitplot/_brand/_logo.py"; p.write_text(p.read_text().replace("def draw(","def draw_removed(",1)); assert any("missing required top-level symbol draw" in x for x in C.check(r)["runtime_errors"])
def test_star_import_leak_detected(tmp_path):
    assert any("star-imports _logo" in x for x in C.check(mini(tmp_path))["runtime_errors"])
def test_package_all_mismatch_detected(tmp_path):
    assert any("package __all__" in x for x in C.check(mini(tmp_path))["runtime_errors"])
def test_nonexistent_cli_alias_detected(tmp_path):
    assert any("scikitplot.logo" in x for x in C.check(mini(tmp_path))["runtime_errors"])
def test_eager_banner_detected(tmp_path):
    assert any("eagerly imports executable _banner" in x for x in C.check(mini(tmp_path))["runtime_errors"])
def test_zero_result_guard_detected(tmp_path):
    assert any("zero-result failure guard" in x for x in C.check(mini(tmp_path))["runtime_errors"])
def test_global_rng_mutation_detected(tmp_path):
    r=mini(tmp_path); p=r/"scikitplot/_brand/_logo.py"; p.write_text(p.read_text()+"\nnp.random.seed(1)\n"); assert any("global RNG" in x for x in C.check(r)["runtime_errors"])
def test_shell_true_detected(tmp_path):
    r=mini(tmp_path); p=r/"scikitplot/_brand/_banner.py"; p.write_text(p.read_text()+"\nsubprocess.run(['figlet'], shell=True)\n"); assert any("shell=True" in x for x in C.check(r)["runtime_errors"])
def test_runtime_plane_violation(tmp_path):
    r=mini(tmp_path); p=r/"scikitplot/_brand/_logo.py"; p.write_text("import maintenances\n"+p.read_text()); assert any("runtime plane violation" in x for x in C.check(r)["runtime_errors"])
def test_maintenance_plane_violation(tmp_path):
    r=mini(tmp_path); p=r/"maintenances/_brand/_maintenance/tools/x.py"; p.write_text("import scikitplot._brand\n"); assert any("maintenance plane violation" in x for x in C.check(r)["maintenance_errors"])
def test_forbidden_metadata_command(tmp_path):
    r=mini(tmp_path); p=r/"maintenances/_brand/REVIEW.json"; o=json.loads(p.read_text()); o["command"]="rm -rf /"; p.write_text(json.dumps(o)); assert any("unsupported executable field" in x for x in C.check(r)["maintenance_errors"])
def test_duplicate_json_key_detected(tmp_path):
    r=mini(tmp_path); p=r/"maintenances/_brand/_maintenance/STATE.json"; p.write_text('{"subsystem":"scikitplot._brand","subsystem":"x"}'); assert any("duplicate JSON key" in x for x in C.check(r)["maintenance_errors"])
def test_missing_skill_detected(tmp_path):
    r=mini(tmp_path); (r/"skills/_brand/SKILL.md").unlink(); assert any("skill" in x.lower() for x in C.check(r)["maintenance_errors"])
def test_foreign_cwd_cli(tmp_path):
    r=mini(tmp_path); cp=subprocess.run([sys.executable,str(r/"maintenances/_brand/_maintenance/tools/check_contract.py"),"--repo",str(r),"--json"],cwd=tmp_path,text=True,capture_output=True); assert cp.returncode==0; assert json.loads(cp.stdout)["maintenance_status"]=="PASS"
