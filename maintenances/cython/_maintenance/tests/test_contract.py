from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if all((p / n).is_dir() for n in ("scikitplot", "maintenances", "skills")))
TOOL = REPO / "maintenances/cython/_maintenance/tools/check_contract.py"
spec = importlib.util.spec_from_file_location("cython_contract", TOOL)
contract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(contract)


def make_repo(tmp_path: Path, *, clean_runtime: bool = True) -> Path:
    root = tmp_path / "repo"
    (root / "scikitplot").mkdir(parents=True)
    (root / "maintenances").mkdir()
    (root / "skills").mkdir()
    shutil.copytree(REPO / "scikitplot/cython", root / "scikitplot/cython")
    shutil.copytree(REPO / "maintenances/cython", root / "maintenances/cython")
    shutil.copytree(REPO / "skills/cython", root / "skills/cython")
    if clean_runtime:
        for p in list((root / "scikitplot/cython").rglob("__pycache__")):
            shutil.rmtree(p)
        for p in list((root / "scikitplot/cython").rglob("*.pyc")):
            p.unlink()
    # Evidence logs belong to the source checkout; a synthetic contract fixture
    # uses PENDING fingerprint and no executable evidence claims.
    ev = root / "maintenances/cython/_maintenance/EVIDENCE.json"
    ev.write_text(json.dumps({"schema_version": 2, "subsystem": "scikitplot.cython", "runtime_fingerprint": "PENDING", "gates": {}}, indent=2) + "\n")
    return root


def errors(root: Path):
    return contract.run_checks(root)[3:]


def test_current_tree_fails_closed_only_for_source_bytecode_hygiene():
    _, _, _, maintenance, runtime = contract.run_checks(REPO)
    assert maintenance == []
    assert any("bytecode/cache" in e for e in runtime)


def test_clean_synthetic_runtime_contract_passes(tmp_path: Path):
    root = make_repo(tmp_path)
    maintenance, runtime = errors(root)
    assert maintenance == []
    assert runtime == []


def test_missing_required_runtime_file_is_detected(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "scikitplot/cython/_security.py").unlink()
    _, runtime = errors(root)
    assert any("missing required runtime file scikitplot/cython/_security.py" in e for e in runtime)


def test_module_scope_cython_dependency_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_utils.py"
    p.write_text("import Cython\n" + p.read_text())
    _, runtime = errors(root)
    assert any("optional toolchain dependency 'Cython'" in e for e in runtime)


def test_sibling_submodule_dependency_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_utils.py"
    p.write_text("import scikitplot.mcp\n" + p.read_text())
    _, runtime = errors(root)
    assert any("sibling subsystem 'scikitplot.mcp'" in e for e in runtime)


def test_runtime_to_maintenance_plane_import_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_utils.py"
    p.write_text("import maintenances.cython\n" + p.read_text())
    _, runtime = errors(root)
    assert any("runtime plane violation" in e for e in runtime)


def test_maintenance_to_runtime_plane_import_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "maintenances/cython/_maintenance/review_subsystem.py"
    p.write_text("import scikitplot.cython\n" + p.read_text())
    maintenance, _ = errors(root)
    assert any("maintenance plane imports runtime" in e for e in maintenance)


def test_security_choke_point_drift_is_detected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_public.py"
    s = p.read_text()
    # Leave the definition, remove visible calls.
    first = s.find("_validate_build_security(")
    s = s[: first + len("_validate_build_security(")] + s[first + len("_validate_build_security(") :].replace("_validate_build_security(", "_security_gate_removed(")
    p.write_text(s)
    _, runtime = errors(root)
    assert any("major entry paths through _validate_build_security" in e for e in runtime)


def test_cache_fingerprint_drift_is_detected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_cache.py"
    p.write_text(p.read_text().replace('fp["resolved_cxx"]', 'fp["removed_resolved_cxx"]'))
    _, runtime = errors(root)
    assert any("resolved_cxx" in e for e in runtime)


def test_lock_exclusivity_marker_drift_is_detected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_lock.py"
    p.write_text(p.read_text().replace("lock_dir.mkdir(exist_ok=False)", "lock_dir.mkdir(exist_ok=True)"))
    _, runtime = errors(root)
    assert any("exist_ok=False" in e for e in runtime)


def test_orphan_template_metadata_is_detected(tmp_path: Path):
    root = make_repo(tmp_path)
    src = next((root / "scikitplot/cython/_templates/basic_python").glob("*.meta.json"))
    orphan = src.with_name("orphan.py.meta.json")
    orphan.write_bytes(src.read_bytes())
    _, runtime = errors(root)
    assert any("orphan.py.meta.json" in e for e in runtime)


def test_probe_catalog_is_contractual(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "scikitplot/cython/_templates/probe/README.md"
    p.write_text(p.read_text().replace("repro_con001.py", "removed_probe.py"))
    _, runtime = errors(root)
    assert any("probe catalog" in e for e in runtime)


def test_skill_must_be_substantive(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "skills/cython/SKILL.md").write_text("---\nname: x\n---\n")
    maintenance, _ = errors(root)
    assert any("substantive SKILL.md" in e for e in maintenance)


def test_executable_metadata_surface_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "maintenances/cython/MAINTENANCE.json"
    data = json.loads(p.read_text()); data["shell"] = "rm -rf /"; p.write_text(json.dumps(data))
    maintenance, _ = errors(root)
    assert any("unsupported executable field" in e for e in maintenance)


def test_unknown_review_check_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "maintenances/cython/REVIEW.json"
    data = json.loads(p.read_text()); data["lanes"][0]["checks"].append("magic"); p.write_text(json.dumps(data))
    maintenance, _ = errors(root)
    assert any("unknown check 'magic'" in e for e in maintenance)


def test_evidence_fingerprint_drift_is_rejected(tmp_path: Path):
    root = make_repo(tmp_path)
    p = root / "maintenances/cython/_maintenance/EVIDENCE.json"
    data = json.loads(p.read_text()); data["runtime_fingerprint"] = "0" * 64; p.write_text(json.dumps(data))
    maintenance, _ = errors(root)
    assert any("evidence runtime fingerprint drift" in e for e in maintenance)


def test_update_refuses_to_bless_failing_runtime():
    cp = subprocess.run([sys.executable, "-B", str(REPO / "maintenances/cython/_maintenance/check_trackers.py"), "--update"], cwd="/tmp", text=True, capture_output=True)
    assert cp.returncode != 0
    assert "refuses to bless" in (cp.stdout + cp.stderr)


def test_wrapper_discovers_repo_from_foreign_cwd():
    cp = subprocess.run([sys.executable, "-B", str(REPO / "maintenances/cython/_maintenance/check_trackers.py"), "--json"], cwd="/tmp", text=True, capture_output=True)
    assert cp.returncode == 1  # current source hygiene finding is intentionally red
    payload = json.loads(cp.stdout)
    assert payload["maintenance_status"] == "PASS"
    assert payload["runtime_status"] == "FAIL"
    assert "scikitplot.cython" == payload["subsystem"]


def test_checked_in_pycache_is_detected_and_clean_tree_clears_it(tmp_path: Path):
    root = make_repo(tmp_path)
    pycdir = root / "scikitplot/cython/__pycache__"; pycdir.mkdir(); (pycdir / "x.pyc").write_bytes(b"x")
    _, runtime = errors(root)
    assert any("bytecode/cache" in e for e in runtime)
    shutil.rmtree(pycdir)
    _, runtime2 = errors(root)
    assert runtime2 == []
