from pathlib import Path
import json
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from check_subsystem import check_subsystem


def _fixture(tmp_path, *, dep_import="from .._sphinx_dep import x", deps=None):
    deps = ["_sphinx_dep"] if deps is None else deps
    runtime = tmp_path / "scikitplot/_externals/_sphinx_ext"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("")
    (runtime / "_extension_setup.py").write_text("")
    pkg = runtime / "_sphinx_demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(dep_import + "\n")
    dep = runtime / "_sphinx_dep"
    dep.mkdir()
    (dep / "__init__.py").write_text("x=1\n")

    mf = tmp_path / "maintenances/_externals/_sphinx_ext/_sphinx_demo"
    (mf / "_maintenance/checkpoints").mkdir(parents=True)
    sf = tmp_path / "skills/_externals/_sphinx_ext/_sphinx_demo"
    sf.mkdir(parents=True)
    (sf / "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\nRead MAINTAINING.md, then STATE.json.\n")

    state = {
        "schema_version": 1,
        "subsystem": "demo",
        "source_anchor": {"sha256": "a" * 64},
        "phase": "X",
        "active_checkpoint": "D-M001",
        "checkpoints": {"D-M001": {}},
        "verification_snapshot": {},
        "next_actions": [],
    }
    tracker = {
        "schema_version": 1,
        "subsystem": "demo",
        "logical_contracts": [{"id": "D-C001"}],
    }
    (mf / "_maintenance/STATE.json").write_text(json.dumps(state))
    (mf / "_maintenance/TRACKER.json").write_text(json.dumps(tracker))
    (mf / "_maintenance/FRESH_CHAT_HANDOFF.md").write_text("Do not rely on previous chat history. Read STATE.json then TRACKER.json.\n")
    (mf / "_maintenance/checkpoints/D-M001_bootstrap.md").write_text("x")
    manifest = {
        "schema_version": 1,
        "subsystem": "demo",
        "runtime_dir": "_sphinx_demo",
        "state": "_maintenance/STATE.json",
        "tracker": "_maintenance/TRACKER.json",
        "handoff": "_maintenance/FRESH_CHAT_HANDOFF.md",
        "checkpoint_dir": "_maintenance/checkpoints",
        "runtime_requires": deps,
        "family_related": [],
        "skill_root": "skills/_externals/_sphinx_ext/_sphinx_demo",
    }
    (mf / "MAINTENANCE.json").write_text(json.dumps(manifest))
    return mf, runtime


def test_happy_fixture(tmp_path):
    mf, _ = _fixture(tmp_path)
    assert check_subsystem(mf / "MAINTENANCE.json") == []


def test_duplicate_contract_ids_fail(tmp_path):
    mf, _ = _fixture(tmp_path)
    p = mf / "_maintenance/TRACKER.json"
    data = json.loads(p.read_text())
    data["logical_contracts"].append({"id": "D-C001"})
    p.write_text(json.dumps(data))
    assert any("duplicate logical contract" in e for e in check_subsystem(mf / "MAINTENANCE.json"))


def test_missing_declared_dependency_fails(tmp_path):
    mf, runtime = _fixture(tmp_path)
    import shutil
    shutil.rmtree(runtime / "_sphinx_dep")
    assert any("dependency missing" in e for e in check_subsystem(mf / "MAINTENANCE.json"))


def test_undeclared_cross_stack_and_plane_leakage_fail(tmp_path):
    mf, runtime = _fixture(tmp_path, deps=[])
    (runtime / "_sphinx_demo/__init__.py").write_text(
        "from .._sphinx_dep import x\nimport maintenances\n"
    )
    errors = check_subsystem(mf / "MAINTENANCE.json")
    assert any("undeclared cross-stack" in e for e in errors)
    assert any("runtime imports maintenance/skill plane" in e for e in errors)


def test_dependency_cycle_fails(tmp_path):
    mf, runtime = _fixture(tmp_path)
    (runtime / "_sphinx_dep/__init__.py").write_text("from .._sphinx_demo import x\nx=1\n")
    assert any("dependency cycle" in e for e in check_subsystem(mf / "MAINTENANCE.json"))


def test_secret_looking_state_key_fails(tmp_path):
    mf, _ = _fixture(tmp_path)
    p = mf / "_maintenance/STATE.json"
    data = json.loads(p.read_text())
    data["api_key"] = "not-for-repository"
    p.write_text(json.dumps(data))
    assert any("secret-looking" in e for e in check_subsystem(mf / "MAINTENANCE.json"))


def test_schema_v3_requires_review_profile_file(tmp_path):
    mf, _ = _fixture(tmp_path)
    p = mf / "MAINTENANCE.json"
    data = json.loads(p.read_text())
    data["schema_version"] = 3
    data.pop("runtime_requires")
    data.pop("family_related")
    data["dependency_edges"] = []
    data["capability_ownership"] = []
    p.write_text(json.dumps(data))
    errors = check_subsystem(p)
    assert any("review_profile" in error for error in errors)


def test_maintenance_state_path_escape_is_rejected(tmp_path):
    mf, _ = _fixture(tmp_path)
    p = mf / "MAINTENANCE.json"
    data = json.loads(p.read_text())
    data["state"] = "../outside-state.json"
    (mf.parent / "outside-state.json").write_text(json.dumps({}))
    p.write_text(json.dumps(data))
    errors = check_subsystem(p)
    assert any("state path escapes subsystem maintenance root" in error for error in errors)
