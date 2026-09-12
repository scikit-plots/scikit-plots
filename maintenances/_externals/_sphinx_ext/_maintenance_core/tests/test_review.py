from pathlib import Path
import json
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from review import build_context, validate_profile
from review_subsystem import review_subsystem


def _fixture(tmp_path: Path, *, skill_has_tracker: bool = True, unavailable: bool = False):
    runtime = tmp_path / "scikitplot/_externals/_sphinx_ext"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("")
    (runtime / "_extension_setup.py").write_text("")
    for name in ("_sphinx_demo", "_sphinx_collection"):
        pkg = runtime / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("x = 1\n")

    maint = tmp_path / "maintenances/_externals/_sphinx_ext/_sphinx_demo"
    (maint / "_maintenance/checkpoints").mkdir(parents=True)
    skill = tmp_path / "skills/_externals/_sphinx_ext/_sphinx_demo"
    skill.mkdir(parents=True)
    tracker_line = " then TRACKER.json" if skill_has_tracker else ""
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo reviewer\n---\n"
        "Read MAINTAINING.md then FRESH_CHAT_HANDOFF.md then STATE.json"
        + tracker_line
        + ".\n"
    )
    (maint / "MAINTAINING.md").write_text("maintenance\n")
    (maint / "_maintenance/FRESH_CHAT_HANDOFF.md").write_text(
        "Do not rely on previous chat history. Read SKILL.md then MAINTAINING.md, this file, STATE.json, TRACKER.json.\n"
    )
    (maint / "_maintenance/checkpoints/D-M001_bootstrap.md").write_text("# checkpoint\n")
    state = {
        "schema_version": 1,
        "subsystem": "demo",
        "source_anchor": {"sha256": "a" * 64},
        "phase": "VERIFY",
        "active_checkpoint": "D-M001",
        "checkpoints": {"D-M001": {"status": "ACTIVE"}},
        "verification_snapshot": (
            {"optional_layer": "UNAVAILABLE_MISSING_demo"} if unavailable else {"core": "GREEN"}
        ),
        "blockers": (["Install demo dependency and rerun optional layer."] if unavailable else []),
        "next_actions": [],
    }
    tracker = {
        "schema_version": 1,
        "subsystem": "demo",
        "logical_contracts": [{"id": "D-C001", "status": "ACTIVE"}],
    }
    (maint / "_maintenance/STATE.json").write_text(json.dumps(state))
    (maint / "_maintenance/TRACKER.json").write_text(json.dumps(tracker))
    profile = {
        "schema_version": 1,
        "subsystem": "demo",
        "mode": "independent_then_reconcile",
        "package_targets": ["_sphinx_demo"],
        "lenses": [
            {
                "id": "fresh-chat",
                "title": "Fresh chat",
                "required": True,
                "focus": "authority",
                "checks": ["fresh-chat-parity", "profile-safety"],
            },
            {
                "id": "state-evidence",
                "title": "State evidence",
                "required": True,
                "focus": "evidence",
                "checks": ["state-tracker-consistency", "verification-state"],
            },
        ],
        "pr_policy": {
            "blocking_severities": ["ERROR"],
            "require_required_lenses": True,
            "unavailable_release_gate_blocks_promotion": True,
        },
    }
    (maint / "_maintenance/REVIEW.json").write_text(json.dumps(profile))
    manifest = {
        "schema_version": 3,
        "subsystem": "demo",
        "runtime_dir": "_sphinx_demo",
        "state": "_maintenance/STATE.json",
        "tracker": "_maintenance/TRACKER.json",
        "handoff": "_maintenance/FRESH_CHAT_HANDOFF.md",
        "checkpoint_dir": "_maintenance/checkpoints",
        "skill_root": "skills/_externals/_sphinx_ext/_sphinx_demo",
        "dependency_edges": [],
        "capability_ownership": [],
        "review_profile": "_maintenance/REVIEW.json",
    }
    manifest_path = maint / "MAINTENANCE.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, profile, runtime


def test_profile_rejects_unknown_registered_check(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    profile["lenses"][0]["checks"].append("run-any-shell")
    errors = validate_profile(profile, manifest)
    assert any("unknown registered check" in error for error in errors)


def test_profile_rejects_executable_metadata_even_without_safety_lens(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    profile["lenses"][0]["checks"] = ["fresh-chat-parity"]
    profile["command"] = "rm -rf /"
    errors = validate_profile(profile, manifest)
    assert any("forbidden executable field" in error for error in errors)


def test_profile_rejects_package_path_escape(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    profile["package_targets"] = ["../../outside"]
    assert any("invalid review package target" in error for error in validate_profile(profile, manifest))


def test_missing_review_target_blocks_pr(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["capability_ownership"].append({
        "id": "demo.missing", "owner": "_sphinx_missing", "description": "missing package"
    })
    manifest_path.write_text(json.dumps(manifest))
    path = manifest_path.parent / "_maintenance/REVIEW.json"
    profile["package_targets"] = ["_sphinx_missing"]
    path.write_text(json.dumps(profile))
    report = review_subsystem(manifest_path, jobs=2)
    assert report["status"] == "BLOCKED"
    assert report["package_reviews"][0]["findings"][0]["code"] == "review.package-missing"


def test_skill_missing_tracker_is_real_fresh_chat_blocker(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path, skill_has_tracker=False)
    report = review_subsystem(manifest_path)
    findings = [f for lens in report["lenses"] for f in lens["findings"]]
    assert any(f["code"] == "review.skill-read-order" for f in findings)
    assert report["status"] == "BLOCKED"


def test_unavailable_layer_blocks_release_not_pr(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path, unavailable=True)
    report = review_subsystem(manifest_path)
    assert report["status"] == "PR_READY"
    assert report["release_promotion"] == "BLOCKED"
    assert report["counts"]["UNAVAILABLE"] == 1


def test_runtime_plane_leak_blocks_package_review(tmp_path):
    manifest_path, _, runtime = _fixture(tmp_path)
    (runtime / "_sphinx_demo/__init__.py").write_text("import maintenances\n")
    report = review_subsystem(manifest_path)
    assert report["status"] == "BLOCKED"
    findings = report["package_reviews"][0]["findings"]
    assert any(f["code"] == "review.package-plane-leak" for f in findings)


def test_serial_and_parallel_review_are_deterministic(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path, unavailable=True)
    assert review_subsystem(manifest_path, jobs=1) == review_subsystem(manifest_path, jobs=8)


def test_profile_rejects_package_outside_declared_architecture_boundary(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    profile["package_targets"] = ["_sphinx_collection"]
    errors = validate_profile(profile, manifest)
    assert any("outside declared subsystem architecture boundary" in error for error in errors)


def test_review_profile_path_cannot_escape_maintenance_root(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    outside = manifest_path.parent.parent / "outside-review.json"
    outside.write_text("{}")
    manifest["review_profile"] = "../outside-review.json"
    manifest_path.write_text(json.dumps(manifest))
    try:
        build_context(manifest_path)
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape must be rejected")


def test_review_subsystem_reports_escaped_profile_as_blocked_not_traceback(tmp_path):
    manifest_path, _, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["review_profile"] = "../outside-review.json"
    (manifest_path.parent.parent / "outside-review.json").write_text("{}")
    manifest_path.write_text(json.dumps(manifest))
    report = review_subsystem(manifest_path)
    assert report["status"] == "BLOCKED"
    assert report["release_promotion"] == "BLOCKED"
    assert report["counts"]["ERROR"] == 1
    assert "safe review context" in report["profile_errors"][0]


def test_profile_rejects_duplicate_blocking_severities(tmp_path):
    manifest_path, profile, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    profile["pr_policy"]["blocking_severities"] = ["ERROR", "ERROR"]
    errors = validate_profile(profile, manifest)
    assert any("blocking_severities must be unique" in error for error in errors)
