from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT, TESTS_ROOT

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SEC = ROOT / "_hf_spaces_proxy" / "security"


def _release_tests():
    security_tests = TESTS_ROOT / "_hf_spaces_proxy" / "security"
    names = (
        "test_promote_release.py",
        "test_publish_release.py",
        "test_finalize_publication.py",
        "test_witness_publication.py",
        "test_preserve_release_history.py",
        "test_govern_release_history.py",
        "test_seal_release_governance.py",
        "test_maintain_release_trust.py",
        "test_continue_recovered_trust.py",
        "test_verify_attestation_lifecycle.py",
        "test_verify_native_status_provenance.py",
        "test_archive_native_status_evidence.py",
        "test_audit_archive_retention.py",
        "test_witness_archive_health.py",
        "test_anchor_archive_health.py",
        "test_verify_archive_merkle_transparency.py",
        "test_govern_archive_merkle_log_authority.py",
        "test_continue_archive_merkle_authority.py",
        "test_rebridge_archive_merkle_authority.py",
        "test_preserve_archive_merkle_rebridge_history.py",
    )
    for name in names:
        yield security_tests / name


def _wall_clock_calls(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and (owner.id, node.func.attr) in {
            ("datetime", "now"), ("datetime", "utcnow"), ("date", "today"), ("time", "time")
        }:
            found.append((node.lineno, f"{owner.id}.{node.func.attr}"))
    return found


def _now_assignment(path: Path) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NOW" for target in node.targets
        ):
            return node.value
    raise AssertionError(f"{path.name}: missing top-level NOW assignment")


def test_run169_run159_clock_is_derived_from_fixed_predecessor():
    path = TESTS_ROOT / "_hf_spaces_proxy" / "security" / "test_verify_native_status_provenance.py"
    value = _now_assignment(path)
    assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
    assert isinstance(value.left, ast.Attribute) and value.left.attr == "NOW"
    assert isinstance(value.left.value, ast.Name) and value.left.value.id == "run158"
    assert isinstance(value.right, ast.Call) and isinstance(value.right.func, ast.Name)
    assert value.right.func.id == "timedelta"
    assert any(
        kw.arg == "hours" and isinstance(kw.value, ast.Constant) and kw.value.value == 1
        for kw in value.right.keywords
    )

def test_run169_release_security_tests_149_168_are_wall_clock_free():
    offenders = {p.name: _wall_clock_calls(p) for p in _release_tests() if _wall_clock_calls(p)}
    assert offenders == {}


def test_run169_downstream_clock_lineage_remains_predecessor_derived():
    expected = {
        "test_archive_native_status_evidence.py": "run159",
        "test_audit_archive_retention.py": "run160_tests",
        "test_witness_archive_health.py": "r161t",
        "test_anchor_archive_health.py": "r162t",
        "test_verify_archive_merkle_transparency.py": "t163",
        "test_govern_archive_merkle_log_authority.py": "t164",
        "test_rebridge_archive_merkle_authority.py": "t164",
    }
    security_tests = TESTS_ROOT / "_hf_spaces_proxy" / "security"
    for name, owner in expected.items():
        value = _now_assignment(security_tests / name)
        assert isinstance(value, ast.Attribute) and value.attr == "NOW", name
        assert isinstance(value.value, ast.Name) and value.value.id == owner, (name, owner)

def test_run169_documentation_and_release_gates_are_wired():
    guide = SEC / "RELEASE_TEST_CLOCK_HERMETICITY_GUIDE.md"
    assert guide.exists()
    text = guide.read_text().lower()
    for phrase in ("run 169", "wall clock", "synthetic", "production", "fail"):
        assert phrase in text
    gates = (SEC / "SECURITY_RELEASE_GATES.md").read_text().lower()
    evidence = (SEC / "RELEASE_EVIDENCE_GUIDE.md").read_text().lower()
    readme = (ROOT / "README.md").read_text().lower()
    assert "run 169" in gates and "run 169" in evidence and "run 169" in readme
