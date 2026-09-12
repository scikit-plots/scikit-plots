from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
ROOT = RUNTIME_ROOT
SECURITY = ROOT / "_hf_spaces_proxy" / "security"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


promote = _load("run149_promote", SECURITY / "promote_release.py")
source_tree = _load("run149_source_tree", SECURITY / "source_tree.py")
REVISION = "b" * 40
NOW = datetime(2026, 9, 5, 3, 45, tzinfo=timezone.utc)
PREFIX = "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, Path]:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "a.txt").write_text("old\n")
    (new / "a.txt").write_text("new\n")
    (new / "bin.sh").write_text("#!/bin/sh\necho ok\n")
    (new / "bin.sh").chmod(0o755)
    return old, new


def _baseline_zip(old: Path, output: Path) -> Path:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(old.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(old).as_posix()
            info = zipfile.ZipInfo(f"{PREFIX}/{rel}", date_time=(2026, 9, 5, 1, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | (path.stat().st_mode & 0o777)) << 16
            zf.writestr(info, path.read_bytes())
    return output


def _git_env(repo: Path) -> dict[str, str]:
    """Return a minimal deterministic environment for synthetic Git evidence."""
    parent = repo.parent
    home = parent / "git-home"
    xdg = parent / "git-xdg"
    template = parent / "git-template"
    for directory in (home, xdg, template):
        directory.mkdir(exist_ok=True)
    empty_config = parent / "empty.gitconfig"
    empty_config.write_text("")
    env: dict[str, str] = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "LC_ALL": "C",
        "LANG": "C",
    }
    for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


def _git(repo: Path, *args: str, check: bool = True, capture_output: bool = False):
    git = shutil.which("git")
    assert git is not None, "git executable required for Run 149 synthetic patch evidence"
    template = repo.parent / "git-template"
    template.mkdir(exist_ok=True)
    command = [
        git,
        "-c", "core.autocrlf=false",
        "-c", "core.filemode=true",
        "-c", "core.ignorecase=false",
        *args,
    ]
    return subprocess.run(
        command,
        cwd=repo,
        env=_git_env(repo),
        check=check,
        capture_output=capture_output,
    )


def _patch(old: Path, new: Path, output: Path) -> Path:
    repo = old.parent / "repo"
    repo.mkdir()
    template = repo.parent / "git-template"
    template.mkdir(exist_ok=True)
    _git(repo, "init", "-q", f"--template={template}")
    for p in old.rglob("*"):
        if p.is_file():
            dest = repo / p.relative_to(old)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            os.chmod(dest, p.stat().st_mode & 0o777)
    _git(repo, "add", "-A")
    for p in list(repo.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            p.unlink()
    for p in new.rglob("*"):
        if p.is_file():
            dest = repo / p.relative_to(new)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            os.chmod(dest, p.stat().st_mode & 0o777)
    _git(repo, "add", "-N", ".")
    proc = _git(repo, "diff", "--binary", "--no-renames", check=False, capture_output=True)
    assert proc.returncode == 0
    output.write_bytes(proc.stdout)
    assert output.stat().st_size > 0
    return output


def _evidence(tmp_path: Path, source: Path) -> tuple[Path, dict]:
    sbom = tmp_path / "image.cdx.json"
    sbom.write_text(json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.6"}))
    source_sha = source_tree.source_tree_sha256(source)
    doc = {
        "source": {"sourceTreeSha256": source_sha, "pythonSbomSha256": "c" * 64},
        "artifacts": {
            "imageSbom": {
                "path": sbom.name,
                "sha256": _sha(sbom),
                "subject": "sha256:" + "d" * 64,
            }
        },
    }
    evidence = tmp_path / "release-evidence.json"
    evidence.write_text(json.dumps(doc, sort_keys=True))
    result = {
        "ok": True,
        "release_id": "run149-test",
        "proxy_version": "7.4.0",
        "source_tree_sha256": source_sha,
        "source_revision": REVISION,
        "evidence_sha256": _sha(evidence),
    }
    return evidence, result


def _fixture(tmp_path: Path):
    old, new = _source(tmp_path)
    baseline = _baseline_zip(old, tmp_path / "baseline.zip")
    patch = _patch(old, new, tmp_path / "change.patch")
    evidence, result = _evidence(tmp_path, new)

    def verifier(path: Path, *, now=None):
        assert path == evidence.resolve()
        return dict(result)

    return old, new, baseline, patch, evidence, result, verifier


def _verifier_evidence(tmp_path: Path) -> Path:
    path = tmp_path / "external-signature-verifier.json"
    path.write_text(json.dumps({"verified": True, "issuer": "test-trust-root"}, sort_keys=True))
    return path


def _signature(tmp_path: Path, statement: Path, *, name: str = "sig.json", verified_at=NOW) -> tuple[Path, Path]:
    verifier_evidence = _verifier_evidence(tmp_path)
    sig = promote.write_signature_verification_record(
        statement=statement, verifier_evidence=verifier_evidence, output=tmp_path / name,
        source_revision=REVISION, verifier_name="test-verifier", verifier_version="1.0", verified_at=verified_at,
    )
    return sig, verifier_evidence


def test_run149_prepare_binds_patch_zip_source_and_sbom_references(tmp_path: Path):
    _, new, baseline, patch, evidence, result, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    out = promote.prepare_release(
        evidence=evidence,
        baseline_zip=baseline,
        patch=patch,
        output_dir=prepared,
        source_root=new,
        evidence_verifier=verifier,
        now=NOW,
    )
    assert out["ok"] is True and out["phase"] == "prepared"
    statement_path = prepared / "release-statement.json"
    statement = json.loads(statement_path.read_text())
    assert statement["subject"] == {
        "sourceTreeSha256": result["source_tree_sha256"],
        "evidenceSha256": result["evidence_sha256"],
    }
    assert statement["verification"] == {
        "evidenceVerified": True,
        "patchRecreatesSourceTree": True,
        "zipRecreatesSourceTree": True,
        "deterministicZip": True,
    }
    assert statement["sbomReferences"]["pythonRuntime"]["sha256"] == "c" * 64
    assert statement["sbomReferences"]["image"]["sha256"] == _sha(tmp_path / "image.cdx.json")
    assert str(tmp_path) not in statement_path.read_text()
    archive = prepared / "run149-test.zip"
    promote._verify_release_zip(archive, new)
    with zipfile.ZipFile(archive) as zf:
        modes = {
            i.filename.rsplit("/", 1)[-1]: ((i.external_attr >> 16) & 0o777)
            for i in zf.infolist()
        }
    assert modes["bin.sh"] == 0o755


def test_run149_prepare_is_deterministic_for_same_source(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=first, source_root=new, evidence_verifier=verifier, now=NOW)
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=second, source_root=new, evidence_verifier=verifier, now=NOW)
    assert (first / "run149-test.zip").read_bytes() == (second / "run149-test.zip").read_bytes()
    assert (first / "release-statement.json").read_bytes() == (second / "release-statement.json").read_bytes()


def test_run149_prepare_rejects_patch_that_does_not_recreate_verified_tree(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    text = patch.read_text()
    patch.write_text(text.replace("+new", "+wrong", 1))
    with pytest.raises(promote.PromotionError, match="PATCH_RESULT_CONTENT_MISMATCH"):
        promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=tmp_path / "prepared", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_baseline_rejects_traversal_and_special_entries(tmp_path: Path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil", b"x")
    with pytest.raises(promote.PromotionError, match="BASELINE_ZIP_PATH_INVALID"):
        promote._safe_extract_baseline(bad, tmp_path / "out1")

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as zf:
        info = zipfile.ZipInfo(f"{PREFIX}/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"target")
    with pytest.raises(promote.PromotionError, match="BASELINE_ZIP_SPECIAL_FILE"):
        promote._safe_extract_baseline(symlink, tmp_path / "out2")


def test_run149_finalize_rechecks_signature_and_all_publishable_hashes(tmp_path: Path):
    _, new, baseline, patch, evidence, result, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json", name="release-statement.signature-verification.json")
    promoted = tmp_path / "promoted"
    out = promote.finalize_release(
        evidence=evidence,
        prepared_dir=prepared,
        baseline_zip=baseline,
        signature_record=sig,
        signature_verifier_evidence=verifier_evidence,
        promotion_dir=promoted,
        source_root=new,
        evidence_verifier=verifier,
        now=NOW,
    )
    assert out["phase"] == "promoted"
    receipt = json.loads((promoted / "promotion-receipt.json").read_text())
    assert receipt["sourceTreeSha256"] == result["source_tree_sha256"]
    assert receipt["status"] == "promoted"
    for item in receipt["publish"]:
        path = promoted / item["name"]
        assert _sha(path) == item["sha256"]
        assert path.stat().st_size == item["size"]


def test_run149_finalize_rejects_tampered_zip_wrong_signature_and_source_drift(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json")

    archive = prepared / "run149-test.zip"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(promote.PromotionError, match="PROMOTION_ZIP_HASH_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "bad1", source_root=new, evidence_verifier=verifier, now=NOW)

    # restore prepared transaction and then corrupt the verification subject
    prepared2 = tmp_path / "prepared2"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared2, source_root=new, evidence_verifier=verifier, now=NOW)
    sig2, verifier_evidence2 = _signature(tmp_path, prepared2 / "release-statement.json", name="sig2.json")
    doc = json.loads(sig2.read_text())
    doc["releaseStatementSha256"] = "0" * 64
    sig2.write_text(json.dumps(doc))
    with pytest.raises(promote.PromotionError, match="SIGNATURE_RECORD_SUBJECT_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared2, baseline_zip=baseline, signature_record=sig2, signature_verifier_evidence=verifier_evidence2, promotion_dir=tmp_path / "bad2", source_root=new, evidence_verifier=verifier, now=NOW)

    sig3, verifier_evidence3 = _signature(tmp_path, prepared2 / "release-statement.json", name="sig3.json")
    (new / "a.txt").write_text("drift\n")
    with pytest.raises(promote.PromotionError, match="RELEASE_STATEMENT_SOURCE_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared2, baseline_zip=baseline, signature_record=sig3, signature_verifier_evidence=verifier_evidence3, promotion_dir=tmp_path / "bad3", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_finalize_rejects_stale_signature_record(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json", verified_at=NOW - timedelta(hours=80))
    with pytest.raises(promote.PromotionError, match="SIGNATURE_RECORD_STALE"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_finalize_rejects_statement_path_traversal_even_if_signature_matches(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    statement_path = prepared / "release-statement.json"
    doc = json.loads(statement_path.read_text())
    doc["artifacts"]["zip"]["name"] = "../escape.zip"
    statement_path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    sig, verifier_evidence = _signature(tmp_path, statement_path)
    with pytest.raises(promote.PromotionError, match="RELEASE_STATEMENT_ZIP_NAME_INVALID"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_finalize_reapplies_patch_and_rejects_wrong_baseline(tmp_path: Path):
    old, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json")
    wrong_old = tmp_path / "wrong-old"
    wrong_old.mkdir()
    (wrong_old / "a.txt").write_text("different baseline\n")
    wrong_baseline = _baseline_zip(wrong_old, tmp_path / "wrong-baseline.zip")
    with pytest.raises(promote.PromotionError, match="PROMOTION_BASELINE_HASH_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=wrong_baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_signature_record_must_bind_verified_identity_and_revision(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json")
    doc = json.loads(sig.read_text())
    doc["signerIdentityVerified"] = False
    sig.write_text(json.dumps(doc, sort_keys=True))
    with pytest.raises(promote.PromotionError, match="SIGNATURE_RECORD_NOT_VERIFIED"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_finalize_rejects_changed_external_verifier_evidence(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json")
    verifier_evidence.write_text(json.dumps({"verified": False, "issuer": "tampered"}))
    with pytest.raises(promote.PromotionError, match="SIGNATURE_VERIFIER_EVIDENCE_HASH_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_prepare_rejects_transaction_input_inside_source_tree(tmp_path: Path):
    old, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    inside = new / "change.patch"
    inside.write_bytes(patch.read_bytes())
    with pytest.raises(promote.PromotionError, match="PATCH_INSIDE_SOURCE_TREE"):
        promote.prepare_release(
            evidence=evidence,
            baseline_zip=baseline,
            patch=inside,
            output_dir=tmp_path / "prepared",
            source_root=new,
            evidence_verifier=verifier,
            now=NOW,
        )


def test_run149_finalize_rejects_signed_zip_file_count_tampering(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    statement_path = prepared / "release-statement.json"
    doc = json.loads(statement_path.read_text())
    doc["artifacts"]["zip"]["fileCount"] += 1
    statement_path.write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    sig, verifier_evidence = _signature(tmp_path, statement_path)
    with pytest.raises(promote.PromotionError, match="PROMOTION_ZIP_FILE_COUNT_MISMATCH"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)


def test_run149_finalize_rejects_fresh_signature_that_predates_statement(tmp_path: Path):
    _, new, baseline, patch, evidence, _, verifier = _fixture(tmp_path)
    prepared = tmp_path / "prepared"
    promote.prepare_release(evidence=evidence, baseline_zip=baseline, patch=patch, output_dir=prepared, source_root=new, evidence_verifier=verifier, now=NOW)
    sig, verifier_evidence = _signature(tmp_path, prepared / "release-statement.json", verified_at=NOW - timedelta(hours=1))
    with pytest.raises(promote.PromotionError, match="SIGNATURE_RECORD_PREDATES_STATEMENT"):
        promote.finalize_release(evidence=evidence, prepared_dir=prepared, baseline_zip=baseline, signature_record=sig, signature_verifier_evidence=verifier_evidence, promotion_dir=tmp_path / "promoted", source_root=new, evidence_verifier=verifier, now=NOW)
