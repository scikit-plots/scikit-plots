from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

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


publish = _load("run150_publish", SECURITY / "publish_release.py")
NOW = datetime(2026, 9, 5, 4, 30, tzinfo=timezone.utc)
REVISION = "e" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _promoted(tmp_path: Path) -> Path:
    root = tmp_path / "promoted"
    root.mkdir(parents=True)
    zip_path = root / "run150-test.zip"
    patch_path = root / "run150-test.patch"
    zip_path.write_bytes(b"zip-bytes\n")
    patch_path.write_bytes(b"patch-bytes\n")
    sboms = {
        "pythonRuntime": {"name": "python-runtime.cdx.json", "sha256": "f" * 64},
        "image": {"name": "image.cdx.json", "sha256": "d" * 64, "subject": "sha256:" + "1" * 64},
    }
    statement = {
        "schemaVersion": 1,
        "predicateType": "https://scikit-plots.org/attestations/release-promotion/v1",
        "generatedAt": "2026-09-05T03:58:00Z",
        "release": {"releaseId": "run150-test", "proxyVersion": "7.4.0", "sourceRevision": REVISION},
        "subject": {"sourceTreeSha256": "a" * 64, "evidenceSha256": "b" * 64},
        "artifacts": {
            "zip": {"name": zip_path.name, "sha256": _sha(zip_path), "size": zip_path.stat().st_size, "fileCount": 345},
            "patch": {"name": patch_path.name, "sha256": _sha(patch_path), "size": patch_path.stat().st_size},
            "baselineZip": {"name": "run149.zip", "sha256": "9" * 64, "size": 12345},
        },
        "sbomReferences": sboms,
        "verification": {
            "evidenceVerified": True, "patchRecreatesSourceTree": True,
            "zipRecreatesSourceTree": True, "deterministicZip": True,
        },
    }
    statement_path = root / "release-statement.json"
    statement_path.write_bytes(publish._canonical_bytes(statement))
    signature = {
        "schemaVersion": 1,
        "verified": True,
        "verifiedAt": "2026-09-05T03:59:00Z",
        "releaseStatementSha256": _sha(statement_path),
        "sourceRevision": REVISION,
        "signerIdentityVerified": True,
        "verifierEvidenceSha256": "c" * 64,
        "verifier": {"name": "test-verifier", "version": "1.0"},
    }
    signature_path = root / "release-statement.signature-verification.json"
    signature_path.write_bytes(publish._canonical_bytes(signature))
    payloads = [zip_path, patch_path, statement_path, signature_path]
    items = [
        {"name": path.name, "sha256": _sha(path), "size": path.stat().st_size}
        for path in payloads
    ]
    receipt = {
        "schemaVersion": 1,
        "finalizedAt": "2026-09-05T04:00:00Z",
        "releaseId": "run150-test",
        "sourceRevision": REVISION,
        "sourceTreeSha256": "a" * 64,
        "evidenceSha256": "b" * 64,
        "releaseStatementSha256": _sha(statement_path),
        "signatureVerificationSha256": _sha(signature_path),
        "signatureVerifierEvidenceSha256": "c" * 64,
        "publish": items,
        "sbomReferences": sboms,
        "status": "promoted",
    }
    (root / "promotion-receipt.json").write_bytes(publish._canonical_bytes(receipt))
    return root


class FakePublisher:
    def __init__(self, *, immutability: str = "object-lock", now: datetime = NOW):
        self.remote: dict[str, bytes] = {}
        self.immutability = immutability
        self.now = now
        self.calls: list[dict] = []
        self.mutate_local = False
        self.force_locator: str | None = None
        self.create_only = True
        self.overwrite = False
        self.readback = True
        self.artifact_override: dict | None = None
        self.on_call = None

    def __call__(self, request: dict) -> dict:
        self.calls.append(request)
        if self.on_call is not None:
            self.on_call(request, len(self.calls))
        art = request["artifact"]
        name = art["name"]
        if request["operation"] == "publish":
            path = Path(art["localPath"])
            data = path.read_bytes()
            if self.mutate_local:
                path.chmod(0o644)
                path.write_bytes(data + b"tamper")
            if name in self.remote:
                assert self.remote[name] == data
                status = "present"
            else:
                self.remote[name] = data
                status = "created"
        else:
            assert name in self.remote
            data = self.remote[name]
            assert hashlib.sha256(data).hexdigest() == art["sha256"]
            assert len(data) == art["size"]
            status = "present"
        artifact = {"name": art["name"], "sha256": art["sha256"], "size": art["size"]}
        if self.artifact_override is not None:
            artifact = dict(self.artifact_override)
        locator = self.force_locator or f"mock://release/{name}"
        return {
            "schemaVersion": 1,
            "operation": request["operation"],
            "publicationId": request["publicationId"],
            "status": status,
            "target": {"publisher": request["target"]["publisher"], "targetId": request["target"]["targetId"], "locator": locator},
            "artifact": artifact,
            "guarantees": {
                "createOnly": self.create_only,
                "overwrite": self.overwrite,
                "remoteReadbackVerified": self.readback,
                "immutability": self.immutability,
            },
            "verifiedAt": self.now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        }


def _run(tmp_path: Path, adapter: FakePublisher, **kwargs):
    promoted = kwargs.pop("promotion_dir", None) or _promoted(tmp_path)
    output = kwargs.pop("output_dir", tmp_path / "publication")
    result = publish.publish_release(
        promotion_dir=promoted,
        output_dir=output,
        publisher=kwargs.pop("publisher", "test-publisher"),
        target_id=kwargs.pop("target_id", "release/run150-test"),
        adapter=adapter,
        now=kwargs.pop("now", NOW),
        **kwargs,
    )
    return promoted, output, result


def test_run150_publishes_only_receipt_objects_and_emits_transparency(tmp_path: Path):
    adapter = FakePublisher()
    promoted, output, result = _run(tmp_path, adapter)
    assert result["phase"] == "published" and result["artifact_count"] == 4
    assert set(adapter.remote) == {
        "run150-test.zip", "run150-test.patch", "release-statement.json",
        "release-statement.signature-verification.json",
    }
    transparency = json.loads((output / "publication-transparency.json").read_text())
    receipt = json.loads((output / "publication-receipt.json").read_text())
    promotion_sha = _sha(promoted / "promotion-receipt.json")
    assert transparency["subject"]["promotionReceiptSha256"] == promotion_sha
    assert transparency["verification"] == {
        "promotionReceiptVerified": True,
        "localSnapshotVerified": True,
        "createOnly": True,
        "remoteReadbackVerified": True,
        "allReceiptObjectsPublished": True,
    }
    assert receipt["promotionReceiptSha256"] == promotion_sha
    assert receipt["transparencySha256"] == _sha(output / "publication-transparency.json")
    assert len(list((output / "publisher-results").glob("*.json"))) == 8
    text = (output / "publication-transparency.json").read_text()
    assert str(tmp_path) not in text and "localPath" not in text


def test_run150_is_resumable_when_exact_remote_objects_already_exist(tmp_path: Path):
    adapter = FakePublisher()
    promoted = _promoted(tmp_path)
    first_out = tmp_path / "first"
    publish.publish_release(promotion_dir=promoted, output_dir=first_out, publisher="test-publisher", target_id="release/run150-test", adapter=adapter, now=NOW)
    second_out = tmp_path / "second"
    out = publish.publish_release(promotion_dir=promoted, output_dir=second_out, publisher="test-publisher", target_id="release/run150-test", adapter=adapter, now=NOW)
    assert out["publication_id"] == json.loads((first_out / "publication-receipt.json").read_text())["publicationId"]
    assert all(call["operation"] in {"publish", "verify"} for call in adapter.calls)
    publish_results = [json.loads(p.read_text()) for p in (second_out / "publisher-results").glob("*.publish.json")]
    assert {item["status"] for item in publish_results} == {"present"}


def test_run150_rejects_tampered_or_extra_promotion_objects(tmp_path: Path):
    promoted = _promoted(tmp_path)
    (promoted / "run150-test.zip").write_bytes(b"changed")
    with pytest.raises(publish.PublicationError, match="PROMOTION_ARTIFACT_REBIND_FAILED"):
        publish.publish_release(promotion_dir=promoted, output_dir=tmp_path / "out1", publisher="p", target_id="t", adapter=FakePublisher(), now=NOW)

    promoted2 = _promoted(tmp_path / "second")
    (promoted2 / "extra.txt").write_text("not authorized")
    with pytest.raises(publish.PublicationError, match="PROMOTION_DIRECTORY_CONTAINS_UNAUTHORIZED_OBJECT"):
        publish.publish_release(promotion_dir=promoted2, output_dir=tmp_path / "out2", publisher="p", target_id="t", adapter=FakePublisher(), now=NOW)


def test_run150_rejects_noncanonical_or_rebound_promotion_receipt(tmp_path: Path):
    promoted = _promoted(tmp_path)
    receipt = json.loads((promoted / "promotion-receipt.json").read_text())
    (promoted / "promotion-receipt.json").write_text(json.dumps(receipt, indent=2))
    with pytest.raises(publish.PublicationError, match="PROMOTION_RECEIPT_NOT_CANONICAL"):
        publish.publish_release(promotion_dir=promoted, output_dir=tmp_path / "out", publisher="p", target_id="t", adapter=FakePublisher(), now=NOW)


def test_run150_rejects_overwrite_capable_or_unverified_publisher(tmp_path: Path):
    adapter = FakePublisher()
    adapter.create_only = False
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_AUTHORITY_INVALID"):
        _run(tmp_path, adapter)
    adapter = FakePublisher()
    adapter.readback = False
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_AUTHORITY_INVALID"):
        _run(tmp_path / "readback", adapter)


def test_run150_rejects_remote_artifact_rebinding_and_locator_collision(tmp_path: Path):
    adapter = FakePublisher()
    adapter.artifact_override = {"name": "wrong", "sha256": "0" * 64, "size": 1}
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_ARTIFACT_MISMATCH"):
        _run(tmp_path, adapter)
    adapter = FakePublisher()
    adapter.force_locator = "mock://release/same"
    with pytest.raises(publish.PublicationError, match="PUBLISHER_LOCATOR_COLLISION"):
        _run(tmp_path / "collision", adapter)


def test_run150_rejects_local_snapshot_mutation_during_upload(tmp_path: Path):
    adapter = FakePublisher()
    adapter.mutate_local = True
    with pytest.raises(publish.PublicationError, match="PUBLICATION_LOCAL_ARTIFACT_CHANGED_DURING_UPLOAD"):
        _run(tmp_path, adapter)


def test_run150_enforces_required_immutability_and_fresh_remote_verification(tmp_path: Path):
    adapter = FakePublisher(immutability="release-create-only")
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_IMMUTABILITY_INSUFFICIENT"):
        _run(tmp_path, adapter, required_immutability={"object-lock", "content-addressed"})
    stale = FakePublisher(now=NOW - timedelta(hours=2))
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_STALE"):
        _run(tmp_path / "stale", stale)
    future = FakePublisher(now=NOW + timedelta(hours=1))
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_FROM_FUTURE"):
        _run(tmp_path / "future", future)


def test_run150_detects_promotion_drift_during_publication(tmp_path: Path):
    promoted = _promoted(tmp_path)
    adapter = FakePublisher()

    def drift(request, call_no):
        if call_no == 2:
            (promoted / "run150-test.patch").write_bytes(b"operator drift")

    adapter.on_call = drift
    with pytest.raises(publish.PublicationError, match="PROMOTION_ARTIFACT_REBIND_FAILED"):
        publish.publish_release(promotion_dir=promoted, output_dir=tmp_path / "out", publisher="p", target_id="t", adapter=adapter, now=NOW)


def test_run150_publication_id_is_bound_to_receipt_publisher_and_target(tmp_path: Path):
    promoted = _promoted(tmp_path)
    doc, receipt_sha, _ = publish._validate_promotion(promoted)
    first = publish._publication_id(receipt_sha, "publisher-a", "release/a")
    assert first == publish._publication_id(receipt_sha, "publisher-a", "release/a")
    assert first != publish._publication_id(receipt_sha, "publisher-b", "release/a")
    assert first != publish._publication_id(receipt_sha, "publisher-a", "release/b")
    assert doc["releaseId"] == "run150-test"


def test_run150_command_publisher_uses_json_protocol_without_shell(tmp_path: Path):
    script = tmp_path / "adapter.py"
    script.write_text(
        "import json,sys\n"
        "r=json.load(sys.stdin)\n"
        "a=r['artifact']\n"
        "a={k:a[k] for k in ('name','sha256','size')}\n"
        "print(json.dumps({'schemaVersion':1,'operation':r['operation'],'publicationId':r['publicationId'],'status':'created' if r['operation']=='publish' else 'present','target':{'publisher':r['target']['publisher'],'targetId':r['target']['targetId'],'locator':'mock://release/'+a['name']},'artifact':a,'guarantees':{'createOnly':True,'overwrite':False,'remoteReadbackVerified':True,'immutability':'object-lock'},'verifiedAt':'2026-09-05T04:30:00Z'}))\n"
    )
    adapter = publish.command_publisher([sys.executable, str(script)])
    promoted = _promoted(tmp_path)
    out = publish.publish_release(promotion_dir=promoted, output_dir=tmp_path / "publication", publisher="p", target_id="t", adapter=adapter, now=NOW)
    assert out["artifact_count"] == 4


def test_run150_rebinds_receipt_to_signed_statement_and_verifier_identity(tmp_path: Path):
    promoted = _promoted(tmp_path)
    receipt_path = promoted / "promotion-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["releaseId"] = "run150-forged"
    receipt_path.write_bytes(publish._canonical_bytes(receipt))
    with pytest.raises(publish.PublicationError, match="PROMOTION_STATEMENT_RELEASE_REBIND_FAILED"):
        publish.publish_release(promotion_dir=promoted, output_dir=tmp_path / "bad-release", publisher="p", target_id="t", adapter=FakePublisher(), now=NOW)

    promoted2 = _promoted(tmp_path / "sig")
    sig_path = promoted2 / "release-statement.signature-verification.json"
    sig = json.loads(sig_path.read_text())
    sig["verifierEvidenceSha256"] = "8" * 64
    sig_path.write_bytes(publish._canonical_bytes(sig))
    receipt_path2 = promoted2 / "promotion-receipt.json"
    receipt2 = json.loads(receipt_path2.read_text())
    new_sig_sha = _sha(sig_path)
    receipt2["signatureVerificationSha256"] = new_sig_sha
    for item in receipt2["publish"]:
        if item["name"] == sig_path.name:
            item["sha256"] = new_sig_sha
            item["size"] = sig_path.stat().st_size
    receipt_path2.write_bytes(publish._canonical_bytes(receipt2))
    with pytest.raises(publish.PublicationError, match="PROMOTION_SIGNATURE_VERIFIER_REBIND_FAILED"):
        publish.publish_release(promotion_dir=promoted2, output_dir=tmp_path / "bad-sig", publisher="p", target_id="t", adapter=FakePublisher(), now=NOW)


def test_run150_rejects_secret_like_or_colliding_remote_locator(tmp_path: Path):
    adapter = FakePublisher()
    adapter.force_locator = "https://example.invalid/release?token=secret"
    with pytest.raises(publish.PublicationError, match="PUBLISHER_RESPONSE_LOCATOR_INVALID"):
        _run(tmp_path, adapter)


def test_run150_command_publisher_bounds_stdout_while_adapter_runs(tmp_path: Path):
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'x' * 400000)\n"
        "sys.stdout.flush()\n"
    )
    adapter = publish.command_publisher([sys.executable, str(script)])
    with pytest.raises(publish.PublicationError, match="PUBLISHER_COMMAND_OUTPUT_TOO_LARGE"):
        adapter({"schemaVersion": 1})


def test_run150_publication_output_cannot_live_inside_promotion_directory(tmp_path: Path):
    promoted = _promoted(tmp_path)
    with pytest.raises(publish.PublicationError, match="PUBLICATION_OUTPUT_INSIDE_PROMOTION"):
        publish.publish_release(
            promotion_dir=promoted,
            output_dir=promoted / "publication-evidence",
            publisher="p",
            target_id="t",
            adapter=FakePublisher(),
            now=NOW,
        )


def test_run150_command_publisher_rejects_duplicate_json_keys(tmp_path: Path):
    script = tmp_path / "duplicate.py"
    script.write_text(
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.write('{\\\"schemaVersion\\\":1,\\\"schemaVersion\\\":1}')\n"
    )
    adapter = publish.command_publisher([sys.executable, str(script)])
    with pytest.raises(publish.PublicationError, match="PUBLISHER_COMMAND_OUTPUT_DUPLICATE_KEY"):
        adapter({"schemaVersion": 1})


def test_run150_command_publisher_closes_pipes_when_child_fails(tmp_path: Path, monkeypatch):
    script = tmp_path / "fails.py"
    script.write_text("raise SystemExit(7)\n")
    seen = []
    real_popen = publish.subprocess.Popen

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        seen.append(proc)
        return proc

    monkeypatch.setattr(publish.subprocess, "Popen", tracking_popen)
    adapter = publish.command_publisher([sys.executable, str(script)])
    with pytest.raises(publish.PublicationError, match="PUBLISHER_COMMAND_FAILED"):
        adapter({"schemaVersion": 1})

    assert len(seen) == 1
    proc = seen[0]
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
