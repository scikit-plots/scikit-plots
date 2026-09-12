# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 139 — server-authoritative ZIP edit artifact orchestration."""

from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import hashlib
import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _zip_workspace as zw
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._zip_artifact import (
    ZIP_EDIT_CONTRACT,
    ZIP_EDIT_MAX_ENTRY_BYTES,
    ZIP_EDIT_MAX_REPLACEMENT_TOTAL_BYTES,
    ZIP_EDIT_MAX_SOURCE_BYTES,
    ZIP_EDIT_RECEIPT_CONTRACT,
    ParsedZipEditRequest,
    ZipArtifactError,
    build_zip_edit_artifact,
    encode_zip_edit_receipt_header,
    parse_zip_edit_manifest,
)


def _zip_bytes() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("pkg/a.py", "print('old')\n", compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("pkg/data.bin", b"\x00\x01\x02", compress_type=zipfile.ZIP_STORED)
        zf.writestr("README.md", "# demo\n", compress_type=zipfile.ZIP_DEFLATED)
    return out.getvalue()


def _manifest(source: bytes, replacement: bytes, *, path: str = "pkg/a.py", auth_paths=None, rid: str = "edit1") -> dict:
    return {
        "contract": ZIP_EDIT_CONTRACT,
        "source": {"size": len(source), "sha256": hashlib.sha256(source).hexdigest()},
        "authorization": {"paths": list(auth_paths or [path])},
        "proposal": {
            "replacements": [
                {
                    "id": rid,
                    "path": path,
                    "size": len(replacement),
                    "sha256": hashlib.sha256(replacement).hexdigest(),
                }
            ]
        },
    }


def _read_zip(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist() if not info.is_dir()}


def test_manifest_separates_authorization_from_model_proposal() -> None:
    source = _zip_bytes()
    replacement = b"print('new')\n"
    raw = _manifest(source, replacement, path="README.md", auth_paths=["pkg/a.py"])
    try:
        parse_zip_edit_manifest(raw)
    except ZipArtifactError as exc:
        assert exc.code == "ZIP_EDIT_NOT_AUTHORIZED"
    else:  # pragma: no cover
        raise AssertionError("proposal outside authorization unexpectedly accepted")


def test_manifest_is_strict_and_alias_safe() -> None:
    source = _zip_bytes()
    replacement = b"x"
    raw = _manifest(source, replacement)
    raw["model"] = "provider-controlled"
    try:
        parse_zip_edit_manifest(raw)
    except ZipArtifactError as exc:
        assert exc.code == "ZIP_EDIT_MANIFEST_INVALID"
    else:  # pragma: no cover
        raise AssertionError("unknown root authority unexpectedly accepted")

    raw = _manifest(source, replacement, auth_paths=["pkg/a.py", "PKG/A.PY"])
    try:
        parse_zip_edit_manifest(raw)
    except ZipArtifactError as exc:
        assert exc.code == "ZIP_EDIT_MANIFEST_INVALID"
    else:  # pragma: no cover
        raise AssertionError("authorization alias unexpectedly accepted")


def test_workspace_streams_seekable_replacement_sources() -> None:
    class NoReadAll(io.BytesIO):
        def read(self, size=-1):  # noqa: ANN001
            if size is None or size < 0:
                raise AssertionError("replacement source must be consumed in bounded chunks")
            return super().read(size)

    source = io.BytesIO(_zip_bytes())
    replacement = NoReadAll(b"A" * (2 * 1024 * 1024 + 17))
    limits = zw.ZipWorkspaceLimits(
        max_entries=32,
        max_entry_uncompressed_bytes=4 * 1024 * 1024,
        max_total_uncompressed_bytes=8 * 1024 * 1024,
        max_replacement_total_bytes=4 * 1024 * 1024,
        max_compression_ratio=500,
        max_source_bytes=8 * 1024 * 1024,
        chunk_bytes=64 * 1024,
        output_spool_bytes=128 * 1024,
        max_path_chars=4096,
    )
    with zw.rewrite_zip_workspace(source, {"pkg/a.py": replacement}, limits=limits) as artifact:
        artifact.file.seek(0)
        rows = _read_zip(artifact.file.read())
    assert rows["pkg/a.py"] == b"A" * (2 * 1024 * 1024 + 17)
    assert replacement.tell() == 0



def test_workspace_rebinds_expected_source_and_replacement_generations() -> None:
    source = _zip_bytes()
    replacement = b"expected"
    source_sha = hashlib.sha256(source).hexdigest()
    replacement_sha = hashlib.sha256(replacement).hexdigest()

    try:
        zw.rewrite_zip_workspace(
            io.BytesIO(source),
            {"pkg/a.py": io.BytesIO(replacement)},
            authorized_paths=("pkg/a.py",),
            expected_source_sha256="0" * 64,
            expected_source_size=len(source),
            replacement_expectations={"pkg/a.py": (len(replacement), replacement_sha)},
        )
    except zw.ZipWorkspaceError as exc:
        assert "expected SHA-256" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale expected source generation unexpectedly accepted")

    try:
        zw.rewrite_zip_workspace(
            io.BytesIO(source),
            {"pkg/a.py": io.BytesIO(replacement)},
            authorized_paths=("pkg/a.py",),
            expected_source_sha256=source_sha,
            expected_source_size=len(source),
            replacement_expectations={"pkg/a.py": (len(replacement), "f" * 64)},
        )
    except zw.ZipWorkspaceError as exc:
        assert "replacement generation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale expected replacement generation unexpectedly accepted")

def test_build_artifact_preserves_complete_tree_and_emits_path_free_receipt() -> None:
    source = _zip_bytes()
    replacement = b"print('changed')\n"
    manifest = parse_zip_edit_manifest(_manifest(source, replacement))
    parsed = ParsedZipEditRequest(
        manifest=manifest,
        source=UploadFile(io.BytesIO(source), filename="../danger project.zip"),
        replacements={"edit1": UploadFile(io.BytesIO(replacement), filename="whatever.py")},
        wire_body_bytes=0,
        wire_body_sha256="0" * 64,
    )
    artifact = build_zip_edit_artifact(parsed)
    try:
        artifact.file.seek(0)
        output = artifact.file.read()
        rows = _read_zip(output)
        assert rows["pkg/a.py"] == replacement
        assert rows["pkg/data.bin"] == b"\x00\x01\x02"
        assert rows["README.md"] == b"# demo\n"
        assert set(rows) == {"pkg/a.py", "pkg/data.bin", "README.md"}
        assert artifact.receipt.source_sha256 == hashlib.sha256(source).hexdigest()
        assert artifact.receipt.output_sha256 == hashlib.sha256(output).hexdigest()
        assert artifact.receipt.authorized_count == 1
        assert artifact.receipt.applied_count == 1
        assert artifact.filename == "danger_project.modified.zip"
        header = encode_zip_edit_receipt_header(artifact.receipt)
        assert "pkg/a.py" not in header
        assert "provider" not in header.lower()
        assert "model" not in header.lower()
    finally:
        artifact.close()


def test_authorization_cannot_add_a_new_archive_path() -> None:
    source = _zip_bytes()
    replacement = b"new"
    manifest = parse_zip_edit_manifest(_manifest(source, replacement, path="pkg/new.py"))
    parsed = ParsedZipEditRequest(
        manifest=manifest,
        source=UploadFile(io.BytesIO(source), filename="project.zip"),
        replacements={"edit1": UploadFile(io.BytesIO(replacement), filename="new.py")},
        wire_body_bytes=0,
        wire_body_sha256="0" * 64,
    )
    try:
        build_zip_edit_artifact(parsed)
    except ZipArtifactError as exc:
        assert exc.code == "ZIP_EDIT_WORKSPACE_REJECTED"
    else:  # pragma: no cover
        raise AssertionError("authorization unexpectedly overrode source-tree authority")



def test_unused_authorized_paths_must_still_exist_in_source_archive() -> None:
    source = _zip_bytes()
    replacement = b"print('changed')\n"
    manifest = parse_zip_edit_manifest(
        _manifest(source, replacement, auth_paths=["pkg/a.py", "pkg/missing.py"])
    )
    parsed = ParsedZipEditRequest(
        manifest=manifest,
        source=UploadFile(io.BytesIO(source), filename="project.zip"),
        replacements={"edit1": UploadFile(io.BytesIO(replacement), filename="a.py")},
        wire_body_bytes=0,
        wire_body_sha256="0" * 64,
    )
    try:
        build_zip_edit_artifact(parsed)
    except ZipArtifactError as exc:
        assert exc.code == "ZIP_EDIT_WORKSPACE_REJECTED"
    else:  # pragma: no cover
        raise AssertionError("nonexistent authorization path unexpectedly accepted")

def test_zip_edit_endpoint_returns_complete_zip_and_bounded_receipt(monkeypatch) -> None:
    source = _zip_bytes()
    replacement = b"print('api')\n"
    manifest = _manifest(source, replacement)
    app._zip_edit_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/zip-edit",
            data={"manifest": json.dumps(manifest)},
            files={
                "archive": ("project.zip", source, "application/zip"),
                "replacement:edit1": ("a.py", replacement, "text/x-python"),
            },
        )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    assert response.headers["content-disposition"] == 'attachment; filename="project.modified.zip"'
    assert response.headers["x-ai-artifact-contract"] == ZIP_EDIT_RECEIPT_CONTRACT
    assert response.headers["x-ai-artifact-sha256"] == hashlib.sha256(response.content).hexdigest()
    receipt = json.loads(response.headers["x-ai-artifact-receipt"])
    assert receipt["contract"] == "scikitplot-zip-edit-receipt-v1"
    assert receipt["applied_count"] == 1
    assert "changed_paths" not in receipt
    assert _read_zip(response.content)["pkg/a.py"] == replacement
    assert set(_read_zip(response.content)) == {"pkg/a.py", "pkg/data.bin", "README.md"}



def test_zip_edit_endpoint_rejects_duplicate_archive_field_before_rewrite(monkeypatch) -> None:
    source = _zip_bytes()
    replacement = b"new"
    manifest = _manifest(source, replacement)
    app._zip_edit_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/zip-edit",
            data={"manifest": json.dumps(manifest)},
            files=[
                ("archive", ("project.zip", source, "application/zip")),
                ("archive", ("duplicate.zip", source, "application/zip")),
                ("replacement:edit1", ("a.py", replacement, "text/plain")),
            ],
        )
    assert response.status_code == 400
    assert response.json()["code"] == "ZIP_EDIT_INVALID"

def test_zip_edit_endpoint_rejects_stale_source_and_replacement_generations(monkeypatch) -> None:
    source = _zip_bytes()
    replacement = b"new"
    app._zip_edit_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)

    stale = _manifest(source, replacement)
    stale["source"]["sha256"] = "0" * 64
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/zip-edit",
            data={"manifest": json.dumps(stale)},
            files={
                "archive": ("project.zip", source, "application/zip"),
                "replacement:edit1": ("a.py", replacement, "text/plain"),
            },
        )
    assert response.status_code == 409
    assert response.json()["code"] == "ZIP_EDIT_SOURCE_MISMATCH"

    app._zip_edit_rl.clear()
    mismatch = _manifest(source, replacement)
    mismatch["proposal"]["replacements"][0]["sha256"] = "f" * 64
    with TestClient(app.app) as client:
        response = client.post(
            "/v1/artifacts/zip-edit",
            data={"manifest": json.dumps(mismatch)},
            files={
                "archive": ("project.zip", source, "application/zip"),
                "replacement:edit1": ("a.py", replacement, "text/plain"),
            },
        )
    assert response.status_code == 422
    assert response.json()["code"] == "ZIP_EDIT_REPLACEMENT_MISMATCH"


def test_zip_edit_capability_is_provider_neutral() -> None:
    caps = app._public_capabilities()["zip_edit_artifact"]
    assert caps["contract"] == ZIP_EDIT_CONTRACT
    assert caps["receipt_contract"] == ZIP_EDIT_RECEIPT_CONTRACT
    assert caps["endpoint"] == "/v1/artifacts/zip-edit"
    assert caps["tree_authority"] == "source-archive"
    assert caps["max_source_bytes"] == ZIP_EDIT_MAX_SOURCE_BYTES
    assert caps["max_entry_bytes"] == ZIP_EDIT_MAX_ENTRY_BYTES
    assert caps["max_replacement_total_bytes"] == ZIP_EDIT_MAX_REPLACEMENT_TOTAL_BYTES
    rendered = json.dumps(caps, sort_keys=True).lower()
    assert "token" not in rendered
    assert "credential" not in rendered
    assert "provider" not in rendered


def test_orchestrator_source_contains_no_provider_adapter_dependency() -> None:
    root = RUNTIME_ROOT
    source = (root / "_hf_spaces_proxy" / "_utils" / "_zip_artifact.py").read_text(encoding="utf-8")
    assert "_providers" not in source
    assert "ProviderExecutor" not in source
    assert "provider_id" not in source
