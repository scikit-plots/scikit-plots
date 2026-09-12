# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Run 144 — ephemeral generated-artifact lifecycle and ZIP correlation."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy import app
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils import _provider_artifact_lifecycle as lifecycle_mod
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact import (
    PROVIDER_ARTIFACT_CONTRACT,
    PROVIDER_ARTIFACT_RECEIPT_CONTRACT,
    ProviderArtifactError,
    ProviderArtifactGeneratorSpec,
    build_provider_artifact_receipt_from_digest,
    parse_provider_artifact_request,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._provider_artifact_lifecycle import (
    PROVIDER_ARTIFACT_CANCEL_CONTRACT,
    PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT,
    ProviderArtifactLifecycleRegistry,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._zip_artifact import (
    ZIP_EDIT_CONTRACT,
    ZIP_EDIT_RECEIPT_CONTRACT,
)


def _request(*, prompt: str = "draw one tiny icon", regenerate_of: str = "", dedupe_key: str = "1" * 32, cancel_token: str = "2" * 64):
    raw = {
        "contract": PROVIDER_ARTIFACT_CONTRACT,
        "generator_id": "stub/generated-png",
        "kind": "image",
        "prompt": prompt,
        "mime_type": "image/png",
        "options": {},
        "cancel_token": cancel_token,
        "dedupe_key": dedupe_key,
    }
    if regenerate_of:
        raw["regenerate_of"] = regenerate_of
    return parse_provider_artifact_request(json.dumps(raw).encode())


def _receipt(request, *, size: int = 123, sha: str = "a" * 64):
    spec = ProviderArtifactGeneratorSpec(
        id="stub/generated-png",
        provider="stub",
        model="deterministic-png",
        kind="image",
        mime_types=("image/png",),
        max_output_bytes=1024 * 1024,
        option_schema={},
        diagnostic=True,
    )
    return build_provider_artifact_receipt_from_digest(
        request=request,
        spec=spec,
        output_size=size,
        output_sha256=sha,
    )


def _artifact_body(**overrides) -> bytes:
    raw = {
        "contract": PROVIDER_ARTIFACT_CONTRACT,
        "generator_id": "stub/generated-png",
        "kind": "image",
        "prompt": "A lifecycle fixture icon",
        "mime_type": "image/png",
        "options": {},
        "cancel_token": "3" * 64,
        "dedupe_key": "4" * 32,
    }
    raw.update(overrides)
    return json.dumps(raw).encode()


def _source_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("assets/icon.png", b"old-icon-bytes")
        zf.writestr("pkg/module.py", b"VALUE = 1\n")
    return out.getvalue()


def test_lifecycle_registry_retains_hashes_not_prompt_bytes_or_cancel_capability() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        request = _request(prompt="secret prompt text that must not be retained raw")
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        row = registry._records[lifecycle_id]  # structural privacy regression guard
        assert "prompt" not in vars(row)
        assert "options" not in vars(row)
        assert "bytes" not in vars(row)
        assert "path" not in vars(row)
        assert "secret" not in repr(vars(row))
        assert request.cancel_token not in repr(vars(row))
        assert row.cancel_token_sha256 == hashlib.sha256(request.cancel_token.encode()).hexdigest()
        assert row.prompt_sha256 == request.prompt_sha256
        public = row.public_status()
        assert "prompt_sha256" not in public
        assert "cancel_token_sha256" not in public
        assert "provider" not in public and "model" not in public
        await registry.clear()
    asyncio.run(run())


def test_duplicate_generation_is_suppressed_until_explicit_regeneration() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry(duplicate_window_seconds=90)
        request = _request()
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        try:
            await registry.begin(_request(cancel_token="5" * 64), provider="stub", model="deterministic-png")
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_DUPLICATE"
        else:  # pragma: no cover
            raise AssertionError("equivalent generation was not suppressed")
        ready = await registry.complete(lifecycle_id, _receipt(request))
        regenerated = _request(regenerate_of=ready.lifecycle_id, cancel_token="6" * 64)
        child_id = await registry.begin(regenerated, provider="stub", model="deterministic-png")
        assert child_id != lifecycle_id
        child = registry._records[child_id]
        assert child.regeneration_of == lifecycle_id
        await registry.clear()
    asyncio.run(run())


def test_cancel_capability_terminates_attached_generation_task() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        request = _request(cancel_token="7" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        task = asyncio.create_task(asyncio.sleep(60))
        await registry.attach_task(lifecycle_id, task)
        status = await registry.cancel(request.cancel_token)
        await asyncio.sleep(0)
        assert status["contract"] == PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT
        assert status["lifecycle_id"] == lifecycle_id
        assert status["status"] == "cancelled"
        assert task.cancelled()
        try:
            await registry.cancel("8" * 64)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_CANCEL_INVALID"
        else:  # pragma: no cover
            raise AssertionError("unknown cancellation capability was accepted")
        await registry.clear()
    asyncio.run(run())


def test_cancel_rejects_non_ascii_before_hashing() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        try:
            await registry.cancel("é" * 64)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_CANCEL_INVALID"
        else:  # pragma: no cover
            raise AssertionError("non-ASCII cancellation token escaped strict validation")
    asyncio.run(run())


def test_applied_candidate_requires_explicit_regeneration_for_same_fingerprint() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry(duplicate_window_seconds=90)
        request = _request(cancel_token="b" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request))
        reservation_id, reserved_ids = await registry.reserve_zip_correlations(
            ((lifecycle_id, "a" * 64, 123),)
        )
        await registry.mark_applied(reserved_ids, reservation_id)
        try:
            await registry.begin(_request(cancel_token="c" * 64), provider="stub", model="deterministic-png")
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_DUPLICATE"
        else:  # pragma: no cover
            raise AssertionError("applied candidate was silently regenerated without explicit lineage")
        child = _request(regenerate_of=lifecycle_id, cancel_token="d" * 64)
        child_id = await registry.begin(child, provider="stub", model="deterministic-png")
        assert child_id != lifecycle_id
        assert registry._records[child_id].regeneration_of == lifecycle_id
        await registry.clear()
    asyncio.run(run())


def test_expired_candidate_loses_zip_correlation_authority(monkeypatch) -> None:
    clock = [1_000]
    monkeypatch.setattr(lifecycle_mod.time, "time", lambda: clock[0])

    async def run():
        registry = ProviderArtifactLifecycleRegistry(candidate_ttl_seconds=30)
        request = _request(cancel_token="9" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request, size=17, sha="b" * 64))
        clock[0] += 31
        try:
            await registry.validate_zip_correlation(lifecycle_id=lifecycle_id, output_sha256="b" * 64, output_size=17)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:  # pragma: no cover
            raise AssertionError("expired provenance remained writable")
        assert await registry.public_status(lifecycle_id) is None
    asyncio.run(run())


def test_zip_correlation_is_hash_size_bound_and_single_use_after_apply() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        request = _request(cancel_token="a" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request, size=19, sha="c" * 64))
        await registry.mark_delivered(lifecycle_id)
        ok = await registry.validate_zip_correlation(lifecycle_id=lifecycle_id, output_sha256="c" * 64, output_size=19)
        assert ok["status"] == "delivered"
        for size, sha in ((18, "c" * 64), (19, "d" * 64)):
            try:
                await registry.validate_zip_correlation(lifecycle_id=lifecycle_id, output_sha256=sha, output_size=size)
            except ProviderArtifactError as exc:
                assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
            else:  # pragma: no cover
                raise AssertionError("mismatched provenance was accepted")
        reservation_id, reserved_ids = await registry.reserve_zip_correlations(
            ((lifecycle_id, "c" * 64, 19),)
        )
        await registry.mark_applied(reserved_ids, reservation_id)
        try:
            await registry.validate_zip_correlation(lifecycle_id=lifecycle_id, output_sha256="c" * 64, output_size=19)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:  # pragma: no cover
            raise AssertionError("applied lifecycle was replayable")
        await registry.clear()
    asyncio.run(run())


def test_simultaneous_zip_reservations_are_atomic_and_single_winner() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        request = _request(cancel_token="e" * 64)
        lifecycle_id = await registry.begin(request, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_id, _receipt(request, size=23, sha="e" * 64))
        await registry.mark_delivered(lifecycle_id)
        correlation = ((lifecycle_id, "e" * 64, 23),)

        async def attempt():
            try:
                reservation_id, lifecycle_ids = await registry.reserve_zip_correlations(correlation)
                return ("reserved", reservation_id, lifecycle_ids)
            except ProviderArtifactError as exc:
                return ("rejected", exc.code, ())

        results = await asyncio.gather(attempt(), attempt())
        winners = [row for row in results if row[0] == "reserved"]
        rejected = [row for row in results if row[0] == "rejected"]
        assert len(winners) == 1
        assert len(rejected) == 1
        assert rejected[0][1] == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"

        reservation_id = winners[0][1]
        reserved_ids = winners[0][2]
        assert reserved_ids == (lifecycle_id,)
        assert registry._records[lifecycle_id].status == "reserved"
        assert registry._records[lifecycle_id].reservation_id == reservation_id

        # An interrupted ZIP build releases authority back to the exact prior
        # ready/delivered state so a later explicit retry may reserve it once.
        await registry.release_zip_reservation(reserved_ids, reservation_id)
        assert registry._records[lifecycle_id].status == "delivered"
        retry_id, retry_ids = await registry.reserve_zip_correlations(correlation)
        assert retry_ids == (lifecycle_id,)
        await registry.mark_applied(retry_ids, retry_id)

        try:
            await registry.reserve_zip_correlations(correlation)
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:  # pragma: no cover
            raise AssertionError("applied lifecycle regained ZIP reservation authority")
        await registry.clear()

    asyncio.run(run())


def test_multi_lifecycle_zip_reservation_is_all_or_nothing() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry()
        request_a = _request(dedupe_key="a" * 32, cancel_token="1" * 64)
        request_b = _request(dedupe_key="b" * 32, cancel_token="2" * 64)
        lifecycle_a = await registry.begin(request_a, provider="stub", model="deterministic-png")
        lifecycle_b = await registry.begin(request_b, provider="stub", model="deterministic-png")
        await registry.complete(lifecycle_a, _receipt(request_a, size=31, sha="1" * 64))
        await registry.complete(lifecycle_b, _receipt(request_b, size=37, sha="2" * 64))
        await registry.mark_delivered(lifecycle_a)
        await registry.mark_delivered(lifecycle_b)

        try:
            await registry.reserve_zip_correlations(
                (
                    (lifecycle_a, "1" * 64, 31),
                    (lifecycle_b, "f" * 64, 37),
                )
            )
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"
        else:  # pragma: no cover
            raise AssertionError("partially invalid provenance transaction was reserved")

        assert registry._records[lifecycle_a].status == "delivered"
        assert registry._records[lifecycle_b].status == "delivered"
        assert registry._records[lifecycle_a].reservation_id == ""
        assert registry._records[lifecycle_b].reservation_id == ""

        reservation_id, lifecycle_ids = await registry.reserve_zip_correlations(
            (
                (lifecycle_a, "1" * 64, 31),
                (lifecycle_b, "2" * 64, 37),
            )
        )
        assert lifecycle_ids == (lifecycle_a, lifecycle_b)
        assert {registry._records[row].reservation_id for row in lifecycle_ids} == {reservation_id}
        assert {registry._records[row].status for row in lifecycle_ids} == {"reserved"}
        await registry.release_zip_reservation(lifecycle_ids, reservation_id)
        assert {registry._records[row].status for row in lifecycle_ids} == {"delivered"}
        await registry.clear()

    asyncio.run(run())


def test_lifecycle_registry_fails_closed_when_bounded_capacity_is_all_generating() -> None:
    async def run():
        registry = ProviderArtifactLifecycleRegistry(max_records=32)
        for idx in range(32):
            await registry.begin(
                _request(dedupe_key=f"{idx + 1:032x}", cancel_token=f"{idx + 1:064x}"),
                provider="stub",
                model="deterministic-png",
            )
        try:
            await registry.begin(_request(dedupe_key="f" * 32, cancel_token="f" * 64), provider="stub", model="deterministic-png")
        except ProviderArtifactError as exc:
            assert exc.code == "PROVIDER_ARTIFACT_LIFECYCLE_BUSY"
        else:  # pragma: no cover
            raise AssertionError("bounded lifecycle registry overcommitted active generations")
        await registry.clear()
    asyncio.run(run())


def test_health_advertises_lifecycle_without_secrets_or_persistence_claim(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        caps = client.get("/health").json()["capabilities"]["provider_artifact_output"]
    assert caps["version"] == 2
    assert caps["lifecycle_contract"] == PROVIDER_ARTIFACT_LIFECYCLE_CONTRACT
    assert caps["cancel_contract"] == PROVIDER_ARTIFACT_CANCEL_CONTRACT
    assert caps["cancel_endpoint"] == "/v1/artifacts/provider-output/cancel"
    assert caps["candidate_ttl_seconds"] > 0
    rendered = json.dumps(caps, sort_keys=True)
    assert "cancel_token" not in rendered
    assert "prompt_sha256" not in rendered
    assert "lifecycle_id" not in rendered
    assert "output_sha256" not in rendered


def test_endpoint_receipt_has_ephemeral_lifecycle_and_no_raw_prompt(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        response = client.post("/v1/artifacts/provider-output", content=_artifact_body(), headers={"content-type": "application/json"})
    assert response.status_code == 200, response.text
    receipt = json.loads(response.headers["x-ai-artifact-receipt"])
    assert receipt["contract"] == PROVIDER_ARTIFACT_RECEIPT_CONTRACT
    assert len(receipt["lifecycle_id"]) == 32
    assert receipt["state"] == "ready"
    assert receipt["expires_at"] > receipt["created_at"]
    assert receipt["regeneration_of"] is None
    assert "A lifecycle fixture icon" not in response.headers["x-ai-artifact-receipt"]
    assert "cancel_token" not in response.headers["x-ai-artifact-receipt"]


def test_endpoint_duplicate_and_explicit_regeneration_semantics(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        first = client.post("/v1/artifacts/provider-output", content=_artifact_body(), headers={"content-type": "application/json"})
        assert first.status_code == 200
        first_receipt = json.loads(first.headers["x-ai-artifact-receipt"])
        duplicate = client.post(
            "/v1/artifacts/provider-output",
            content=_artifact_body(cancel_token="5" * 64),
            headers={"content-type": "application/json"},
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["code"] == "PROVIDER_ARTIFACT_DUPLICATE"
        child = client.post(
            "/v1/artifacts/provider-output",
            content=_artifact_body(cancel_token="6" * 64, regenerate_of=first_receipt["lifecycle_id"]),
            headers={"content-type": "application/json"},
        )
        assert child.status_code == 200, child.text
        child_receipt = json.loads(child.headers["x-ai-artifact-receipt"])
        assert child_receipt["lifecycle_id"] != first_receipt["lifecycle_id"]
        assert child_receipt["regeneration_of"] == first_receipt["lifecycle_id"]


def test_cancel_endpoint_is_strict_and_no_store(monkeypatch) -> None:
    app._provider_artifact_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        bad = client.post(
            "/v1/artifacts/provider-output/cancel",
            json={"contract": PROVIDER_ARTIFACT_CANCEL_CONTRACT, "cancel_token": "not-a-capability"},
        )
        extra = client.post(
            "/v1/artifacts/provider-output/cancel",
            json={"contract": PROVIDER_ARTIFACT_CANCEL_CONTRACT, "cancel_token": "1" * 64, "path": "asset.png"},
        )
    assert bad.status_code == 409
    assert bad.headers["cache-control"] == "no-store"
    assert extra.status_code == 409
    assert extra.json()["code"] == "PROVIDER_ARTIFACT_CANCEL_INVALID"


def test_generated_candidate_correlates_into_zip_receipt_then_becomes_nonreplayable(monkeypatch) -> None:
    app._provider_artifact_rl.clear(); app._zip_edit_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        generated = client.post("/v1/artifacts/provider-output", content=_artifact_body(), headers={"content-type": "application/json"})
        assert generated.status_code == 200, generated.text
        replacement = generated.content
        provider_receipt = json.loads(generated.headers["x-ai-artifact-receipt"])
        lifecycle_id = provider_receipt["lifecycle_id"]
        source = _source_zip()
        replacement_sha = hashlib.sha256(replacement).hexdigest()
        manifest = {
            "contract": ZIP_EDIT_CONTRACT,
            "source": {"size": len(source), "sha256": hashlib.sha256(source).hexdigest()},
            "authorization": {"paths": ["assets/icon.png"]},
            "proposal": {"replacements": [{
                "id": "r1",
                "path": "assets/icon.png",
                "size": len(replacement),
                "sha256": replacement_sha,
                "provider_artifact_id": lifecycle_id,
            }]},
        }
        files = {
            "manifest": (None, json.dumps(manifest), "application/json"),
            "archive": ("project.zip", source, "application/zip"),
            "replacement:r1": ("icon.png", replacement, "image/png"),
        }
        edited = client.post("/v1/artifacts/zip-edit", files=files)
        assert edited.status_code == 200, edited.text
        receipt = json.loads(edited.headers["x-ai-artifact-receipt"])
        assert receipt["contract"] == ZIP_EDIT_RECEIPT_CONTRACT
        assert receipt["provider_artifact_ids"] == [lifecycle_id]
        assert "assets/icon.png" not in edited.headers["x-ai-artifact-receipt"]
        with zipfile.ZipFile(io.BytesIO(edited.content)) as zf:
            assert zf.read("assets/icon.png") == replacement
            assert zf.read("pkg/module.py") == b"VALUE = 1\n"
        replay = client.post("/v1/artifacts/zip-edit", files=files)
        assert replay.status_code == 409
        assert replay.json()["code"] == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"


def test_one_provider_lifecycle_cannot_ambiguously_correlate_to_two_paths(monkeypatch) -> None:
    app._provider_artifact_rl.clear(); app._zip_edit_rl.clear()
    monkeypatch.setattr(app, "_SHARED_RATE_LIMITER", None)
    with TestClient(app.app) as client:
        generated = client.post("/v1/artifacts/provider-output", content=_artifact_body(), headers={"content-type": "application/json"})
        assert generated.status_code == 200
        replacement = generated.content
        lifecycle_id = json.loads(generated.headers["x-ai-artifact-receipt"])["lifecycle_id"]
        source_io = io.BytesIO()
        with zipfile.ZipFile(source_io, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("a.png", b"old-a")
            zf.writestr("b.png", b"old-b")
        source = source_io.getvalue(); sha = hashlib.sha256(replacement).hexdigest()
        manifest = {
            "contract": ZIP_EDIT_CONTRACT,
            "source": {"size": len(source), "sha256": hashlib.sha256(source).hexdigest()},
            "authorization": {"paths": ["a.png", "b.png"]},
            "proposal": {"replacements": [
                {"id": "r1", "path": "a.png", "size": len(replacement), "sha256": sha, "provider_artifact_id": lifecycle_id},
                {"id": "r2", "path": "b.png", "size": len(replacement), "sha256": sha, "provider_artifact_id": lifecycle_id},
            ]},
        }
        files = [
            ("manifest", (None, json.dumps(manifest), "application/json")),
            ("archive", ("project.zip", source, "application/zip")),
            ("replacement:r1", ("a.png", replacement, "image/png")),
            ("replacement:r2", ("b.png", replacement, "image/png")),
        ]
        response = client.post("/v1/artifacts/zip-edit", files=files)
    assert response.status_code == 409
    assert response.json()["code"] == "PROVIDER_ARTIFACT_PROVENANCE_INVALID"

# Large-contract case fragments are collected only through this canonical owner.
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._case_loader import export_case_tests as _export_case_tests

_export_case_tests(globals(), package=__package__, case_package='_cases._provider_artifact_lifecycle', cases=('redis_chaos', 'shared'))
del _export_case_tests
