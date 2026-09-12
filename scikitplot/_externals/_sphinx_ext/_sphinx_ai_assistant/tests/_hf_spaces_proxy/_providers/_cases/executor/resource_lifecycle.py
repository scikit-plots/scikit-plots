from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass

import pytest
from starlette.datastructures import UploadFile

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.base import ResourceRoute
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._providers.executor import (
    ProviderExecutorRegistry,
    ProviderPrivateResource,
    ResourceExecutionError,
    ResourceExecutionSession,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_contract import (
    ResourceDescriptor,
    VerifiedResource,
)
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._hf_spaces_proxy._utils._resource_transport import ResourceUpload


def _upload(rid: str, data: bytes = b"abc", modality: str = "document") -> ResourceUpload:
    import hashlib

    descriptor = ResourceDescriptor(
        id=rid,
        name=f"{rid}.bin",
        mime_type="application/octet-stream",
        size=len(data),
        modality=modality,
        intent="auto",
    )
    verified = VerifiedResource(
        descriptor=descriptor,
        actual_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        detected_mime="application/octet-stream",
        detected_modality=modality,
        signature="binary",
    )
    return ResourceUpload(verified=verified, upload=UploadFile(io.BytesIO(data), filename=descriptor.name))


def _route(upload: ResourceUpload, route: str = "native") -> ResourceRoute:
    return ResourceRoute(
        resource_id=upload.verified.id,
        route=route,
        metadata={"modality": upload.verified.detected_modality, "intent": upload.verified.intent},
    )


@dataclass
class FakeExecutor:
    name: str = "openai"
    enabled: bool = True
    fail_prepare_id: str = ""
    fail_release_id: str = ""
    pause_prepare_id: str = ""

    def __post_init__(self):
        self.events: list[str] = []
        self.pause_started = asyncio.Event()
        self.pause_gate = asyncio.Event()

    async def prepare(self, *, model, route, upload):
        del model
        rid = upload.verified.id
        self.events.append(f"prepare:{rid}")
        if rid == self.pause_prepare_id:
            self.pause_started.set()
            await self.pause_gate.wait()
        if rid == self.fail_prepare_id:
            raise RuntimeError("provider prepare failed: PRIVATE_FILE_ID")
        return ProviderPrivateResource(
            resource_id=rid,
            route=route.route,
            provider=self.name,
            source_sha256=upload.verified.sha256,
            source_size=upload.verified.actual_size,
            opaque={"provider_file_id": f"PRIVATE-{rid}"},
        )

    async def release(self, handle):
        self.events.append(f"release:{handle.resource_id}")
        await asyncio.sleep(0)
        if handle.resource_id == self.fail_release_id:
            raise RuntimeError("cleanup failed PRIVATE_FILE_ID")


@pytest.mark.asyncio
async def test_prepare_is_sequential_and_cleanup_is_lifo_and_idempotent():
    a, b, c = _upload("r0"), _upload("r1"), _upload("r2")
    ex = FakeExecutor()
    session = ResourceExecutionSession(
        provider="openai", model="gpt-x", executor=ex, routes=[_route(a), _route(b), _route(c)]
    )
    receipts = await session.prepare([a, b, c])
    assert [r.resource_id for r in receipts] == ["r0", "r1", "r2"]
    assert ex.events == ["prepare:r0", "prepare:r1", "prepare:r2"]
    await session.close()
    assert ex.events[-3:] == ["release:r2", "release:r1", "release:r0"]
    before = list(ex.events)
    await session.close()
    assert ex.events == before
    assert all(r.state == "released" for r in session.receipts)


@pytest.mark.asyncio
async def test_partial_prepare_failure_cleans_prepared_handles():
    a, b = _upload("r0"), _upload("r1")
    ex = FakeExecutor(fail_prepare_id="r1")
    session = ResourceExecutionSession(
        provider="openai", model="gpt-x", executor=ex, routes=[_route(a), _route(b)]
    )
    with pytest.raises(RuntimeError):
        await session.prepare([a, b])
    assert ex.events == ["prepare:r0", "prepare:r1", "release:r0"]
    assert session.receipts[0].state == "released"


@pytest.mark.asyncio
async def test_unsupported_or_mismatched_plan_fails_before_provider_io():
    a, b = _upload("r0"), _upload("r1")
    ex = FakeExecutor()
    with pytest.raises(ResourceExecutionError):
        await ResourceExecutionSession(
            provider="openai", model="gpt-x", executor=ex,
            routes=[_route(a, "unsupported")],
        ).prepare([a])
    assert ex.events == []

    with pytest.raises(ResourceExecutionError):
        await ResourceExecutionSession(
            provider="openai", model="gpt-x", executor=ex, routes=[_route(a)]
        ).prepare([a, b])
    assert ex.events == []


@pytest.mark.asyncio
async def test_malformed_provider_handle_is_registered_then_released():
    a = _upload("r0")

    class BadHandleExecutor(FakeExecutor):
        async def prepare(self, *, model, route, upload):
            del model
            self.events.append("prepare:r0")
            return ProviderPrivateResource(
                resource_id="wrong",
                route=route.route,
                provider=self.name,
                source_sha256=upload.verified.sha256,
                source_size=upload.verified.actual_size,
                opaque={"provider_file_id": "PRIVATE-WRONG"},
            )

    ex = BadHandleExecutor()
    session = ResourceExecutionSession(
        provider="openai", model="gpt-x", executor=ex, routes=[_route(a)]
    )
    with pytest.raises(ResourceExecutionError):
        await session.prepare([a])
    assert ex.events == ["prepare:r0", "release:wrong"]


@pytest.mark.asyncio
async def test_cancellation_waits_for_cleanup_before_propagating():
    a, b = _upload("r0"), _upload("r1")
    ex = FakeExecutor(pause_prepare_id="r1")
    session = ResourceExecutionSession(
        provider="openai", model="gpt-x", executor=ex, routes=[_route(a), _route(b)]
    )
    task = asyncio.create_task(session.prepare([a, b]))
    await asyncio.wait_for(ex.pause_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ex.events[-1] == "release:r0"
    assert session.receipts[0].state == "released"


@pytest.mark.asyncio
async def test_cleanup_failure_is_sanitized_and_does_not_leak_private_handle():
    a = _upload("r0")
    ex = FakeExecutor(fail_release_id="r0")
    session = ResourceExecutionSession(
        provider="openai", model="gpt-x", executor=ex, routes=[_route(a)]
    )
    await session.prepare([a])
    await session.close()
    public = session.public_receipts()[0]
    assert public["state"] == "cleanup_failed"
    assert public["cleanup_error"] is True
    encoded = json.dumps(public)
    assert "PRIVATE" not in encoded
    assert "provider_file_id" not in encoded


def test_executor_registry_is_separate_fail_closed_authority():
    reg = ProviderExecutorRegistry(["openai", "anthropic"])
    assert reg.execution_state("openai") == "plan-only"
    assert reg.execution_state("anthropic") == "plan-only"
    assert reg.execution_state("missing") == "plan-only"
    ex = FakeExecutor(name="openai")
    reg.register("openai", ex)
    assert reg.execution_state("openai") == "enabled"
    with pytest.raises(ValueError):
        reg.register("anthropic", ex)

def test_route_plan_has_no_provider_identifier_field():
    fields = set(ResourceRoute.__dataclass_fields__)
    assert "provider_file_id" not in fields
    route = ResourceRoute(resource_id="r0", route="native")
    encoded = json.dumps(route.__dict__)
    assert "provider_file_id" not in encoded
