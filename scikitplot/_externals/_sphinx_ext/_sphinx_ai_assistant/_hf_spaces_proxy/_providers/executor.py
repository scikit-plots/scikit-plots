# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Credential-neutral provider resource execution lifecycle.

Run 128 defines the state machine every later provider executor must obey.  It
contains no provider credentials and performs no provider I/O by itself.  The
important invariant is ownership: once a provider-side resource is created it
is registered for cleanup immediately, and cleanup is attempted in reverse
creation order on success, error, timeout, or cancellation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Sequence

from typing_extensions import Self

from .base import ResourceRoute

try:
    from .._utils._resource_transport import ResourceUpload
except ImportError:  # standalone HF Space: _providers is top-level
    from _utils._resource_transport import ResourceUpload


class ResourceExecutionError(RuntimeError):
    """A provider resource could not enter a safe executable state."""


@dataclass
class ProviderPrivateResource:
    """Provider-owned resource handle that must never be serialized publicly."""

    resource_id: str
    route: str
    provider: str
    source_sha256: str
    source_size: int
    opaque: Any = None
    cleanup_required: bool = True
    released: bool = False


@dataclass
class ResourceExecutionReceipt:
    """Sanitized chain-of-custody record for one resource lifecycle."""

    resource_id: str
    provider: str
    model: str
    route: str
    source_sha256: str
    source_size: int
    state: str = "prepared"
    cleanup_required: bool = True
    cleanup_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_public_dict(self) -> dict[str, Any]:
        """Return a bounded receipt that contains no provider opaque identifier."""
        return {
            "resource_id": self.resource_id,
            "provider": self.provider,
            "model": self.model,
            "route": self.route,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "state": self.state,
            "cleanup_required": self.cleanup_required,
            "cleanup_error": self.cleanup_error,
            "metadata": dict(self.metadata),
        }


class ProviderResourceExecutor(Protocol):
    """Provider-specific prepare/release boundary with route-level authority."""

    name: str
    enabled: bool

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        """Return exact modality routes this executor can perform for *model*."""
        ...

    async def prepare(
        self,
        *,
        model: str,
        route: ResourceRoute,
        upload: ResourceUpload,
    ) -> ProviderPrivateResource: ...

    async def release(self, handle: ProviderPrivateResource) -> None: ...


class ProviderChatExecutor(ProviderResourceExecutor, Protocol):
    """
    Executable provider boundary for one prepared resource chat request.

    Provider planning and temporary-resource ownership remain generic.  Only an
    executor registered in the separate execution registry may implement this
    credential-bearing boundary.  ``chat`` and ``upstream`` deliberately stay
    provider-neutral here so the common proxy lifecycle does not depend on one
    SDK's public types.
    """

    async def open_chat(
        self,
        *,
        chat: Any,
        handles: Sequence[ProviderPrivateResource],
    ) -> Any: ...

    async def buffered_chat_response(self, upstream: Any, *, maximum: int) -> dict: ...

    def chat_sse(self, upstream: Any, *, maximum: int) -> AsyncIterator[bytes]: ...


@dataclass
class PendingProviderExecutor:
    """Explicit non-executor used until a credential-bearing adapter is enabled."""

    name: str
    enabled: bool = False

    def executable_routes(self, model: str) -> dict[str, tuple[str, ...]]:
        del model
        return {}

    async def prepare(
        self,
        *,
        model: str,
        route: ResourceRoute,
        upload: ResourceUpload,
    ) -> ProviderPrivateResource:
        del model, route, upload
        raise ResourceExecutionError(f"{self.name} resource executor is not enabled")

    async def release(self, handle: ProviderPrivateResource) -> None:
        del handle


class ProviderExecutorRegistry:
    """Execution registry separate from provider capability/route planning."""

    def __init__(self, names: Sequence[str] = ()) -> None:
        names = [str(name).strip().lower() for name in names]
        self._executors: dict[str, ProviderResourceExecutor] = {
            name: PendingProviderExecutor(name) for name in names if str(name).strip()
        }

    def register(self, name: str, executor: ProviderResourceExecutor) -> None:
        key = str(name or "").strip().lower()
        if not key or key != str(executor.name or "").strip().lower():
            raise ValueError("executor registry name mismatch")
        self._executors[key] = executor

    def unregister(self, name: str) -> None:
        """Return one provider to explicit plan-only state."""
        key = str(name or "").strip().lower()
        if key:
            self._executors[key] = PendingProviderExecutor(key)

    def get(self, name: str) -> ProviderResourceExecutor:
        key = str(name or "custom").strip().lower() or "custom"
        return self._executors.get(key, PendingProviderExecutor(key))

    def execution_state(self, name: str) -> str:
        return "enabled" if bool(self.get(name).enabled) else "plan-only"


class ResourceExecutionSession:
    """Sequential prepare + shielded LIFO cleanup for one provider request."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        executor: ProviderResourceExecutor,
        routes: Sequence[ResourceRoute],
    ) -> None:
        self.provider = str(provider or "").strip().lower()
        self.model = str(model or "")
        self.executor = executor
        self.routes = tuple(routes)
        self._handles: list[ProviderPrivateResource] = []
        self._receipts: list[ResourceExecutionReceipt] = []
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def receipts(self) -> tuple[ResourceExecutionReceipt, ...]:
        return tuple(self._receipts)

    @property
    def private_handles(self) -> tuple[ProviderPrivateResource, ...]:
        """
        Return request-private provider handles for the in-process executor only.

        These handles are intentionally never exposed by receipts, health, logs,
        persistence, or browser responses.  Provider execution consumes them only
        after :meth:`prepare` has validated the full upload/route identity.
        """
        return tuple(self._handles)

    def public_receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(row.as_public_dict() for row in self._receipts)

    def _validate_plan(
        self, uploads: Sequence[ResourceUpload]
    ) -> dict[str, ResourceUpload]:
        if not bool(getattr(self.executor, "enabled", False)):
            raise ResourceExecutionError(
                f"{self.provider} resource executor is not enabled"
            )
        by_id: dict[str, ResourceUpload] = {}
        for upload in uploads:
            rid = upload.verified.id
            if rid in by_id:
                raise ResourceExecutionError("duplicate resource upload id")
            by_id[rid] = upload
        route_ids: set[str] = set()
        for route in self.routes:
            if route.resource_id in route_ids:
                raise ResourceExecutionError("duplicate resource route id")
            route_ids.add(route.resource_id)
            if route.route == "unsupported":
                raise ResourceExecutionError("resource plan contains unsupported route")
        if route_ids != set(by_id):
            raise ResourceExecutionError("resource plan/upload ids do not match")
        return by_id

    def _register_handle(
        self,
        *,
        route: ResourceRoute,
        upload: ResourceUpload,
        handle: ProviderPrivateResource,
    ) -> None:
        """
        Validate and register one private handle before it can be used.

        Batch-capable executors may create shared provider resources (for
        example one retrieval store for many files).  The common lifecycle
        still validates every returned per-resource handle against the exact
        verified upload and route so batch preparation cannot weaken identity
        or receipt authority.
        """
        if (
            handle.resource_id != route.resource_id
            or handle.route != route.route
            or handle.provider != self.provider
            or handle.source_sha256 != upload.verified.sha256
            or int(handle.source_size) != int(upload.verified.actual_size)
        ):
            # Register first so even a malformed provider response is released
            # before the validation error escapes.
            self._handles.append(handle)
            raise ResourceExecutionError(
                "provider resource handle failed identity validation"
            )
        self._handles.append(handle)
        self._receipts.append(
            ResourceExecutionReceipt(
                resource_id=route.resource_id,
                provider=self.provider,
                model=self.model,
                route=route.route,
                source_sha256=upload.verified.sha256,
                source_size=upload.verified.actual_size,
                cleanup_required=bool(handle.cleanup_required),
                metadata={
                    "modality": route.metadata.get("modality", ""),
                    "intent": route.metadata.get("intent", ""),
                },
            )
        )

    async def prepare(
        self, uploads: Sequence[ResourceUpload]
    ) -> tuple[ResourceExecutionReceipt, ...]:
        by_id = self._validate_plan(uploads)
        ordered_uploads = tuple(by_id[route.resource_id] for route in self.routes)
        try:
            prepare_many = getattr(self.executor, "prepare_many", None)
            if callable(prepare_many):
                # A batch executor owns rollback for provider objects created
                # before this method returns.  Once returned, the common
                # session immediately registers every handle and owns cleanup.
                handles = tuple(
                    await prepare_many(
                        model=self.model,
                        routes=self.routes,
                        uploads=ordered_uploads,
                    )
                )
                if len(handles) != len(self.routes):
                    # Returned handles are the only provider cleanup authority
                    # available to the session. Register all of them so a bad
                    # batch count cannot silently leak those that are usable.
                    self._handles.extend(handles)
                    raise ResourceExecutionError(
                        "provider batch prepare returned the wrong handle count"
                    )
                for route, upload, handle in zip(self.routes, ordered_uploads, handles):
                    self._register_handle(route=route, upload=upload, handle=handle)
            else:
                for route, upload in zip(self.routes, ordered_uploads):
                    handle = await self.executor.prepare(
                        model=self.model,
                        route=route,
                        upload=upload,
                    )
                    self._register_handle(route=route, upload=upload, handle=handle)
            return self.receipts
        except BaseException:
            await self.close_shielded()
            raise

    async def _release_all(self) -> None:
        # Receipts and handles are aligned only for handles that passed identity
        # validation. A malformed last handle may therefore have no receipt.
        receipt_by_id = {row.resource_id: row for row in self._receipts}
        for handle in reversed(self._handles):
            if handle.released or not handle.cleanup_required:
                handle.released = True
                receipt = receipt_by_id.get(handle.resource_id)
                if receipt is not None:
                    receipt.state = "released"
                continue
            try:
                await self.executor.release(handle)
            except Exception:  # noqa: BLE001
                receipt = receipt_by_id.get(handle.resource_id)
                if receipt is not None:
                    receipt.state = "cleanup_failed"
                    receipt.cleanup_error = True
            else:
                handle.released = True
                receipt = receipt_by_id.get(handle.resource_id)
                if receipt is not None:
                    receipt.state = "released"

    async def close(self) -> tuple[ResourceExecutionReceipt, ...]:
        async with self._close_lock:
            if self._closed:
                return self.receipts
            await self._release_all()
            self._closed = True
            return self.receipts

    async def close_shielded(self) -> tuple[ResourceExecutionReceipt, ...]:
        """Finish cleanup even when the caller task is being cancelled."""
        task = asyncio.create_task(self.close())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Do not abandon provider cleanup. Preserve cancellation only after
            # the cleanup task has reached its terminal state.
            try:
                await task
            finally:
                raise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        await self.close_shielded()
        return False
