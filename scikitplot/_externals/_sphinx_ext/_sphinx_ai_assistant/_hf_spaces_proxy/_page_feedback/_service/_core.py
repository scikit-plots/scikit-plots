"""Provider-neutral feedback service coordinator."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
    build_feedback_event,
    feedback_request_hash,
    parse_feedback_request,
)
from ._config import (
    FeedbackServiceConfig,
    FeedbackServiceConfigError,
    StorageTarget,
    load_service_config,
)
from ._github import ProviderWriteError, submit_github_review
from ._sqlite import SQLiteFeedbackStore


class FeedbackServiceUnavailable(  # ruff: ignore[error-suffix-on-exception-name]
    RuntimeError,
):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PageFeedbackService:
    """Accept one minimal feedback event and coordinate primary/mirror writes."""

    def __init__(
        self,
        config: FeedbackServiceConfig,
        *,
        credential_env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._credential_env = os.environ if credential_env is None else credential_env
        self._page_authority = (
            None
            if config.page_authority is None
            else {
                (site_id, page_id): revision
                for site_id, page_id, revision in config.page_authority
            }
        )

    @classmethod
    def from_env(cls, env=None) -> PageFeedbackService:
        source = os.environ if env is None else env
        return cls(load_service_config(source), credential_env=source)

    async def _write_target(
        self,
        target: StorageTarget,
        *,
        request: dict[str, Any],
        event: dict[str, Any],
        request_hash: str,
        http_client=None,
    ) -> dict[str, Any]:
        if target.provider == "sqlite":
            store = SQLiteFeedbackStore(Path(target.database))
            try:
                return await asyncio.to_thread(
                    store.put,
                    feedback_id=event["feedback"]["id"],
                    request_hash=request_hash,
                    event=event,
                )
            except FeedbackConflictError:
                raise
            except (OSError, sqlite3.Error) as exc:
                raise FeedbackServiceUnavailable(
                    "sqlite_write_failed", "SQLite feedback storage is unavailable"
                ) from exc
        if target.provider == "github":
            try:
                token, _token_name = target.resolve_token(self._credential_env)
            except FeedbackServiceConfigError as exc:
                raise FeedbackServiceUnavailable(
                    "credential_invalid",
                    "GitHub feedback credential is invalid",
                ) from exc
            if http_client is None:
                try:
                    import httpx  # ruff: ignore[import-outside-top-level]
                except (  # pragma: no cover - runtime dependency boundary
                    Exception
                ) as exc:
                    raise FeedbackServiceUnavailable(
                        "http_client_unavailable",
                        "GitHub feedback transport requires httpx",
                    ) from exc
                timeout = httpx.Timeout(
                    20.0, connect=10.0, read=20.0, write=20.0, pool=10.0
                )
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=False
                ) as client:
                    return await submit_github_review(
                        client=client,
                        target=target,
                        token=token,
                        event=event,
                        request_hash=request_hash,
                    )
            return await submit_github_review(
                client=http_client,
                target=target,
                token=token,
                event=event,
                request_hash=request_hash,
            )
        raise FeedbackServiceUnavailable(
            "provider_not_implemented",
            f"Feedback provider {target.provider!r} is modeled but not implemented in this release",
        )

    def validate_request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize one request and enforce server-side submission authority."""
        request = parse_feedback_request(payload)
        if (
            self.config.allowed_site_ids
            and request["site_id"] not in self.config.allowed_site_ids
        ):
            raise FeedbackValidationError(
                "site_id is not authorized by this feedback service",
                code="site_not_allowed",
            )
        if self._page_authority is not None:
            authority_key = (request["site_id"], request["page_id"])
            if authority_key not in self._page_authority:
                raise FeedbackValidationError(
                    "page_id is not authorized by this feedback service",
                    code="page_not_allowed",
                )
            expected_revision = self._page_authority[authority_key]
            if (
                expected_revision
                and request.get("page_revision", "") != expected_revision
            ):
                raise FeedbackValidationError(
                    "page_revision does not match the server-side page authority",
                    code="page_revision_mismatch",
                )
        return request

    @staticmethod
    def _valid_target_receipt(
        target: StorageTarget,
        receipt: Any,
        *,
        feedback_id: str,
        request_hash: str,
    ) -> bool:
        return (
            isinstance(receipt, dict)
            and receipt.get("status") in {"accepted", "replay"}
            and receipt.get("provider") == target.provider
            and receipt.get("feedback_id") == feedback_id
            and receipt.get("request_hash") == request_hash
        )

    async def submit(
        self,
        payload: dict[str, Any],
        *,
        http_client=None,
    ) -> dict[str, Any]:
        if self.config.review_mode == "disabled":
            raise FeedbackServiceUnavailable(
                "feedback_disabled",
                "Generic page feedback is disabled on this service",
            )
        request = self.validate_request(payload)
        request_hash = feedback_request_hash(request)
        event = build_feedback_event(request)
        primary = self.config.primary
        if primary is None:
            raise FeedbackServiceUnavailable(
                "primary_missing",
                "No primary feedback storage target is configured",
            )
        try:
            primary_receipt = await self._write_target(
                primary,
                request=request,
                event=event,
                request_hash=request_hash,
                http_client=http_client,
            )
        except ProviderWriteError as exc:
            raise FeedbackServiceUnavailable(exc.code, str(exc)) from exc

        if not self._valid_target_receipt(
            primary,
            primary_receipt,
            feedback_id=event["feedback"]["id"],
            request_hash=request_hash,
        ):
            raise FeedbackServiceUnavailable(
                "provider_receipt_invalid",
                "Primary feedback provider returned an invalid receipt",
            )
        primary_status = primary_receipt["status"]

        mirror_status: dict[str, str] = {}

        async def write_mirror(mirror: StorageTarget) -> tuple[str, str]:
            try:
                mirror_receipt = await self._write_target(
                    mirror,
                    request=request,
                    event=event,
                    request_hash=request_hash,
                    http_client=http_client,
                )
                if not self._valid_target_receipt(
                    mirror,
                    mirror_receipt,
                    feedback_id=event["feedback"]["id"],
                    request_hash=request_hash,
                ):
                    return mirror.id, "degraded"
                return mirror.id, "ok"
            except asyncio.CancelledError:
                raise
            except (
                FeedbackConflictError,
                FeedbackServiceUnavailable,
                ProviderWriteError,
                OSError,
            ):
                # Mirrors are durability only. Primary authority already accepted.
                return mirror.id, "degraded"

        tasks = {
            asyncio.create_task(write_mirror(mirror)): mirror.id
            for mirror in self.config.mirrors
        }
        if tasks:
            pending: set[asyncio.Task] = set()
            try:
                done, pending = await asyncio.wait(
                    tasks, timeout=self.config.mirror_timeout_seconds
                )
                for task in done:
                    mirror_id = tasks[task]
                    try:
                        resolved_id, status = task.result()
                    except asyncio.CancelledError:
                        # A child mirror can be cancelled independently by its
                        # transport/runtime. Primary authority has already
                        # succeeded, so treat that as durability degradation;
                        # parent cancellation is handled by the outer finally.
                        resolved_id, status = mirror_id, "degraded"
                    except Exception:  # ruff: ignore[blind-except]
                        resolved_id, status = mirror_id, "degraded"
                    mirror_status[resolved_id] = status
                for task in pending:
                    mirror_status[tasks[task]] = "degraded"
            finally:
                # If the request task itself is cancelled after primary authority
                # has committed, never leave mirror writes detached from their
                # parent lifecycle.  Cancelling and gathering unfinished children
                # makes shutdown/disconnect behavior deterministic; a later retry
                # can safely resume every provider with the same event nonce.
                unfinished = [task for task in tasks if not task.done()]
                for task in unfinished:
                    task.cancel()
                if unfinished:
                    await asyncio.gather(*unfinished, return_exceptions=True)
        return {
            "ok": True,
            "contract": "page.feedback-receipt.v1",
            "feedback_id": event["feedback"]["id"],
            "status": primary_status,
            "provider": primary.provider,
            "request_hash": request_hash,
            "review_url": (
                primary_receipt.get("review_url", "") if primary.expose_links else ""
            ),
            "mirrors": {
                mirror_id: mirror_status[mirror_id]
                for mirror_id in sorted(mirror_status)
            },
        }


__all__ = [
    "FeedbackConflictError",
    "FeedbackServiceConfigError",
    "FeedbackServiceUnavailable",
    "PageFeedbackService",
]
