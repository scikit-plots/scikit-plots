"""Provider-neutral feedback service coordinator."""

from __future__ import annotations

import asyncio
import inspect
import logging
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
from ._bitbucket import submit_bitbucket_review
from ._config import (
    FeedbackServiceConfig,
    FeedbackServiceConfigError,
    StorageTarget,
    load_service_config,
)
from ._custom import load_custom_adapter
from ._git import put_git_event
from ._github import ProviderWriteError, submit_github_review
from ._gitlab import submit_gitlab_review
from ._sqlite import SQLiteFeedbackStore
from ._webhook import submit_webhook

#: Service logger. Records carry stable codes, provider/target identifiers,
#: site_id, page_id and the feedback_id event nonce; never comment text,
#: contributor credit, network addresses, headers or credentials. Structured
#: fields are attached as ``record.feedback`` (a dict) for JSON formatters.
logger = logging.getLogger("sphinx_feedback.service")


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
        # Custom adapters are imported once, at start, so a typo in a factory
        # path stops the service instead of failing the first submission.
        self._custom_adapters = {
            target.id: load_custom_adapter(target, self._credential_env)
            for target in config.targets
            if target.provider == "custom"
        }
        primary = config.primary
        logger.info(
            "page feedback service ready: mode=%s primary=%s mirrors=%d "
            "allowed_sites=%s page_authority=%s",
            config.review_mode,
            f"{primary.id}({primary.provider})" if primary else "none",
            len(config.mirrors),
            ",".join(config.allowed_site_ids) or "any",
            "on" if config.page_authority is not None else "off",
            extra={
                "feedback": {
                    "event": "service_ready",
                    "mode": config.review_mode,
                    "targets": [
                        {"id": t.id, "provider": t.provider, "role": t.role}
                        for t in config.targets
                    ],
                }
            },
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
        if target.provider == "git":
            try:
                return await asyncio.to_thread(
                    put_git_event, target, event=event, request_hash=request_hash
                )
            except OSError as exc:
                raise FeedbackServiceUnavailable(
                    "git_write_failed", "Local git feedback storage is unavailable"
                ) from exc
        if target.provider == "custom":
            return await self._write_custom(
                target, event=event, request_hash=request_hash
            )
        if target.provider in _NETWORK_SUBMITTERS:
            label, submit = _NETWORK_SUBMITTERS[target.provider]
            try:
                token, _token_name = target.resolve_token(self._credential_env)
            except FeedbackServiceConfigError as exc:
                raise FeedbackServiceUnavailable(
                    "credential_invalid",
                    f"{label} feedback credential is invalid",
                ) from exc
            if http_client is None:
                try:
                    import httpx  # ruff: ignore[import-outside-top-level]
                except (  # pragma: no cover - runtime dependency boundary
                    Exception
                ) as exc:
                    raise FeedbackServiceUnavailable(
                        "http_client_unavailable",
                        f"{label} feedback transport requires httpx",
                    ) from exc
                timeout = httpx.Timeout(
                    20.0, connect=10.0, read=20.0, write=20.0, pool=10.0
                )
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=False
                ) as client:
                    return await submit(
                        client=client,
                        target=target,
                        token=token,
                        event=event,
                        request_hash=request_hash,
                    )
            return await submit(
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

    async def _write_custom(
        self, target: StorageTarget, *, event: dict[str, Any], request_hash: str
    ) -> dict[str, Any]:
        """Call an operator adapter; never surface its exception text."""
        put = self._custom_adapters[target.id].put
        try:
            if inspect.iscoroutinefunction(put):
                return await put(event=event, request_hash=request_hash)
            return await asyncio.to_thread(put, event=event, request_hash=request_hash)
        except (FeedbackConflictError, ProviderWriteError, FeedbackServiceUnavailable):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # ruff: ignore[blind-except]
            raise FeedbackServiceUnavailable(
                "custom_provider_failed", "Custom feedback storage failed"
            ) from exc

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
        """
        Validate, store and acknowledge one feedback request, logging the outcome.

        Parameters
        ----------
        payload : dict
            A decoded ``page.feedback-request.v1`` object.
        http_client : httpx.AsyncClient, optional
            Shared transport for network providers.

        Returns
        -------
        dict
            A ``page.feedback-receipt.v1`` receipt.

        Raises
        ------
        FeedbackValidationError
            The request is malformed or not authorized (logged at WARNING).
        FeedbackConflictError
            The feedback_id is stored with different content (WARNING).
        FeedbackServiceUnavailable
            Storage failed; the stable ``code`` is logged at ERROR, the chained
            provider exception only at DEBUG.
        """
        fields = _request_fields(payload)
        try:
            receipt = await self._submit(payload, http_client=http_client)
        except FeedbackValidationError as exc:
            logger.warning(
                "feedback rejected: code=%s site=%s page=%s",
                getattr(exc, "code", "invalid"),
                fields["site_id"],
                fields["page_id"],
                extra={
                    "feedback": {
                        "event": "rejected",
                        **fields,
                        "code": getattr(exc, "code", "invalid"),
                    }
                },
            )
            raise
        except FeedbackConflictError:
            logger.warning(
                "feedback conflict: feedback_id=%s site=%s page=%s",
                fields["feedback_id"],
                fields["site_id"],
                fields["page_id"],
                extra={"feedback": {"event": "conflict", **fields}},
            )
            raise
        except FeedbackServiceUnavailable as exc:
            logger.error(
                "feedback storage failed: code=%s feedback_id=%s site=%s page=%s",
                exc.code,
                fields["feedback_id"],
                fields["site_id"],
                fields["page_id"],
                extra={
                    "feedback": {"event": "unavailable", **fields, "code": exc.code}
                },
            )
            logger.debug("feedback storage failure detail", exc_info=exc)
            raise
        degraded = sorted(k for k, v in receipt["mirrors"].items() if v != "ok")
        logger.info(
            "feedback stored: status=%s provider=%s feedback_id=%s site=%s page=%s%s",
            receipt["status"],
            receipt["provider"],
            receipt["feedback_id"],
            fields["site_id"],
            fields["page_id"],
            f" degraded_mirrors={','.join(degraded)}" if degraded else "",
            extra={
                "feedback": {
                    "event": "stored",
                    **fields,
                    "status": receipt["status"],
                    "provider": receipt["provider"],
                    "degraded_mirrors": degraded,
                }
            },
        )
        return receipt

    async def _submit(
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


def _request_fields(payload: Any) -> dict[str, str]:
    """Return the loggable identifiers of a request, bounded and printable."""

    def clean(key: str) -> str:
        value = payload.get(key, "") if isinstance(payload, dict) else ""
        text = value if isinstance(value, str) else ""
        return "".join(ch for ch in text[:200] if ch.isprintable()) or "-"

    return {key: clean(key) for key in ("site_id", "page_id", "feedback_id")}


async def _submit_webhook(*, client, target, token, event, request_hash):
    """Adapt :func:`submit_webhook` to the shared ``token`` keyword."""
    return await submit_webhook(
        client=client,
        target=target,
        secret=token,
        event=event,
        request_hash=request_hash,
    )


#: Network providers: receipt label and submit coroutine (``token`` keyword).
_NETWORK_SUBMITTERS = {
    "github": ("GitHub", submit_github_review),
    "gitlab": ("GitLab", submit_gitlab_review),
    "bitbucket": ("Bitbucket", submit_bitbucket_review),
    "webhook": ("Webhook", _submit_webhook),
}


__all__ = [
    "FeedbackConflictError",
    "FeedbackServiceConfigError",
    "FeedbackServiceUnavailable",
    "PageFeedbackService",
]
