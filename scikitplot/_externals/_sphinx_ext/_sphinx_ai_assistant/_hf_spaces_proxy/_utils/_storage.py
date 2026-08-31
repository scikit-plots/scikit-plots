# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Provider-neutral record storage for the sphinx AI assistant proxy.

The browser never receives storage credentials.  A canonical UTF-8 payload is
written to one primary repository and, optionally, mirrored to additional
repositories.  Existing TRAINING_DATASET_REPO/HF_* deployments are synthesized
as a single Hugging Face primary target by :func:`load_storage_targets`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import quote, urlencode, urlsplit

import httpx

Provider = Literal["huggingface", "github", "gitlab", "bitbucket"]
Role = Literal["primary", "mirror"]

_MAX_TARGETS = 8
_MAX_REPO = 240
_MAX_BRANCH = 120
_MAX_PATH = 240
_TOKEN_ENV_RE = re.compile(r"^AI_RECORD_STORAGE_TOKEN_[A-Z0-9_]{1,80}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_SEG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOWED_PROVIDERS = {"huggingface", "github", "gitlab", "bitbucket"}
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_CONTROL_RESPONSE_DEFAULT = 4 * 1024 * 1024
_CONTROL_RESPONSE_HARD_MAX = 16 * 1024 * 1024
_HF_FACTORY_LOCK = threading.RLock()


class _ProviderResponseTooLarge(  # ruff: ignore[error-suffix-on-exception-name]
    RuntimeError
):
    pass


def _control_response_limit() -> int:
    raw = os.environ.get("AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES", "").strip()
    try:
        value = int(raw) if raw else _CONTROL_RESPONSE_DEFAULT
    except ValueError:
        value = _CONTROL_RESPONSE_DEFAULT
    return max(1024, min(_CONTROL_RESPONSE_HARD_MAX, value))


def _safe_gitlab_api_base(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return "https://gitlab.com/api/v4"
    if any(
        ord(ch) < 0x20  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 0x7F  # ruff: ignore[magic-value-comparison]
        for ch in raw
    ):
        raise StorageConfigError("TARGET_API_BASE")
    try:
        parsed = urlsplit(raw)
    except Exception as exc:
        raise StorageConfigError("TARGET_API_BASE") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise StorageConfigError("TARGET_API_BASE")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise StorageConfigError("TARGET_API_BASE")
    if ".." in [seg for seg in parsed.path.split("/") if seg]:
        raise StorageConfigError("TARGET_API_BASE")
    if "\\" in parsed.path:
        raise StorageConfigError("TARGET_API_BASE")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise StorageConfigError("TARGET_API_BASE") from exc
    return raw


class _BoundedSyncStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._seen = 0

    def __iter__(self):
        for chunk in self._stream:
            self._seen += len(chunk)
            if self._seen > self._limit:
                raise _ProviderResponseTooLarge(
                    "provider control response exceeds configured limit"
                )
            yield chunk

    def close(self) -> None:
        self._stream.close()


class _BoundedTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport, limit: int) -> None:
        self._transport = transport
        self._limit = limit

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._transport.handle_request(request)
        length = response.headers.get("content-length")
        if length and length.isdigit() and int(length) > self._limit:
            response.close()
            raise _ProviderResponseTooLarge(
                "provider control response exceeds configured limit"
            )
        response.stream = _BoundedSyncStream(response.stream, self._limit)
        return response

    def close(self) -> None:
        self._transport.close()


def _with_bounded_hf_client(call):
    """
    Run one huggingface_hub control-plane call with bounded HTTP bodies.

    Minimal test doubles and old SDK shims may not expose the factory module;
    production is pinned to a version that does. In that compatibility-only
    case there is no underlying SDK HTTP client to wrap, so execute directly.
    """
    try:
        from huggingface_hub.utils import _http as hf_http  # noqa: PLC0415
    except (ImportError, ModuleNotFoundError):
        return call()

    with _HF_FACTORY_LOCK:
        previous = hf_http._GLOBAL_CLIENT_FACTORY

        def factory():
            client = previous()
            transport = getattr(client, "_transport", None)
            if transport is None:
                client.close()
                raise RuntimeError("HF_CLIENT_TRANSPORT")
            client._transport = _BoundedTransport(transport, _control_response_limit())
            return client

        hf_http.set_client_factory(factory)
        try:
            return call()
        finally:
            hf_http.set_client_factory(previous)


class StorageConfigError(ValueError):
    """Raised for invalid server-side storage target configuration."""


class StorageWriteError(RuntimeError):
    """Raised when a storage target cannot persist a record."""

    def __init__(self, code: str, *, transient: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.transient = transient


@dataclass(slots=True)
class StorageTarget:
    id: str
    label: str
    provider: Provider
    role: Role
    repo: str
    branch: str = "main"
    feedback_path: str = "feedback"
    contributions_path: str = "contributions"
    token_env: str = ""
    token_type: str = "unknown"  # ruff: ignore[hardcoded-password-string]
    expose_links: bool = True
    api_base: str = ""

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "").strip() if self.token_env else ""

    def folder_for(self, kind: str) -> str:
        return self.feedback_path if kind == "feedback" else self.contributions_path


@dataclass(slots=True)
class TargetRuntimeState:
    status: str = "configured"
    write_capability: str = "unknown"
    failures: int = 0
    open_until: float = 0.0
    last_error_code: str = ""
    last_success_ms: int | None = None
    last_failure_ms: int | None = None
    pending_retries: int = 0


@dataclass(slots=True)
class StorageReceipt:
    accepted: bool
    record_id: str
    primary: str | None
    mirrors: dict[str, str] = field(default_factory=dict)
    # Exact logical record path per configured target.  Paths are control-plane
    # metadata used for best-effort current-view removal after participant
    # withdrawal.  Removing these files creates another repository commit and
    # therefore does NOT imply physical erasure from Git/provider history.
    paths: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ReviewReceipt:
    """Provider-neutral code-review receipt for one quarantined contribution."""

    provider: Provider
    target_id: str
    repo: str
    base_branch: str
    review_branch: str
    review_key: str
    review_id: str
    review_url: str
    status: str
    record_id: str
    path: str

    def storage_metadata(self) -> dict[str, Any]:
        return {
            "recordId": self.record_id,
            "primary": self.review_url or None,
            "mirrors": {},
            "paths": {self.target_id: self.path},
            "review": {
                "provider": self.provider,
                "targetId": self.target_id,
                "repo": self.repo,
                "baseBranch": self.base_branch,
                "reviewBranch": self.review_branch,
                "reviewKey": self.review_key,
                "reviewId": self.review_id,
                "reviewUrl": self.review_url,
                "status": self.status,
            },
        }


def review_key_for(receipt_id: str) -> str:
    """Return a non-identifying deterministic key safe for refs and titles."""
    raw = str(receipt_id or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def review_branch_for(receipt_id: str) -> str:
    return f"ai-contrib-{review_key_for(receipt_id)}"


def review_title_for(receipt_id: str) -> str:
    return f"Dataset contribution {review_key_for(receipt_id)}"


def _safe_id(value: Any, fallback: str) -> str:
    s = str(value or "").strip().lower()
    if _ID_RE.fullmatch(s):
        return s
    if fallback and _ID_RE.fullmatch(fallback):
        return fallback
    raise StorageConfigError("TARGET_ID")


def _safe_repo(value: Any, provider: str = "") -> str:
    s = str(value or "").strip().strip("/")
    if not s or len(s) > _MAX_REPO:
        raise StorageConfigError("TARGET_REPO")
    parts = s.split("/")
    # GitLab project paths may contain nested groups. The other supported
    # providers use an owner/workspace + repository pair.
    if provider == "gitlab":  # ruff: ignore[if-else-block-instead-of-if-exp]
        valid_count = 2 <= len(parts) <= 8  # ruff: ignore[magic-value-comparison]
    else:
        valid_count = len(parts) == 2  # ruff: ignore[magic-value-comparison]
    if not valid_count or any(not p or not _SEG_RE.fullmatch(p) for p in parts):
        raise StorageConfigError("TARGET_REPO")
    return s


def _safe_branch(value: Any) -> str:
    s = str(value or "main").strip()
    if not s or len(s) > _MAX_BRANCH or any(c in s for c in "\\\r\n\x00"):
        raise StorageConfigError("TARGET_BRANCH")
    if s.startswith(("-", "/")) or ".." in s or s.endswith("/"):
        raise StorageConfigError("TARGET_BRANCH")
    return s


def _safe_folder(value: Any, default: str) -> str:
    s = str(value if value is not None else default).strip().strip("/")
    if not s or len(s) > _MAX_PATH or "\\" in s or "\x00" in s:
        raise StorageConfigError("TARGET_PATH")
    parts = s.split("/")
    if len(parts) > 12 or any(  # ruff: ignore[magic-value-comparison]
        p in {"", ".", ".."} or not _SEG_RE.fullmatch(p) for p in parts
    ):
        raise StorageConfigError("TARGET_PATH")
    return s


def _safe_token_env(value: Any) -> str:
    s = str(value or "").strip()
    if not s or not _TOKEN_ENV_RE.fullmatch(s):
        raise StorageConfigError("TARGET_TOKEN_ENV")
    return s


def _normalize_token_type(value: Any) -> str:
    s = str(value or "unknown").strip().lower().replace("_", "-")
    if s in {"finegrained", "fine-grained"}:
        return "fine-grained"
    return s if s in {"read", "write", "unknown"} else "unknown"


def _parse_target(raw: dict[str, Any], index: int) -> StorageTarget:
    provider = str(raw.get("provider") or "").strip().lower()
    if provider not in _ALLOWED_PROVIDERS:
        raise StorageConfigError("TARGET_PROVIDER")
    role = (
        str(raw.get("role") or ("primary" if index == 0 else "mirror")).strip().lower()
    )
    if role not in {"primary", "mirror"}:
        raise StorageConfigError("TARGET_ROLE")
    fallback_id = f"{provider}-{index + 1}"
    target_id = _safe_id(raw.get("id"), fallback_id)
    label = str(raw.get("label") or target_id).strip()[:96] or target_id
    paths = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    token_env = _safe_token_env(raw.get("token_env"))
    token_type = _normalize_token_type(
        raw.get("token_type") or os.environ.get(token_env + "_TYPE")
    )
    raw_api_base = raw.get("api_base")
    if provider == "gitlab":
        api_base = _safe_gitlab_api_base(raw_api_base)
    else:
        if str(raw_api_base or "").strip():
            raise StorageConfigError("TARGET_API_BASE_UNSUPPORTED")
        api_base = ""
    return StorageTarget(
        id=target_id,
        label=label,
        provider=provider,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        repo=_safe_repo(raw.get("repo"), provider),
        branch=_safe_branch(raw.get("branch")),
        feedback_path=_safe_folder(paths.get("feedback"), "feedback"),
        contributions_path=_safe_folder(paths.get("contributions"), "contributions"),
        token_env=token_env,
        token_type=token_type,
        expose_links=raw.get("expose_links", True) is not False,
        api_base=api_base,
    )


def load_storage_targets(
    raw_json: str,
    *,
    legacy_repo: str = "",
    legacy_token: str = "",
    legacy_token_type: str = "unknown",  # ruff: ignore[hardcoded-password-default]
) -> list[StorageTarget]:
    """Parse configured targets or synthesize the legacy HF target."""
    raw_json = (raw_json or "").strip()
    targets: list[StorageTarget] = []
    if raw_json:
        try:
            data = json.loads(raw_json)
        except Exception as exc:  # noqa: BLE001
            raise StorageConfigError("TARGETS_JSON") from exc
        if not isinstance(data, list) or not 1 <= len(data) <= _MAX_TARGETS:
            raise StorageConfigError("TARGETS_COUNT")
        seen: set[str] = set()
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise StorageConfigError("TARGET_OBJECT")
            target = _parse_target(item, i)
            if target.id in seen:
                raise StorageConfigError("TARGET_DUPLICATE")
            seen.add(target.id)
            targets.append(target)
        primaries = [t for t in targets if t.role == "primary"]
        if len(primaries) != 1:
            raise StorageConfigError("TARGET_PRIMARY_COUNT")
        return targets

    # 100% backwards-compatible legacy synthesis.  We cannot reference the
    # actual token value from a synthetic env name, so mirror it into a private
    # process env slot used only by this module.
    if legacy_repo:
        env_name = "AI_RECORD_STORAGE_TOKEN_LEGACY_HF"
        if legacy_token:
            os.environ[env_name] = legacy_token
        return [
            StorageTarget(
                id="hf-primary",
                label="Hugging Face Dataset",
                provider="huggingface",
                role="primary",
                repo=_safe_repo(legacy_repo, "huggingface"),
                token_env=env_name,
                token_type=_normalize_token_type(legacy_token_type),
            )
        ]
    return []


def _repo_parts(repo: str) -> tuple[str, str]:
    return tuple(repo.split("/", 1))  # type: ignore[return-value]


def public_links(target: StorageTarget) -> dict[str, str]:
    """Return public browser links without exposing credentials."""
    if not target.expose_links:
        return {}
    owner, repo = _repo_parts(target.repo)
    b = quote(target.branch, safe="")
    fp = quote(target.feedback_path, safe="/")
    cp = quote(target.contributions_path, safe="/")
    if target.provider == "huggingface":
        root = f"https://huggingface.co/datasets/{quote(owner)}/{quote(repo)}"
        return {
            "root": root,
            "feedback": f"{root}/tree/{b}/{fp}",
            "contributions": f"{root}/tree/{b}/{cp}",
        }
    if target.provider == "github":
        root = f"https://github.com/{quote(owner)}/{quote(repo)}"
        return {
            "root": root,
            "feedback": f"{root}/tree/{b}/{fp}",
            "contributions": f"{root}/tree/{b}/{cp}",
        }
    if target.provider == "gitlab":
        # Public links default to gitlab.com even when a custom API base is
        # configured; self-managed instances can supply `public_base` in a
        # future schema version without exposing credentials.
        root = f"https://gitlab.com/{quote(owner)}/{quote(repo)}"
        return {
            "root": root,
            "feedback": f"{root}/-/tree/{b}/{fp}",
            "contributions": f"{root}/-/tree/{b}/{cp}",
        }
    root = f"https://bitbucket.org/{quote(owner)}/{quote(repo)}"
    return {
        "root": root,
        "feedback": f"{root}/src/{b}/{fp}",
        "contributions": f"{root}/src/{b}/{cp}",
    }


def canonical_record_path(
    target: StorageTarget, kind: str, record_id: str, now: float | None = None
) -> str:
    dt = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    folder = target.folder_for(kind)
    prefix = "fb" if kind == "feedback" else "ct"
    return f"{folder}/{dt:%Y/%m/%d}/{prefix}_{record_id}.jsonl"


def record_id_for(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:24]


class StorageCoordinator:
    """Persist canonical records to a primary target and optional mirrors."""

    def __init__(
        self, targets: list[StorageTarget], client: httpx.AsyncClient | None = None
    ) -> None:
        self.targets = targets
        self.client = client
        self._locks = {t.id: asyncio.Lock() for t in targets}
        self._state = {t.id: TargetRuntimeState() for t in targets}
        self._max_attempts = 2
        self._circuit_seconds = 60.0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # A withdrawal must not allow an already-scheduled degraded-mirror retry
        # to resurrect the original eligible file after current-view removal.
        self._suppressed_retry_record_ids: set[str] = set()

    @property
    def primary(self) -> StorageTarget | None:
        return next((t for t in self.targets if t.role == "primary"), None)

    def primary_ready(self) -> bool:
        target = self.primary
        if target is None or not target.token:
            return False
        state = self._state[target.id]
        return state.write_capability not in {
            "missing-token",
            "denied",
            "denied-read-token",
        }

    def set_client(self, client: httpx.AsyncClient | None) -> None:
        self.client = client

    async def initialize(self) -> None:
        """Best-effort capability checks.  Never raises at application startup."""
        for target in self.targets:
            state = self._state[target.id]
            if target.provider == "huggingface":
                state.write_capability = await self._hf_write_capability(target)
            else:
                state.write_capability = (
                    "configured" if target.token else "missing-token"
                )
            if state.write_capability in {
                "missing-token",
                "denied",
                "denied-read-token",
            }:
                state.status = "degraded"

    async def _hf_write_capability(  # ruff: ignore[too-many-return-statements]
        self,
        target: StorageTarget,
    ) -> str:
        token = target.token
        if not token:
            return "missing-token"
        token_type = _normalize_token_type(target.token_type)
        if token_type == "read":  # ruff: ignore[hardcoded-password-string]
            return "denied-read-token"
        # Modern huggingface_hub can verify repo-specific write access without a
        # mutation.  Keep compatibility with older pinned versions by feature-
        # detecting the `write` parameter.
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415

            api = HfApi(token=token)
            auth_check = getattr(api, "auth_check", None)
            if auth_check and "write" in inspect.signature(auth_check).parameters:
                await asyncio.to_thread(
                    _with_bounded_hf_client,
                    lambda: auth_check(
                        repo_id=target.repo,
                        repo_type="dataset",
                        write=True,
                    ),
                )
                return "verified"
        except Exception as exc:  # ruff: ignore[blind-except]
            # Permission denial is materially different from a network/version
            # uncertainty.  Modern huggingface_hub exceptions normally carry an
            # HTTP response; classify only the status code and keep every response
            # body / exception message private.  A 401/403 means the token cannot
            # write this repo and should be blocked before any mutation attempt.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                return "denied"
            # Older huggingface_hub releases do not support auth_check(write=True),
            # and transient network failures are also possible.  Preserve
            # compatibility by allowing the first real commit to prove capability.
            return "unverified"
        if token_type == "write":  # ruff: ignore[hardcoded-password-string]
            return "broad-write"
        if token_type == "fine-grained":  # ruff: ignore[hardcoded-password-string]
            return "unverified"
        return "legacy-unverified"

    def manifest(self) -> dict[str, Any]:
        targets = []
        for t in self.targets:
            st = self._state[t.id]
            targets.append(
                {
                    "id": t.id,
                    "label": t.label,
                    "provider": t.provider,
                    "role": t.role,
                    "repo": t.repo if t.expose_links else None,
                    "branch": t.branch,
                    "paths": {
                        "feedback": t.feedback_path,
                        "contributions": t.contributions_path,
                    },
                    "capabilities": {
                        "feedback": True,
                        "contributions": True,
                        "native_review": t.role == "primary",
                        "write": (
                            bool(t.token)
                            and st.write_capability
                            not in {"missing-token", "denied", "denied-read-token"}
                        ),
                    },
                    "token": {
                        "type": (
                            _normalize_token_type(t.token_type)
                            if t.provider == "huggingface"
                            else "server-managed"
                        ),
                        "write_capability": st.write_capability,
                    },
                    "status": (
                        "circuit-open" if st.open_until > time.time() else st.status
                    ),
                    "last_success_ms": st.last_success_ms,
                    "last_failure_ms": st.last_failure_ms,
                    "pending_retries": st.pending_retries,
                    "links": public_links(t),
                }
            )
        return {
            "schema_version": 1,
            "policy": "primary_then_mirrors",
            "targets": targets,
        }

    async def write(
        self,
        *,
        kind: str,
        content: bytes,
        commit_message: str,
        path_timestamp: float | None = None,
    ) -> StorageReceipt:
        primary = self.primary
        if primary is None:
            raise StorageWriteError("NO_PRIMARY")
        rid = record_id_for(content)
        # Freeze one logical path timestamp for every target.  Retries must write
        # the same path rather than drifting across a UTC day boundary, because
        # the receipt lifecycle later uses these paths for best-effort current-
        # view removal.
        # Callers with a durable lifecycle can supply a receipt-stable timestamp
        # so a crash/restart replay targets the exact same logical provider path
        # instead of creating a second dated file.
        path_now = float(path_timestamp) if path_timestamp is not None else time.time()
        paths = {
            t.id: canonical_record_path(t, kind, rid, now=path_now)
            for t in self.targets
        }
        await self._write_target(
            primary, kind, rid, content, commit_message, path=paths[primary.id]
        )
        mirrors: dict[str, str] = {}
        mirror_targets = [t for t in self.targets if t.role == "mirror"]
        if mirror_targets:
            results = await asyncio.gather(
                *(
                    self._write_target(
                        t, kind, rid, content, commit_message, path=paths[t.id]
                    )
                    for t in mirror_targets
                ),
                return_exceptions=True,
            )
            for target, result in zip(mirror_targets, results, strict=True):
                if not isinstance(result, Exception):
                    mirrors[target.id] = "ok"
                else:
                    mirrors[target.id] = "degraded"
                    self._schedule_mirror_retry(
                        target, kind, rid, content, commit_message, paths[target.id]
                    )
        return StorageReceipt(True, rid, primary.id, mirrors, paths)

    def _schedule_mirror_retry(  # ruff: ignore[too-many-positional-arguments]
        self,
        target: StorageTarget,
        kind: str,
        rid: str,
        content: bytes,
        message: str,
        path: str,
    ) -> None:
        state = self._state[target.id]
        state.pending_retries += 1

        async def _runner() -> None:
            try:
                for delay in (2.0, 10.0, 30.0):
                    await asyncio.sleep(delay)
                    if rid in self._suppressed_retry_record_ids:
                        return
                    # A circuit opened by prior failures is allowed to cool down
                    # before the next scheduled retry rather than busy-looping.
                    if self._state[target.id].open_until > time.time():
                        continue
                    try:
                        await self._write_target(
                            target, kind, rid, content, message, path=path
                        )
                        return
                    except StorageWriteError:
                        continue
            finally:
                state.pending_retries = max(0, state.pending_retries - 1)

        task = asyncio.create_task(_runner())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def close(self) -> None:
        """Cancel pending in-memory mirror retries during graceful shutdown."""
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._suppressed_retry_record_ids.clear()

    async def _write_target(
        self,
        target: StorageTarget,
        kind: str,
        rid: str,
        content: bytes,
        message: str,
        *,
        path: str | None = None,
    ) -> str:
        state = self._state[target.id]
        now = time.time()
        if state.open_until > now:
            raise StorageWriteError("CIRCUIT_OPEN", transient=True)
        if not target.token:
            self._mark_failure(target, "MISSING_TOKEN")
            raise StorageWriteError("MISSING_TOKEN")
        if (
            target.provider == "huggingface"
            and _normalize_token_type(target.token_type) == "read"
        ):
            self._mark_failure(target, "READ_TOKEN")
            raise StorageWriteError("READ_TOKEN")

        async with self._locks[target.id]:
            # Re-check withdrawal suppression *after* taking the target lock. A
            # degraded-mirror retry may have passed the scheduler's earlier check
            # and then waited behind current-view deletion; without this second
            # check it could resurrect the withdrawn eligible file after DELETE.
            if kind == "contributions" and rid in self._suppressed_retry_record_ids:
                raise StorageWriteError("WITHDRAWN")
            last: StorageWriteError | None = None
            for attempt in range(self._max_attempts):
                try:
                    logical_path = path or canonical_record_path(target, kind, rid)
                    await self._dispatch(target, logical_path, content, message)
                    self._mark_success(target)
                    return logical_path
                except StorageWriteError as exc:
                    last = exc
                    if not exc.transient or attempt + 1 >= self._max_attempts:
                        break
                    await asyncio.sleep(0.25 * (2**attempt))
            self._mark_failure(target, last.code if last else "WRITE_FAILED")
            raise last or StorageWriteError("WRITE_FAILED")

    def _mark_success(self, target: StorageTarget) -> None:
        st = self._state[target.id]
        st.status = "healthy"
        st.failures = 0
        st.open_until = 0.0
        st.last_error_code = ""
        st.last_success_ms = int(time.time() * 1000)
        if target.provider == "huggingface" and st.write_capability in {
            "unverified",
            "legacy-unverified",
            "broad-write",
        }:
            st.write_capability = "verified"

    def _mark_failure(self, target: StorageTarget, code: str) -> None:
        st = self._state[target.id]
        st.failures += 1
        st.status = "degraded"
        st.last_error_code = code
        st.last_failure_ms = int(time.time() * 1000)
        if st.failures >= 3:  # ruff: ignore[magic-value-comparison]
            st.open_until = time.time() + self._circuit_seconds

    async def open_contribution_review(
        self,
        *,
        receipt_id: str,
        content: bytes,
        commit_message: str,
        path_timestamp: float | None = None,
    ) -> ReviewReceipt:
        """
        Create or recover a native provider review for one contribution.

        The contribution is written to its final canonical path on an isolated
        review ref.  The configured canonical branch remains the only
        training-eligible authority.  The operation is idempotent by a
        receipt-derived, non-identifying review branch/title.
        """
        target = self.primary
        if target is None:
            raise StorageWriteError("NO_PRIMARY_TARGET")
        if not target.token:
            raise StorageWriteError("PRIMARY_TOKEN_MISSING")
        rid = record_id_for(content)
        path = canonical_record_path(target, "contributions", rid, path_timestamp)
        key = review_key_for(receipt_id)
        branch = review_branch_for(receipt_id)
        title = review_title_for(receipt_id)
        async with self._locks[target.id]:
            return await self._open_review_target(
                target,
                branch=branch,
                key=key,
                title=title,
                path=path,
                record_id=rid,
                content=content,
                message=commit_message,
            )

    async def get_contribution_review(self, receipt_id: str) -> ReviewReceipt | None:
        """Return the provider review state without exposing provider bodies."""
        target = self.primary
        if target is None or not target.token:
            return None
        key = review_key_for(receipt_id)
        branch = review_branch_for(receipt_id)
        title = review_title_for(receipt_id)
        async with self._locks[target.id]:
            return await self._discover_review_target(
                target, branch=branch, key=key, title=title
            )

    async def close_contribution_review(self, receipt_id: str) -> str:
        """Close/reject an open provider review and remove its temporary branch."""
        target = self.primary
        if target is None or not target.token:
            return "not-configured"
        key = review_key_for(receipt_id)
        branch = review_branch_for(receipt_id)
        title = review_title_for(receipt_id)
        async with self._locks[target.id]:
            review = await self._discover_review_target(
                target, branch=branch, key=key, title=title
            )
            if review is None:
                return "already-absent"
            if review.status == "merged":
                return "already-merged"
            if review.status not in {"closed", "rejected"}:
                await self._close_review_target(target, review)
            await self._delete_review_branch(target, branch)
            return "closed"

    async def merge_contribution_review(self, receipt_id: str) -> ReviewReceipt:
        """
        Merge an existing provider review through the provider API.

        Reviewers may instead merge in the native web UI.  This method remains
        for the authenticated legacy promote endpoint and automation.
        """
        target = self.primary
        if target is None or not target.token:
            raise StorageWriteError("NO_PRIMARY_TARGET")
        key = review_key_for(receipt_id)
        branch = review_branch_for(receipt_id)
        title = review_title_for(receipt_id)
        async with self._locks[target.id]:
            review = await self._discover_review_target(
                target, branch=branch, key=key, title=title
            )
            if review is None:
                raise StorageWriteError("REVIEW_NOT_FOUND")
            if review.status == "merged":
                return review
            if review.status in {"closed", "rejected"}:
                raise StorageWriteError("REVIEW_CLOSED")
            await self._merge_review_target(target, review)
            merged = await self._discover_review_target(
                target, branch=branch, key=key, title=title
            )
            if merged is None:
                raise StorageWriteError("REVIEW_MERGE_CONFIRM", transient=True)
            if merged.status != "merged":
                raise StorageWriteError("REVIEW_MERGE_PENDING", transient=True)
            await self._delete_review_branch(target, branch)
            return merged

    async def _open_review_target(
        self,
        target: StorageTarget,
        *,
        branch: str,
        key: str,
        title: str,
        path: str,
        record_id: str,
        content: bytes,
        message: str,
    ) -> ReviewReceipt:
        existing = await self._discover_review_target(
            target, branch=branch, key=key, title=title
        )
        if existing is not None:
            return replace(existing, record_id=record_id, path=path)
        if target.provider == "huggingface":
            return await self._open_hf_review(
                target, branch, key, title, path, record_id, content, message
            )
        if target.provider == "github":
            return await self._open_github_review(
                target, branch, key, title, path, record_id, content, message
            )
        if target.provider == "gitlab":
            return await self._open_gitlab_review(
                target, branch, key, title, path, record_id, content, message
            )
        return await self._open_bitbucket_review(
            target, branch, key, title, path, record_id, content, message
        )

    async def _discover_review_target(
        self,
        target: StorageTarget,
        *,
        branch: str,
        key: str,
        title: str,
    ) -> ReviewReceipt | None:
        if target.provider == "huggingface":
            return await self._discover_hf_review(target, branch, key, title)
        if target.provider == "github":
            return await self._discover_github_review(target, branch, key, title)
        if target.provider == "gitlab":
            return await self._discover_gitlab_review(target, branch, key, title)
        return await self._discover_bitbucket_review(target, branch, key, title)

    @staticmethod
    def _review_receipt(
        target: StorageTarget,
        *,
        branch: str,
        key: str,
        review_id: Any,
        review_url: Any,
        status: str,
        record_id: str = "",
        path: str = "",
    ) -> ReviewReceipt:
        return ReviewReceipt(
            provider=target.provider,
            target_id=target.id,
            repo=target.repo,
            base_branch=target.branch,
            review_branch=branch,
            review_key=key,
            review_id=str(review_id or ""),
            review_url=str(review_url or "")[:2048],
            status=str(status or "unknown").lower(),
            record_id=record_id,
            path=path,
        )

    async def _open_hf_review(  # ruff: ignore[too-many-positional-arguments]
        self, target, branch, key, title, path, record_id, content, message
    ):
        try:
            from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

            api = HfApi(token=target.token)
            info = await asyncio.to_thread(
                _with_bounded_hf_client,
                lambda: api.create_commit(
                    repo_id=target.repo,
                    repo_type="dataset",
                    revision=target.branch,
                    operations=[
                        CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
                    ],
                    commit_message=title,
                    commit_description=message,
                    create_pr=True,
                ),
            )
            url = str(getattr(info, "pr_url", "") or "")
            match = re.search(r"/(?:discussions|pulls?)/(\d+)(?:[/?#]|$)", url)
            rid = match.group(1) if match else ""
            if not rid:
                found = await self._discover_hf_review(target, branch, key, title)
                if found is None:
                    raise StorageWriteError("HF_REVIEW_DISCOVERY", transient=True)
                return replace(found, record_id=record_id, path=path)
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=rid,
                review_url=url,
                status="open",
                record_id=record_id,
                path=path,
            )
        except StorageWriteError:
            raise
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise StorageWriteError(
                "HF_REVIEW_OPEN",
                transient=status is None or status in _TRANSIENT_STATUS,
            ) from exc

    async def _discover_hf_review(self, target, branch, key, title):
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415

            api = HfApi(token=target.token)

            def _scan():
                out = []
                for i, item in enumerate(
                    api.get_repo_discussions(
                        target.repo, repo_type="dataset", discussion_type="pull_request"
                    )
                ):
                    if i >= 100:  # ruff: ignore[magic-value-comparison]
                        break
                    out.append(item)
                return out

            items = await asyncio.to_thread(_with_bounded_hf_client, _scan)
            for item in items:
                if getattr(item, "title", "") == title and bool(
                    getattr(item, "is_pull_request", False)
                ):
                    num = int(getattr(item, "num", 0) or 0)
                    url = f"https://huggingface.co/datasets/{target.repo}/discussions/{num}"
                    return self._review_receipt(
                        target,
                        branch=branch,
                        key=key,
                        review_id=num,
                        review_url=url,
                        status=str(getattr(item, "status", "unknown")),
                    )
            return None
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise StorageWriteError(
                "HF_REVIEW_LOOKUP",
                transient=status is None or status in _TRANSIENT_STATUS,
            ) from exc

    async def _open_github_review(  # ruff: ignore[too-many-positional-arguments]
        self, target, branch, key, title, path, record_id, content, message
    ):
        owner, repo = _repo_parts(target.repo)
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        ref_url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/git/ref/heads/{quote(target.branch, safe='')}"
        status, data = await self._request_bounded_json(
            "GET", ref_url, headers=headers, timeout=15.0
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise StorageWriteError(
                "GITHUB_REVIEW_BASE", transient=status in _TRANSIENT_STATUS
            )
        sha = str(((data or {}).get("object") or {}).get("sha") or "")
        if not sha:
            raise StorageWriteError("GITHUB_REVIEW_BASE_SHA")
        create_ref = (
            f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/git/refs"
        )
        status = await self._request_no_body(
            "POST",
            create_ref,
            headers=headers,
            json={"ref": f"refs/heads/{branch}", "sha": sha},
            timeout=15.0,
        )
        if status not in {201, 422}:
            raise StorageWriteError(
                "GITHUB_REVIEW_BRANCH", transient=status in _TRANSIENT_STATUS
            )
        await self._write_github(replace(target, branch=branch), path, content, message)
        pulls = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/pulls"
        status, data = await self._request_bounded_json(
            "POST",
            pulls,
            headers=headers,
            json={
                "title": title,
                "head": branch,
                "base": target.branch,
                "body": (
                    "Automated dataset contribution review. Merge to make the record training-eligible; close to reject."
                ),
            },
            timeout=20.0,
        )
        if status == 201:  # ruff: ignore[magic-value-comparison]
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=(data or {}).get("number"),
                review_url=(data or {}).get("html_url"),
                status="open",
                record_id=record_id,
                path=path,
            )
        if status in {409, 422}:
            found = await self._discover_github_review(target, branch, key, title)
            if found is not None:
                return replace(found, record_id=record_id, path=path)
        raise StorageWriteError(
            "GITHUB_REVIEW_OPEN", transient=status in _TRANSIENT_STATUS
        )

    async def _discover_github_review(self, target, branch, key, title):
        owner, repo = _repo_parts(target.repo)
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/pulls"
        status, data = await self._request_bounded_json(
            "GET",
            url,
            headers=headers,
            params={
                "state": "all",
                "head": f"{owner}:{branch}",
                "base": target.branch,
                "per_page": 10,
            },
            timeout=15.0,
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise StorageWriteError(
                "GITHUB_REVIEW_LOOKUP", transient=status in _TRANSIENT_STATUS
            )
        for item in data if isinstance(data, list) else []:
            if str((item.get("head") or {}).get("ref") or "") != branch:
                continue
            state = (
                "merged"
                if item.get("merged_at")
                else (
                    "draft"
                    if item.get("draft") and item.get("state") == "open"
                    else str(item.get("state") or "unknown")
                )
            )
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=item.get("number"),
                review_url=item.get("html_url"),
                status=state,
            )
        return None

    async def _open_gitlab_review(  # ruff: ignore[too-many-positional-arguments]
        self, target, branch, key, title, path, record_id, content, message
    ):
        base = target.api_base or "https://gitlab.com/api/v4"
        project = quote(target.repo, safe="")
        headers = {"PRIVATE-TOKEN": target.token}
        branch_url = f"{base}/projects/{project}/repository/branches"
        status = await self._request_no_body(
            "POST",
            branch_url,
            headers=headers,
            params={"branch": branch, "ref": target.branch},
            timeout=15.0,
        )
        if status not in {201, 400}:
            raise StorageWriteError(
                "GITLAB_REVIEW_BRANCH", transient=status in _TRANSIENT_STATUS
            )
        await self._write_gitlab(replace(target, branch=branch), path, content, message)
        mr_url = f"{base}/projects/{project}/merge_requests"
        status, data = await self._request_bounded_json(
            "POST",
            mr_url,
            headers=headers,
            json={
                "source_branch": branch,
                "target_branch": target.branch,
                "title": title,
                "description": (
                    "Automated dataset contribution review. Merge to make the record training-eligible; close to reject."
                ),
                "remove_source_branch": True,
            },
            timeout=20.0,
        )
        if status == 201:  # ruff: ignore[magic-value-comparison]
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=(data or {}).get("iid"),
                review_url=(data or {}).get("web_url"),
                status="open",
                record_id=record_id,
                path=path,
            )
        if status in {400, 409}:
            found = await self._discover_gitlab_review(target, branch, key, title)
            if found is not None:
                return replace(found, record_id=record_id, path=path)
        raise StorageWriteError(
            "GITLAB_REVIEW_OPEN", transient=status in _TRANSIENT_STATUS
        )

    async def _discover_gitlab_review(self, target, branch, key, title):
        base = target.api_base or "https://gitlab.com/api/v4"
        project = quote(target.repo, safe="")
        headers = {"PRIVATE-TOKEN": target.token}
        url = f"{base}/projects/{project}/merge_requests"
        status, data = await self._request_bounded_json(
            "GET",
            url,
            headers=headers,
            params={
                "scope": "all",
                "state": "all",
                "source_branch": branch,
                "target_branch": target.branch,
                "per_page": 20,
            },
            timeout=15.0,
        )
        if status != 200:  # ruff: ignore[magic-value-comparison]
            raise StorageWriteError(
                "GITLAB_REVIEW_LOOKUP", transient=status in _TRANSIENT_STATUS
            )
        for item in data if isinstance(data, list) else []:
            if str(item.get("source_branch") or "") != branch:
                continue
            state = str(item.get("state") or "unknown")
            if state == "opened":
                state = "open"
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=item.get("iid"),
                review_url=item.get("web_url"),
                status=state,
            )
        return None

    async def _open_bitbucket_review(  # ruff: ignore[too-many-positional-arguments]
        self, target, branch, key, title, path, record_id, content, message
    ):
        workspace, repo = _repo_parts(target.repo)
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/json",
        }
        branch_url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/refs/branches"
        status = await self._request_no_body(
            "POST",
            branch_url,
            headers=headers,
            json={"name": branch, "target": {"hash": target.branch}},
            timeout=15.0,
        )
        if status not in {201, 400}:
            raise StorageWriteError(
                "BITBUCKET_REVIEW_BRANCH", transient=status in _TRANSIENT_STATUS
            )
        await self._write_bitbucket(
            replace(target, branch=branch), path, content, message
        )
        pr_url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/pullrequests"
        status, data = await self._request_bounded_json(
            "POST",
            pr_url,
            headers=headers,
            json={
                "title": title,
                "source": {"branch": {"name": branch}},
                "destination": {"branch": {"name": target.branch}},
                "close_source_branch": True,
                "description": (
                    "Automated dataset contribution review. Merge to make the record training-eligible; decline to reject."
                ),
            },
            timeout=20.0,
        )
        if status == 201:  # ruff: ignore[magic-value-comparison]
            html_url = (((data or {}).get("links") or {}).get("html") or {}).get(
                "href"
            ) or ""
            return self._review_receipt(
                target,
                branch=branch,
                key=key,
                review_id=(data or {}).get("id"),
                review_url=html_url,
                status="open",
                record_id=record_id,
                path=path,
            )
        if status in {400, 409}:
            found = await self._discover_bitbucket_review(target, branch, key, title)
            if found is not None:
                return replace(found, record_id=record_id, path=path)
        raise StorageWriteError(
            "BITBUCKET_REVIEW_OPEN", transient=status in _TRANSIENT_STATUS
        )

    async def _discover_bitbucket_review(self, target, branch, key, title):
        workspace, repo = _repo_parts(target.repo)
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/json",
        }
        url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/pullrequests"
        query = f'source.branch.name="{branch}"'
        for state in ("OPEN", "MERGED", "DECLINED", "SUPERSEDED"):
            status, data = await self._request_bounded_json(
                "GET",
                url,
                headers=headers,
                params={"state": state, "q": query, "pagelen": 10},
                timeout=15.0,
            )
            if status != 200:  # ruff: ignore[magic-value-comparison]
                raise StorageWriteError(
                    "BITBUCKET_REVIEW_LOOKUP", transient=status in _TRANSIENT_STATUS
                )
            for item in (
                (data or {}).get("values", []) if isinstance(data, dict) else []
            ):
                if (
                    str(
                        ((item.get("source") or {}).get("branch") or {}).get("name")
                        or ""
                    )
                    != branch
                ):
                    continue
                raw = str(item.get("state") or state).upper()
                mapped = {
                    "OPEN": "open",
                    "MERGED": "merged",
                    "DECLINED": "closed",
                    "SUPERSEDED": "closed",
                }.get(raw, "unknown")
                html_url = ((item.get("links") or {}).get("html") or {}).get(
                    "href"
                ) or ""
                return self._review_receipt(
                    target,
                    branch=branch,
                    key=key,
                    review_id=item.get("id"),
                    review_url=html_url,
                    status=mapped,
                )
        return None

    async def _close_review_target(
        self, target: StorageTarget, review: ReviewReceipt
    ) -> None:
        if target.provider == "huggingface":
            try:
                from huggingface_hub import HfApi  # noqa: PLC0415

                api = HfApi(token=target.token)
                await asyncio.to_thread(
                    _with_bounded_hf_client,
                    lambda: api.change_discussion_status(
                        target.repo,
                        int(review.review_id),
                        "closed",
                        repo_type="dataset",
                    ),
                )
                return
            except Exception as exc:  # noqa: BLE001
                raise StorageWriteError("HF_REVIEW_CLOSE", transient=True) from exc
        if target.provider == "github":
            owner, repo = _repo_parts(target.repo)
            headers = {
                "Authorization": f"Bearer {target.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/pulls/{quote(review.review_id)}"
            status = await self._request_no_body(
                "PATCH", url, headers=headers, json={"state": "closed"}, timeout=15.0
            )
        elif target.provider == "gitlab":
            base = target.api_base or "https://gitlab.com/api/v4"
            project = quote(target.repo, safe="")
            headers = {"PRIVATE-TOKEN": target.token}
            url = f"{base}/projects/{project}/merge_requests/{quote(review.review_id)}"
            status = await self._request_no_body(
                "PUT", url, headers=headers, json={"state_event": "close"}, timeout=15.0
            )
        else:
            workspace, repo = _repo_parts(target.repo)
            headers = {"Authorization": f"Bearer {target.token}"}
            url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/pullrequests/{quote(review.review_id)}/decline"
            status = await self._request_no_body(
                "POST", url, headers=headers, timeout=15.0
            )
        if status not in {200, 201, 204}:
            raise StorageWriteError(
                f"{target.provider.upper()}_REVIEW_CLOSE",
                transient=status in _TRANSIENT_STATUS,
            )

    async def _merge_review_target(
        self, target: StorageTarget, review: ReviewReceipt
    ) -> None:
        if target.provider == "huggingface":
            try:
                from huggingface_hub import HfApi  # noqa: PLC0415

                api = HfApi(token=target.token)
                await asyncio.to_thread(
                    _with_bounded_hf_client,
                    lambda: api.merge_pull_request(
                        target.repo, int(review.review_id), repo_type="dataset"
                    ),
                )
                return
            except Exception as exc:  # noqa: BLE001
                raise StorageWriteError("HF_REVIEW_MERGE", transient=True) from exc
        if target.provider == "github":
            owner, repo = _repo_parts(target.repo)
            headers = {
                "Authorization": f"Bearer {target.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/pulls/{quote(review.review_id)}/merge"
            status = await self._request_no_body(
                "PUT",
                url,
                headers=headers,
                json={
                    "commit_title": f"Merge dataset contribution {review.review_key}"
                },
                timeout=20.0,
            )
        elif target.provider == "gitlab":
            base = target.api_base or "https://gitlab.com/api/v4"
            project = quote(target.repo, safe="")
            headers = {"PRIVATE-TOKEN": target.token}
            url = f"{base}/projects/{project}/merge_requests/{quote(review.review_id)}/merge"
            status = await self._request_no_body(
                "PUT", url, headers=headers, timeout=20.0
            )
        else:
            workspace, repo = _repo_parts(target.repo)
            headers = {"Authorization": f"Bearer {target.token}"}
            url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/pullrequests/{quote(review.review_id)}/merge"
            status = await self._request_no_body(
                "POST", url, headers=headers, timeout=20.0
            )
        if status not in {200, 201, 202}:
            raise StorageWriteError(
                f"{target.provider.upper()}_REVIEW_MERGE",
                transient=status in _TRANSIENT_STATUS,
            )

    async def _delete_review_branch(self, target: StorageTarget, branch: str) -> None:
        try:
            if target.provider == "huggingface":
                # HF pull requests use refs/pr/* rather than ordinary source branches.
                return
            if target.provider == "github":
                owner, repo = _repo_parts(target.repo)
                headers = {
                    "Authorization": f"Bearer {target.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                }
                url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/git/refs/heads/{quote(branch, safe='')}"
            elif target.provider == "gitlab":
                base = target.api_base or "https://gitlab.com/api/v4"
                project = quote(target.repo, safe="")
                headers = {"PRIVATE-TOKEN": target.token}
                url = f"{base}/projects/{project}/repository/branches/{quote(branch, safe='')}"
            else:
                workspace, repo = _repo_parts(target.repo)
                headers = {"Authorization": f"Bearer {target.token}"}
                url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/refs/branches/{quote(branch, safe='')}"
            status = await self._request_no_body(
                "DELETE", url, headers=headers, timeout=15.0
            )
            if status not in {200, 204, 404}:
                raise StorageWriteError(
                    f"{target.provider.upper()}_REVIEW_BRANCH_DELETE",
                    transient=status in _TRANSIENT_STATUS,
                )
        except StorageWriteError:
            raise
        except Exception:  # ruff: ignore[blind-except]
            # Source-branch cleanup is hygiene, not review-state authority.
            return

    async def remove_current_view(
        self,
        paths: dict[str, str],
        *,
        record_id: str | None = None,
        commit_message: str = "Withdraw reviewed contribution from current branch view",
    ) -> dict[str, str]:
        """
        Best-effort remove previously written record files from current branches.

        This operation intentionally returns per-target status and never claims
        physical erasure.  All bundled providers are versioned repositories; a
        deletion commit removes the current branch view while prior Git/provider
        history may retain the original bytes.
        """
        results: dict[str, str] = {}
        if record_id:
            self._suppressed_retry_record_ids.add(str(record_id))
        for target in self.targets:
            path = str((paths or {}).get(target.id) or "")
            if not path:
                results[target.id] = "unknown-path"
                continue
            try:
                results[target.id] = await self._delete_target_current_view(
                    target, path, commit_message
                )
            except StorageWriteError:
                results[target.id] = "degraded"
        return results

    async def _delete_target_current_view(
        self, target: StorageTarget, path: str, message: str
    ) -> str:
        if not target.token:
            raise StorageWriteError("MISSING_TOKEN")
        if (
            target.provider == "huggingface"
            and _normalize_token_type(target.token_type) == "read"
        ):
            raise StorageWriteError("READ_TOKEN")
        async with self._locks[target.id]:
            if target.provider == "huggingface":
                return await self._delete_hf(target, path, message)
            if target.provider == "github":
                return await self._delete_github(target, path, message)
            if target.provider == "gitlab":
                return await self._delete_gitlab(target, path, message)
            return await self._delete_bitbucket(target, path, message)

    async def _delete_hf(self, target: StorageTarget, path: str, message: str) -> str:
        try:
            from huggingface_hub import CommitOperationDelete, HfApi  # noqa: PLC0415

            api = HfApi(token=target.token)
            await asyncio.to_thread(
                _with_bounded_hf_client,
                lambda: api.create_commit(
                    repo_id=target.repo,
                    repo_type="dataset",
                    revision=target.branch,
                    operations=[CommitOperationDelete(path_in_repo=path)],
                    commit_message=message,
                ),
            )
            return "removed-current-view"
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:  # ruff: ignore[magic-value-comparison]
                return "already-absent"
            raise StorageWriteError(
                "HF_DELETE", transient=status in _TRANSIENT_STATUS
            ) from exc

    async def _delete_github(
        self, target: StorageTarget, path: str, message: str
    ) -> str:
        owner, repo = _repo_parts(target.repo)
        url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/contents/{quote(path, safe='/')}"
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        current_status, current_json = await self._request_bounded_json(
            "GET", url, headers=headers, params={"ref": target.branch}, timeout=15.0
        )
        if current_status == 404:  # ruff: ignore[magic-value-comparison]
            return "already-absent"
        if current_status != 200:  # ruff: ignore[magic-value-comparison]
            raise StorageWriteError(
                "GITHUB_DELETE_LOOKUP", transient=current_status in _TRANSIENT_STATUS
            )
        sha = str((current_json or {}).get("sha") or "")
        if not sha:
            raise StorageWriteError("GITHUB_DELETE_SHA")
        status = await self._request_no_body(
            "DELETE",
            url,
            headers=headers,
            json={"message": message, "sha": sha, "branch": target.branch},
            timeout=20.0,
        )
        if status in {200, 204}:
            return "removed-current-view"
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return "already-absent"
        raise StorageWriteError("GITHUB_DELETE", transient=status in _TRANSIENT_STATUS)

    async def _delete_gitlab(
        self, target: StorageTarget, path: str, message: str
    ) -> str:
        base = target.api_base or "https://gitlab.com/api/v4"
        project = quote(target.repo, safe="")
        file_path = quote(path, safe="")
        url = f"{base}/projects/{project}/repository/files/{file_path}"
        headers = {"PRIVATE-TOKEN": target.token}
        status = await self._request_no_body(
            "DELETE",
            url,
            headers=headers,
            json={"branch": target.branch, "commit_message": message},
            timeout=20.0,
        )
        if status in {200, 204}:
            return "removed-current-view"
        if status == 404:  # ruff: ignore[magic-value-comparison]
            return "already-absent"
        raise StorageWriteError("GITLAB_DELETE", transient=status in _TRANSIENT_STATUS)

    async def _delete_bitbucket(
        self, target: StorageTarget, path: str, message: str
    ) -> str:
        workspace, repo = _repo_parts(target.repo)
        read_url = (
            f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}"
            f"/src/{quote(target.branch, safe='')}/{quote(path, safe='/')}"
        )
        headers = {"Authorization": f"Bearer {target.token}"}
        current_status = await self._request_no_body(
            "GET", read_url, headers=headers, timeout=15.0
        )
        if current_status == 404:  # ruff: ignore[magic-value-comparison]
            return "already-absent"
        if current_status != 200:  # ruff: ignore[magic-value-comparison]
            raise StorageWriteError(
                "BITBUCKET_DELETE_LOOKUP", transient=current_status in _TRANSIENT_STATUS
            )
        url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/src"
        form = urlencode(
            [("branch", target.branch), ("message", message), ("files", "/" + path)]
        )
        delete_headers = {
            **headers,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        status = await self._request_no_body(
            "POST",
            url,
            headers=delete_headers,
            content=form.encode("utf-8"),
            timeout=25.0,
        )
        if status in {200, 201}:
            return "removed-current-view"
        raise StorageWriteError(
            "BITBUCKET_DELETE", transient=status in _TRANSIENT_STATUS
        )

    async def _dispatch(
        self, target: StorageTarget, path: str, content: bytes, message: str
    ) -> None:
        try:
            if target.provider == "huggingface":
                await self._write_hf(target, path, content, message)
            elif target.provider == "github":
                await self._write_github(target, path, content, message)
            elif target.provider == "gitlab":
                await self._write_gitlab(target, path, content, message)
            else:
                await self._write_bitbucket(target, path, content, message)
        except StorageWriteError:
            raise
        except (httpx.HTTPError, OSError, TimeoutError) as exc:
            # A transport failure can happen after the provider accepted the
            # mutation but before this process received the response. Treat the
            # outcome as ambiguous so the contribution lifecycle fails safe to
            # reconciliation instead of reopening quarantine/re-promotion.
            raise StorageWriteError(
                f"{target.provider.upper()}_TRANSPORT", transient=True
            ) from exc

    async def _write_hf(
        self, target: StorageTarget, path: str, content: bytes, message: str
    ) -> None:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

            api = HfApi(token=target.token)
            await asyncio.to_thread(
                _with_bounded_hf_client,
                lambda: api.create_commit(
                    repo_id=target.repo,
                    repo_type="dataset",
                    revision=target.branch,
                    operations=[
                        CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
                    ],
                    commit_message=message,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # Keep the exception private.  HTTP permission/transient distinction is
            # intentionally coarse here because huggingface_hub exception classes
            # differ across supported versions.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            text = str(status or "")
            # No response status means the commit outcome is unknown (timeout,
            # connection reset, provider client transport failure). Conservatively
            # classify that as ambiguous/transient rather than retry-safe failure.
            transient = status is None or text in {
                "408",
                "409",
                "425",
                "429",
                "500",
                "502",
                "503",
                "504",
            }
            raise StorageWriteError("HF_WRITE", transient=transient) from exc

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise StorageWriteError("NO_HTTP_CLIENT", transient=True)
        return self.client

    async def _request_no_body(self, method: str, url: str, **kwargs: Any) -> int:
        client = self._client()
        request = client.build_request(method, url, **kwargs)
        response = await client.send(request, stream=True)
        try:
            return response.status_code
        finally:
            await response.aclose()

    async def _request_bounded_json(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[int, Any]:
        client = self._client()
        request = client.build_request(method, url, **kwargs)
        response = await client.send(request, stream=True)
        try:
            limit = _control_response_limit()
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise StorageWriteError("PROVIDER_RESPONSE_TOO_LARGE")
            buf = bytearray()
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > limit:
                    raise StorageWriteError("PROVIDER_RESPONSE_TOO_LARGE")
            if not buf:
                payload = {}
            else:
                try:
                    payload = json.loads(bytes(buf))
                except Exception as exc:
                    raise StorageWriteError("PROVIDER_RESPONSE_JSON") from exc
            return response.status_code, payload
        finally:
            await response.aclose()

    async def _write_github(
        self, target: StorageTarget, path: str, content: bytes, message: str
    ) -> None:
        owner, repo = _repo_parts(target.repo)
        url = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}/contents/{quote(path, safe='/')}"
        headers = {
            "Authorization": f"Bearer {target.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        payload = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": target.branch,
        }
        status = await self._request_no_body(
            "PUT", url, headers=headers, json=payload, timeout=20.0
        )
        if status in {200, 201}:
            return
        # A retry may encounter an already-created idempotent path.  Confirm
        # content equality before treating the conflict as success.
        if status in {409, 422}:
            get_status, get_json = await self._request_bounded_json(
                "GET", url, headers=headers, params={"ref": target.branch}, timeout=15.0
            )
            if get_status == 200:  # ruff: ignore[magic-value-comparison]
                try:
                    existing = base64.b64decode(
                        (get_json.get("content") or "").replace("\n", "")
                    )
                    if existing == content:
                        return
                except Exception:  # ruff: ignore[blind-except]
                    pass
        raise StorageWriteError("GITHUB_WRITE", transient=status in _TRANSIENT_STATUS)

    async def _write_gitlab(
        self, target: StorageTarget, path: str, content: bytes, message: str
    ) -> None:
        base = target.api_base or "https://gitlab.com/api/v4"
        project = quote(target.repo, safe="")
        file_path = quote(path, safe="")
        url = f"{base}/projects/{project}/repository/files/{file_path}"
        headers = {"PRIVATE-TOKEN": target.token}
        payload = {
            "branch": target.branch,
            "commit_message": message,
            "content": content.decode("utf-8"),
        }
        status = await self._request_no_body(
            "POST", url, headers=headers, json=payload, timeout=20.0
        )
        if status in {200, 201}:
            return
        if status == 400:  # ruff: ignore[magic-value-comparison]
            get_status, get_json = await self._request_bounded_json(
                "GET", url, headers=headers, params={"ref": target.branch}, timeout=15.0
            )
            if get_status == 200:  # ruff: ignore[magic-value-comparison]
                try:
                    if (
                        get_json.get("content_sha256")
                        == hashlib.sha256(content).hexdigest()
                    ):
                        return
                except Exception:  # ruff: ignore[blind-except]
                    pass
        raise StorageWriteError("GITLAB_WRITE", transient=status in _TRANSIENT_STATUS)

    async def _write_bitbucket(
        self, target: StorageTarget, path: str, content: bytes, message: str
    ) -> None:
        workspace, repo = _repo_parts(target.repo)
        url = f"https://api.bitbucket.org/2.0/repositories/{quote(workspace)}/{quote(repo)}/src"
        headers = {"Authorization": f"Bearer {target.token}"}
        files = {"/" + path: (path.rsplit("/", 1)[-1], content, "application/x-ndjson")}
        data = {"branch": target.branch, "message": message}
        status = await self._request_no_body(
            "POST", url, headers=headers, data=data, files=files, timeout=25.0
        )
        if status in {200, 201}:
            return
        raise StorageWriteError(
            "BITBUCKET_WRITE", transient=status in _TRANSIENT_STATUS
        )
