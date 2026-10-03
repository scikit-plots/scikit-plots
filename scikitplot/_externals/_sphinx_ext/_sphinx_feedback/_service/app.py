"""
Dependency-free ASGI adapter for the generic page-feedback service.

This module intentionally does not depend on FastAPI/Starlette. It is small enough to
run directly under any ASGI server while the provider-neutral service remains reusable
inside an existing application. The adapter stores no page-view data and keeps abuse
control identity separate from durable feedback events.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from .._contracts import (
    FeedbackConflictError,
    FeedbackValidationError,
    decode_feedback_request,
    feedback_request_hash,
)
from ._config import FeedbackServiceConfig, load_service_config
from ._core import FeedbackServiceUnavailable, PageFeedbackService

_ENDPOINT = "/v1/feedback"
_WINDOW_SECONDS = 3600.0
_MAX_ORIGINS = 64
_MAX_RATE_IDENTITIES = 10_000
_MAX_RETRY_KEYS = 50_000
_MAX_SAME_REQUEST_RETRIES = 3
_MAX_REQUEST_FRAMES = 256
_MAX_TRUSTED_PROXY_CIDRS = 64
_MAX_FORWARDED_HOPS = 32
_MAX_SECURITY_HEADER_BYTES = 8192
_RATE_PRUNE_INTERVAL_SECONDS = 60.0


class FeedbackASGIConfigError(ValueError):
    """Raised for unsafe standalone-adapter configuration."""


def _origin(value: str) -> str:
    text = str(value or "").strip()
    _len = len(text) > 2048  # ruff: ignore[magic-value-comparison]
    if (
        not text
        or _len
        or any(
            ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
            for ch in text
        )
    ):
        raise FeedbackASGIConfigError(
            "feedback allowed origin is invalid",
        )
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise FeedbackASGIConfigError(
            "feedback allowed origin is invalid",
        ) from exc
    host = (parsed.hostname or "").lower()
    local_http = parsed.scheme == "http" and host in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not local_http:
        raise FeedbackASGIConfigError(
            "feedback allowed origins must use HTTPS or localhost HTTP",
        )
    if (
        not host
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise FeedbackASGIConfigError(
            "feedback allowed origin must contain only scheme and authority",
        )
    if parsed.scheme == "https" and port not in (None, 443):
        raise FeedbackASGIConfigError(
            "feedback allowed HTTPS origins must use port 443",
        )
    authority = host
    if ":" in host and not host.startswith("["):
        authority = f"[{host}]"
    if (
        port is not None
        and not (
            parsed.scheme == "https"  # lint
            and port == 443  # ruff: ignore[magic-value-comparison]
        )
        and not (
            parsed.scheme == "http"  # lint
            and port == 80  # ruff: ignore[magic-value-comparison]
        )
    ):
        authority += f":{port}"
    return f"{parsed.scheme}://{authority}"


def parse_allowed_origins(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize an optional exact CORS allowlist; blank means no CORS headers."""
    if value in (None, ""):
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 0:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise FeedbackASGIConfigError(
            "feedback allowed origins must be a string or sequence",
        )
    if len(items) > _MAX_ORIGINS:
        raise FeedbackASGIConfigError(
            "feedback allowed origins contains too many entries",
        )
    result: list[str] = []
    for item in items:
        normalized = _origin(str(item))
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _singleton_header(scope: dict[str, Any], name: str) -> str:
    values: list[str] = []
    expected = name.lower()
    headers = scope.get("headers") or []
    if not isinstance(headers, (list, tuple)):
        raise FeedbackASGIConfigError("ASGI headers must be a sequence of byte pairs")
    for item in headers:
        _len = len(item) != 2  # ruff: ignore[magic-value-comparison]
        if not isinstance(item, (list, tuple)) or _len:
            raise FeedbackASGIConfigError("ASGI headers must contain byte pairs")
        key, value = item
        if not isinstance(key, bytes) or not isinstance(value, bytes):
            raise FeedbackASGIConfigError("ASGI header names and values must be bytes")
        if (
            len(key) > 256  # ruff: ignore[magic-value-comparison]
            or len(value) > _MAX_SECURITY_HEADER_BYTES
            or any(ch in key for ch in (0, 10, 13))
            or any(ch in value for ch in (0, 10, 13))
        ):
            raise FeedbackASGIConfigError("ASGI header framing is invalid")
        header_name = key.decode("latin-1").lower()
        header_value = value.decode("latin-1")
        if header_name == expected:
            values.append(header_value)
    if len(values) > 1:
        raise FeedbackASGIConfigError(f"duplicate {name} header is not allowed")
    return values[0] if values else ""


def parse_trusted_proxy_cidrs(  # ruff: ignore[too-many-branches]
    value: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """
    Return canonical trusted-proxy CIDRs for X-Forwarded-For processing.

    Forwarded identity is ignored unless the direct ASGI peer belongs to one of
    these explicitly trusted networks.  Networks must already be canonical (host
    bits are rejected), which makes deployment policy deterministic and auditable.
    """
    if value in (None, ""):
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 0:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise FeedbackASGIConfigError(
            "trusted proxy CIDRs must be a string or sequence",
        )
    if not items or len(items) > _MAX_TRUSTED_PROXY_CIDRS:
        raise FeedbackASGIConfigError(
            "trusted proxy CIDRs must contain 1..64 entries",
        )
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise FeedbackASGIConfigError(
                "trusted proxy CIDR entries must be strings",
            )
        text = item.strip()
        _len = len(text) > 128  # ruff: ignore[magic-value-comparison]
        if (
            not text
            or _len
            or any(
                ord(ch) < 33 or ord(ch) > 126  # ruff: ignore[magic-value-comparison]
                for ch in text
            )
        ):
            raise FeedbackASGIConfigError("trusted proxy CIDR is invalid")
        try:
            network = ipaddress.ip_network(text, strict=True)
        except ValueError as exc:
            raise FeedbackASGIConfigError("trusted proxy CIDR is invalid") from exc
        if network.prefixlen == 0:
            raise FeedbackASGIConfigError(
                "trusted proxy CIDR must not trust an entire address family"
            )
        canonical = network.with_prefixlen
        if canonical != text:
            raise FeedbackASGIConfigError(
                "trusted proxy CIDR must already be canonical",
            )
        if canonical not in result:
            result.append(canonical)

    # Reject a collectively universal trust policy too. A literal /0 is not the
    # only way to trust an entire address family: two complementary /1 networks
    # (or several narrower networks) can collapse to the same /0. Treat that as
    # the same unsafe deployment mistake instead of letting spelling bypass the
    # guard above. Keep IPv4 and IPv6 separate because collapse_addresses does
    # not accept mixed address families.
    for version in (4, 6):
        networks = [
            ipaddress.ip_network(item, strict=True)
            for item in result
            if ipaddress.ip_network(item, strict=True).version == version
        ]
        if networks and any(
            network.prefixlen == 0 for network in ipaddress.collapse_addresses(networks)
        ):
            raise FeedbackASGIConfigError(
                "trusted proxy CIDRs must not collectively trust an entire address family"
            )
    return tuple(result)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class FeedbackASGIApp:
    """Minimal standalone ASGI transport for :class:`PageFeedbackService`."""

    def __init__(
        self,
        config: FeedbackServiceConfig,
        *,
        allowed_origins: Sequence[str] = (),
        service: PageFeedbackService | None = None,
        trusted_proxy_cidrs: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.service = service or PageFeedbackService(config)
        self.allowed_origins = parse_allowed_origins(allowed_origins)
        self.trusted_proxy_cidrs = parse_trusted_proxy_cidrs(trusted_proxy_cidrs)
        self._trusted_proxy_networks = tuple(
            ipaddress.ip_network(item, strict=True) for item in self.trusted_proxy_cidrs
        )
        self._rate_secret = secrets.token_bytes(32)
        self._rate: dict[str, tuple[int, float]] = {}
        self._rate_retries: dict[tuple[str, str], tuple[int, float]] = {}
        self._rate_lock = asyncio.Lock()
        self._rate_last_prune = 0.0

    def _cors_origin(self, scope: dict[str, Any]) -> str:
        origin = _singleton_header(scope, "origin").strip()
        if not origin or not self.allowed_origins:
            return ""
        try:
            normalized = _origin(origin)
        except FeedbackASGIConfigError:
            return "!"
        return normalized if normalized in self.allowed_origins else "!"

    def _is_trusted_proxy(
        self,
        address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        return any(address in network for network in self._trusted_proxy_networks)

    def _rate_address(  # ruff: ignore[too-many-return-statements]
        self,
        scope: dict[str, Any],
    ) -> str:
        """
        Resolve the abuse-control address without trusting caller headers.

        Direct peers are authoritative by default.  ``X-Forwarded-For`` is used
        only when the immediate peer is explicitly trusted, and the chain is
        walked from right to left until the first untrusted hop.  Malformed
        forwarding metadata falls back to the direct peer, which can over-limit
        but cannot create attacker-controlled identities.
        """
        client = scope.get("client")
        if not isinstance(client, (list, tuple)) or not client:
            return "unknown"
        try:
            peer = ipaddress.ip_address(str(client[0] or ""))
        except ValueError:
            return "unknown"
        if not self._trusted_proxy_networks or not self._is_trusted_proxy(peer):
            return peer.compressed
        try:
            forwarded = _singleton_header(scope, "x-forwarded-for").strip()
        except FeedbackASGIConfigError:
            return peer.compressed
        if not forwarded:
            return peer.compressed
        parts = [part.strip() for part in forwarded.split(",")]
        if (
            not parts
            or len(parts) > _MAX_FORWARDED_HOPS
            or any(not part for part in parts)
        ):
            return peer.compressed
        try:
            chain = [ipaddress.ip_address(part) for part in parts]
        except ValueError:
            return peer.compressed
        current = peer
        for candidate in reversed(chain):
            if not self._is_trusted_proxy(current):
                break
            current = candidate
        return current.compressed

    def _rate_identity(self, scope: dict[str, Any]) -> str:
        address = self._rate_address(scope)
        message = f"page-feedback-asgi:v1:submit\0{address}".encode("utf-8", "replace")
        return hmac.new(self._rate_secret, message, hashlib.sha256).hexdigest()

    def _prune_rate_locked(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._rate_last_prune < _RATE_PRUNE_INTERVAL_SECONDS:
            return
        stale = [
            key
            for key, (_count, start) in self._rate.items()
            if now - start >= _WINDOW_SECONDS
        ]
        for key in stale:
            self._rate.pop(key, None)
        stale_retries = [
            key
            for key, (_count, start) in self._rate_retries.items()
            if now - start >= _WINDOW_SECONDS
        ]
        for key in stale_retries:
            self._rate_retries.pop(key, None)
        self._rate_last_prune = now

    async def _consume_rate(self, identity: str, request_hash: str) -> bool:
        """
        Admit a new event or a bounded exact retry without tracking people.

        The main quota counts distinct request commitments. Once a commitment has
        been admitted, up to ``_MAX_SAME_REQUEST_RETRIES`` exact retries may pass
        without consuming another new-event slot. This preserves broken-pipe
        recovery even at a 1/hour quota while keeping retries bounded. The cache
        contains only the HMAC abuse identity plus a request commitment.
        """
        now = time.monotonic()
        async with self._rate_lock:
            self._prune_rate_locked(now)

            retry_key = (identity, request_hash)
            retry = self._rate_retries.get(retry_key)
            if retry is not None:
                retry_count, retry_start = retry
                if retry_count >= _MAX_SAME_REQUEST_RETRIES:
                    return False
                self._rate_retries[retry_key] = (retry_count + 1, retry_start)
                return True

            if (
                identity not in self._rate and len(self._rate) >= _MAX_RATE_IDENTITIES
            ) or len(self._rate_retries) >= _MAX_RETRY_KEYS:
                self._prune_rate_locked(now, force=True)
            if identity not in self._rate and len(self._rate) >= _MAX_RATE_IDENTITIES:
                return False
            if len(self._rate_retries) >= _MAX_RETRY_KEYS:
                # Fail closed rather than allowing attacker-controlled commitment
                # cardinality to grow process memory without bound.
                return False

            count, start = self._rate.get(identity, (0, now))
            if now - start >= _WINDOW_SECONDS:
                count, start = 0, now
            if count >= self.config.rate_limit_per_hour:
                return False
            self._rate[identity] = (count + 1, start)
            self._rate_retries[retry_key] = (0, now)
            return True

    async def _send_json(
        self,
        send,
        status: int,
        payload: dict[str, Any],
        *,
        cors_origin: str = "",
        extra_headers: Sequence[tuple[bytes, bytes]] = (),
    ) -> None:
        body = (
            b""
            if status == 204  # ruff: ignore[magic-value-comparison]
            else _json_bytes(payload)
        )
        headers = [
            (b"cache-control", b"no-store"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-content-type-options", b"nosniff"),
        ]
        if status != 204:  # ruff: ignore[magic-value-comparison]
            headers.insert(0, (b"content-type", b"application/json; charset=utf-8"))
        if cors_origin and cors_origin != "!":
            headers.extend(
                [
                    (b"access-control-allow-origin", cors_origin.encode("ascii")),
                    (b"vary", b"Origin"),
                ]
            )
        headers.extend(extra_headers)
        try:
            await send(
                {"type": "http.response.start", "status": status, "headers": headers}
            )
            await send({"type": "http.response.body", "body": body})
        except OSError:
            # ASGI servers are allowed to raise OSError when the client disconnects
            # after a durable write. The browser retry uses the same event id, so
            # suppressing this transport-only broken pipe preserves exact replay.
            return

    async def _read_body(self, receive) -> bytes:
        body = bytearray()
        messages = 0
        while True:
            message = await receive()
            messages += 1
            if messages > _MAX_REQUEST_FRAMES:
                raise OverflowError("request is too fragmented")
            if message.get("type") == "http.disconnect":
                raise ConnectionError("client disconnected")
            if message.get("type") != "http.request":
                continue
            raw_chunk = message.get("body", b"")
            if not isinstance(raw_chunk, bytes):
                raise ValueError(  # ruff: ignore[type-check-without-type-error]
                    "request body frame must contain bytes",
                )
            if len(body) + len(raw_chunk) > self.config.max_body_bytes:
                raise OverflowError("request too large")
            body.extend(raw_chunk)
            if not message.get("more_body", False):
                return bytes(body)

    async def __call__(  # ruff: ignore[too-many-branches, too-many-return-statements]
        self,
        scope,
        receive,
        send,
    ) -> None:
        if scope.get("type") != "http":
            return
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "GET").upper()
        try:
            # These headers are security-relevant singletons. Ambiguous duplicates
            # are rejected instead of relying on first/last-value proxy behavior.
            _singleton_header(scope, "content-type")
            declared_length = _singleton_header(scope, "content-length")
            cors_origin = self._cors_origin(scope)
        except FeedbackASGIConfigError:
            await self._send_json(send, 400, {"detail": "Ambiguous request headers."})
            return
        if cors_origin == "!":
            await self._send_json(
                send, 403, {"detail": "Origin is not allowed for page feedback."}
            )
            return
        if path != _ENDPOINT:
            await self._send_json(
                send,
                404,
                {"detail": "Not found."},
                cors_origin=cors_origin,
            )
            return
        if method == "OPTIONS":
            if not cors_origin:
                await self._send_json(
                    send,
                    204,
                    {},
                    cors_origin="",
                )
                return
            await self._send_json(
                send,
                204,
                {},
                cors_origin=cors_origin,
                extra_headers=(
                    (b"access-control-allow-methods", b"POST, OPTIONS"),
                    (b"access-control-allow-headers", b"Content-Type, Accept"),
                    (b"access-control-max-age", b"600"),
                ),
            )
            return
        if method != "POST":
            await self._send_json(
                send, 405, {"detail": "Method not allowed."}, cors_origin=cors_origin
            )
            return
        declared_bytes: int | None = None
        if declared_length:
            if not declared_length.isascii() or not declared_length.isdecimal():
                await self._send_json(
                    send,
                    400,
                    {"detail": "Invalid Content-Length."},
                    cors_origin=cors_origin,
                )
                return
            try:
                declared_bytes = int(declared_length, 10)
            except ValueError:
                await self._send_json(
                    send,
                    400,
                    {"detail": "Invalid Content-Length."},
                    cors_origin=cors_origin,
                )
                return
            if declared_bytes < 0:
                await self._send_json(
                    send,
                    400,
                    {"detail": "Invalid Content-Length."},
                    cors_origin=cors_origin,
                )
                return
            if declared_bytes > self.config.max_body_bytes:
                await self._send_json(
                    send,
                    413,
                    {
                        "detail": (
                            "Page feedback request exceeds the configured safety limit."
                        ),
                    },
                    cors_origin=cors_origin,
                )
                return
        media_type = (
            _singleton_header(
                scope,
                "content-type",
            )
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if media_type != "application/json":
            await self._send_json(
                send,
                415,
                {"detail": "Page feedback requires application/json."},
                cors_origin=cors_origin,
            )
            return
        try:
            raw = await self._read_body(receive)
        except OverflowError:
            await self._send_json(
                send,
                413,
                {
                    "detail": (
                        "Page feedback request exceeds the configured safety limit."
                    ),
                },
                cors_origin=cors_origin,
            )
            return
        except ConnectionError:
            return
        except ValueError:
            await self._send_json(
                send,
                400,
                {"detail": "Malformed page feedback request framing."},
                cors_origin=cors_origin,
            )
            return
        if declared_bytes is not None and declared_bytes != len(raw):
            await self._send_json(
                send,
                400,
                {"detail": "Content-Length does not match the request body."},
                cors_origin=cors_origin,
            )
            return
        try:
            validated = decode_feedback_request(raw)
            validated = self.service.validate_request(validated)
        except FeedbackValidationError as exc:
            await self._send_json(
                send, 422, {"detail": str(exc)}, cors_origin=cors_origin
            )
            return
        request_hash = feedback_request_hash(validated)
        if not await self._consume_rate(self._rate_identity(scope), request_hash):
            await self._send_json(
                send,
                429,
                {"detail": "Rate limit exceeded for page feedback submissions."},
                cors_origin=cors_origin,
                extra_headers=((b"retry-after", b"3600"),),
            )
            return
        try:
            receipt = await self.service.submit(validated)
        except FeedbackConflictError as exc:
            await self._send_json(
                send, 409, {"detail": str(exc)}, cors_origin=cors_origin
            )
            return
        except FeedbackValidationError as exc:
            await self._send_json(
                send, 422, {"detail": str(exc)}, cors_origin=cors_origin
            )
            return
        except FeedbackServiceUnavailable:
            await self._send_json(
                send,
                503,
                {"detail": "Page feedback storage is temporarily unavailable."},
                cors_origin=cors_origin,
            )
            return
        await self._send_json(send, 202, receipt, cors_origin=cors_origin)


def create_app(
    env: Mapping[str, str] | None = None,
    *,
    allowed_origins: str | Sequence[str] | None = None,
) -> FeedbackASGIApp:
    """Create a standalone ASGI app from explicit/process environment config."""
    source = os.environ if env is None else env
    config = load_service_config(source)
    origins = (
        source.get("FEEDBACK_ALLOWED_ORIGINS", "")
        if allowed_origins is None
        else allowed_origins
    )
    trusted_proxies = parse_trusted_proxy_cidrs(
        source.get("FEEDBACK_TRUSTED_PROXY_CIDRS", "")
    )
    return FeedbackASGIApp(
        config,
        allowed_origins=parse_allowed_origins(origins),
        service=PageFeedbackService(config, credential_env=source),
        trusted_proxy_cidrs=trusted_proxies,
    )


app = create_app()

__all__ = [
    "FeedbackASGIApp",
    "FeedbackASGIConfigError",
    "app",
    "create_app",
    "parse_allowed_origins",
    "parse_trusted_proxy_cidrs",
]
