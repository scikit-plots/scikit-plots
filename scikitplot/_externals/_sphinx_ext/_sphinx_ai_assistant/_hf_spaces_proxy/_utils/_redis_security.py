"""
Redis transport-security policy shared by all proxy control planes.

Connection URLs are credentials/configuration, never diagnostics.  This module
validates them without returning/logging their authority component and applies
one TLS policy consistently to rate limiting, Global Share, and contribution
lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


class RedisSecurityError(RuntimeError):
    """Stable, non-sensitive Redis configuration error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class RedisTransportPolicy:
    """Validated non-secret transport properties."""

    tls: bool
    scheme: str
    database: int

    def manifest(self) -> dict[str, Any]:
        return {
            "transport": "tls_verified" if self.tls else "plaintext",
            "tls": self.tls,
            "certificate_verification": "required" if self.tls else "not_applicable",
        }


def validate_redis_url(url: str, *, require_tls: bool = False) -> RedisTransportPolicy:
    """
    Validate a Redis URL without externalizing credentials or host details.

    Query parameters are intentionally rejected.  Redis-py accepts TLS controls
    such as ``ssl_cert_reqs=none`` through URL queries; allowing caller-provided
    query policy would make a deployment-wide ``require_tls`` setting
    downgradeable from the URL itself.  Database selection belongs in the path.
    """
    raw = str(url or "").strip()
    if not raw:
        raise RedisSecurityError("REDIS_URL_REQUIRED")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise RedisSecurityError("REDIS_URL_INVALID") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"redis", "rediss"}:
        raise RedisSecurityError("REDIS_SCHEME_UNSUPPORTED")
    if not parsed.hostname:
        raise RedisSecurityError("REDIS_HOST_REQUIRED")
    if parsed.fragment:
        raise RedisSecurityError("REDIS_FRAGMENT_FORBIDDEN")
    if parsed.query:
        raise RedisSecurityError("REDIS_QUERY_FORBIDDEN")
    if parsed.path in {"", "/"}:
        database = 0
    else:
        text = parsed.path[1:] if parsed.path.startswith("/") else parsed.path
        if not text.isdigit() or not (
            0 <= int(text) <= 2_147_483_647  # ruff: ignore[magic-value-comparison]
        ):
            raise RedisSecurityError("REDIS_DATABASE_INVALID")
        database = int(text)
    tls = scheme == "rediss"
    if require_tls and not tls:
        raise RedisSecurityError("REDIS_TLS_REQUIRED")
    return RedisTransportPolicy(tls=tls, scheme=scheme, database=database)


def redis_connection_kwargs(
    url: str,
    *,
    require_tls: bool,
    socket_timeout_seconds: float,
) -> tuple[RedisTransportPolicy, dict[str, Any]]:
    """Return validated non-secret policy plus hardened redis-py kwargs."""
    policy = validate_redis_url(url, require_tls=require_tls)
    timeout = max(0.25, min(float(socket_timeout_seconds), 10.0))
    kwargs: dict[str, Any] = {
        "decode_responses": False,
        "socket_connect_timeout": timeout,
        "socket_timeout": timeout,
        "health_check_interval": 30,
    }
    if policy.tls:
        # Never inherit a URL-supplied certificate downgrade.  Query strings are
        # rejected above and verification is explicitly required here.
        kwargs.update(ssl_cert_reqs="required", ssl_check_hostname=True)
    return policy, kwargs


__all__ = [
    "RedisSecurityError",
    "RedisTransportPolicy",
    "redis_connection_kwargs",
    "validate_redis_url",
]
