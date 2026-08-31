"""
Rate-limit control plane for the HF proxy.

The default backend is deliberately process-local and is an abuse gate only.
Operators that need one quota decision shared by multiple proxy replicas may
select the optional ``redis`` backend.  Redis mode uses one atomic server-side
Lua operation per request and HMACs the client identity before it leaves the
process, so raw IP-like identifiers are never stored as Redis keys.

The Redis guarantee is scoped to one Redis consistency domain.  This module does
not claim billing/accounting correctness across independent Redis deployments,
Active-Active conflict domains, or a gateway that bypasses this service.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Any

from ._redis_security import RedisSecurityError, redis_connection_kwargs


class RateLimitBackendError(RuntimeError):
    """Stable, non-sensitive rate-limit backend error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_REDIS_FIXED_WINDOW_LUA = r"""
local key = KEYS[1]
local window_seconds = tonumber(ARGV[1])
local current = redis.call('INCR', key)
if current == 1 then
  redis.call('EXPIRE', key, window_seconds)
end
local ttl = redis.call('TTL', key)
return { current, ttl }
""".strip()


def _safe_component(value: str, fallback: str = "generic") -> str:
    out = "".join(ch for ch in str(value or "").lower() if ch.isalnum() or ch in "_-:")
    return out[:64] or fallback


class RedisRateLimiter:
    """Shared fixed-window limiter backed by Redis atomic scripting."""

    backend = "redis"
    shared = True
    authoritative = True
    consistency_scope = "single_redis_consistency_domain"

    def __init__(
        self,
        url: str,
        *,
        identity_secret: str,
        key_prefix: str = "sphinx-ai-assistant",
        socket_timeout_seconds: float = 2.0,
        client: Any | None = None,
        require_tls: bool = False,
    ) -> None:
        if not str(url or "").strip():
            raise RateLimitBackendError("REDIS_URL_REQUIRED")
        if len(str(identity_secret or "").encode("utf-8")) < (
            32  # ruff: ignore[magic-value-comparison]
        ):
            raise RateLimitBackendError("IDENTITY_SECRET_TOO_SHORT")
        self.url = str(url).strip()
        self.require_tls = bool(require_tls)
        try:
            self._transport, self._connection_kwargs = redis_connection_kwargs(
                self.url,
                require_tls=self.require_tls,
                socket_timeout_seconds=socket_timeout_seconds,
            )
        except RedisSecurityError as exc:
            raise RateLimitBackendError(exc.code) from exc
        self._secret = str(identity_secret).encode("utf-8")
        self.key_prefix = _safe_component(key_prefix, "sphinx-ai-assistant")
        self.socket_timeout_seconds = max(
            0.25, min(float(socket_timeout_seconds), 10.0)
        )
        self._client = client
        self._owns_client = client is None
        self._init_lock = asyncio.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "shared": self.shared,
            "authoritative": self.authoritative,
            "consistency_scope": self.consistency_scope,
            "identity_externalized": "hmac_sha256",
            **self._transport.manifest(),
        }

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._client is None:
                try:
                    import redis.asyncio as redis_async  # type: ignore[import-not-found]  # ruff: ignore[import-outside-top-level]
                except Exception as exc:  # pragma: no cover - deployment dependency
                    raise RateLimitBackendError("REDIS_DEPENDENCY_UNAVAILABLE") from exc
                self._client = redis_async.from_url(self.url, **self._connection_kwargs)
            try:
                await self._client.ping()
            except Exception as exc:
                raise RateLimitBackendError("REDIS_UNAVAILABLE") from exc

    async def close(self) -> None:
        if self._client is None or not self._owns_client:
            return
        closer = getattr(self._client, "aclose", None)
        if closer is None:
            closer = getattr(self._client, "close", None)
        if closer is not None:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        self._client = None

    def _identity_key(self, identity: str, scope: str) -> str:
        digest = hmac.new(
            self._secret, str(identity or "unknown").encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{self.key_prefix}:rl:{_safe_component(scope)}:{digest}"

    async def consume(
        self,
        identity: str,
        *,
        scope: str,
        limit: int,
        window_seconds: int = 3600,
    ) -> tuple[bool, int, int]:
        if self._client is None:
            raise RateLimitBackendError("REDIS_NOT_INITIALIZED")
        bounded_limit = max(1, min(int(limit), 1_000_000))
        bounded_window = max(1, min(int(window_seconds), 86_400))
        key = self._identity_key(identity, scope)
        try:
            result = await self._client.eval(
                _REDIS_FIXED_WINDOW_LUA, 1, key, bounded_window
            )
            count = int(result[0])
            ttl = int(result[1])
        except Exception as exc:
            raise RateLimitBackendError("REDIS_CONSUME_FAILED") from exc
        retry_after = max(1, ttl if ttl > 0 else bounded_window)
        return count <= bounded_limit, count, retry_after


__all__ = ["_REDIS_FIXED_WINDOW_LUA", "RateLimitBackendError", "RedisRateLimiter"]
